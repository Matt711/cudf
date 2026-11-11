/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "native_actors.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <cudf_streaming/channel_metadata.hpp>
#include <cudf_streaming/table_chunk.hpp>

#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/fanout.hpp>
#include <rapidsmpf/streaming/core/lineariser.hpp>
#include <rapidsmpf/streaming/core/queue.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <optional>
#include <thread>

namespace cudf_polars_cucascade::actor {

namespace rs = rapidsmpf::streaming;
using cudf_streaming::table_chunk;

namespace {

// CUDF_POLARS_NATIVE_DEBUG=1: one stderr line per actor phase (to locate a stall).
bool debug_enabled()
{
  static bool const enabled = [] {
    auto const* v = std::getenv("CUDF_POLARS_NATIVE_DEBUG");
    return v != nullptr && std::string{v} == "1";
  }();
  return enabled;
}

int next_actor_id()
{
  static std::atomic<int> id{0};
  return id++;
}

template <typename... Args>
void debug(int actor, char const* what, Args... args)
{
  if (!debug_enabled()) { return; }
  std::fprintf(stderr, "[native %d] ", actor);
  std::fprintf(stderr, what, args...);
  std::fputc('\n', stderr);
}

// How long a producer waiting on the scan service sleeps between checks (after yielding its
// executor thread), so waiting costs neither a thread nor a spinning core.
constexpr auto poll_interval = std::chrono::microseconds(100);

std::unique_ptr<cudf::table> empty_table(std::vector<int> const& types)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  for (auto t : types) {
    cols.push_back(cudf::make_empty_column(cudf::data_type{static_cast<cudf::type_id>(t)}));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

// ---------------------------------------------------------------- scan source

struct planned_task {
  std::int64_t id{-1};
  bool empty{false};
  std::size_t staged{0};
  std::size_t decoded{0};
  bool consumed{false};
};

struct scan_state {
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<std::optional<planned_task>> plans;
  std::size_t queued{0};  // staged bytes planned but not yet decoded
  bool stop{false};
  std::exception_ptr error;
};

void plan_tasks(scan_io* io,
                std::int64_t stream_id,
                std::vector<scan_task> const& tasks,
                cudf::io::parquet_reader_options const* options,
                bool prune_with_stats,
                std::size_t merge_gap,
                std::size_t max_ahead_bytes,
                scan_state& st)
{
  auto const stream = cudf::get_default_stream();
  for (std::size_t i = 0; i < tasks.size(); ++i) {
    {
      std::unique_lock lock(st.mutex);
      st.cv.wait(lock, [&] { return st.stop || st.queued < max_ahead_bytes; });
      if (st.stop) { return; }
    }
    planned_task p;
    try {
      auto const r = io->plan_task(stream_id,
                                   tasks[i].paths,
                                   tasks[i].split_index,
                                   tasks[i].total_splits,
                                   *options,
                                   prune_with_stats,
                                   merge_gap,
                                   stream.get());
      CUDF_EXPECTS(!r.unsupported && !r.too_big,
                   "native scan: a task the native planner cannot plan");
      p = {r.id, r.empty, r.staged_bytes, r.decoded_bytes, false};
    } catch (...) {
      std::lock_guard lock(st.mutex);
      st.error = std::current_exception();
      st.stop  = true;
      st.cv.notify_all();
      return;
    }
    {
      std::lock_guard lock(st.mutex);
      st.plans[i] = p;
      st.queued += p.staged;
    }
    st.cv.notify_all();
  }
}

rs::Actor produce(std::shared_ptr<rs::Context> ctx,
                  std::shared_ptr<rs::BoundedQueue> queue,
                  scan_io* io,
                  std::shared_ptr<scan_state> st,
                  std::size_t first,
                  std::size_t step,
                  std::vector<std::string> const& columns,
                  std::vector<int> const& column_types)
{
  co_await ctx->executor()->schedule();
  auto const n = st->plans.size();
  for (auto i = first; i < n; i += step) {
    // Wait for the task to be planned, then for its bytes, without holding an executor thread.
    planned_task p;
    while (true) {
      {
        std::lock_guard lock(st->mutex);
        if (st->error) { std::rethrow_exception(st->error); }
        if (st->plans[i].has_value()) {
          p = *st->plans[i];
          break;
        }
      }
      co_await ctx->executor()->yield();
      std::this_thread::sleep_for(poll_interval);
    }
    while (!p.empty && !io->ready(p.id)) {
      co_await ctx->executor()->yield();
      std::this_thread::sleep_for(poll_interval);
    }
    auto ticket = co_await queue->acquire();
    if (!ticket.has_value()) { break; }  // the output is shut down
    co_await ctx->executor()->schedule();
    auto const stream = ctx->br()->stream_pool()->get_stream();
    std::unique_ptr<cudf::table> table;
    if (p.empty) {
      table = empty_table(column_types);
    } else {
      auto reservation =
        ctx->br()->reserve_device_memory_and_spill(p.decoded, rapidsmpf::AllowOverbooking::YES);
      auto decoded = io->materialize(p.id, stream.get());
      io->release(p.id);
      {
        std::lock_guard lock(st->mutex);
        st->plans[i]->consumed = true;
        st->queued -= p.staged;
      }
      st->cv.notify_all();
      // The decoder returns the file's column order: put the columns in the scan's.
      auto const& info = decoded.metadata.schema_info;
      auto owned       = decoded.tbl->release();
      std::vector<std::unique_ptr<cudf::column>> cols;
      for (auto const& name : columns) {
        auto it = std::find_if(
          info.begin(), info.end(), [&](auto const& c) { return c.name == name; });
        CUDF_EXPECTS(it != info.end(), "native scan: decoded table has no column " + name);
        cols.push_back(std::move(owned[static_cast<std::size_t>(it - info.begin())]));
      }
      table = std::make_unique<cudf::table>(std::move(cols));
    }
    if (!co_await ticket->send(cudf_streaming::to_message(
          i, std::make_unique<table_chunk>(std::move(table), stream)))) {
      break;
    }
  }
  co_await queue->drain(ctx->executor());
}

// ---------------------------------------------------------------- chain

rs::Actor collect(std::shared_ptr<rs::Context> ctx,
                  std::shared_ptr<rs::Channel> ch,
                  std::vector<table_chunk>& chunks)
{
  co_await ctx->executor()->schedule();
  co_await ch->receive_metadata();
  while (true) {
    auto msg = co_await ch->receive();
    if (msg.empty()) { break; }
    auto chunk = msg.release<table_chunk>();
    chunks.push_back(co_await chunk.make_available(ctx));
  }
}

struct chain_state {
  std::vector<build_table> builds;
  std::mutex mutex;
  std::vector<std::unique_ptr<cudf::table>> partials;
  // ordered sends: chunks are numbered as they are received and sent in that order
  bool ordered{false};
  std::uint64_t next_received{0};
  std::atomic<std::uint64_t> next_send{0};
};

std::unique_ptr<cudf::table> concat_tables(std::vector<std::unique_ptr<cudf::table>>& tables,
                                           ::cuda::stream_ref stream)
{
  if (tables.size() == 1) { return std::move(tables.front()); }
  std::vector<cudf::table_view> views;
  for (auto const& t : tables) {
    views.push_back(t->view());
  }
  return cudf::concatenate(views, stream);
}

rs::Actor work(std::shared_ptr<rs::Context> ctx,
               std::shared_ptr<rs::Channel> ch_in,
               std::shared_ptr<rs::Channel> ch_out,
               std::vector<std::vector<stage>> const& branches,
               int fold_keys,
               std::shared_ptr<chain_state> st)
{
  constexpr std::size_t fold_every = 16;
  co_await ctx->executor()->schedule();
  while (true) {
    auto msg = co_await ch_in->receive();
    if (msg.empty()) { break; }
    std::uint64_t turn = 0;
    if (st->ordered) {
      std::lock_guard lock(st->mutex);
      turn = st->next_received++;
    }
    auto const seq = msg.sequence_number();
    auto input     = msg.release<table_chunk>();
    auto chunk     = co_await input.make_available(ctx);
    auto stream    = chunk.stream();
    std::vector<std::unique_ptr<cudf::table>> outs;
    for (auto const& stages : branches) {
      outs.push_back(run_stages(stages, chunk.table_view(), st->builds, stream));
    }
    auto out = concat_tables(outs, stream);
    if (fold_keys < 0) {
      while (st->ordered && st->next_send.load() != turn) {
        co_await ctx->executor()->yield();
      }
      auto const sent = co_await ch_out->send(
        cudf_streaming::to_message(seq, std::make_unique<table_chunk>(std::move(out), stream)));
      if (st->ordered) { st->next_send.store(turn + 1); }
      if (!sent) { break; }
      continue;
    }
    // Another worker may fold this partial on its own stream: it must be complete first.
    stream.sync();
    std::vector<std::unique_ptr<cudf::table>> batch;
    {
      std::lock_guard lock(st->mutex);
      st->partials.push_back(std::move(out));
      if (st->partials.size() >= fold_every) { batch.swap(st->partials); }
    }
    if (!batch.empty()) {
      auto all    = concat_tables(batch, stream);
      auto folded = fold_sums(all->view(), fold_keys, stream);
      stream.sync();
      std::lock_guard lock(st->mutex);
      st->partials.push_back(std::move(folded));
    }
  }
}

}  // namespace

rs::Actor scan_source(std::shared_ptr<rs::Context> ctx,
                      std::shared_ptr<rs::Channel> ch_out,
                      scan_io* io,
                      std::vector<scan_task> tasks,
                      cudf::io::parquet_reader_options const* options,
                      bool prune_with_stats,
                      std::size_t merge_gap,
                      std::size_t num_producers,
                      std::vector<std::string> columns,
                      std::vector<int> column_types,
                      std::size_t max_ahead_bytes)
{
  rs::ShutdownAtExit c{ch_out};
  auto const me = next_actor_id();
  co_await ctx->executor()->schedule();
  debug(me, "scan source: %zu tasks, sending metadata", tasks.size());
  co_await ch_out->send_metadata(cudf_streaming::to_message(
    0, std::make_unique<cudf_streaming::channel_metadata>(tasks.size())));
  debug(me, "scan source: metadata sent");
  if (tasks.empty()) {
    co_await ch_out->drain(ctx->executor());
    co_return;
  }
  auto st = std::make_shared<scan_state>();
  st->plans.resize(tasks.size());
  auto const stream_id = io->open_stream();
  std::thread planner(plan_tasks,
                      io,
                      stream_id,
                      std::cref(tasks),
                      options,
                      prune_with_stats,
                      merge_gap,
                      max_ahead_bytes,
                      std::ref(*st));
  num_producers = std::max<std::size_t>(1, std::min(num_producers, tasks.size()));
  std::exception_ptr error;
  try {
    rs::Lineariser lineariser(ctx, ch_out, num_producers);
    auto& queues = lineariser.get_queues();
    std::vector<rs::Actor> actors;
    for (std::size_t k = 0; k < num_producers; ++k) {
      actors.push_back(produce(ctx, queues[k], io, st, k, num_producers, columns, column_types));
    }
    actors.push_back(lineariser.drain());
    rs::coro_results(co_await coro::when_all(std::move(actors)));
  } catch (...) {
    error = std::current_exception();
  }
  // Stop planning, then give back every planned task no decode took (their reads may still be
  // in flight: release waits for them).
  {
    std::lock_guard lock(st->mutex);
    st->stop = true;
  }
  st->cv.notify_all();
  planner.join();
  for (auto const& p : st->plans) {
    if (p.has_value() && !p->empty && !p->consumed) { io->release(p->id); }
  }
  io->close_stream(stream_id);
  debug(me, "scan source: done%s", error ? " (error)" : "");
  if (error) { std::rethrow_exception(error); }
  co_await ch_out->drain(ctx->executor());
}

rs::Actor chain(std::shared_ptr<rs::Context> ctx,
                std::vector<std::shared_ptr<rs::Channel>> ch_builds,
                std::vector<std::vector<int>> build_keys,
                std::vector<std::vector<int>> build_types,
                std::shared_ptr<rs::Channel> ch_in,
                std::shared_ptr<rs::Channel> ch_out,
                std::vector<std::vector<stage>> branches,
                std::size_t concurrency,
                int fold_keys,
                std::vector<int> out_types,
                bool forward_metadata,
                bool ordered)
{
  std::vector<std::shared_ptr<rs::Channel>> all_channels{ch_in, ch_out};
  all_channels.insert(all_channels.end(), ch_builds.begin(), ch_builds.end());
  rs::ShutdownAtExit c{all_channels};
  auto const me = next_actor_id();
  co_await ctx->executor()->schedule();
  debug(me, "chain: collecting %zu build sides", ch_builds.size());

  // Build sides: collected concurrently (they may share upstream fanouts), then hashed.
  std::vector<std::vector<table_chunk>> build_chunks(ch_builds.size());
  {
    std::vector<rs::Actor> collectors;
    for (std::size_t i = 0; i < ch_builds.size(); ++i) {
      collectors.push_back(collect(ctx, ch_builds[i], build_chunks[i]));
    }
    rs::coro_results(co_await coro::when_all(std::move(collectors)));
  }
  auto st                 = std::make_shared<chain_state>();
  auto const build_stream = ctx->br()->stream_pool()->get_stream();
  std::vector<std::unique_ptr<cudf::table>> build_tables;
  for (std::size_t i = 0; i < build_chunks.size(); ++i) {
    std::vector<cudf::table_view> views;
    for (auto& chunk : build_chunks[i]) {
      chunk.stream().sync();
      views.push_back(chunk.table_view());
    }
    build_tables.push_back(views.empty() ? empty_table(build_types[i])
                                          : cudf::concatenate(views, build_stream));
    st->builds.push_back(make_build_table(build_tables.back()->view(), build_keys[i], build_stream));
  }
  build_stream.sync();
  build_chunks.clear();
  debug(me, "chain: builds ready, receiving input metadata");

  if (forward_metadata) {
    auto metadata = co_await ch_in->receive_metadata();
    debug(me, "chain: input metadata received");
    if (fold_keys < 0) {
      co_await ch_out->send_metadata(std::move(metadata));
    } else {
      co_await ch_out->send_metadata(cudf_streaming::to_message(
        0, std::make_unique<cudf_streaming::channel_metadata>(1)));
    }
  }
  st->ordered = ordered;
  debug(me, "chain: metadata sent, running %zu workers", std::max<std::size_t>(1, concurrency));
  {
    std::vector<rs::Actor> workers;
    for (std::size_t k = 0; k < std::max<std::size_t>(1, concurrency); ++k) {
      workers.push_back(work(ctx, ch_in, ch_out, branches, fold_keys, st));
    }
    rs::coro_results(co_await coro::when_all(std::move(workers)));
  }
  if (fold_keys >= 0) {
    auto const stream = ctx->br()->stream_pool()->get_stream();
    std::unique_ptr<cudf::table> result;
    if (st->partials.empty()) {
      result = empty_table(out_types);
    } else {
      auto all = concat_tables(st->partials, stream);
      result   = fold_sums(all->view(), fold_keys, stream);
    }
    auto cols = result->release();
    for (std::size_t i = 0; i < cols.size() && i < out_types.size(); ++i) {
      auto const type = cudf::data_type{static_cast<cudf::type_id>(out_types[i])};
      if (cols[i]->type() != type) { cols[i] = cudf::cast(cols[i]->view(), type, stream); }
    }
    co_await ch_out->send(cudf_streaming::to_message(
      0,
      std::make_unique<table_chunk>(std::make_unique<cudf::table>(std::move(cols)), stream)));
  }
  debug(me, "chain: done");
  co_await ch_out->drain(ctx->executor());
}

rs::Actor broadcast(std::shared_ptr<rs::Context> ctx,
                    std::shared_ptr<rs::Channel> ch_in,
                    std::vector<std::shared_ptr<rs::Channel>> chs_out)
{
  co_await ctx->executor()->schedule();
  {
    auto msg = co_await ch_in->receive_metadata();
    auto const metadata = msg.release<cudf_streaming::channel_metadata>();
    for (auto const& ch : chs_out) {
      co_await ch->send_metadata(cudf_streaming::to_message(
        0, std::make_unique<cudf_streaming::channel_metadata>(metadata)));
    }
  }
  co_await rs::actor::fanout(ctx, ch_in, chs_out, rs::actor::FanoutPolicy::BOUNDED);
}

}  // namespace cudf_polars_cucascade::actor
