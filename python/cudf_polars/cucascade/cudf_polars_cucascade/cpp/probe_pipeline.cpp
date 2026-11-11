/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "probe_pipeline.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <tuple>
#include <string>

namespace cudf_polars_cucascade {

namespace {

double now()
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

enum counter : std::size_t {
  tasks,
  decode_s,
  join_s,
  compute_s,
  groupby_s,
  fold_s,
  wall_s,
  joins,
  rows_in,
  rows_out,
  n_counters
};

// Worker streams come from a process-wide pool that is never destroyed: tables built on them
// (streamed outputs, results) are freed on them after the pipeline is gone.
cudaStream_t pooled_stream(std::size_t i)
{
  static std::mutex mutex;
  static std::vector<cudaStream_t> pool;
  std::lock_guard lock(mutex);
  while (pool.size() <= i) {
    cudaStream_t s = nullptr;
    CUDF_CUDA_TRY(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    pool.push_back(s);
  }
  return pool[i];
}

}  // namespace

probe_pipeline::probe_pipeline(scan_io& io,
                               std::vector<std::vector<stage>> branches,
                               std::vector<cudf::table_view> builds,
                               std::vector<std::vector<int>> build_keys,
                               int threads,
                               std::vector<std::string> input_columns)
  : _io{io},
    _branches(std::move(branches)),

    _input_columns(std::move(input_columns)),
    _fold_mutex(_branches.size()),
    _partials(_branches.size()),
    _stats(n_counters, 0.0)
{
  CUDF_EXPECTS(builds.size() == build_keys.size(), "probe_pipeline: one key list per build");
  _streamed = _branches.size() == 1 && !_branches[0].empty() &&
              _branches[0].back().op != stage::groupby;
  if (!_streamed) {
    for (auto const& stages : _branches) {
      CUDF_EXPECTS(!stages.empty() && stages.back().op == stage::groupby,
                   "probe_pipeline: every branch must end in a group-by (or one branch streams)");
    }
  }
  for (int i = 0; i < std::max(threads, 1); ++i) {
    _streams.push_back(pooled_stream(static_cast<std::size_t>(i)));
  }
  _out_cap  = 2 * _streams.size();
  _running  = threads;
  for (std::size_t i = 0; i < builds.size(); ++i) {
    _builds.push_back(
      make_build_table(builds[i], build_keys[i], ::cuda::stream_ref{_streams[0]}));
  }
  CUDF_CUDA_TRY(cudaStreamSynchronize(_streams[0]));
  for (int i = 0; i < threads; ++i) {
    _workers.emplace_back(&probe_pipeline::worker_main, this, i);
  }
}

probe_pipeline::~probe_pipeline()
{
  {
    std::lock_guard lock(_mutex);
    _closed = true;
    _cancel = true;
    _queue.clear();
  }
  _cv.notify_all();
  _out_cv.notify_all();
  for (auto& w : _workers) {
    if (w.joinable()) { w.join(); }
  }
  for (auto s : _streams) {
    cudaStreamSynchronize(s);
  }
}

void probe_pipeline::submit(std::int64_t task_id)
{
  {
    std::lock_guard lock(_mutex);
    _queue.emplace_back(task_id, _next_seq++);
  }
  _cv.notify_one();
}

void probe_pipeline::close()
{
  {
    std::lock_guard lock(_mutex);
    _closed = true;
  }
  _cv.notify_all();
  _out_cv.notify_all();
}

std::pair<std::int64_t, std::unique_ptr<cudf::table>> probe_pipeline::next_output()
{
  std::unique_lock lock(_mutex);
  _out_cv.wait(lock, [&] { return !_out.empty() || _error || (_closed && _running == 0); });
  if (_error) { std::rethrow_exception(_error); }
  if (_out.empty()) { return {-1, nullptr}; }
  auto item = std::move(_out.front());
  _out.pop_front();
  lock.unlock();
  _out_cv.notify_all();
  return item;
}

void probe_pipeline::worker_main(int index)
{
  auto stream = ::cuda::stream_ref{_streams[index]};
  while (true) {
    std::int64_t id;
    std::int64_t seq;
    {
      std::unique_lock lock(_mutex);
      _cv.wait(lock, [&] { return _closed || !_queue.empty(); });
      if (_queue.empty()) {
        --_running;
        lock.unlock();
        _out_cv.notify_all();
        return;
      }
      std::tie(id, seq) = _queue.front();
      _queue.pop_front();
      if (_error) {
        lock.unlock();
        _io.release(id);
        continue;
      }
    }
    try {
      run_task(id, seq, stream);
    } catch (...) {
      {
        std::lock_guard lock(_mutex);
        if (!_error) { _error = std::current_exception(); }
      }
      _out_cv.notify_all();
    }
    _io.release(id);
  }
}

void probe_pipeline::run_task(std::int64_t id, std::int64_t seq, ::cuda::stream_ref stream)
{
  auto const t0 = now();
  auto decoded  = _io.materialize(id, stream.get());
  auto const t1 = now();
  {
    std::lock_guard lock(_stats_mutex);
    _stats[tasks] += 1;
    _stats[decode_s] += t1 - t0;
    if (_t_first < 0 || t0 < _t_first) { _t_first = t0; }
  }
  auto input = decoded.tbl->view();
  if (!_input_columns.empty()) {
    // The stages address columns by position: put the decoded ones in the order they expect.
    std::vector<int> order;
    auto const& info = decoded.metadata.schema_info;
    for (auto const& name : _input_columns) {
      auto it = std::find_if(
        info.begin(), info.end(), [&](auto const& c) { return c.name == name; });
      CUDF_EXPECTS(it != info.end(), "probe_pipeline: decoded table has no column " + name);
      order.push_back(static_cast<int>(it - info.begin()));
    }
    input = select_columns(input, order);
  }
  if (_streamed) {
    stage_counters counters;
    auto out = run_stages(_branches[0], input, _builds, stream, &counters);
    add_counters(counters);
    stream.sync();  // complete before another thread hands it on
    std::unique_lock lock(_mutex);
    _out_cv.wait(lock, [&] { return _out.size() < _out_cap || _cancel; });
    if (!_cancel) { _out.emplace_back(seq, std::move(out)); }
    lock.unlock();
    _out_cv.notify_all();
  } else {
    for (std::size_t b = 0; b < _branches.size(); ++b) {
      stage_counters counters;
      auto partial = run_stages(_branches[b], input, _builds, stream, &counters);
      add_counters(counters);
      fold(b, std::move(partial), stream);
    }
  }
  std::lock_guard lock(_stats_mutex);
  _t_last = std::max(_t_last, now());
}

void probe_pipeline::add_counters(stage_counters const& c)
{
  std::lock_guard lock(_stats_mutex);
  _stats[join_s] += c.join_s;
  _stats[compute_s] += c.compute_s;
  _stats[groupby_s] += c.groupby_s;
  _stats[joins] += c.joins;
  _stats[rows_in] += c.rows_in;
  _stats[rows_out] += c.rows_out;
}

void probe_pipeline::fold(std::size_t branch,
                          std::unique_ptr<cudf::table> partial,
                          ::cuda::stream_ref stream)
{
  // Partials are kept as they come and folded into one every few, so they stay small. Another
  // worker (on its own stream) may fold this one: it must be complete first.
  constexpr std::size_t fold_every = 16;
  stream.sync();
  std::vector<std::unique_ptr<cudf::table>> batch;
  {
    std::lock_guard lock(_fold_mutex[branch]);
    _partials[branch].push_back(std::move(partial));
    if (_partials[branch].size() < fold_every) { return; }
    batch.swap(_partials[branch]);
  }
  auto const t0 = now();
  std::vector<cudf::table_view> views;
  for (auto const& t : batch) {
    views.push_back(t->view());
  }
  auto all    = cudf::concatenate(views, stream);
  auto folded =
    fold_sums(all->view(), static_cast<int>(_branches[branch].back().keys.size()), stream);
  stream.sync();
  {
    std::lock_guard lock(_fold_mutex[branch]);
    _partials[branch].push_back(std::move(folded));
  }
  std::lock_guard lock(_stats_mutex);
  _stats[fold_s] += now() - t0;
}

std::vector<std::unique_ptr<cudf::table>> probe_pipeline::finish()
{
  {
    std::lock_guard lock(_mutex);
    _closed = true;
  }
  _cv.notify_all();
  for (auto& w : _workers) {
    w.join();
  }
  _workers.clear();
  if (_error) { std::rethrow_exception(_error); }
  if (_streamed) { return {}; }
  // The results outlive the pipeline and its streams: build them on the default stream.
  auto stream = cudf::get_default_stream();
  std::vector<std::unique_ptr<cudf::table>> out;
  for (std::size_t b = 0; b < _branches.size(); ++b) {
    std::vector<cudf::table_view> views;
    for (auto const& t : _partials[b]) {
      views.push_back(t->view());
    }
    CUDF_EXPECTS(!views.empty(), "probe_pipeline: no partial aggregates");
    auto all = cudf::concatenate(views, stream);
    out.push_back(
      fold_sums(all->view(), static_cast<int>(_branches[b].back().keys.size()), stream));
  }
  stream.sync();
  for (auto& p : _partials) {
    p.clear();
  }
  {
    std::lock_guard lock(_stats_mutex);
    _stats[wall_s] = _t_last - _t_first;
  }
  return out;
}

std::vector<double> probe_pipeline::stats() const
{
  std::lock_guard lock(_stats_mutex);
  return _stats;
}

}  // namespace cudf_polars_cucascade
