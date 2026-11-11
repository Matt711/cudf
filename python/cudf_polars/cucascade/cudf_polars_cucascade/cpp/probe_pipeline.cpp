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

cudf::table_view select(cudf::table_view t, std::vector<int> const& columns)
{
  std::vector<cudf::column_view> cols;
  cols.reserve(columns.size());
  std::string sizes;
  for (auto c : columns) {
    CUDF_EXPECTS(c >= 0 && c < t.num_columns(),
                 "probe_pipeline: column " + std::to_string(c) + " of a table with " +
                   std::to_string(t.num_columns()) + " columns");
    cols.push_back(t.column(c));
    sizes += std::to_string(c) + ":" + std::to_string(cols.back().size()) + " ";
  }
  for (auto const& col : cols) {
    CUDF_EXPECTS(col.size() == cols.front().size(),
                 "probe_pipeline: select of mismatched columns " + sizes + "of a table with " +
                   std::to_string(t.num_rows()) + " rows");
  }
  return cudf::table_view(cols);
}

// Evaluates a flattened expression against `t`; returns the result column.
std::unique_ptr<cudf::column> evaluate(std::vector<expr_node> const& nodes,
                                       cudf::table_view t,
                                       ::cuda::stream_ref stream)
{
  CUDF_EXPECTS(!nodes.empty(), "probe_pipeline: empty expression");
  // Each node's value: a column (owned or a view into `t`) or a literal.
  std::vector<std::unique_ptr<cudf::column>> owned(nodes.size());
  std::vector<cudf::column_view> views(nodes.size());
  std::vector<std::unique_ptr<cudf::numeric_scalar<double>>> scalars(nodes.size());
  auto const f64 = cudf::data_type{cudf::type_id::FLOAT64};
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    auto const& n = nodes[i];
    switch (n.op) {
      default: CUDF_FAIL("probe_pipeline: unknown expression node");
      case expr_node::column: views[i] = t.column(n.column_index); break;
      case expr_node::literal:
        scalars[i] = std::make_unique<cudf::numeric_scalar<double>>(n.value, true, stream);
        break;
      case expr_node::year:
        owned[i] = cudf::datetime::extract_datetime_component(
          views[n.lhs], cudf::datetime::datetime_component::YEAR, stream);
        views[i] = owned[i]->view();
        break;
      case expr_node::cast:
        owned[i] = cudf::cast(views[n.lhs],
                              cudf::data_type{static_cast<cudf::type_id>(n.binop)},
                              stream);
        views[i] = owned[i]->view();
        break;
      case expr_node::binary: {
        auto const l_lit = scalars[n.lhs] != nullptr;
        auto const r_lit = scalars[n.rhs] != nullptr;
        CUDF_EXPECTS(!(l_lit && r_lit), "probe_pipeline: binary of two literals");
        if (l_lit) {
          owned[i] = cudf::binary_operation(*scalars[n.lhs], views[n.rhs], static_cast<cudf::binary_operator>(n.binop), f64, stream);
        } else if (r_lit) {
          owned[i] = cudf::binary_operation(views[n.lhs], *scalars[n.rhs], static_cast<cudf::binary_operator>(n.binop), f64, stream);
        } else {
          owned[i] = cudf::binary_operation(views[n.lhs], views[n.rhs], static_cast<cudf::binary_operator>(n.binop), f64, stream);
        }
        views[i] = owned[i]->view();
        break;
      }
    }
  }
  CUDF_EXPECTS(owned.back() != nullptr, "probe_pipeline: an expression must compute a column");
  return std::move(owned.back());
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
    _builds(std::move(builds)),
    _input_columns(std::move(input_columns)),
    _fold_mutex(_branches.size()),
    _partials(_branches.size()),
    _stats(n_counters, 0.0)
{
  CUDF_EXPECTS(_builds.size() == build_keys.size(), "probe_pipeline: one key list per build");
  for (auto const& stages : _branches) {
    CUDF_EXPECTS(!stages.empty() && stages.back().op == stage::groupby,
                 "probe_pipeline: every branch must end in a group-by");
  }
  for (int i = 0; i < std::max(threads, 1); ++i) {
    cudaStream_t s = nullptr;
    CUDF_CUDA_TRY(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    _streams.push_back(s);
  }
  for (std::size_t i = 0; i < _builds.size(); ++i) {
    _joins.push_back(std::make_unique<cudf::hash_join>(select(_builds[i], build_keys[i]),
                                                       cudf::null_equality::UNEQUAL,
                                                       ::cuda::stream_ref{_streams[0]}));
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
    _queue.clear();
  }
  _cv.notify_all();
  for (auto& w : _workers) {
    if (w.joinable()) { w.join(); }
  }
  // Device memory allocated on the worker streams goes back on them: free it before they go.
  for (auto s : _streams) {
    cudaStreamSynchronize(s);
  }
  _partials.clear();
  _joins.clear();
  for (auto s : _streams) {
    cudaStreamDestroy(s);
  }
}

void probe_pipeline::submit(std::int64_t task_id)
{
  {
    std::lock_guard lock(_mutex);
    _queue.push_back(task_id);
  }
  _cv.notify_one();
}

void probe_pipeline::worker_main(int index)
{
  auto stream = ::cuda::stream_ref{_streams[index]};
  while (true) {
    std::int64_t id;
    {
      std::unique_lock lock(_mutex);
      _cv.wait(lock, [&] { return _closed || !_queue.empty(); });
      if (_queue.empty()) { return; }
      id = _queue.front();
      _queue.pop_front();
      if (_error) {
        lock.unlock();
        _io.release(id);
        continue;
      }
    }
    try {
      run_task(id, stream);
    } catch (...) {
      std::lock_guard lock(_mutex);
      if (!_error) { _error = std::current_exception(); }
    }
    _io.release(id);
  }
}

void probe_pipeline::run_task(std::int64_t id, ::cuda::stream_ref stream)
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
    input = select(input, order);
  }
  for (std::size_t b = 0; b < _branches.size(); ++b) {
    auto partial = run_branch(_branches[b], input, stream);
    fold(b, std::move(partial), stream);
  }
  std::lock_guard lock(_stats_mutex);
  _t_last = std::max(_t_last, now());
}

std::unique_ptr<cudf::table> probe_pipeline::run_branch(std::vector<stage> const& stages,
                                                        cudf::table_view input,
                                                        ::cuda::stream_ref stream)
{
  std::unique_ptr<cudf::table> current;  // owns the current view's data after the first stage
  auto view = input;
  for (auto const& s : stages) {
    auto const t0 = now();
    std::size_t c = 0;
    switch (s.op) {
      default: CUDF_FAIL("probe_pipeline: unknown stage");
      case stage::join: {
        auto const rows = view.num_rows();
        auto [probe_idx, build_idx] =
          _joins[s.build]->inner_join(select(view, s.probe_keys), std::nullopt, stream);
        auto const n = static_cast<cudf::size_type>(probe_idx->size());
        auto const i32 = cudf::data_type{cudf::type_id::INT32};
        auto const probe_map = cudf::column_view(
          i32, n, static_cast<void const*>(probe_idx->data()), nullptr, 0);
        auto const build_map = cudf::column_view(
          i32, n, static_cast<void const*>(build_idx->data()), nullptr, 0);
        auto probe_out = cudf::gather(select(view, s.keep_probe), probe_map,
                                      cudf::out_of_bounds_policy::DONT_CHECK, stream);
        auto build_out = cudf::gather(select(_builds[s.build], s.keep_build), build_map,
                                      cudf::out_of_bounds_policy::DONT_CHECK, stream);
        auto cols = probe_out->release();
        for (auto& col : build_out->release()) {
          cols.push_back(std::move(col));
        }
        current = std::make_unique<cudf::table>(std::move(cols));
        view    = current->view();
        c       = join_s;
        std::lock_guard lock(_stats_mutex);
        _stats[joins] += 1;
        if (&s == &stages.front()) {
          _stats[rows_in] += rows;
          _stats[rows_out] += n;
        }
        break;
      }
      case stage::compute: {
        std::vector<std::unique_ptr<cudf::column>> cols;
        for (auto i = 0; i < view.num_columns(); ++i) {
          cols.push_back(std::make_unique<cudf::column>(view.column(i), stream));
        }
        for (auto const& e : s.exprs) {
          cols.push_back(evaluate(e, view, stream));
        }
        current = std::make_unique<cudf::table>(std::move(cols));
        view    = current->view();
        c       = compute_s;
        break;
      }
      case stage::select: {
        view = select(view, s.columns);
        break;
      }
      case stage::groupby: {
        std::vector<cudf::groupby::aggregation_request> requests;
        for (auto v : s.sums) {
          cudf::groupby::aggregation_request r;
          r.values = view.column(v);
          r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
          requests.push_back(std::move(r));
        }
        cudf::groupby::groupby gb(select(view, s.keys));
        auto [keys, results] = gb.aggregate(requests, stream);
        auto cols = keys->release();
        for (auto& r : results) {
          cols.push_back(std::move(r.results.front()));
        }
        current = std::make_unique<cudf::table>(std::move(cols));
        view    = current->view();
        c       = groupby_s;
        break;
      }
    }
    if (c != 0) {
      std::lock_guard lock(_stats_mutex);
      _stats[c] += now() - t0;
    }
  }
  return current;
}

std::unique_ptr<cudf::table> probe_pipeline::aggregate(std::size_t branch,
                                                       cudf::table_view partials,
                                                       ::cuda::stream_ref stream) const
{
  // Partials hold the keys, then one sum per value column: summing the sums again folds them.
  auto const& last  = _branches[branch].back();
  auto const n_keys = static_cast<int>(last.keys.size());
  std::vector<int> keys(n_keys);
  for (int i = 0; i < n_keys; ++i) {
    keys[i] = i;
  }
  std::vector<cudf::groupby::aggregation_request> requests;
  for (int v = n_keys; v < partials.num_columns(); ++v) {
    cudf::groupby::aggregation_request r;
    r.values = partials.column(v);
    r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    requests.push_back(std::move(r));
  }
  cudf::groupby::groupby gb(select(partials, keys));
  auto [k, results] = gb.aggregate(requests, stream);
  auto cols = k->release();
  for (auto& r : results) {
    cols.push_back(std::move(r.results.front()));
  }
  return std::make_unique<cudf::table>(std::move(cols));
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
  auto folded = aggregate(branch, all->view(), stream);
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
    out.push_back(aggregate(b, all->view(), stream));
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
