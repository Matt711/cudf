/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "scan_io.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace cudf_polars_cucascade {

/**
 * @brief One node of a flattened expression: its operands are earlier nodes of the same list, and
 * the last node is the result.
 */
struct expr_node {
  enum kind : int { column = 0, literal = 1, binary = 2, year = 3 };
  int op{column};
  int column_index{0};  ///< column: index into the current table
  double value{0.0};    ///< literal: a float64 scalar
  int binop{0};         ///< binary: a cudf::binary_operator
  int lhs{0};           ///< binary, year: operand node
  int rhs{0};           ///< binary: operand node
};

/** @brief One step of a probe chain, applied to each decoded chunk in order. */
struct stage {
  enum kind : int { join = 0, compute = 1, select = 2, groupby = 3 };
  int op{select};
  // join: inner join against a build table; the output is the kept probe columns, then the kept
  // build columns.
  int build{0};
  std::vector<int> probe_keys;
  std::vector<int> keep_probe;
  std::vector<int> keep_build;
  // compute: one appended column per expression
  std::vector<std::vector<expr_node>> exprs;
  // select
  std::vector<int> columns;
  // groupby (last stage): sum of each value column per key
  std::vector<int> keys;
  std::vector<int> sums;
};

/**
 * @brief Runs probe chains over the decoded tasks of a scan, natively, to partial aggregates.
 *
 * Experiment (the Q7 vertical slice of a native executor): worker threads take planned tasks of
 * a @ref scan_io in submission order, decode each one, and run every branch's stages on it on the
 * worker's stream: hash-join probes against prebuilt build tables, column expressions,
 * selections, and a partial group-by whose results are folded together. No Python runs per
 * chunk; only the final aggregates come back.
 */
class probe_pipeline {
 public:
  /**
   * @param io The scan service the tasks were planned on.
   * @param branches Each branch's stages; the last must be a group-by.
   * @param builds Build tables (views must outlive the pipeline) and their key columns.
   * @param threads Worker threads.
   */
  probe_pipeline(scan_io& io,
                 std::vector<std::vector<stage>> branches,
                 std::vector<cudf::table_view> builds,
                 std::vector<std::vector<int>> build_keys,
                 int threads);
  ~probe_pipeline();

  probe_pipeline(probe_pipeline const&)            = delete;
  probe_pipeline& operator=(probe_pipeline const&) = delete;

  /** @brief Queue a planned task (decoded and released by the pipeline). */
  void submit(std::int64_t task_id);

  /** @brief No more tasks: wait for the workers; returns each branch's final aggregate. */
  std::vector<std::unique_ptr<cudf::table>> finish();

  /**
   * @brief Counters: tasks, decode s, join s, compute s, groupby s, fold s, wall s of the first
   * task start to the last task end, joins, rows in, rows out of the first branch's joins.
   */
  [[nodiscard]] std::vector<double> stats() const;

 private:
  void worker_main(int index);
  void run_task(std::int64_t id, ::cuda::stream_ref stream);
  std::unique_ptr<cudf::table> run_branch(std::vector<stage> const& stages,
                                          cudf::table_view input,
                                          ::cuda::stream_ref stream);
  void fold(std::size_t branch, std::unique_ptr<cudf::table> partial, ::cuda::stream_ref stream);
  std::unique_ptr<cudf::table> aggregate(std::size_t branch,
                                         cudf::table_view partials,
                                         ::cuda::stream_ref stream) const;

  scan_io& _io;
  std::vector<std::vector<stage>> _branches;
  std::vector<cudf::table_view> _builds;
  std::vector<std::unique_ptr<cudf::hash_join>> _joins;
  std::vector<cudaStream_t> _streams;  // one per worker, owned
  std::vector<std::thread> _workers;

  std::mutex _mutex;
  std::condition_variable _cv;
  std::deque<std::int64_t> _queue;
  bool _closed{false};
  std::exception_ptr _error;

  // per branch: folded partial aggregates (key columns, then one sum per value column)
  std::vector<std::mutex> _fold_mutex;
  std::vector<std::vector<std::unique_ptr<cudf::table>>> _partials;

  mutable std::mutex _stats_mutex;
  std::vector<double> _stats;
  double _t_first{-1.0};
  double _t_last{0.0};
};

}  // namespace cudf_polars_cucascade
