/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "scan_io.hpp"
#include "stages.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/column/column.hpp>
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
#include <string>
#include <thread>
#include <vector>

namespace cudf_polars_cucascade {

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
   * @param input_columns Names of the decoded columns the stages see, in that order (empty: the
   * decoder's order).
   */
  probe_pipeline(scan_io& io,
                 std::vector<std::vector<stage>> branches,
                 std::vector<cudf::table_view> builds,
                 std::vector<std::vector<int>> build_keys,
                 int threads,
                 std::vector<std::string> input_columns = {});
  ~probe_pipeline();

  probe_pipeline(probe_pipeline const&)            = delete;
  probe_pipeline& operator=(probe_pipeline const&) = delete;

  /** @brief Queue a planned task (decoded and released by the pipeline). */
  void submit(std::int64_t task_id);

  /** @brief No more tasks: wait for the workers; returns each branch's final aggregate. */
  std::vector<std::unique_ptr<cudf::table>> finish();

  /** @brief Streamed mode (one branch, no group-by): no more tasks will be submitted. */
  void close();

  /**
   * @brief Streamed mode: the next task's output and its submission index, or index -1 once
   * every task is done. Blocks; a full output queue holds the workers back.
   */
  std::pair<std::int64_t, std::unique_ptr<cudf::table>> next_output();

  /**
   * @brief Counters: tasks, decode s, join s, compute s, groupby s, fold s, wall s of the first
   * task start to the last task end, joins, rows in, rows out of the first branch's joins.
   */
  [[nodiscard]] std::vector<double> stats() const;

 private:
  void worker_main(int index);
  void run_task(std::int64_t id, std::int64_t seq, ::cuda::stream_ref stream);
  void fold(std::size_t branch, std::unique_ptr<cudf::table> partial, ::cuda::stream_ref stream);
  void add_counters(stage_counters const& c);

  scan_io& _io;
  std::vector<std::vector<stage>> _branches;
  std::vector<build_table> _builds;
  std::vector<std::string> _input_columns;
  std::vector<cudaStream_t> _streams;  // one per worker, from a process-wide pool
  std::vector<std::thread> _workers;

  std::mutex _mutex;
  std::condition_variable _cv;
  std::deque<std::pair<std::int64_t, std::int64_t>> _queue;  // (task id, submission index)
  std::int64_t _next_seq{0};
  bool _closed{false};
  std::exception_ptr _error;

  // streamed mode: each task's output, in completion order
  bool _streamed{false};
  std::size_t _out_cap{0};
  int _running{0};
  bool _cancel{false};
  std::condition_variable _out_cv;
  std::deque<std::pair<std::int64_t, std::unique_ptr<cudf::table>>> _out;

  // per branch: folded partial aggregates (key columns, then one sum per value column)
  std::vector<std::mutex> _fold_mutex;
  std::vector<std::vector<std::unique_ptr<cudf::table>>> _partials;

  mutable std::mutex _stats_mutex;
  std::vector<double> _stats;
  double _t_first{-1.0};
  double _t_last{0.0};
};

}  // namespace cudf_polars_cucascade
