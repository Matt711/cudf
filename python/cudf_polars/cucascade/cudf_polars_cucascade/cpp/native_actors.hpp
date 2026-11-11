/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "scan_io.hpp"
#include "stages.hpp"

#include <cudf/io/parquet.hpp>

#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

/**
 * @file
 * @brief rapidsmpf C++ actors of native pipelines: a remote scan source and a chain of stages.
 */

namespace cudf_polars_cucascade::actor {

/** @brief One task of a scan: its files, or one split of one file. */
struct scan_task {
  std::vector<std::string> paths;
  int split_index{0};
  int total_splits{1};
};

/**
 * @brief Read a remote parquet scan through the scan service and send one chunk per task.
 *
 * A planning thread plans the tasks in order in @p io, which fetches their bytes ahead of the
 * decode; at most @p max_ahead_bytes of planned reads wait for their decode. @p num_producers
 * coroutines decode the tasks once their bytes are in (without blocking an executor thread on the
 * network), reserving device memory first, and a lineariser sends the chunks in task order. Sends
 * channel metadata with one chunk per task first; an empty task sends an empty chunk of
 * @p column_types.
 *
 * @param ctx Streaming context.
 * @param ch_out Output channel.
 * @param io The scan service (outlives the actor).
 * @param tasks The scan's tasks.
 * @param options Reader options: columns and filter (outlive the actor).
 * @param prune_with_stats Whether to prune row groups with footer statistics.
 * @param merge_gap Byte gap below which neighbouring reads are merged.
 * @param num_producers Concurrent decodes.
 * @param columns Output column names, in order.
 * @param column_types Output column type ids, in order.
 * @param max_ahead_bytes Planned reads allowed to wait for their decode.
 */
rapidsmpf::streaming::Actor scan_source(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
                                        scan_io* io,
                                        std::vector<scan_task> tasks,
                                        cudf::io::parquet_reader_options const* options,
                                        bool prune_with_stats,
                                        std::size_t merge_gap,
                                        std::size_t num_producers,
                                        std::vector<std::string> columns,
                                        std::vector<int> column_types,
                                        std::size_t max_ahead_bytes);

/**
 * @brief Run branches of stages on every chunk of @p ch_in.
 *
 * First collects each build channel into one table and builds its hash table; join stages probe
 * them by index. Then @p concurrency coroutines take chunks, reserve device memory, and run every
 * branch on each (the branch outputs are concatenated: a union). With @p fold_keys < 0 each result
 * is sent with its chunk's sequence number (in arrival order when @p ordered); otherwise every
 * result holds @p fold_keys key columns then sums, and one chunk with the folded sums, cast to
 * @p out_types, is sent at the end. With @p forward_metadata false the caller has already
 * received the input's channel metadata and sent the output's.
 */
rapidsmpf::streaming::Actor chain(
  std::shared_ptr<rapidsmpf::streaming::Context> ctx,
  std::vector<std::shared_ptr<rapidsmpf::streaming::Channel>> ch_builds,
  std::vector<std::vector<int>> build_keys,
  std::vector<std::vector<int>> build_types,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
  std::vector<std::vector<stage>> branches,
  std::size_t concurrency,
  int fold_keys,
  std::vector<int> out_types,
  bool forward_metadata = true,
  bool ordered          = false);

/**
 * @brief Send every chunk of @p ch_in to each of @p chs_out, channel metadata included.
 *
 * rapidsmpf's fanout forwards data messages only; this sends a copy of the input's channel
 * metadata to every output first, then fans out the data with backpressure from the slowest
 * output.
 */
rapidsmpf::streaming::Actor broadcast(
  std::shared_ptr<rapidsmpf::streaming::Context> ctx,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
  std::vector<std::shared_ptr<rapidsmpf::streaming::Channel>> chs_out);

}  // namespace cudf_polars_cucascade::actor
