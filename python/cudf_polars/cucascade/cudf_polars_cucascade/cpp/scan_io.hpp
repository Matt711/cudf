/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cucascade/io/datasource_factory.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/types.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cudf_polars_cucascade {

/**
 * @brief Settings of a @ref scan_io.
 */
struct scan_io_config {
  std::string endpoint;  ///< `https://host` or `http://host` of the object store
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string session_token;
  std::size_t reactors{8};        ///< REST reactor threads
  std::size_t connections{32};    ///< connections per reactor
  std::size_t stall_seconds{5};   ///< a GET slower than 64 KiB/s for this long is retried
  std::size_t arena_bytes{0};     ///< pinned staging arena, pinned once at construction
  std::size_t footer_threads{32};  ///< concurrent footer requests
  /// Bytes of reads in flight on the network at once (0: no limit). Reads are issued in the order
  /// the tasks got their blocks, so with a limit of about bandwidth x latency the tasks complete
  /// roughly in order and decodes start as soon as the first one lands, instead of every task in
  /// flight sharing the bandwidth and finishing together.
  std::size_t inflight_bytes{0};
};

/**
 * @brief Remote parquet I/O for a scan: footers and column chunks of S3 objects, fetched ahead of
 * their decode into one pinned host arena with cuCascade's REST engine.
 *
 * Footers follow Sirius: one metadata job per file on a pool of threads, each one suffix-range
 * request (`Range: bytes=-N`) that returns the object size, its version tag and the footer; the
 * footer is parsed here and kept, keyed by path, until @ref clear_footers. The open object is kept
 * too, so data requests need no further round trip and are checked against the version tag.
 *
 * Data: each scan opens a stream and submits its tasks (the byte ranges one decode needs) in task
 * order. A scheduler thread gives every task a block of the arena in its stream's order and issues
 * the reads at once, straight into the block. Streams share the arena: a stream holding no block is
 * always given its next one when it fits (the others keep room for it), and beyond its first block
 * a stream keeps to an equal share. So the reads of every scan of a query run from its start, as far
 * ahead as the arena allows, and a scan whose consumer waits on another scan cannot take the room
 * that other scan needs. A decode copies its block to the device in one copy and releases it.
 */
class scan_io {
 public:
  explicit scan_io(scan_io_config const& cfg);
  ~scan_io();

  scan_io(scan_io const&)            = delete;
  scan_io& operator=(scan_io const&) = delete;
  scan_io(scan_io&&)                 = delete;
  scan_io& operator=(scan_io&&)      = delete;

  /** @brief Open connections to the object store now (one bucket URL, e.g. `s3://bucket/key`). */
  void warmup(std::string const& url);

  // ---------------------------------------------------------------- footers

  /** @brief Queue the footer jobs of the paths not already requested (non-blocking). */
  void request_footers(std::vector<std::string> const& paths);

  /**
   * @brief The footer of @p path: waits for its job (requesting it if needed). Rethrows its error.
   * @return The object size and the parsed footer (a copy).
   */
  std::pair<std::size_t, std::unique_ptr<cudf::io::parquet::FileMetaData>> footer(
    std::string const& path);

  /**
   * @brief A minimal parquet file holding only @p path's footer (`PAR1`, footer, length, `PAR1`):
   * enough for metadata readers, with no further request.
   */
  std::vector<std::uint8_t> footer_file(std::string const& path);

  /**
   * @brief Per path, from its footer (waiting for it): the bytes of the column chunks of
   * @p columns (top-level names; every column when empty), then its row-group count, flattened
   * as `{bytes_0, row_groups_0, bytes_1, ...}`.
   */
  std::vector<std::size_t> projected_bytes(std::vector<std::string> const& paths,
                                           std::vector<std::string> const& columns);

  /** @brief Forget every footer and open object (so the next query fetches them again). */
  void clear_footers();

  // ---------------------------------------------------------------- data

  [[nodiscard]] std::size_t arena_bytes() const noexcept { return _arena_bytes; }

  /** @brief Open a stream: a scan's tasks, given blocks in their own submission order. */
  std::int64_t open_stream();

  /** @brief Close a stream whose tasks have all been released. */
  void close_stream(std::int64_t stream);

  /**
   * @brief Queue the reads of one task; returns its id, or -1 if it is larger than the arena.
   *
   * Range @c i is the @c sizes[i] bytes at @c offsets[i] of @c paths[sources[i]] (paths whose
   * footers were requested). The ranges are placed back to back in the task's block, in order.
   */
  std::int64_t submit(std::int64_t stream,
                      std::vector<std::string> const& paths,
                      std::vector<std::int32_t> const& sources,
                      std::vector<std::size_t> const& offsets,
                      std::vector<std::size_t> const& sizes);

  /** @brief Whether every read of task @p id has completed (never blocks). */
  [[nodiscard]] bool ready(std::int64_t id);

  /** @brief Block until every read of task @p id has completed; rethrows a read error. */
  void wait(std::int64_t id);

  /** @brief Copy task @p id's staged bytes, back to back, to a new device buffer on @p stream. */
  [[nodiscard]] std::unique_ptr<rmm::device_buffer> to_device(std::int64_t id, cudaStream_t stream);

  /** @brief Give task @p id's block back (after the work that reads it has completed). */
  void release(std::int64_t id);

  /** @brief What @ref plan_task decided for a task. */
  struct planned_task {
    std::int64_t id{-1};       ///< task id (when neither empty nor unsupported)
    bool empty{false};         ///< no row group survives: nothing to read
    bool unsupported{false};   ///< a split this planner does not handle (plan it another way)
    bool too_big{false};       ///< larger than the arena
    std::size_t staged_bytes{0};
    std::size_t decoded_bytes{0};  ///< estimated peak device bytes of the decode
  };

  /**
   * @brief Plan a task from its footers and queue its reads, without the GIL.
   *
   * Builds the multi-file hybrid scan reader from the held footers and @p options (columns and
   * filter of the scan), takes the task's row groups (all of them, or the stride of a split of a
   * single file), prunes them with footer statistics when @p prune_with_stats and the options
   * hold a filter, computes the column-chunk byte ranges, merges neighbours closer than
   * @p merge_gap and queues the reads on @p stream_id. The reader stays with the task for
   * @ref materialize.
   */
  planned_task plan_task(std::int64_t stream_id,
                         std::vector<std::string> const& paths,
                         int split_index,
                         int total_splits,
                         cudf::io::parquet_reader_options const& options,
                         bool prune_with_stats,
                         std::size_t merge_gap,
                         cudaStream_t stream);

  /**
   * @brief Decode a planned task from its staged bytes: one copy to the device, then the hybrid
   * scan reader's `materialize_all_columns` (which applies the filter).
   */
  cudf::io::table_with_metadata materialize(std::int64_t id, cudaStream_t stream);

  /** @brief Bytes requested from the object store so far (data, not footers). */
  [[nodiscard]] std::size_t fetched_bytes() const noexcept { return _fetched_bytes; }

  /**
   * @brief A snapshot of the data path: tasks waiting for a block, bytes of reads in flight,
   * bytes of granted reads not yet issued, and free bytes of the arena.
   */
  [[nodiscard]] std::vector<std::size_t> snapshot();

 private:
  struct footer_entry {
    std::mutex mutex;
    std::condition_variable cv;
    bool done{false};
    std::exception_ptr error;
    std::shared_ptr<cucascade::io::io_object> object;
    std::unique_ptr<cudf::io::parquet::FileMetaData> metadata;
    std::vector<std::uint8_t> footer;  // the Thrift-encoded footer bytes
  };

  std::shared_ptr<footer_entry> wait_footer(std::string const& path);

  struct decode_state {
    std::unique_ptr<cudf::io::parquet::experimental::hybrid_scan_multifile> reader;
    std::vector<std::vector<cudf::size_type>> row_groups;
    cudf::io::parquet_reader_options options;
    std::vector<std::size_t> positions;  // block position of each column-chunk range
    std::vector<std::size_t> sizes;      // size of each column-chunk range
  };

  struct task {
    std::shared_ptr<decode_state> decode;  // set by plan_task
    std::int64_t stream{0};
    std::vector<std::shared_ptr<cucascade::io::io_object>> objects;
    std::vector<std::int32_t> sources;
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> sizes;
    std::size_t size{0};
    std::size_t block_offset{0};
    bool allocated{false};
    bool done{false};
    std::size_t pending{0};  // reads in flight
    std::exception_ptr error;
  };

  struct stream_state {
    std::deque<std::int64_t> queue;  // submitted, not yet given a block, in order
    std::size_t held{0};             // arena bytes of its granted, unreleased tasks
  };

  void footer_main();
  std::shared_ptr<footer_entry> footer_entry_for(std::string const& path, bool request);
  void load_footer(std::string const& path, footer_entry& entry);

  struct read_batch {
    std::size_t file;
    std::vector<cucascade::io::slice> slices;
    std::size_t bytes;
  };
  struct queued_batch {
    std::int64_t id;
    std::shared_ptr<task> t;
    read_batch batch;
  };
  /// Bytes per vectored read call (the engine spreads one call over at most two reactors).
  static constexpr std::size_t batch_bytes = 64UL << 20;
  [[nodiscard]] std::vector<read_batch> plan_reads(task const& t) const;

  std::int64_t enqueue(std::shared_ptr<task> t);  // -1 if larger than the arena
  void scheduler_main();
  [[nodiscard]] std::pair<std::int64_t, std::shared_ptr<task>> next_grant();  // mutex_ held
  [[nodiscard]] bool try_allocate(task& t);                                    // mutex_ held
  void issue(queued_batch q);  // no lock
  void finish_read(std::shared_ptr<task> const& t, std::size_t bytes, std::exception_ptr error);
  std::shared_ptr<task> find(std::int64_t id);

  std::unique_ptr<cucascade::memory::memory_reservation_manager> _memory;
  std::unique_ptr<cucascade::io::io_context_registry> _registry;
  std::shared_ptr<cucascade::io::ioctx> _context;

  // footers
  std::mutex _footer_mutex;
  std::condition_variable _footer_cv;
  std::deque<std::string> _footer_queue;
  std::unordered_map<std::string, std::shared_ptr<footer_entry>> _footers;
  std::vector<std::thread> _footer_threads;
  bool _footer_stop{false};

  // data
  std::uint8_t* _arena{nullptr};
  std::size_t _arena_bytes{0};
  std::mutex _mutex;
  std::condition_variable _cv;
  std::unordered_map<std::int64_t, stream_state> _streams;
  std::int64_t _next_stream{1};
  std::int64_t _rr{0};
  std::size_t _largest_task{0};
  std::unordered_map<std::int64_t, std::shared_ptr<task>> _tasks;
  std::vector<std::pair<std::size_t, std::size_t>> _free;  // sorted (offset, size)
  std::int64_t _next_id{0};
  bool _stop{false};
  std::size_t _fetched_bytes{0};
  std::size_t _inflight_limit{0};
  std::size_t _inflight{0};             // bytes of issued, unfinished reads
  std::deque<queued_batch> _to_issue;  // read batches of granted tasks, in grant order
  std::thread _scheduler;
};

}  // namespace cudf_polars_cucascade
