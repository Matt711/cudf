/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cucascade/io/datasource_factory.hpp>
#include <cucascade/io/io_context.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

#include <cudf/io/parquet_schema.hpp>

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

  /** @brief Bytes requested from the object store so far (data, not footers). */
  [[nodiscard]] std::size_t fetched_bytes() const noexcept { return _fetched_bytes; }

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

  struct task {
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

  void scheduler_main();
  [[nodiscard]] std::pair<std::int64_t, std::shared_ptr<task>> next_grant();  // mutex_ held
  [[nodiscard]] bool try_allocate(task& t);                                    // mutex_ held
  void issue_reads(std::int64_t id, std::shared_ptr<task> const& t);           // no lock
  void finish_read(std::int64_t id, std::shared_ptr<task> const& t, std::exception_ptr error);
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
  std::thread _scheduler;
};

}  // namespace cudf_polars_cucascade
