// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cudf_streaming/experimental/prefetch_export.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <cuda/stream_ref>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace cudf_streaming::prefetch {

// ---------------------------------------------------------------------------
// scan_split — one entry in the ordered split list passed to start_readahead
// ---------------------------------------------------------------------------

/**
 * @brief A single scan split: one file path plus the byte ranges that will
 * be read from it.  The readahead thread opens the file, calls fadvise with
 * the ranges, and begins the IO so data is in cache when the reader arrives.
 */
struct CUDF_STREAMING_PREFETCH_API scan_split {
  std::string path;
  std::vector<cudf::io::text::byte_range_info> ranges;
};

// ---------------------------------------------------------------------------
// PrefetchDatasource — public wrapper around the internal sirius_datasource
// ---------------------------------------------------------------------------

/**
 * @brief Public prefetch-backed cudf datasource.
 *
 * Subclasses @c cudf::io::datasource; all reads go through the in-process
 * prefetching cache built by @c ScanContext.  Every @c PrefetchDatasource
 * instance is scoped to one scan and holds its own @c prefetching_handle so
 * it can call @c fadvise() without stomping on sibling scans that share the
 * same file.
 *
 * Acquire via @c ScanContext::open_datasource (one per file) and then
 * @c duplicate (one per split-scan within the same file).
 */
class CUDF_STREAMING_PREFETCH_API PrefetchDatasource : public cudf::io::datasource {
 public:
  // Non-copyable; duplication is explicit via duplicate().
  PrefetchDatasource(PrefetchDatasource const&)            = delete;
  PrefetchDatasource& operator=(PrefetchDatasource const&) = delete;

  PrefetchDatasource(PrefetchDatasource&&)            = default;
  PrefetchDatasource& operator=(PrefetchDatasource&&) = default;

  ~PrefetchDatasource() override;

  // ---- cudf::io::datasource virtuals ---------------------------------------
  std::size_t size() const override;
  bool supports_device_read() const override;
  bool is_device_read_preferred(std::size_t size) const override;
  std::unique_ptr<buffer> host_read(std::size_t offset, std::size_t size) override;
  std::size_t host_read(std::size_t offset, std::size_t size, uint8_t* dst) override;
  std::unique_ptr<buffer> device_read(std::size_t offset,
                                      std::size_t size,
                                      ::cuda::stream_ref stream) override;
  std::size_t device_read(std::size_t offset,
                           std::size_t size,
                           uint8_t* dst,
                           ::cuda::stream_ref stream) override;
  std::future<std::size_t> device_read_async(std::size_t offset,
                                              std::size_t size,
                                              uint8_t* dst,
                                              ::cuda::stream_ref stream) override;

  // ---- Advisory IO ---------------------------------------------------------

  /**
   * @brief Hint the cache about @p ranges that this scan will read soon.
   *
   * @p ranges   Byte ranges (offset + size) to prefetch into pinned host memory.
   * @p dev_id   GPU device to stage data near (preferred NUMA node).  Pass
   *             @c std::nullopt for "no preference."
   */
  void fadvise(std::span<const cudf::io::text::byte_range_info> ranges,
               std::optional<int> dev_id = std::nullopt);

  /**
   * @brief Return a fresh datasource sharing this file's I/O context.
   *
   * Shares the same underlying @c sirius_io_object (same file, same cache
   * bucket) but carries an empty prefetch handle.  Use once per split scan
   * within a single file so each split can issue its own @c fadvise().
   */
  [[nodiscard]] std::unique_ptr<PrefetchDatasource> duplicate() const;

 private:
  struct impl;
  explicit PrefetchDatasource(std::unique_ptr<impl>);
  std::unique_ptr<impl> _impl;

  friend class scan_context;
};

// ---------------------------------------------------------------------------
// scan_context_config
// ---------------------------------------------------------------------------

/**
 * @brief Configuration for the prefetch scan context.
 */
struct CUDF_STREAMING_PREFETCH_API scan_context_config {
  /// Bytes of pinned host memory to reserve for the prefetch staging pool.
  std::size_t host_memory_capacity{8ULL * 1024ULL * 1024ULL * 1024ULL};
  /// Block size (bytes) used by the fixed-size host allocator.
  std::size_t host_block_size{1UL << 20};  // 1 MiB
  /// GPU device ID to register a GPU memory space for (-1 = skip).
  int gpu_device_id{0};
  /// Fraction of GPU memory to reserve for the GPU-tier memory space.
  double gpu_reservation_fraction{0.9};
  // S3 / REST configuration (empty = disabled)
  std::string s3_endpoint{};
  std::string s3_region{};
  std::string s3_access_key_id{};
  std::string s3_secret_access_key{};
  std::string s3_session_token{};
  /// Route local files through io_uring (requires CUDF_STREAMING_HAS_URING).
  bool use_sirius_datasource_for_local{false};
};

// ---------------------------------------------------------------------------
// scan_context
// ---------------------------------------------------------------------------

/**
 * @brief Per-engine prefetch scan context.
 *
 * Holds the background memory pools, I/O reactors, and the prefetching
 * cache that back every @c PrefetchDatasource created by this context.
 * Construct once per rank at engine startup and destroy at engine shutdown.
 *
 * Thread-safe: @c open_datasource may be called concurrently.
 */
class CUDF_STREAMING_PREFETCH_API scan_context {
 public:
  [[nodiscard]] static std::unique_ptr<scan_context> create(
    const scan_context_config& cfg = {});

  ~scan_context();

  scan_context(scan_context const&)            = delete;
  scan_context& operator=(scan_context const&) = delete;

  /**
   * @brief Open a prefetch-backed datasource for @p path.
   */
  [[nodiscard]] std::unique_ptr<PrefetchDatasource> open_datasource(std::string path);

  /**
   * @brief Signal that a new query epoch is beginning (for telemetry).
   */
  void prepare_for_query(uint64_t query_id) noexcept;

  // ---- Eager readahead API (Stage 5) ---------------------------------------

  /**
   * @brief Start a background readahead thread that prefetches @p splits in
   * order, keeping at most @p budget IOs in flight at once.
   *
   * The thread opens each split's file, calls fadvise with its byte ranges,
   * allocates staging buffers, and issues async IO.  Subsequent calls to
   * @c get_datasource(i) block until split @p i is ready.
   *
   * May only be called once per @c scan_context instance.  Call
   * @c stop_readahead (or destroy the context) to join the thread.
   *
   * @param splits   Ordered list of (path, ranges) pairs.
   * @param gpu_id   GPU device to stage data near (NUMA preference).
   * @param budget   Max IOs in flight at once (default 16).
   */
  void start_readahead(std::vector<scan_split> splits,
                       std::optional<int> gpu_id = std::nullopt,
                       std::size_t budget        = 16);

  /**
   * @brief Return the prefetched datasource for split @p i.
   *
   * Blocks until the readahead thread has issued IO for split @p i (which
   * may or may not have completed; the cache handles the wait on first read).
   * Drives the split's consumer stage to @c reading.
   *
   * @returns Pointer valid until @c release_datasource(i) is called.
   */
  [[nodiscard]] PrefetchDatasource* get_datasource(std::size_t i);

  /**
   * @brief Signal that the reader has finished with split @p i.
   *
   * Drives the split's consumer stage to @c disposed so the evictor may
   * reclaim the staging buffers.  The slot remains valid until
   * @c stop_readahead is called or the context is destroyed.
   */
  void release_datasource(std::size_t i) noexcept;

  /**
   * @brief Stop the readahead thread and join it.
   *
   * Called automatically by the destructor; may also be called explicitly
   * to join early.  Safe to call even if @c start_readahead was never called.
   */
  void stop_readahead() noexcept;

 private:
  struct impl;
  explicit scan_context(std::unique_ptr<impl>);
  std::unique_ptr<impl> _impl;
};

}  // namespace cudf_streaming::prefetch
