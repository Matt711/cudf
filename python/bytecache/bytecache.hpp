// SPDX-License-Identifier: Apache-2.0
// Prototype: cross-query byte-range cache exposed as a cudf::io::datasource (see DESIGN.md).
#pragma once

#include <cudf/io/datasource.hpp>

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace bytecache {

struct config {
  std::size_t capacity_bytes{0};            ///< arena size; 0 disables caching (reads go straight to the remote handle)
  std::size_t block_size{1 << 20};          ///< cache block size
  std::size_t max_get_bytes{16 << 20};      ///< largest fused remote read
};

struct stats {
  std::uint64_t bytes_requested{0};
  std::uint64_t bytes_hit{0};       ///< served from cache
  std::uint64_t bytes_fetched{0};   ///< fetched from the remote object (whole blocks)
  std::uint64_t blocks_evicted{0};
  std::uint64_t blocks_resident{0};
  std::uint64_t gets_issued{0};
  std::uint64_t waits_on_inflight{0};
  // Time (ns, summed over threads) spent in each stage; for finding overheads.
  std::uint64_t ns_open{0};         ///< opening remote objects (HEAD)
  std::uint64_t ns_pinned_wait{0};  ///< waiting for / creating a pinned staging buffer
  std::uint64_t ns_read{0};         ///< cache read (lookup, fetch, copies) inside device reads
  std::uint64_t ns_h2d{0};          ///< host-to-device copy plus stream synchronise
  std::uint64_t ns_budget_wait{0};  ///< waiting for GET staging buffers
  std::uint64_t device_reads{0};    ///< device_read_async calls
  std::uint64_t ns_queue_wait{0};   ///< time jobs waited between submit and start on the I/O pool
  std::uint64_t host_reads{0};      ///< synchronous host_read calls (footers, metadata)
  std::uint64_t ns_host_read{0};    ///< time inside synchronous host_read calls
  std::uint64_t bytes_prefetched{0};  ///< bytes fetched by read-ahead (included in bytes_fetched)
  std::uint64_t prefetch_jobs{0};     ///< read-ahead requests submitted
  std::uint64_t max_inflight{0};    ///< peak number of device reads running at once
};

/// Per-column accounting (columns are known once a file's layout is registered; id 0 is "(unknown)").
struct column_stats {
  std::string name;                       ///< "<table>|<column>"
  std::uint64_t registered_bytes{0};      ///< compressed bytes of this column over all registered files
  std::uint64_t requested_bytes{0};       ///< bytes asked for by reads
  std::uint64_t hit_bytes{0};             ///< served from the cache
  std::uint64_t fetched_bytes{0};         ///< fetched from S3 (whole blocks)
  std::uint64_t resident_blocks{0};       ///< blocks of this column currently cached
};

/// Register the byte layout of one remote object: for every column chunk its (offset, size), so the cache can
/// attribute blocks to columns. `col_index[i]` indexes `columns` for chunk i. Safe to call more than once per path (later calls are ignored).
void register_layout(std::string const& path, std::string const& table, std::vector<std::string> const& columns,
                     std::vector<std::uint32_t> const& col_index, std::vector<std::uint64_t> const& offsets,
                     std::vector<std::uint64_t> const& sizes);
std::vector<column_stats> get_column_stats();

/// Read-ahead: start fetching, on the cache's prefetch pool, the byte ranges of `columns` of the file at `path`
/// (its layout must be registered) into the cache. Blocks that the policy would not admit are skipped, and demand
/// reads are never blocked by it (separate thread pool and staging budget). Returns the bytes of the ranges (0 if the
/// file or layout is unknown).
std::uint64_t prefetch_columns(std::string const& path, std::string const& table, std::vector<std::string> const& columns);

/// Set the process-wide configuration. Must be called before the first read; later calls with a different config throw.
void configure(config const& cfg);
stats get_stats();
void reset_stats();
/// Per-second counts of device reads submitted and completed since the first submit (index = seconds, last bucket is a catch-all).
std::vector<std::uint64_t> timeline_submit();
std::vector<std::uint64_t> timeline_done();
/// Device reads that took longer than 1 s: "path offset size milliseconds" (diagnostic, capped at 64 entries).
std::vector<std::string> slow_reads();
void clear();

class cached_source : public cudf::io::datasource {
 public:
  /// @param path remote URI (s3://...). @param size object size in bytes if known (0 = look it up with HEAD).
  cached_source(std::string path, std::size_t size);
  ~cached_source() override;

  std::unique_ptr<buffer> host_read(std::size_t offset, std::size_t size) override;
  std::size_t host_read(std::size_t offset, std::size_t size, std::uint8_t* dst) override;
  /// Async reads run on the cache's own thread pool (BYTECACHE_THREADS, default 128). libcudf's default
  /// async implementation would run the blocking read on its small shared host worker pool and cap concurrency.
  std::future<std::unique_ptr<buffer>> host_read_async(std::size_t offset, std::size_t size) override;
  std::future<std::size_t> host_read_async(std::size_t offset, std::size_t size, std::uint8_t* dst) override;
  [[nodiscard]] std::size_t size() const override;

  /// Device reads: libcudf's parquet reader only overlaps remote reads for sources that prefer device reads
  /// (it otherwise calls the synchronous host_read on its small shared worker pool). The data is fetched on the
  /// cache's thread pool, staged in pinned memory and copied to `dst` on `stream` before the future completes.
  [[nodiscard]] bool supports_device_read() const override { return true; }
  [[nodiscard]] bool is_device_read_preferred(std::size_t) const override { return true; }
  std::size_t device_read(std::size_t offset, std::size_t size, std::uint8_t* dst, cuda::stream_ref stream) override;
  std::future<std::size_t> device_read_async(std::size_t offset, std::size_t size, std::uint8_t* dst,
                                             cuda::stream_ref stream) override;

 private:
  struct impl;
  std::unique_ptr<impl> p_;
};

}  // namespace bytecache
