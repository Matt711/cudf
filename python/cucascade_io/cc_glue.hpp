// SPDX-License-Identifier: Apache-2.0
// Minimal glue: build a cuCascade REST io context with the prefetching cache and open cudf datasources from it.
#pragma once

#include <cudf/io/datasource.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ccglue {

struct options {
  std::size_t cache_bytes{std::size_t{100} << 30};  ///< host (pinned) memory space capacity = prefetching cache budget
  std::size_t block_bytes{std::size_t{256} << 10};     ///< cache chunk size (= pinned host block = REST staging block)
  std::size_t initial_pools{0};                     ///< 128 MiB pinned pools allocated up front; 0 = enough for the whole cache (a growing pool makes every prefetch wait, see DESIGN)
  std::size_t pool_threads{8};                      ///< threads that pin the initial pools
  std::size_t rest_reactors{16};
  std::size_t max_connections{128};
  std::string endpoint, region, access_key, secret_key, session_token;
  std::size_t advise_min_bytes{std::size_t{256} << 10};  ///< reads of at least this size are advised to the cache first (0 = never)
  double evict_batch{0.02};  ///< cuCascade eviction_batch_fraction (patched cuCascade): share of the pool freed per eviction pass
  double evict_threshold{0.99};  ///< cuCascade eviction_threshold_fraction: usable share of the pool (and where LRU eviction starts); its default is 0.8
  double budget_fraction{0.9};  ///< admitted pieces (counted in chunk-aligned bytes) may fill this fraction of the cache
  bool column_policy{true};  ///< advise only columns that rank within the cache budget (needs register_layout); false = advise every read
  bool lru{true};  ///< cache eviction: LRU (keep idle chunks) vs idle (drop on idle)
};

/// Build the manager, registry and REST ioctx, start the reactors and arm the cache. Idempotent per process.
void init(options const& o);
/// Open an s3:// object (size 0 = discover with a footer-probing HEAD).
std::unique_ptr<cudf::io::datasource> open(std::string const& path, std::uint64_t known_size);
/// Register the byte ranges (offset, size) a scan is about to read so the cache keeps those chunks (cuCascade only caches advised ranges).
/// With start=true also attaches staging buffers and issues the (low priority) prefetch; returns prepare_result*10 + prefetch_refusal, or -1 when not started.
int fadvise(cudf::io::datasource* ds, std::vector<std::pair<std::uint64_t, std::uint64_t>> const& ranges, bool start);
/// Column layout of one file (column chunk byte ranges), used by the column-aware admission policy: only columns that rank high enough by how often
/// they are read, and fit the cache budget, are advised to the cache. dir groups files of one table; names/col_index/offsets/sizes as in bytecache.
void register_layout(std::string const& path, std::string const& dir, std::vector<std::string> const& names, std::vector<std::uint32_t> const& col_index,
                     std::vector<std::uint64_t> const& offsets, std::vector<std::uint64_t> const& sizes);
/// Per-column admission accounting (registered / requested / admitted / refused bytes), one line per column.
std::string column_report();
/// fs_cache counters (reads, hits, misses, evictions) as text; empty when the cache is off.
std::string cache_summary();
/// Drain the cache and reactors; datasources must be destroyed first.
void shutdown();

}  // namespace ccglue
