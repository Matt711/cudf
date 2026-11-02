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
  std::size_t initial_pools{4};                     ///< pinned pools allocated up front (see cuCascade host_memory_space_config)
  std::size_t rest_reactors{2};
  std::size_t max_connections{64};
  std::string endpoint, region, access_key, secret_key, session_token;
  std::size_t advise_min_bytes{std::size_t{256} << 10};  ///< reads of at least this size are advised to the cache first (0 = never)
  bool lru{true};  ///< cache eviction: LRU (keep idle chunks) vs idle (drop on idle)
};

/// Build the manager, registry and REST ioctx, start the reactors and arm the cache. Idempotent per process.
void init(options const& o);
/// Open an s3:// object (size 0 = discover with a footer-probing HEAD).
std::unique_ptr<cudf::io::datasource> open(std::string const& path, std::uint64_t known_size);
/// Register the byte ranges (offset, size) a scan is about to read so the cache keeps those chunks (cuCascade only caches advised ranges).
/// With start=true also attaches staging buffers and issues the (low priority) prefetch; returns prepare_result*10 + prefetch_refusal, or -1 when not started.
int fadvise(cudf::io::datasource* ds, std::vector<std::pair<std::uint64_t, std::uint64_t>> const& ranges, bool start);
/// fs_cache counters (reads, hits, misses, evictions) as text; empty when the cache is off.
std::string cache_summary();
/// Drain the cache and reactors; datasources must be destroyed first.
void shutdown();

}  // namespace ccglue
