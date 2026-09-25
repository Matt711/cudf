// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// C++ wrapper for the Sirius prefetch-cache IO stack.
// Included by sirius_cache.pyx via cdef extern from.
//
// Lifetime model
// --------------
// SiriusSession owns, in order of construction:
//   1. memory_reservation_manager  (HOST-tier pinned-memory pool)
//   2. io_context_registry         (holds ref to (1); auto-registers kvikio/uring/restful)
//   3. shared_ptr<ioctx>           (the restful backend for s3:// → REST reactor)
//
// SiriusDatasource wraps a unique_ptr<sirius_datasource>.  A datasource is
// opened from the session's ioctx; it borrows the ioctx by raw pointer (same
// as every other datasource in the C++ tree).  The session must outlive all
// its datasources.
//
// PrefetchingHandle (C++ side) is held *inside* sirius_datasource.  Python
// exposes it as an opaque object that holds a Python reference to its parent
// SiriusDatasource, which is sufficient to keep the C++ handle alive.

#pragma once

#include <cucascade/memory/config.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cudf/io/streaming/io/cache/config.hpp>
#include <cudf/io/streaming/io/cache/prefetching_cache.hpp>
#include <cudf/io/streaming/io/datasource_factory.hpp>
#include <cudf/io/streaming/io/io_context.hpp>
#include <cudf/io/streaming/io/object_store_config.hpp>
#include <cudf/io/streaming/io/sirius_datasource.hpp>
#include <cudf/io/streaming/scan_manager/config.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius_cache_impl {

// ─── Configuration bundle passed from Python ─────────────────────────────────

struct session_config {
  // kvikio thread pool
  std::size_t num_threads{20};

  // Reactor counts — rest_n_reactors drives S3 parallelism via the REST reactor
  std::size_t rest_n_reactors{16};
  std::size_t uring_n_reactors{4};

  // HOST-tier pinned-memory pool for the prefetch cache
  std::size_t host_capacity_bytes{200UL * 1024 * 1024 * 1024};  // 200 GiB
  std::size_t host_pool_size_mib{512};      // pool_size in 1-MiB blocks
  std::size_t host_initial_number_pools{300};
  double eviction_threshold_fraction{0.8};

  // S3 credentials (empty = no object-store backend)
  std::string object_store_endpoint;
  std::string object_store_region;
  std::string access_key;
  std::string secret_key;
  std::string session_token;

  // Whether to arm the prefetch cache at all
  bool use_prefetch_cache{true};
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

inline std::vector<cucascade::memory::memory_space_config> make_rm_configs(
  const session_config& cfg)
{
  cucascade::memory::host_memory_space_config hcfg;
  hcfg.numa_id              = -1;                                  // any NUMA node
  hcfg.memory_capacity      = cfg.host_capacity_bytes;
  hcfg.block_size           = 1UL << 20;                           // 1 MiB per block
  hcfg.pool_size            = cfg.host_pool_size_mib;              // blocks per pool
  hcfg.initial_number_pools = cfg.host_initial_number_pools;
  hcfg.make_portable        = true;                                 // cudaMallocHost (DMA-safe)
  return {hcfg};
}

inline cudf_streaming::scan_manager::scan_manager_config make_scan_cfg(const session_config& cfg)
{
  cudf_streaming::scan_manager::scan_manager_config sc;
  // backend defaults to io_backend::sirius: s3:// → REST reactor, local → uring reactor.
  sc.uring_n_reactors = cfg.uring_n_reactors;
  sc.rest_n_reactors  = cfg.rest_n_reactors;

  if (cfg.num_threads > 0) {
    sc.kvikio.nthreads = static_cast<unsigned int>(cfg.num_threads);
  }

  sc.object_store.endpoint      = cfg.object_store_endpoint;
  sc.object_store.region        = cfg.object_store_region;
  sc.object_store.access_key    = cfg.access_key;
  sc.object_store.secret_key    = cfg.secret_key;
  sc.object_store.session_token = cfg.session_token;

  return sc;
}

// ─── SiriusSession ───────────────────────────────────────────────────────────

// Non-copyable/non-movable: memory_reservation_manager is neither.
// Python holds this via unique_ptr so the Cython layer never copies it.
class SiriusSession {
 public:
  explicit SiriusSession(const session_config& cfg)
    : _rm(make_rm_configs(cfg))
    , _registry(make_scan_cfg(cfg), _rm)
  {
    using namespace cudf_streaming::io;

    // The io_context_registry constructor auto-registers all backends
    // (kvikio, uring, restful) via datasource_factory.  We use the restful
    // backend here so that s3:// paths route to the REST reactor, which
    // returns supports_vector_host_read()=true and
    // supports_host_to_device_read()=true — arming the prefetch cache
    // (_armed=true).  kvikio returns false for both, so the cache would
    // never activate if we used the kvikio ioctx for S3 reads.
    _ioctx = _registry.make_ioctx(io_context_type::restful);
    if (!_ioctx) {
      throw std::runtime_error("sirius_cache: failed to build restful ioctx");
    }
    _ioctx->start();

    if (cfg.use_prefetch_cache) {
      cache::config ccfg;
      ccfg.mode                         = cache::cache_mode::sirius;
      ccfg.eviction                     = cache::eviction_policy::lru;
      ccfg.eviction_threshold_fraction  = cfg.eviction_threshold_fraction;
      ccfg.apply_mode();
      // topology_index = nullptr → no NUMA affinity preference.
      _ioctx->initialize_cache(_rm, ccfg, nullptr);
    }
  }

  ~SiriusSession()
  {
    if (_ioctx) {
      _ioctx->shutdown_cache();
      _ioctx->shutdown();
    }
  }

  SiriusSession(const SiriusSession&)            = delete;
  SiriusSession& operator=(const SiriusSession&) = delete;
  SiriusSession(SiriusSession&&)                 = delete;
  SiriusSession& operator=(SiriusSession&&)      = delete;

  // Open one sirius_datasource for @p path (unique_ptr; caller takes ownership).
  std::unique_ptr<cudf_streaming::io::sirius_datasource> open(const std::string& path)
  {
    if (!_ioctx) { throw std::runtime_error("sirius_cache: session not initialized"); }
    return _ioctx->open_datasource(path);
  }

  // Flush and reset the prefetch cache (call between benchmark iterations).
  void reset_caches() noexcept
  {
    if (!_ioctx) { return; }
    auto* c = _ioctx->cache();
    if (!c) { return; }
    c->evict(c->claimed_bytes());  // async eviction of everything
    c->prepare_for_query();        // reset per-query epoch counter
  }

  // Return the prefetch cache hit/miss/eviction summary string.
  // Returns an empty string if the cache is not active.
  std::string cache_summary() const noexcept
  {
    if (!_ioctx) { return ""; }
    auto* c = _ioctx->cache();
    if (!c) { return ""; }
    return c->summary();
  }

 private:
  cucascade::memory::memory_reservation_manager _rm;
  cudf_streaming::io::io_context_registry       _registry;
  std::shared_ptr<cudf_streaming::io::ioctx>    _ioctx;
};

// ─── Thin free-function wrappers ─────────────────────────────────────────────
// These are called from Cython and avoid the Cython layer needing to know about
// std::span or std::optional directly (both can be awkward to declare in pxd).

// Call sirius_datasource::fadvise with Python-friendly argument types.
inline void call_fadvise(cudf_streaming::io::sirius_datasource& ds,
                         const std::vector<cudf::io::text::byte_range_info>& ranges,
                         int device_id_or_neg1)
{
  std::optional<int> dev_id;
  if (device_id_or_neg1 >= 0) { dev_id = device_id_or_neg1; }
  ds.fadvise(
    std::span<const cudf::io::text::byte_range_info>(ranges.data(), ranges.size()),
    dev_id);
}

// Step 2: allocate staging buffers from the pinned host pool.
// Returns true if preparation succeeded (memory available).
// wait_for_eviction=true means block until LRU eviction frees space if needed.
inline bool call_prepare_prefetch(cudf_streaming::io::sirius_datasource& ds,
                                   bool wait_for_eviction)
{
  return ds.prepare_prefetch(wait_for_eviction) ==
         cudf_streaming::io::prepare_result::prepared;
}

// Step 3: dispatch actual background S3 → host-pool reads (fire-and-forget).
// on_done callback is ignored — caller doesn't need to wait for completion.
inline void call_prefetch_async(cudf_streaming::io::sirius_datasource& ds)
{
  ds.prefetch_async([](bool) noexcept {});
}

// Step 3b (blocking variant): dispatch S3 → host-pool reads and wait until
// the transfer has fully landed in pinned host memory before returning.
// Call this from a worker thread when you need the data to be ready before
// the main read loop starts (e.g. to guarantee Q1 cold-start hits the cache).
inline void call_await_inflight_prefetch(cudf_streaming::io::sirius_datasource& ds) noexcept
{
  ds.prefetch_async([](bool) noexcept {});
  ds.await_inflight_prefetch();
}

// Vectored form of device_read: all byte-range H2D copies in one batch.
//
// Avoids N × device_read_async calls on the same CUDA stream (serial H2D) by
// calling sirius_datasource::device_read_ranges_async() which lets the cache
// back-end fuse and schedule all copies together.
//
// dst_ptrs[i] must be valid device pointers allocated by the caller
// (e.g. via RMM) that are large enough to hold sizes[i] bytes.
// The function blocks until the async future is ready.
inline std::size_t call_device_read_ranges_async(
    cudf_streaming::io::sirius_datasource& ds,
    const std::vector<std::size_t>& file_offsets,
    const std::vector<std::size_t>& sizes,
    const std::vector<std::uintptr_t>& dst_ptrs,
    cudaStream_t raw_stream)
{
    if (file_offsets.empty()) { return 0; }
    rmm::cuda_stream_view stream{raw_stream};
    std::vector<cudf_streaming::io::slice> slices;
    slices.reserve(file_offsets.size());
    for (std::size_t i = 0; i < file_offsets.size(); ++i) {
        if (sizes[i] == 0) { continue; }
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        slices.emplace_back(
            file_offsets[i],
            sizes[i],
            reinterpret_cast<std::uint8_t*>(dst_ptrs[i]));
    }
    if (slices.empty()) { return 0; }
    return ds.device_read_ranges_async(
               std::span<const cudf_streaming::io::slice>(slices.data(), slices.size()),
               stream)
           .get();
}

}  // namespace sirius_cache_impl
