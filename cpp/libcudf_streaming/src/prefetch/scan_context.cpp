// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cudf_streaming/experimental/prefetch_context.hpp>

#include "sirius/io/datasource_factory.hpp"
#include "sirius/io/io_context.hpp"
#include "sirius/io/sirius_datasource.hpp"
#include "sirius/memory/common.hpp"
#include "sirius/memory/config.hpp"
#include "sirius/memory/memory_reservation_manager.hpp"
#include "sirius/memory/topology_discovery.hpp"
#include "sirius/memory/topology_index.hpp"
#include "sirius/prefetch_config.hpp"

#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace cudf_streaming::prefetch {

// ---------------------------------------------------------------------------
// PrefetchDatasource::impl
// ---------------------------------------------------------------------------

struct PrefetchDatasource::impl {
  std::unique_ptr<io::sirius_datasource> ds;
};

// ---------------------------------------------------------------------------
// PrefetchDatasource — delegate all virtuals to the wrapped sirius_datasource
// ---------------------------------------------------------------------------

PrefetchDatasource::PrefetchDatasource(std::unique_ptr<impl> p) : _impl(std::move(p)) {}

PrefetchDatasource::~PrefetchDatasource() = default;

std::size_t PrefetchDatasource::size() const { return _impl->ds->size(); }

bool PrefetchDatasource::supports_device_read() const
{
  return _impl->ds->supports_device_read();
}

bool PrefetchDatasource::is_device_read_preferred(std::size_t sz) const
{
  return _impl->ds->is_device_read_preferred(sz);
}

std::unique_ptr<cudf::io::datasource::buffer> PrefetchDatasource::host_read(std::size_t offset,
                                                                              std::size_t sz)
{
  return _impl->ds->host_read(offset, sz);
}

std::size_t PrefetchDatasource::host_read(std::size_t offset, std::size_t sz, uint8_t* dst)
{
  return _impl->ds->host_read(offset, sz, dst);
}

std::unique_ptr<cudf::io::datasource::buffer> PrefetchDatasource::device_read(
  std::size_t offset, std::size_t sz, ::cuda::stream_ref stream)
{
  return _impl->ds->device_read(offset, sz, stream);
}

std::size_t PrefetchDatasource::device_read(std::size_t offset,
                                             std::size_t sz,
                                             uint8_t* dst,
                                             ::cuda::stream_ref stream)
{
  return _impl->ds->device_read(offset, sz, dst, stream);
}

std::future<std::size_t> PrefetchDatasource::device_read_async(std::size_t offset,
                                                                 std::size_t sz,
                                                                 uint8_t* dst,
                                                                 ::cuda::stream_ref stream)
{
  return _impl->ds->device_read_async(offset, sz, dst, stream);
}

void PrefetchDatasource::fadvise(std::span<const cudf::io::text::byte_range_info> ranges,
                                  std::optional<int> dev_id)
{
  _impl->ds->fadvise(ranges, dev_id);
}

std::unique_ptr<PrefetchDatasource> PrefetchDatasource::duplicate() const
{
  auto dup_ds  = _impl->ds->duplicate();
  auto dup_impl = std::make_unique<PrefetchDatasource::impl>();
  dup_impl->ds  = std::move(dup_ds);
  return std::unique_ptr<PrefetchDatasource>(new PrefetchDatasource(std::move(dup_impl)));
}

// ---------------------------------------------------------------------------
// scan_context::impl
// ---------------------------------------------------------------------------

struct scan_context::impl {
  using registry_type = io::io_context_registry;

  std::unique_ptr<memory::memory_reservation_manager> reservation_manager;
  std::unique_ptr<registry_type> registry;
  std::shared_ptr<const memory::topology_index> topology;
  std::unordered_map<io::io_context_type, std::shared_ptr<io::ioctx>> live_ioctxs;
  scan_manager::scan_manager_config config{};

  [[nodiscard]] std::shared_ptr<io::ioctx> get_ioctx(io::io_context_type type) const
  {
    auto it = live_ioctxs.find(type);
    if (it == live_ioctxs.end()) return nullptr;
    return it->second;
  }

  // ---- Eager readahead state (populated by start_readahead) ----------------

  struct readahead_state {
    std::vector<std::unique_ptr<PrefetchDatasource>> slots;
    std::vector<bool>                                slot_ready;
    std::mutex                                       mtx;
    std::condition_variable                          cv;
    // counting_semaphore<N>: N is compile-time max; actual budget is runtime.
    // 64 is a safe upper bound (Sirius uses 16).
    std::unique_ptr<std::counting_semaphore<64>>     gatekeeper;
    std::atomic<bool>                                stop{false};
    std::thread                                      thread;
  };

  std::unique_ptr<readahead_state> readahead;
};

// ---------------------------------------------------------------------------
// scan_context::create
// ---------------------------------------------------------------------------

/*static*/
std::unique_ptr<scan_context> scan_context::create(const scan_context_config& cfg)
{
  auto pimpl = std::make_unique<impl>();

  // topology discovery
  memory::topology_discovery discovery;
  (void)discovery.discover();
  auto sys_topology = discovery.is_discovered()
                        ? discovery.get_topology()
                        : memory::system_topology_info{};

  // memory spaces
  std::vector<memory::memory_space_config> space_configs;

  if (cfg.gpu_device_id >= 0) {
    std::size_t free_bytes = 0, total_gpu = 0;
    cudaMemGetInfo(&free_bytes, &total_gpu);
    auto gpu_capacity =
      static_cast<std::size_t>(static_cast<double>(free_bytes) * cfg.gpu_reservation_fraction);

    memory::gpu_memory_space_config gpu_cfg;
    gpu_cfg.device_id       = cfg.gpu_device_id;
    gpu_cfg.memory_capacity = gpu_capacity;
    gpu_cfg.mr_factory_fn   = memory::make_default_allocator_for_tier(memory::Tier::GPU);
    space_configs.emplace_back(gpu_cfg);
  }

  {
    memory::host_memory_space_config host_cfg;
    host_cfg.numa_id              = -1;
    host_cfg.memory_capacity      = cfg.host_memory_capacity;
    host_cfg.block_size           = cfg.host_block_size;
    host_cfg.pool_size            = 128;
    host_cfg.initial_number_pools = 4;
    host_cfg.make_portable        = true;
    host_cfg.mr_factory_fn        = memory::make_default_allocator_for_tier(memory::Tier::HOST);
    space_configs.emplace_back(host_cfg);
  }

  pimpl->reservation_manager =
    std::make_unique<memory::memory_reservation_manager>(std::move(space_configs));

  pimpl->topology = std::make_shared<memory::topology_index>(
    std::move(sys_topology), *pimpl->reservation_manager);

  // scan_manager_config — when uring is unavailable, kvikio is the local backend
  pimpl->config.backend = cfg.use_sirius_datasource_for_local
                            ? scan_manager::io_backend::sirius
                            : scan_manager::io_backend::kvikio;
  if (!cfg.s3_endpoint.empty()) {
    pimpl->config.object_store.endpoint      = cfg.s3_endpoint;
    pimpl->config.object_store.region        = cfg.s3_region;
    pimpl->config.object_store.access_key    = cfg.s3_access_key_id;
    pimpl->config.object_store.secret_key    = cfg.s3_secret_access_key;
    pimpl->config.object_store.session_token = cfg.s3_session_token;
  }

  // io_context_registry
  pimpl->registry =
    std::make_unique<io::io_context_registry>(pimpl->config, *pimpl->reservation_manager);

  // instantiate live ioctxs
  auto kvikio_ctx = pimpl->registry->make_ioctx(io::io_context_type::kvikio);
  if (kvikio_ctx) { pimpl->live_ioctxs[io::io_context_type::kvikio] = std::move(kvikio_ctx); }

#ifdef CUDF_STREAMING_HAS_URING
  auto uring_ctx = pimpl->registry->make_ioctx(io::io_context_type::uring);
  if (uring_ctx) {
    uring_ctx->start();
    uring_ctx->initialize_cache(*pimpl->reservation_manager, pimpl->config.cache, pimpl->topology);
    pimpl->live_ioctxs[io::io_context_type::uring] = std::move(uring_ctx);
  }
#endif

  auto rest_ctx = pimpl->registry->make_ioctx(io::io_context_type::restful);
  if (rest_ctx) {
    rest_ctx->start();
    rest_ctx->initialize_cache(*pimpl->reservation_manager, pimpl->config.cache, pimpl->topology);
    pimpl->live_ioctxs[io::io_context_type::restful] = std::move(rest_ctx);
  }

  return std::unique_ptr<scan_context>(new scan_context(std::move(pimpl)));
}

// ---------------------------------------------------------------------------
// scan_context — constructor / destructor / methods
// ---------------------------------------------------------------------------

scan_context::scan_context(std::unique_ptr<impl> p) : _impl(std::move(p)) {}

scan_context::~scan_context() { stop_readahead(); }

std::unique_ptr<PrefetchDatasource> scan_context::open_datasource(std::string path)
{
  auto type_opt = _impl->registry->lookup_path(path);
  if (!type_opt) {
    throw std::runtime_error("scan_context::open_datasource: no backend for path: " + path);
  }
  auto ioctx = _impl->get_ioctx(*type_opt);
  if (!ioctx) {
    throw std::runtime_error(
      "scan_context::open_datasource: backend not live for path: " + path);
  }
  auto raw_ds   = ioctx->open_datasource(std::move(path));
  auto ds_impl  = std::make_unique<PrefetchDatasource::impl>();
  ds_impl->ds   = std::move(raw_ds);
  return std::unique_ptr<PrefetchDatasource>(new PrefetchDatasource(std::move(ds_impl)));
}

void scan_context::prepare_for_query([[maybe_unused]] uint64_t query_id) noexcept
{
  for (auto& [type, ioctx] : _impl->live_ioctxs) {
    if (ioctx && ioctx->cache()) { ioctx->cache()->prepare_for_query(); }
  }
}

// ---------------------------------------------------------------------------
// Stage 5: eager readahead thread
// ---------------------------------------------------------------------------

void scan_context::start_readahead(std::vector<scan_split> splits,
                                    std::optional<int> gpu_id,
                                    std::size_t budget)
{
  if (_impl->readahead) {
    throw std::logic_error("scan_context::start_readahead: already started");
  }
  if (budget == 0 || budget > 64) {
    throw std::invalid_argument("scan_context::start_readahead: budget must be 1..64");
  }

  auto ra = std::make_unique<impl::readahead_state>();
  ra->gatekeeper = std::make_unique<std::counting_semaphore<64>>(
    static_cast<std::ptrdiff_t>(budget));
  ra->slots.resize(splits.size());
  ra->slot_ready.resize(splits.size(), false);
  ra->stop.store(false, std::memory_order_relaxed);

  // Thread body — captured by value for splits, by pointer for ra/impl.
  auto* pimpl = _impl.get();
  auto* pra   = ra.get();

  ra->thread = std::thread([pimpl, pra, splits = std::move(splits), gpu_id]() mutable {
    const std::size_t n = splits.size();

    for (std::size_t cursor = 0; cursor < n; ++cursor) {
      if (pra->stop.load(std::memory_order_acquire)) break;

      // Acquire a budget slot; blocks when 'budget' IOs are in flight.
      pra->gatekeeper->acquire();
      if (pra->stop.load(std::memory_order_acquire)) {
        pra->gatekeeper->release();
        break;
      }

      const auto& split = splits[cursor];

      // Resolve backend for this path.
      auto type_opt = pimpl->registry->lookup_path(split.path);
      if (!type_opt) {
        pra->gatekeeper->release();
        std::lock_guard lk{pra->mtx};
        pra->slot_ready[cursor] = true;
        pra->cv.notify_all();
        continue;
      }
      auto ioctx_ptr = pimpl->get_ioctx(*type_opt);
      if (!ioctx_ptr) {
        pra->gatekeeper->release();
        std::lock_guard lk{pra->mtx};
        pra->slot_ready[cursor] = true;
        pra->cv.notify_all();
        continue;
      }

      // Open the datasource.
      auto raw_ds = ioctx_ptr->open_datasource(split.path);

      // Register the ranges with the prefetching cache.
      if (!split.ranges.empty()) { raw_ds->fadvise(split.ranges, gpu_id); }

      // Allocate staging buffers; wait if memory is temporarily exhausted.
      auto prep = raw_ds->prepare_prefetch(/*wait_for_eviction=*/true);
      if (prep == io::prepare_result::allocation_failed) {
        // Pool is empty and we are not allowed to wait further — skip prefetch.
        pra->gatekeeper->release();
      } else {
        // Issue async IO.  on_done fires from the IO completion thread.
        auto* gate = pra->gatekeeper.get();
        raw_ds->prefetch_async([gate](bool /*success*/) noexcept { gate->release(); });
      }

      // Wrap in PrefetchDatasource and stash in the slot.
      auto ds_impl    = std::make_unique<PrefetchDatasource::impl>();
      ds_impl->ds     = std::move(raw_ds);
      auto pds        = std::unique_ptr<PrefetchDatasource>(
        new PrefetchDatasource(std::move(ds_impl)));

      {
        std::lock_guard lk{pra->mtx};
        pra->slots[cursor] = std::move(pds);
        pra->slot_ready[cursor] = true;
      }
      pra->cv.notify_all();
    }
  });

  _impl->readahead = std::move(ra);
}

PrefetchDatasource* scan_context::get_datasource(std::size_t i)
{
  if (!_impl->readahead) {
    throw std::logic_error("scan_context::get_datasource: readahead not started");
  }
  auto& ra = *_impl->readahead;
  std::unique_lock lk{ra.mtx};
  ra.cv.wait(lk, [&] {
    return ra.stop.load(std::memory_order_acquire) || ra.slot_ready[i];
  });
  auto* ds = ra.slots[i].get();
  if (ds) { ds->_impl->ds->update(io::cache::scan_stage::reading); }
  return ds;
}

void scan_context::release_datasource(std::size_t i) noexcept
{
  if (!_impl->readahead) return;
  auto& ra = *_impl->readahead;
  std::lock_guard lk{ra.mtx};
  if (i < ra.slots.size() && ra.slots[i]) {
    ra.slots[i]->_impl->ds->update(io::cache::scan_stage::disposed);
  }
}

void scan_context::stop_readahead() noexcept
{
  if (!_impl->readahead) return;
  auto& ra = *_impl->readahead;
  ra.stop.store(true, std::memory_order_release);
  // Unblock the readahead thread if it is waiting on a gatekeeper slot.
  if (ra.gatekeeper) { ra.gatekeeper->release(); }
  ra.cv.notify_all();
  if (ra.thread.joinable()) { ra.thread.join(); }
  _impl->readahead.reset();
}

}  // namespace cudf_streaming::prefetch

// ---------------------------------------------------------------------------
// C API — called from Python via ctypes
// ---------------------------------------------------------------------------
extern "C" {

using scan_ctx_t = cudf_streaming::prefetch::scan_context;
using scan_cfg_t = cudf_streaming::prefetch::scan_context_config;

CUDF_STREAMING_PREFETCH_EXPORT
void* cudf_streaming_scan_context_create(
    std::size_t host_memory_bytes,
    int         gpu_device_id,
    const char* s3_endpoint,
    const char* s3_region,
    const char* s3_access_key,
    const char* s3_secret_key,
    const char* s3_session_token)
{
  scan_cfg_t cfg;
  cfg.host_memory_capacity = host_memory_bytes;
  cfg.gpu_device_id        = gpu_device_id;
  if (s3_endpoint && *s3_endpoint) {
    cfg.s3_endpoint           = s3_endpoint;
    cfg.s3_region             = s3_region     ? s3_region     : "";
    cfg.s3_access_key_id      = s3_access_key ? s3_access_key : "";
    cfg.s3_secret_access_key  = s3_secret_key ? s3_secret_key : "";
    cfg.s3_session_token      = s3_session_token ? s3_session_token : "";
  }
  try {
    return cudf_streaming::prefetch::scan_context::create(cfg).release();
  } catch (...) {
    return nullptr;
  }
}

CUDF_STREAMING_PREFETCH_EXPORT
void cudf_streaming_scan_context_destroy(void* ctx)
{
  delete static_cast<scan_ctx_t*>(ctx);
}

CUDF_STREAMING_PREFETCH_EXPORT
void cudf_streaming_scan_context_start_readahead(
    void*              ctx,
    const char**       paths,
    std::size_t        n_splits,
    const std::size_t* flat_range_offsets,
    const std::size_t* flat_range_sizes,
    const std::size_t* split_range_counts,
    int                gpu_id,
    std::size_t        budget)
{
  auto* sc = static_cast<scan_ctx_t*>(ctx);
  std::vector<cudf_streaming::prefetch::scan_split> splits(n_splits);
  std::size_t flat_idx = 0;
  for (std::size_t i = 0; i < n_splits; ++i) {
    splits[i].path = paths[i];
    if (flat_range_offsets && split_range_counts) {
      std::size_t n_ranges = split_range_counts[i];
      splits[i].ranges.reserve(n_ranges);
      for (std::size_t r = 0; r < n_ranges; ++r, ++flat_idx) {
        splits[i].ranges.emplace_back(flat_range_offsets[flat_idx],
                                       flat_range_sizes[flat_idx]);
      }
    }
  }
  std::optional<int> gpu_opt;
  if (gpu_id >= 0) gpu_opt = gpu_id;
  sc->start_readahead(std::move(splits), gpu_opt, budget);
}

CUDF_STREAMING_PREFETCH_EXPORT
void* cudf_streaming_scan_context_get_datasource(void* ctx, std::size_t split_idx)
{
  return static_cast<scan_ctx_t*>(ctx)->get_datasource(split_idx);
}

CUDF_STREAMING_PREFETCH_EXPORT
void cudf_streaming_scan_context_release_datasource(void* ctx, std::size_t split_idx)
{
  static_cast<scan_ctx_t*>(ctx)->release_datasource(split_idx);
}

CUDF_STREAMING_PREFETCH_EXPORT
void cudf_streaming_scan_context_stop_readahead(void* ctx)
{
  static_cast<scan_ctx_t*>(ctx)->stop_readahead();
}

CUDF_STREAMING_PREFETCH_EXPORT
void* cudf_streaming_scan_context_open_datasource(void* ctx, const char* path)
{
  try {
    return static_cast<scan_ctx_t*>(ctx)->open_datasource(path).release();
  } catch (...) {
    return nullptr;
  }
}

CUDF_STREAMING_PREFETCH_EXPORT
void cudf_streaming_scan_context_destroy_datasource(void* ds_ptr)
{
  delete static_cast<cudf_streaming::prefetch::PrefetchDatasource*>(ds_ptr);
}

}  // extern "C"
