// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright 2025, NVIDIA CORPORATION & AFFILIATES.
 * (Ported from cuCascade, commit 1b0e7b6c28dafa43bfe2c48011a3657c0dd6f127)
 */

#include "common.hpp"
#include "fixed_size_host_memory_resource.hpp"
#include "null_device_memory_resource.hpp"
#include "numa_region_pinned_host_allocator.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <mutex>
#include <unordered_map>
#include <vector>

namespace cudf_streaming {
namespace prefetch {
namespace memory {

namespace {

constexpr int kMaxDevices                      = 16;
bool g_p2p_supported[kMaxDevices][kMaxDevices] = {};
bool g_p2p_probed                              = false;
std::mutex& p2p_probe_mutex()
{
  static std::mutex m;
  return m;
}

void run_p2p_probe_locked(int device_count)
{
  int saved_device = 0;
  (void)cudaGetDevice(&saved_device);

  for (int i = 0; i < device_count; ++i) {
    for (int j = 0; j < device_count; ++j) {
      if (i == j) continue;
      int can = 0;
      if (cudaDeviceCanAccessPeer(&can, i, j) != cudaSuccess || !can) {
        (void)cudaGetLastError();
        continue;
      }
      if (cudaSetDevice(i) != cudaSuccess) {
        (void)cudaGetLastError();
        continue;
      }
      cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
      if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) { (void)cudaGetLastError(); }
    }
  }

  constexpr std::size_t kProbeBytes       = 64;
  unsigned char src_pat[kProbeBytes]      = {};
  unsigned char dst_sentinel[kProbeBytes] = {};
  for (std::size_t k = 0; k < kProbeBytes; ++k) {
    src_pat[k]      = static_cast<unsigned char>(0x40 + (k & 0x3F));
    dst_sentinel[k] = 0xAA;
  }
  for (int i = 0; i < device_count; ++i) {
    for (int j = 0; j < device_count; ++j) {
      if (i == j) {
        g_p2p_supported[i][j] = true;
        continue;
      }
      int can = 0;
      if (cudaDeviceCanAccessPeer(&can, i, j) != cudaSuccess || !can) {
        (void)cudaGetLastError();
        g_p2p_supported[i][j] = false;
        continue;
      }
      void* src = nullptr;
      void* dst = nullptr;
      bool ok   = false;
      if (cudaSetDevice(i) == cudaSuccess && cudaMalloc(&src, kProbeBytes) == cudaSuccess &&
          cudaMemcpy(src, src_pat, kProbeBytes, cudaMemcpyHostToDevice) == cudaSuccess &&
          cudaSetDevice(j) == cudaSuccess && cudaMalloc(&dst, kProbeBytes) == cudaSuccess &&
          cudaMemcpy(dst, dst_sentinel, kProbeBytes, cudaMemcpyHostToDevice) == cudaSuccess) {
        if (cudaMemcpyPeer(dst, j, src, i, kProbeBytes) == cudaSuccess &&
            cudaDeviceSynchronize() == cudaSuccess) {
          unsigned char readback[kProbeBytes] = {};
          if (cudaMemcpy(readback, dst, kProbeBytes, cudaMemcpyDeviceToHost) == cudaSuccess) {
            ok = std::memcmp(readback, src_pat, kProbeBytes) == 0;
          }
        }
      }
      if (dst) {
        cudaSetDevice(j);
        cudaFree(dst);
      }
      if (src) {
        cudaSetDevice(i);
        cudaFree(src);
      }
      g_p2p_supported[i][j] = ok;
    }
  }

  for (int i = 0; i < device_count; ++i) {
    for (int j = 0; j < device_count; ++j) {
      if (i == j) continue;
      if (g_p2p_supported[i][j]) continue;
      int can = 0;
      if (cudaDeviceCanAccessPeer(&can, i, j) != cudaSuccess || !can) {
        (void)cudaGetLastError();
        continue;
      }
      cudaSetDevice(i);
      cudaError_t e = cudaDeviceDisablePeerAccess(j);
      if (e != cudaSuccess && e != cudaErrorPeerAccessNotEnabled) { (void)cudaGetLastError(); }
    }
  }
  cudaSetDevice(saved_device);
  (void)cudaGetLastError();

  int broken = 0;
  for (int i = 0; i < device_count; ++i) {
    for (int j = 0; j < device_count; ++j) {
      if (i != j && !g_p2p_supported[i][j]) {
        int can = 0;
        if (cudaDeviceCanAccessPeer(&can, i, j) == cudaSuccess && can) ++broken;
        (void)cudaGetLastError();
      }
    }
  }
  if (broken > 0) {
    fprintf(stderr,
            "[cudf_streaming::prefetch] direct GPU↔GPU peer DMA broken on %d direction(s); "
            "cudaMemcpyPeer* will host-stage automatically.\n",
            broken);
  }
}

bool ensure_p2p_probed()
{
  std::lock_guard<std::mutex> lk(p2p_probe_mutex());
  if (g_p2p_probed) return true;
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
    (void)cudaGetLastError();
    g_p2p_probed = true;
    return false;
  }
  if (device_count > kMaxDevices) device_count = kMaxDevices;
  run_p2p_probe_locked(device_count);
  g_p2p_probed = true;
  return true;
}

bool p2p_dma_works_cached(int src, int dst)
{
  if (src < 0 || dst < 0 || src >= kMaxDevices || dst >= kMaxDevices) return false;
  ensure_p2p_probed();
  return g_p2p_supported[src][dst];
}

void set_access_on_pool(cudaMemPool_t pool, int owner_device_id, int device_count)
{
  for (int peer = 0; peer < device_count; ++peer) {
    if (peer == owner_device_id) { continue; }
    int can_access = 0;
    if (cudaDeviceCanAccessPeer(&can_access, peer, owner_device_id) != cudaSuccess || !can_access) {
      (void)cudaGetLastError();
      continue;
    }
    if (!p2p_dma_works_cached(peer, owner_device_id)) continue;
    cudaMemAccessDesc desc{};
    desc.location.type = cudaMemLocationTypeDevice;
    desc.location.id   = peer;
    desc.flags         = cudaMemAccessFlagsProtReadWrite;
    if (cudaMemPoolSetAccess(pool, &desc, 1) != cudaSuccess) {
      (void)cudaGetLastError();
    }
  }
}
}  // namespace

void enable_pool_peer_access_for_all_visible_devices(cudaMemPool_t pool, int owner_device_id)
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
    (void)cudaGetLastError();
    return;
  }
  ensure_p2p_probed();
  set_access_on_pool(pool, owner_device_id, device_count);

  rmm::cuda_set_device_raii set_device(rmm::cuda_device_id{owner_device_id});
  cudaMemPool_t default_pool{};
  if (cudaDeviceGetMemPool(&default_pool, owner_device_id) == cudaSuccess) {
    set_access_on_pool(default_pool, owner_device_id, device_count);
  } else {
    (void)cudaGetLastError();
  }
}

cuda::mr::any_resource<cuda::mr::device_accessible> make_default_gpu_memory_resource(
  int device_id, std::size_t capacity)
{
  rmm::cuda_set_device_raii set_device(rmm::cuda_device_id{device_id});
  return {rmm::mr::cuda_async_memory_resource(capacity)};
}

cuda::mr::any_resource<cuda::mr::device_accessible, cuda::mr::host_accessible>
make_default_host_memory_resource(int numa_node_id, [[maybe_unused]] std::size_t capacity)
{
  return make_default_host_memory_resource(numa_node_id, capacity, false);
}

cuda::mr::any_resource<cuda::mr::device_accessible, cuda::mr::host_accessible>
make_default_host_memory_resource(int numa_node_id,
                                  [[maybe_unused]] std::size_t capacity,
                                  bool make_portable)
{
  return {numa_region_pinned_host_memory_resource(numa_node_id, make_portable)};
}

DeviceMemoryResourceFactoryFn make_default_allocator_for_tier(Tier tier)
{
  if (tier == Tier::GPU) {
    return make_default_gpu_memory_resource;
  } else if (tier == Tier::HOST) {
    return [](int numa_node_id, std::size_t capacity) {
      return make_default_host_memory_resource(numa_node_id, capacity);
    };
  } else {
    return [](int, std::size_t) {
      return cuda::mr::any_resource<cuda::mr::device_accessible>{null_device_memory_resource{}};
    };
  }
}

bool probe_peer_dma_works(int src_device, int dst_device)
{
  if (src_device == dst_device) return true;
  return p2p_dma_works_cached(src_device, dst_device);
}

int disable_peer_access_where_broken(std::vector<cudaMemPool_t> const& pools_by_device)
{
  (void)pools_by_device;
  ensure_p2p_probed();
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  if (device_count > kMaxDevices) device_count = kMaxDevices;
  int disabled = 0;
  for (int i = 0; i < device_count; ++i) {
    for (int j = 0; j < device_count; ++j) {
      if (i == j) continue;
      int can = 0;
      if (cudaDeviceCanAccessPeer(&can, i, j) != cudaSuccess || !can) {
        (void)cudaGetLastError();
        continue;
      }
      if (!g_p2p_supported[j][i]) ++disabled;
    }
  }
  return disabled;
}

//===----------------------------------------------------------------------===//
// HOST pool registry
//===----------------------------------------------------------------------===//

namespace {
std::mutex& host_pool_registry_mutex()
{
  static std::mutex m;
  return m;
}

std::unordered_map<int, std::vector<fixed_size_host_memory_resource*>>& host_pool_registry()
{
  static std::unordered_map<int, std::vector<fixed_size_host_memory_resource*>> r;
  return r;
}
}  // namespace

void register_host_pool(int numa_id, fixed_size_host_memory_resource* pool)
{
  if (pool == nullptr) { return; }
  std::lock_guard<std::mutex> g(host_pool_registry_mutex());
  auto& pools = host_pool_registry()[numa_id];
  for (auto* existing : pools) {
    if (existing == pool) { return; }
  }
  pools.push_back(pool);
}

void unregister_host_pool(int numa_id, fixed_size_host_memory_resource* pool) noexcept
{
  std::lock_guard<std::mutex> g(host_pool_registry_mutex());
  auto& reg = host_pool_registry();
  auto it   = reg.find(numa_id);
  if (it == reg.end()) { return; }
  auto& pools = it->second;
  for (auto pit = pools.begin(); pit != pools.end(); ++pit) {
    if (*pit == pool) {
      pools.erase(pit);
      break;
    }
  }
  if (pools.empty()) { reg.erase(it); }
}

fixed_size_host_memory_resource* find_host_pool(int numa_id) noexcept
{
  std::lock_guard<std::mutex> g(host_pool_registry_mutex());
  auto& reg = host_pool_registry();
  if (auto it = reg.find(numa_id); it != reg.end() && !it->second.empty()) {
    return it->second.back();
  }
  for (auto& entry : reg) {
    if (!entry.second.empty()) { return entry.second.back(); }
  }
  return nullptr;
}

}  // namespace memory
}  // namespace prefetch
}  // namespace cudf_streaming

namespace std {
std::size_t
hash<cudf_streaming::prefetch::memory::memory_space_id>::operator()(
  const cudf_streaming::prefetch::memory::memory_space_id& p) const
{
  return p.uuid();
}
}  // namespace std
