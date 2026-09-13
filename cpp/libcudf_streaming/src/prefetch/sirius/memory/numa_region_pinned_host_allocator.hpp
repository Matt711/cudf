// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda/memory_resource>
#include <cuda/stream>

#include <cstddef>

namespace cudf_streaming {
namespace prefetch {
namespace memory {

class numa_region_pinned_host_memory_resource final {
 public:
  explicit numa_region_pinned_host_memory_resource(int numa_node, bool make_portable = false);
  ~numa_region_pinned_host_memory_resource()                                              = default;
  numa_region_pinned_host_memory_resource(numa_region_pinned_host_memory_resource const&) = default;
  numa_region_pinned_host_memory_resource(numa_region_pinned_host_memory_resource&&)      = default;
  numa_region_pinned_host_memory_resource& operator=(
    numa_region_pinned_host_memory_resource const&) = default;
  numa_region_pinned_host_memory_resource& operator=(numa_region_pinned_host_memory_resource&&) =
    default;

  /**
   * @brief Allocates pinned host memory of size at least \p bytes bytes.
   */
  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t));

  /**
   * @brief Deallocate memory pointed to by \p ptr.
   */
  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept;

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t));

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept;

  [[nodiscard]] bool operator==(
    numa_region_pinned_host_memory_resource const& other) const noexcept;

  /**
   * @brief Enables the `::cuda::mr::device_accessible` property
   */
  friend void get_property(numa_region_pinned_host_memory_resource const&,
                           ::cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Enables the `::cuda::mr::host_accessible` property
   */
  friend void get_property(numa_region_pinned_host_memory_resource const&,
                           ::cuda::mr::host_accessible) noexcept
  {
  }

 private:
  static int cuda_host_flags(int numa_node, bool make_portable) noexcept;

  int _numa_node{-1};
  int _cuda_host_flags{0};
};

}  // namespace memory
}  // namespace prefetch
}  // namespace cudf_streaming
