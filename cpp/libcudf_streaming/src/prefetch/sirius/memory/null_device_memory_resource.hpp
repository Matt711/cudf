// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda/memory_resource>
#include <cuda/stream>

#include <cstddef>

namespace cudf_streaming {
namespace prefetch {
namespace memory {

/**
 * A no-op memory resource used for DISK tier to satisfy API requirements.
 * - allocate always returns nullptr
 * - deallocate is a no-op
 */
class null_device_memory_resource {
 public:
  null_device_memory_resource()  = default;
  ~null_device_memory_resource() = default;

  void* allocate([[maybe_unused]] ::cuda::stream_ref stream,
                 [[maybe_unused]] std::size_t bytes,
                 [[maybe_unused]] std::size_t alignment = alignof(std::max_align_t))
  {
    return nullptr;
  }

  void deallocate([[maybe_unused]] ::cuda::stream_ref stream,
                  [[maybe_unused]] void* p,
                  [[maybe_unused]] std::size_t bytes,
                  [[maybe_unused]] std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
  }

  void* allocate_sync([[maybe_unused]] std::size_t bytes,
                      [[maybe_unused]] std::size_t alignment = alignof(std::max_align_t))
  {
    return nullptr;
  }

  void deallocate_sync([[maybe_unused]] void* p,
                       [[maybe_unused]] std::size_t bytes,
                       [[maybe_unused]] std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
  }

  bool operator==(null_device_memory_resource const& other) const noexcept
  {
    return this == &other;
  }

  friend void get_property(null_device_memory_resource const&,
                           ::cuda::mr::device_accessible) noexcept
  {
  }
};

}  // namespace memory
}  // namespace prefetch
}  // namespace cudf_streaming
