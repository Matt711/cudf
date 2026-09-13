// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright 2025, NVIDIA CORPORATION & AFFILIATES.
 * (Ported from cuCascade, commit 1b0e7b6c28dafa43bfe2c48011a3657c0dd6f127)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "detail/reservation_aware_resource_adaptor_impl.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <memory>

namespace cudf_streaming {
namespace prefetch {
namespace memory {

class reservation_aware_resource_adaptor
  : public ::cuda::mr::shared_resource<detail::reservation_aware_resource_adaptor_impl> {
  using shared_base = ::cuda::mr::shared_resource<detail::reservation_aware_resource_adaptor_impl>;
  using impl_type   = detail::reservation_aware_resource_adaptor_impl;

 public:
  using device_reserved_arena        = impl_type::device_reserved_arena;
  using stream_ordered_tracker_state = impl_type::stream_ordered_tracker_state;
  using allocation_tracker_iface     = impl_type::allocation_tracker_iface;
  using AllocationTrackingScope      = impl_type::AllocationTrackingScope;

  friend void get_property(reservation_aware_resource_adaptor const&,
                           ::cuda::mr::device_accessible) noexcept
  {
  }

  explicit reservation_aware_resource_adaptor(
    memory_space_id space_id,
    rmm::device_async_resource_ref upstream,
    std::size_t capacity,
    std::unique_ptr<reservation_limit_policy> stream_reservation_policy = nullptr,
    std::unique_ptr<oom_handling_policy> default_oom_policy             = nullptr,
    AllocationTrackingScope tracking_scope = AllocationTrackingScope::PER_STREAM,
    cudaMemPool_t pool_handle              = nullptr);

  explicit reservation_aware_resource_adaptor(
    memory_space_id space_id,
    rmm::device_async_resource_ref upstream,
    std::size_t memory_limit,
    std::size_t capacity,
    std::unique_ptr<reservation_limit_policy> stream_reservation_policy = nullptr,
    std::unique_ptr<oom_handling_policy> default_oom_policy             = nullptr,
    AllocationTrackingScope tracking_scope = AllocationTrackingScope::PER_STREAM,
    cudaMemPool_t pool_handle              = nullptr);

  rmm::device_async_resource_ref get_upstream_resource() const noexcept;
  std::size_t get_available_memory() const noexcept;
  std::size_t get_available_memory(rmm::cuda_stream_view stream) const noexcept;
  std::size_t get_available_memory_print(rmm::cuda_stream_view stream) const noexcept;
  std::size_t get_allocated_bytes(rmm::cuda_stream_view stream) const;
  std::size_t get_peak_allocated_bytes(rmm::cuda_stream_view stream) const;
  std::size_t get_total_allocated_bytes() const;
  std::size_t get_peak_total_allocated_bytes() const;
  void reset_peak_allocated_bytes(rmm::cuda_stream_view stream);
  std::size_t get_total_reserved_bytes() const;
  bool is_stream_tracked(rmm::cuda_stream_view stream) const;

  std::unique_ptr<reserved_arena> reserve(
    std::size_t bytes, std::unique_ptr<event_notifier> release_notifer = nullptr);

  std::unique_ptr<reserved_arena> reserve_upto(
    std::size_t bytes, std::unique_ptr<event_notifier> release_notifer = nullptr);

  std::size_t get_active_reservation_count() const noexcept;

  bool attach_reservation_to_tracker(
    rmm::cuda_stream_view stream,
    std::unique_ptr<reservation> reserved_bytes,
    std::unique_ptr<reservation_limit_policy> stream_reservation_policy = nullptr,
    std::unique_ptr<oom_handling_policy> stream_oom_policy              = nullptr);

  void reset_stream_reservation(rmm::cuda_stream_view stream);
  void set_default_policy(std::unique_ptr<reservation_limit_policy> policy);
  const reservation_limit_policy& get_default_reservation_policy() const;
  const oom_handling_policy& get_default_oom_handling_policy() const;

  void* allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    return get().allocate(stream, bytes, alignment);
  }

  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    get().deallocate(stream, ptr, bytes, alignment);
  }
};

}  // namespace memory
}  // namespace prefetch
}  // namespace cudf_streaming
