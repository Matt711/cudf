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

#include "../common.hpp"
#include "../error.hpp"
#include "../memory_reservation.hpp"
#include "../notification_channel.hpp"
#include "../oom_handling_policy.hpp"
#include "../../error.hpp"
#include "../../utils/atomics.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <atomic>
#include <memory>
#include <set>

namespace cudf_streaming {
namespace prefetch {
namespace memory {
namespace detail {

class reservation_aware_resource_adaptor_impl {
 public:
  struct device_reserved_arena : public reserved_arena {
    friend class reservation_aware_resource_adaptor_impl;

    explicit device_reserved_arena(reservation_aware_resource_adaptor_impl& impl,
                                   std::size_t bytes,
                                   std::unique_ptr<event_notifier> notifier)
      : reserved_arena(static_cast<int64_t>(bytes), std::move(notifier)), _impl(&impl)
    {
    }

    ~device_reserved_arena() noexcept { _impl->do_release_reservation(this); }

    bool grow_by(std::size_t additional_bytes) final
    {
      return _impl->grow_reservation_by(*this, additional_bytes);
    }

    void shrink_to_fit() final { _impl->shrink_reservation_to_fit(*this); }

    [[nodiscard]] std::size_t get_available_memory() const noexcept
    {
      auto current = allocated_bytes.value();
      auto sz      = this->size();
      return current < sz ? static_cast<std::size_t>(sz - current) : 0UL;
    }

    utils::atomic_bounded_counter<std::int64_t> allocated_bytes{0LL};
    utils::atomic_peak_tracker<std::int64_t> peak_allocated_bytes{0LL};

   private:
    reservation_aware_resource_adaptor_impl* _impl;
  };

  struct stream_ordered_tracker_state {
    std::unique_ptr<device_reserved_arena> memory_reservation;
    std::unique_ptr<reservation_limit_policy> reservation_policy;
    std::unique_ptr<oom_handling_policy> oom_policy;

    friend class reservation_aware_resource_adaptor_impl;

    explicit stream_ordered_tracker_state(
      std::unique_ptr<device_reserved_arena> arena,
      std::unique_ptr<reservation_limit_policy> reservation_policy,
      std::unique_ptr<oom_handling_policy> oom_policy);

    std::size_t check_reservation_and_handle_overflow(reservation_aware_resource_adaptor_impl& impl,
                                                      std::size_t allocation_size,
                                                      rmm::cuda_stream_view stream);

   private:
    mutable std::mutex _arbitration_mutex;
  };

  struct allocation_tracker_iface {
    virtual ~allocation_tracker_iface() = default;

    virtual void reset_tracker_state(rmm::cuda_stream_view stream) = 0;

    virtual void assign_reservation_to_tracker(rmm::cuda_stream_view stream,
                                               std::unique_ptr<device_reserved_arena> reservation,
                                               std::unique_ptr<reservation_limit_policy> policy,
                                               std::unique_ptr<oom_handling_policy> oom_policy) = 0;

    virtual stream_ordered_tracker_state* get_tracker_state(rmm::cuda_stream_view stream) = 0;

    virtual const stream_ordered_tracker_state* get_tracker_state(
      rmm::cuda_stream_view stream) const = 0;
  };

  enum class AllocationTrackingScope {
    PER_STREAM,
    PER_THREAD
  };

  explicit reservation_aware_resource_adaptor_impl(
    memory_space_id space_id,
    rmm::device_async_resource_ref upstream,
    std::size_t capacity,
    std::unique_ptr<reservation_limit_policy> stream_reservation_policy = nullptr,
    std::unique_ptr<oom_handling_policy> default_oom_policy             = nullptr,
    AllocationTrackingScope tracking_scope = AllocationTrackingScope::PER_STREAM,
    cudaMemPool_t pool_handle              = nullptr);

  explicit reservation_aware_resource_adaptor_impl(
    memory_space_id space_id,
    rmm::device_async_resource_ref upstream,
    std::size_t memory_limit,
    std::size_t capacity,
    std::unique_ptr<reservation_limit_policy> stream_reservation_policy = nullptr,
    std::unique_ptr<oom_handling_policy> default_oom_policy             = nullptr,
    AllocationTrackingScope tracking_scope = AllocationTrackingScope::PER_STREAM,
    cudaMemPool_t pool_handle              = nullptr);

  ~reservation_aware_resource_adaptor_impl() = default;

  reservation_aware_resource_adaptor_impl(const reservation_aware_resource_adaptor_impl&) = delete;
  reservation_aware_resource_adaptor_impl& operator=(
    const reservation_aware_resource_adaptor_impl&)                                  = delete;
  reservation_aware_resource_adaptor_impl(reservation_aware_resource_adaptor_impl&&) = delete;
  reservation_aware_resource_adaptor_impl& operator=(reservation_aware_resource_adaptor_impl&&) =
    delete;

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

  void* allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment);

  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment) noexcept;

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    auto def_stream = ::cuda::stream_ref{cudaStream_t{nullptr}};
    auto* ptr       = allocate(def_stream, bytes, alignment);
    def_stream.sync();
    return ptr;
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    auto def_stream = ::cuda::stream_ref{cudaStream_t{nullptr}};
    deallocate(def_stream, ptr, bytes, alignment);
    def_stream.sync();
  }

  bool operator==(reservation_aware_resource_adaptor_impl const& other) const noexcept;

  friend void get_property(reservation_aware_resource_adaptor_impl const&,
                           ::cuda::mr::device_accessible) noexcept
  {
  }

 private:
  bool grow_reservation_by(device_reserved_arena& arena, std::size_t bytes);
  void shrink_reservation_to_fit(device_reserved_arena& arena);
  void* do_allocate_managed(std::size_t bytes, rmm::cuda_stream_view stream);
  void* do_allocate_managed(std::size_t bytes,
                            stream_ordered_tracker_state* state,
                            rmm::cuda_stream_view stream);
  void* do_allocate_unmanaged(std::size_t bytes,
                              std::size_t tracking_bytes,
                              rmm::cuda_stream_view stream);
  bool do_reserve(std::size_t size_bytes, std::size_t limit_bytes);
  std::size_t do_reserve_upto(std::size_t size_bytes, std::size_t limit_bytes);
  void do_release_reservation(device_reserved_arena* reservation) noexcept;

  memory_space_id _space_id;
  rmm::device_async_resource_ref _upstream;
  cudaMemPool_t _pool_handle{nullptr};
  const std::size_t _memory_limit;
  const std::size_t _capacity;
  std::unique_ptr<allocation_tracker_iface> _allocation_tracker;
  std::atomic<size_t> _total_reserved_bytes{0UL};
  std::atomic<size_t> _number_of_allocations{0UL};
  utils::atomic_bounded_counter<std::size_t> _total_allocated_bytes{0UL};
  utils::atomic_peak_tracker<std::size_t> _peak_total_allocated_bytes{0UL};
  std::unique_ptr<reservation_limit_policy> _default_reservation_policy;
  std::unique_ptr<oom_handling_policy> _default_oom_policy;
};

}  // namespace detail
}  // namespace memory
}  // namespace prefetch
}  // namespace cudf_streaming
