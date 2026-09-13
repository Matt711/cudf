// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright 2025, NVIDIA CORPORATION & AFFILIATES.
 * (Ported from cuCascade, commit 1b0e7b6c28dafa43bfe2c48011a3657c0dd6f127)
 */

#include "common.hpp"
#include "memory_reservation.hpp"
#include "memory_reservation_manager.hpp"
#include "memory_space.hpp"

#include <rmm/cuda_device.hpp>

namespace cudf_streaming {
namespace prefetch {
namespace memory {

// Provide out-of-line definition for the virtual destructor to satisfy linker
reservation_limit_policy::~reservation_limit_policy() = default;

//===----------------------------------------------------------------------===//
// reservation Implementation
//===----------------------------------------------------------------------===//

std::unique_ptr<reservation> reservation::create(memory_space& space,
                                                 std::unique_ptr<reserved_arena> arena)
{
  return std::unique_ptr<reservation>(new reservation(&space, std::move(arena)));
}

reservation::reservation(const memory_space* space, std::unique_ptr<reserved_arena> arena)
  : _space(space), _arena(std::move(arena))
{
  if (_arena == nullptr) {
    throw std::runtime_error(
      "cudf_streaming::prefetch::memory::reservation: trying to create reservation without a "
      "reserved arena\n");
  }
}

reservation::~reservation() = default;

rmm::device_async_resource_ref reservation::get_memory_resource() const noexcept
{
  return _space->get_default_allocator();
}

const memory_space& reservation::get_memory_space() const noexcept { return *_space; }

std::size_t reservation::size() const noexcept
{
  return static_cast<std::size_t>(std::max(int64_t{0}, _arena->size()));
}

[[nodiscard]] Tier reservation::tier() const noexcept { return _space->get_tier(); }

[[nodiscard]] int reservation::device_id() const noexcept { return _space->get_device_id(); }

bool reservation::grow_by(std::size_t additional_bytes) { return _arena->grow_by(additional_bytes); }

void reservation::shrink_to_fit() { return _arena->shrink_to_fit(); }

//===----------------------------------------------------------------------===//
// Reservation Limit Policy Implementations
//===----------------------------------------------------------------------===//

ignore_reservation_limit_policy::ignore_reservation_limit_policy() = default;

void ignore_reservation_limit_policy::handle_over_reservation(
  [[maybe_unused]] rmm::cuda_stream_view stream,
  [[maybe_unused]] std::size_t requested_bytes,
  [[maybe_unused]] std::size_t current_allocated,
  [[maybe_unused]] reserved_arena* reserved_bytes)
{
}

std::string ignore_reservation_limit_policy::get_policy_name() const { return "ignore"; }

fail_reservation_limit_policy::fail_reservation_limit_policy() = default;

void fail_reservation_limit_policy::handle_over_reservation(
  [[maybe_unused]] rmm::cuda_stream_view stream,
  std::size_t requested_bytes,
  std::size_t current_allocated,
  reserved_arena* reserved_bytes)
{
  auto reservation_size =
    reserved_bytes ? static_cast<std::size_t>(std::max(int64_t{0}, reserved_bytes->size())) : 0UL;
  RMM_FAIL("Allocation of " + std::to_string(requested_bytes) +
             " bytes would exceed stream reservation of " + std::to_string(reservation_size) +
             " bytes (current: " + std::to_string(current_allocated) + " bytes)",
           rmm::out_of_memory);
}

std::string fail_reservation_limit_policy::get_policy_name() const { return "fail"; }

increase_reservation_limit_policy::increase_reservation_limit_policy() = default;

increase_reservation_limit_policy::increase_reservation_limit_policy(double padding_factor,
                                                                     bool allow_beyond_limit)
  : _padding_factor(padding_factor), allow_reservation_beyond_limit(allow_beyond_limit)
{
}

void increase_reservation_limit_policy::handle_over_reservation(
  [[maybe_unused]] rmm::cuda_stream_view stream,
  std::size_t requested_bytes,
  std::size_t current_allocated,
  reserved_arena* reserved_bytes)
{
  if (!reserved_bytes) { RMM_FAIL("No reservation set for stream", rmm::out_of_memory); }

  std::size_t post_allocation_size = current_allocated + requested_bytes;
  auto current_reserved = static_cast<std::size_t>(std::max(int64_t{0}, reserved_bytes->size()));
  std::size_t extra_space_needed = post_allocation_size - current_reserved;

  std::size_t padded_reservation =
    static_cast<std::size_t>(static_cast<double>(extra_space_needed) * _padding_factor);

  if (!reserved_bytes->grow_by(padded_reservation)) {
    if (!reserved_bytes->grow_by(extra_space_needed)) {
      RMM_FAIL("Failed to increase stream reservation from " +
                 std::to_string(reserved_bytes->size()) + " to " +
                 std::to_string(extra_space_needed) + " bytes",
               rmm::out_of_memory);
    }
  }
}

std::string increase_reservation_limit_policy::get_policy_name() const { return "increase"; }

std::unique_ptr<reservation_limit_policy> make_default_reservation_limit_policy()
{
  return std::make_unique<ignore_reservation_limit_policy>();
}

}  // namespace memory
}  // namespace prefetch
}  // namespace cudf_streaming
