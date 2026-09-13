// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright 2025, NVIDIA CORPORATION & AFFILIATES.
 * (Ported from cuCascade, commit 1b0e7b6c28dafa43bfe2c48011a3657c0dd6f127)
 */

#include "disk_access_limiter.hpp"

namespace cudf_streaming {
namespace prefetch {
namespace memory {

disk_access_limiter::disk_access_limiter(memory_space_id space_id,
                                         std::size_t capacity,
                                         std::string_view mount_path)
  : _space_id(space_id), _memory_limit(capacity), _capacity(capacity), mounting_path_(mount_path)
{
}

disk_access_limiter::disk_access_limiter(memory_space_id space_id,
                                         std::size_t memory_limit,
                                         std::size_t capacity,
                                         std::string_view mount_path)
  : _space_id(space_id),
    _memory_limit(memory_limit),
    _capacity(capacity),
    mounting_path_(mount_path)
{
}

std::size_t disk_access_limiter::get_available_memory() const noexcept
{
  auto current = _total_allocated_bytes.load();
  if (current <= _capacity) {
    return _capacity - current;
  } else {
    return 0;
  }
}

std::size_t disk_access_limiter::get_peak_reserved_bytes() const
{
  return _peak_total_allocated_bytes.peak();
}

std::size_t disk_access_limiter::get_total_reserved_bytes() const
{
  return _total_allocated_bytes.load();
}

std::unique_ptr<reserved_arena> disk_access_limiter::reserve(
  std::size_t bytes, std::unique_ptr<event_notifier> release_notifer)
{
  if (do_reserve(bytes, _memory_limit)) {
    auto slot = std::make_unique<disk_reserved_arena>(
      *this, bytes, mounting_path_, std::move(release_notifer));
    _total_reservation_count.fetch_add(1);
    return slot;
  }
  return nullptr;
}

std::unique_ptr<reserved_arena> disk_access_limiter::reserve_upto(
  std::size_t bytes, std::unique_ptr<event_notifier> release_notifer)
{
  auto reserved_bytes = do_reserve_upto(bytes, _memory_limit);
  auto slot           = std::make_unique<disk_reserved_arena>(
    *this, reserved_bytes, mounting_path_, std::move(release_notifer));
  _total_reservation_count.fetch_add(1);
  return slot;
}

bool disk_access_limiter::grow_reservation_by(reserved_arena& arena, std::size_t bytes)
{
  auto* disk_reservation = dynamic_cast<disk_reserved_arena*>(&arena);
  if (do_reserve(bytes, _memory_limit)) {
    disk_reservation->_size += static_cast<int64_t>(bytes);
    return true;
  }
  return false;
}

void disk_access_limiter::shrink_reservation_to_fit(reserved_arena& arena)
{
  auto size_to_release = static_cast<std::size_t>(std::max(int64_t{0}, arena.size()));
  _total_allocated_bytes.sub(size_to_release);
  arena._size = 0;
}

std::size_t disk_access_limiter::get_active_reservation_count() const noexcept
{
  return _total_reservation_count.load();
}

bool disk_access_limiter::do_reserve(std::size_t size_bytes, std::size_t limit_bytes)
{
  auto [success, post_allocation_size] = _total_allocated_bytes.try_add(size_bytes, limit_bytes);
  if (success) { _peak_total_allocated_bytes.update_peak(post_allocation_size); }
  return success;
}

std::size_t disk_access_limiter::do_reserve_upto(std::size_t size_bytes, std::size_t limit_bytes)
{
  auto post_allocation_size = _total_allocated_bytes.add_bounded(size_bytes, limit_bytes);
  if (size_bytes > 0) { _peak_total_allocated_bytes.update_peak(post_allocation_size); }
  return size_bytes;
}

void disk_access_limiter::do_release_reservation(disk_reserved_arena* arena) noexcept
{
  if (!arena) return;
  auto size_to_release = static_cast<std::size_t>(std::max(int64_t{0}, arena->size()));
  _total_allocated_bytes.sub(size_to_release);
  _total_reservation_count.fetch_sub(1);
  arena->_size = 0;
}

}  // namespace memory
}  // namespace prefetch
}  // namespace cudf_streaming
