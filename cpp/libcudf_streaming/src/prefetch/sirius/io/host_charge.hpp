// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// host_charge.hpp — Bridge between the prefetch cache's internal host-memory
// accounting and rapidsmpf's MemoryReservation system.
//
// Design note (A03): The prefetch cache tracks a "live" byte count (pages
// currently pinned or in-flight) and maintains a "floor" reservation that
// ensures the memory manager can never reclaim pages still needed by active
// reads.  To integrate with rapidsmpf's cooperative-scheduling model we must
// mirror the logical delta max(floor, live) → mpf_reservation.  host_charge
// owns that retained MemoryReservation and keeps it sized to max(floor, live)
// on every adjust() call.

#pragma once

#include <rapidsmpf/memory/memory_reservation.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <mutex>
#include <optional>

namespace cudf_streaming::prefetch::io {

/**
 * @brief Tracks pinned host-memory charge against a rapidsmpf BufferResource.
 *
 * The prefetch layer has two parallel accounting dimensions:
 *   - floor_bytes: minimum bytes that must always stay reserved (e.g. pages
 *     that have been handed to the caller and cannot be evicted).
 *   - live_bytes: bytes actually allocated right now (in-flight + cached).
 *
 * host_charge keeps a rapidsmpf::MemoryReservation sized to
 *   charged_bytes = max(floor_bytes, live_bytes)
 * and lazily splits or clears it on every adjust() to track the high-water
 * mark.  When the reservation is smaller than the new target we try to grow
 * it; if the BufferResource can't accommodate the growth we return false so
 * the caller can stall or shed load.
 */
class host_charge {
 public:
  /**
   * @brief Construct with a reference to the rapidsmpf BufferResource.
   *
   * @param br  The buffer resource to draw host reservations from.
   *            Must outlive this object.
   * @param mem_type  Memory type to reserve (typically HOST_PINNED).
   */
  explicit host_charge(rapidsmpf::BufferResource& br,
                       rapidsmpf::MemoryType mem_type = rapidsmpf::MemoryType::HOST)
    : _br{br}, _mem_type{mem_type}, _reserved{0}
  {
  }

  host_charge(host_charge const&)            = delete;
  host_charge& operator=(host_charge const&) = delete;

  host_charge(host_charge&&)            = delete;
  host_charge& operator=(host_charge&&) = delete;

  ~host_charge() { release(); }

  /**
   * @brief Update the charge to reflect a new floor and/or live byte count.
   *
   * Computes target = max(floor_bytes, live_bytes) and adjusts the retained
   * MemoryReservation to match.  Returns false if growth was requested but
   * the BufferResource could not satisfy it — the caller should stall.
   *
   * @param floor_bytes  Bytes that must be kept reserved unconditionally.
   * @param live_bytes   Bytes currently allocated by the prefetch cache.
   * @return true if the reservation now covers the requested charge.
   */
  [[nodiscard]] bool adjust(std::size_t floor_bytes, std::size_t live_bytes)
  {
    std::size_t const target = std::max(floor_bytes, live_bytes);
    std::lock_guard<std::mutex> lk{_mu};
    return _set_reserved(target);
  }

  /**
   * @brief Unconditionally release all held reservation bytes back to the pool.
   */
  void release()
  {
    std::lock_guard<std::mutex> lk{_mu};
    if (_reservation.has_value()) {
      _reservation->clear();
      _reservation.reset();
    }
    _reserved = 0;
  }

  /**
   * @brief Return the number of bytes currently held in the reservation.
   */
  [[nodiscard]] std::size_t reserved_bytes() const noexcept
  {
    std::lock_guard<std::mutex> lk{_mu};
    return _reserved;
  }

 private:
  bool _set_reserved(std::size_t target)
  {
    if (target == _reserved) return true;

    if (target == 0) {
      if (_reservation.has_value()) {
        _reservation->clear();
        _reservation.reset();
      }
      _reserved = 0;
      return true;
    }

    if (target < _reserved) {
      // Shrink: split off the surplus and discard it.
      if (_reservation.has_value()) {
        std::size_t const surplus = _reserved - target;
        auto discard              = _reservation->split(surplus);
        // discard is released on scope exit
        (void)discard;
        _reserved = target;
      }
      return true;
    }

    // Grow: try to reserve the additional bytes.
    // rapidsmpf::BufferResource::reserve(MemoryType, size_t, AllowOverbooking)
    // returns pair<MemoryReservation, size_t> (actual bytes granted).
    try {
      if (_reservation.has_value()) {
        // Simplest safe merge: release existing and re-reserve the full target.
        _reservation->clear();
        _reservation.reset();
        _reserved = 0;
      }
      auto [full_res, granted] =
        _br.reserve(_mem_type, target, rapidsmpf::AllowOverbooking::NO);
      if (granted < target) {
        // BufferResource granted fewer bytes than requested.
        full_res.clear();
        return false;
      }
      _reservation.emplace(std::move(full_res));
      _reserved = target;
      return true;
    } catch (...) {
      return false;
    }
  }

  rapidsmpf::BufferResource& _br;
  rapidsmpf::MemoryType      _mem_type;
  mutable std::mutex         _mu;
  std::size_t                _reserved{0};
  std::optional<rapidsmpf::MemoryReservation> _reservation;
};

}  // namespace cudf_streaming::prefetch::io
