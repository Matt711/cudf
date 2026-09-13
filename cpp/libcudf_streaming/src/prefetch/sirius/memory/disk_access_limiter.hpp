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

#include "common.hpp"
#include "memory_reservation.hpp"
#include "notification_channel.hpp"

#include "../utils/atomics.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <atomic>
#include <memory>
#include <string_view>

namespace cudf_streaming {
namespace prefetch {
namespace memory {

class disk_access_limiter {
 public:
  struct disk_reserved_arena : public reserved_arena {
    friend class disk_access_limiter;

    explicit disk_reserved_arena(disk_access_limiter& mr,
                                 std::size_t bytes,
                                 std::string_view base_fname,
                                 std::unique_ptr<event_notifier> notifier)
      : reserved_arena(static_cast<int64_t>(bytes), std::move(notifier)),
        _base_name(base_fname.data(), base_fname.size()),
        _mr(&mr)
    {
    }

    ~disk_reserved_arena() noexcept { _mr->do_release_reservation(this); }

    std::string_view base_name() const noexcept { return _base_name; }

    bool grow_by(std::size_t) final { return false; }

    void shrink_to_fit() final {}

   private:
    std::string _base_name;
    disk_access_limiter* _mr;
  };

  explicit disk_access_limiter(memory_space_id space_id,
                               std::size_t capacity,
                               std::string_view mount_path);

  explicit disk_access_limiter(memory_space_id space_id,
                               std::size_t memory_limit,
                               std::size_t capacity,
                               std::string_view mount_path);

  ~disk_access_limiter() = default;

  [[nodiscard]] std::string_view get_mount_path() const noexcept { return mounting_path_; }

  [[nodiscard]] std::size_t get_available_memory() const noexcept;

  [[nodiscard]] std::size_t get_total_reserved_bytes() const;

  [[nodiscard]] std::size_t get_peak_reserved_bytes() const;

  std::unique_ptr<reserved_arena> reserve(std::size_t bytes,
                                          std::unique_ptr<event_notifier> release_notifer = nullptr);

  std::unique_ptr<reserved_arena> reserve_upto(
    std::size_t bytes, std::unique_ptr<event_notifier> release_notifer = nullptr);

  std::size_t get_active_reservation_count() const noexcept;

 private:
  bool do_reserve(std::size_t size_bytes, std::size_t limit_bytes);
  std::size_t do_reserve_upto(std::size_t size_bytes, std::size_t limit_bytes);
  void do_release_reservation(disk_reserved_arena* reservation) noexcept;
  bool grow_reservation_by(reserved_arena& res, std::size_t bytes);
  void shrink_reservation_to_fit(reserved_arena& res);

  memory_space_id _space_id;
  const std::size_t _memory_limit;
  const std::size_t _capacity;

  utils::atomic_bounded_counter<std::size_t> _total_allocated_bytes{0};
  utils::atomic_peak_tracker<std::size_t> _peak_total_allocated_bytes{0};
  std::atomic<std::size_t> _total_reservation_count{0};
  std::string mounting_path_;
};

}  // namespace memory
}  // namespace prefetch
}  // namespace cudf_streaming
