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

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>

namespace cudf_streaming::prefetch::cuda {

namespace event {
enum class query_result { success, in_progress, error };
}

/**
 * @brief Non-owning view of a CUDA event.
 */
class cuda_event_view {
 public:
  cuda_event_view()                                           = default;
  ~cuda_event_view()                                          = default;
  cuda_event_view(cuda_event_view const&) noexcept            = default;
  cuda_event_view(cuda_event_view&&) noexcept                 = default;
  cuda_event_view& operator=(cuda_event_view const&) noexcept = default;
  cuda_event_view& operator=(cuda_event_view&&) noexcept      = default;

  cuda_event_view(int)            = delete;
  cuda_event_view(std::nullptr_t) = delete;

  cuda_event_view(cudaEvent_t event) noexcept : event_{event} {}

  [[nodiscard]] cudaEvent_t value() const noexcept { return event_; }
  operator cudaEvent_t() const noexcept { return event_; }

  void record(rmm::cuda_stream_view stream = rmm::cuda_stream_default);
  void wait(rmm::cuda_stream_view stream = rmm::cuda_stream_default) const;
  void synchronize() const;

  [[nodiscard]] cudaError_t synchronize_no_throw() const noexcept;

  [[nodiscard]] std::chrono::duration<float, std::milli> elapsed_time(cuda_event_view start) const;

  [[nodiscard]] event::query_result query() const noexcept;

  [[nodiscard]] cudaError_t query_raw_status() const noexcept;

 private:
  cudaEvent_t event_{};
};

class cuda_event {
 public:
  explicit cuda_event(unsigned int flags = cudaEventDisableTiming);
  ~cuda_event() noexcept;

  cuda_event(cuda_event const&)            = delete;
  cuda_event& operator=(cuda_event const&) = delete;

  cuda_event(cuda_event&& other) noexcept;
  cuda_event& operator=(cuda_event&& other) noexcept;

  [[nodiscard]] cudaEvent_t get() const noexcept;
  [[nodiscard]] explicit operator cudaEvent_t() const noexcept;

  [[nodiscard]] cuda_event_view view() const noexcept;
  operator cuda_event_view() const noexcept;

  void record(rmm::cuda_stream_view stream = rmm::cuda_stream_default);
  void wait(rmm::cuda_stream_view stream = rmm::cuda_stream_default) const;
  void synchronize() const;

  [[nodiscard]] cudaError_t synchronize_no_throw() const noexcept;

  [[nodiscard]] std::chrono::duration<float, std::milli> elapsed_time(
    cuda_event const& start) const;

  [[nodiscard]] event::query_result query() const noexcept;

  [[nodiscard]] cudaError_t query_raw_status() const noexcept;

 private:
  cudaEvent_t event_{nullptr};
};

}  // namespace cudf_streaming::prefetch::cuda
