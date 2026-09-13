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

#include "cuda/event.hpp"

#include <rmm/error.hpp>

#include <stdexcept>
#include <utility>

// Use RMM's CUDA error-checking pattern adapted for this module.
#define PREFETCH_CUDA_TRY(call)                                        \
  do {                                                                 \
    cudaError_t const _e = (call);                                     \
    if (_e != cudaSuccess) {                                           \
      throw std::runtime_error(std::string("CUDA error: ") +          \
                               ::cudaGetErrorString(_e));              \
    }                                                                  \
  } while (0)

#define PREFETCH_CUDA_TRY_NOEXCEPT(call) \
  do { (void)(call); } while (0)

namespace cudf_streaming::prefetch::cuda {

cuda_event::cuda_event(unsigned int flags)
{
  PREFETCH_CUDA_TRY(::cudaEventCreateWithFlags(&event_, flags));
}

cuda_event::~cuda_event() noexcept
{
  if (event_ != nullptr) { PREFETCH_CUDA_TRY_NOEXCEPT(::cudaEventDestroy(event_)); }
}

cuda_event::cuda_event(cuda_event&& other) noexcept : event_(std::exchange(other.event_, nullptr))
{
}

cuda_event& cuda_event::operator=(cuda_event&& other) noexcept
{
  if (this != &other) {
    if (event_ != nullptr) { PREFETCH_CUDA_TRY_NOEXCEPT(::cudaEventDestroy(event_)); }
    event_ = std::exchange(other.event_, nullptr);
  }
  return *this;
}

cudaEvent_t cuda_event::get() const noexcept { return event_; }

cuda_event::operator cudaEvent_t() const noexcept { return event_; }

cuda_event_view cuda_event::view() const noexcept { return cuda_event_view{event_}; }

cuda_event::operator cuda_event_view() const noexcept { return cuda_event_view{event_}; }

void cuda_event::record(rmm::cuda_stream_view stream) { view().record(stream); }

void cuda_event::wait(rmm::cuda_stream_view stream) const { view().wait(stream); }

void cuda_event::synchronize() const { view().synchronize(); }

cudaError_t cuda_event::synchronize_no_throw() const noexcept
{
  return view().synchronize_no_throw();
}

std::chrono::duration<float, std::milli> cuda_event::elapsed_time(cuda_event const& start) const
{
  return view().elapsed_time(start.view());
}

event::query_result cuda_event::query() const noexcept { return view().query(); }

cudaError_t cuda_event::query_raw_status() const noexcept { return view().query_raw_status(); }

void cuda_event_view::record(rmm::cuda_stream_view stream)
{
  PREFETCH_CUDA_TRY(::cudaEventRecord(event_, stream.value()));
}

void cuda_event_view::wait(rmm::cuda_stream_view stream) const
{
  PREFETCH_CUDA_TRY(::cudaStreamWaitEvent(stream.value(), event_, 0));
}

void cuda_event_view::synchronize() const
{
  PREFETCH_CUDA_TRY(::cudaEventSynchronize(event_));
}

cudaError_t cuda_event_view::synchronize_no_throw() const noexcept
{
  return ::cudaEventSynchronize(event_);
}

std::chrono::duration<float, std::milli> cuda_event_view::elapsed_time(
  cuda_event_view start) const
{
  float ms{0.F};
  PREFETCH_CUDA_TRY(::cudaEventElapsedTime(&ms, start.value(), event_));
  return std::chrono::duration<float, std::milli>{ms};
}

event::query_result cuda_event_view::query() const noexcept
{
  cudaError_t const status = query_raw_status();
  if (status == cudaSuccess) { return event::query_result::success; }
  if (status == cudaErrorNotReady) { return event::query_result::in_progress; }
  return event::query_result::error;
}

cudaError_t cuda_event_view::query_raw_status() const noexcept
{
  return ::cudaEventQuery(event_);
}

}  // namespace cudf_streaming::prefetch::cuda

#undef PREFETCH_CUDA_TRY
#undef PREFETCH_CUDA_TRY_NOEXCEPT
