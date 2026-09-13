// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// error.hpp — CUDA/memory error-checking macros for the cudf prefetch layer.
// Replaces <cucascade/error.hpp> from the original cuCascade source.

#pragma once

#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <cassert>
#include <stdexcept>
#include <string>

namespace cudf_streaming::prefetch {

struct cuda_error : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct logic_error : public std::logic_error {
  using std::logic_error::logic_error;
};

}  // namespace cudf_streaming::prefetch

// ---------------------------------------------------------------------------
// Macro compatibility shims (map CUCASCADE_* → equivalent prefetch checks)
// ---------------------------------------------------------------------------

#define CUCASCADE_STRINGIFY_DETAIL(x) #x
#define CUCASCADE_STRINGIFY(x) CUCASCADE_STRINGIFY_DETAIL(x)

#define CUCASCADE_CUDA_TRY_2(_call, _exception_type)                                          \
  do {                                                                                        \
    cudaError_t const _err = (_call);                                                         \
    if (cudaSuccess != _err) {                                                                \
      cudaGetLastError();                                                                     \
      throw _exception_type{std::string{"CUDA error at: "} + __FILE__ + ":" +                \
                            CUCASCADE_STRINGIFY(__LINE__) + ": " + cudaGetErrorName(_err) +   \
                            " " + cudaGetErrorString(_err)};                                  \
    }                                                                                         \
  } while (0)

#define CUCASCADE_CUDA_TRY_1(_call) \
  CUCASCADE_CUDA_TRY_2(_call, cudf_streaming::prefetch::cuda_error)

#define GET_CUCASCADE_CUDA_TRY_MACRO(_1, _2, NAME, ...) NAME
#define CUCASCADE_CUDA_TRY(...)                                                               \
  GET_CUCASCADE_CUDA_TRY_MACRO(__VA_ARGS__, CUCASCADE_CUDA_TRY_2, CUCASCADE_CUDA_TRY_1)      \
  (__VA_ARGS__)

// Allocation-specific: OOM → rmm::out_of_memory, other → rmm::bad_alloc
#define CUCASCADE_CUDA_TRY_ALLOC_2(_call, _num_bytes)                                        \
  do {                                                                                        \
    cudaError_t const _err = (_call);                                                         \
    if (cudaSuccess != _err) {                                                                \
      cudaGetLastError();                                                                     \
      if (cudaErrorMemoryAllocation == _err) {                                                \
        throw rmm::out_of_memory{"CUDA out of memory"};                                       \
      }                                                                                       \
      throw rmm::bad_alloc{cudaGetErrorString(_err)};                                        \
    }                                                                                         \
  } while (0)

#define CUCASCADE_CUDA_TRY_ALLOC_1(_call) CUCASCADE_CUDA_TRY_ALLOC_2(_call, 0)

#define GET_CUCASCADE_CUDA_TRY_ALLOC_MACRO(_1, _2, NAME, ...) NAME
#define CUCASCADE_CUDA_TRY_ALLOC(...)                                                         \
  GET_CUCASCADE_CUDA_TRY_ALLOC_MACRO(__VA_ARGS__,                                            \
                                     CUCASCADE_CUDA_TRY_ALLOC_2,                             \
                                     CUCASCADE_CUDA_TRY_ALLOC_1)                             \
  (__VA_ARGS__)

// Noexcept-safe assert: calls the CUDA API unconditionally in all builds.
// In debug we assert success; in release we silently discard the return.
#ifdef NDEBUG
#define CUCASCADE_ASSERT_CUDA_SUCCESS(_call) (void)(_call)
#else
#define CUCASCADE_ASSERT_CUDA_SUCCESS(_call)                                                  \
  do {                                                                                        \
    cudaError_t const _err = (_call);                                                         \
    assert(_err == cudaSuccess);                                                              \
  } while (0)
#endif

// General logic-error assertion
#define CUCASCADE_FAIL(_msg)                                                                   \
  throw cudf_streaming::prefetch::logic_error{std::string{_msg} + " (" __FILE__ ":" +        \
                                              CUCASCADE_STRINGIFY(__LINE__) + ")"}

// NVTX function-range tracing (no-op: we don't link against cuCascade's NVTX domain here)
#define CUCASCADE_FUNC_RANGE() ((void)0)

// clang-format on
