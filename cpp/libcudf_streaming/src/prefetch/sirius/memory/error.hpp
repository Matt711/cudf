// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <cstring>
#include <string_view>
#include <system_error>

namespace cudf_streaming {
namespace prefetch {
namespace memory {

enum class MemoryError { SUCCESS, ALLOCATION_FAILED, LIMIT_EXCEEDED, POOL_EXHAUSTED, SIZE };

struct memory_error_category : std::error_category {
  const char* name() const noexcept final;

  std::string message(int ev) const final;
};

const memory_error_category& memory_category();

inline std::error_code make_error_code(MemoryError e);

struct cucascade_out_of_memory : public rmm::out_of_memory {
  explicit cucascade_out_of_memory(std::string_view message,
                                   MemoryError error_kind,
                                   std::size_t requested_bytes,
                                   std::size_t global_usage,
                                   cudaMemPool_t pool_handle);

  const MemoryError error_kind;
  const std::size_t requested_bytes;
  const std::size_t global_usage;
  const cudaMemPool_t pool_handle;
};

}  // namespace memory
}  // namespace prefetch
}  // namespace cudf_streaming

namespace std {

template <>
struct is_error_code_enum<cudf_streaming::prefetch::memory::MemoryError> : true_type {};

}  // namespace std
