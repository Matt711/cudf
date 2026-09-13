// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright 2025, NVIDIA CORPORATION & AFFILIATES.
 * (Ported from cuCascade, commit 1b0e7b6c28dafa43bfe2c48011a3657c0dd6f127)
 */

#include "error.hpp"

#include <cuda_runtime_api.h>

#include <system_error>

namespace cudf_streaming {
namespace prefetch {
namespace memory {

const char* memory_error_category::name() const noexcept { return "cudfStreamingPrefetchMemory"; }

std::string memory_error_category::message(int ev) const
{
  switch (static_cast<MemoryError>(ev)) {
    case MemoryError::SUCCESS: return "Success";
    case MemoryError::ALLOCATION_FAILED: return "System allocation failed";
    case MemoryError::LIMIT_EXCEEDED: return "Reservation limit exceeded";
    case MemoryError::POOL_EXHAUSTED: return "Internal memory pool exhausted";
    default: return "Unknown memory error";
  }
}

const memory_error_category& memory_category()
{
  static const memory_error_category instance;
  return instance;
}

std::error_code make_error_code(MemoryError e)
{
  return std::error_code(static_cast<int>(e), memory_category());
}

cucascade_out_of_memory::cucascade_out_of_memory(std::string_view message,
                                                 MemoryError error_kind_,
                                                 std::size_t requested_bytes_,
                                                 std::size_t global_usage_,
                                                 cudaMemPool_t pool_handle_)
  : rmm::out_of_memory(message.data()),
    error_kind(error_kind_),
    requested_bytes(requested_bytes_),
    global_usage(global_usage_),
    pool_handle(pool_handle_)
{
}

}  // namespace memory
}  // namespace prefetch
}  // namespace cudf_streaming
