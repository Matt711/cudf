// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright 2025, NVIDIA CORPORATION & AFFILIATES.
 * (Ported from cuCascade, commit 1b0e7b6c28dafa43bfe2c48011a3657c0dd6f127)
 */

#include "error.hpp"
#include "oom_handling_policy.hpp"

#include <exception>

namespace cudf_streaming {
namespace prefetch {
namespace memory {

void* throw_on_oom_policy::do_handle_oom([[maybe_unused]] std::size_t bytes,
                                         [[maybe_unused]] rmm::cuda_stream_view stream,
                                         std::exception_ptr eptr,
                                         [[maybe_unused]] RetryFunc retry_function)
{
  std::rethrow_exception(eptr);
}

std::string throw_on_oom_policy::get_policy_name() const noexcept { return "rethrow"; }

std::unique_ptr<oom_handling_policy> make_default_oom_policy()
{
  return std::make_unique<throw_on_oom_policy>();
}

}  // namespace memory
}  // namespace prefetch
}  // namespace cudf_streaming
