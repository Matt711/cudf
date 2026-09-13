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

#include <cstddef>

namespace cudf_streaming {
namespace prefetch {
namespace memory {

/**
 * @brief Opt-in mixin for memory resources that allocate in fixed-size chunks.
 *
 * A memory resource that hands out bounded-size chunks can inherit from this
 * interface to advertise its chunk size.  Probing code can detect chunked
 * allocators via `dynamic_cast` without coupling to specific allocator types.
 */
struct chunked_resource_info {
  virtual ~chunked_resource_info() = default;

  [[nodiscard]] virtual std::size_t max_chunk_bytes() const = 0;
};

}  // namespace memory
}  // namespace prefetch
}  // namespace cudf_streaming
