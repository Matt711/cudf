// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Ported from Sirius PR #1661 (aminaramoon:feature/frugal_caching_and_dynamic_io)
// src/include/exec/invocable.hpp

#pragma once

#include <absl/functional/any_invocable.h>

namespace cudf_streaming::prefetch::exec {

/// Move-only type-erased callable. Unlike @c std::function it accepts move-only
/// targets, and unlike @c std::move_only_function (C++23) it is available today.
template <typename Signature>
using invocable = absl::AnyInvocable<Signature>;

}  // namespace cudf_streaming::prefetch::exec
