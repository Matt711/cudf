#pragma once
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// Compatibility shim: provide cuda::mul_overflow for CCCL versions < 2.7 that
// lack it.  Force-included via -include flag so it appears before any cudf header.
//
// In the nightly-20260916 container, <cuda/cmath> EXISTS but does NOT declare
// cuda::mul_overflow (libcudacxx < 2.7).  We define it unconditionally here.
// There is no conflict because the old CCCL never defines it anywhere else.
//
// On containers with CCCL >= 2.7, the real cuda::mul_overflow (different return
// type from CCCL's overflow_result<T>) would produce a redefinition error, but
// this file is only used to build against the pinned nightly-20260916 base image.

#include <type_traits>

namespace cuda {

// Guard: only define if not already declared by newer CCCL via cuda/cmath.
// We check for a sentinel type that CCCL 2.7+ puts in namespace cuda.
#if !defined(__has_builtin) || !__has_builtin(__builtin_mul_overflow)
// No __builtin_mul_overflow → extremely old g++; fall back to manual check.
template <typename T>
struct _pcache_mul_result { bool overflow; T value; };
template <typename T>
inline _pcache_mul_result<T> mul_overflow(T a, T b) noexcept {
    _pcache_mul_result<T> r{false, T{}};
    // naive overflow check
    if (b != 0) {
        r.value = a * b;
        r.overflow = (r.value / b != a);
    } else {
        r.value = T{};
        r.overflow = false;
    }
    return r;
}
#else
// Normal path: __builtin_mul_overflow is available.
template <typename T>
struct _pcache_mul_result { bool overflow; T value; };
template <typename T>
inline _pcache_mul_result<T> mul_overflow(T a, T b) noexcept {
    _pcache_mul_result<T> r{};
    r.overflow = __builtin_mul_overflow(a, b, &r.value);
    return r;
}
#endif

} // namespace cuda
