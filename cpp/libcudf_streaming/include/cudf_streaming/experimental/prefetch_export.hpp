// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Facade symbol visibility for the prefetch extension.
// Generated equivalent: symbols in cudf_streaming::prefetch are visible when
// building the shared library; hidden otherwise.
#if defined(_WIN32)
#  define CUDF_STREAMING_PREFETCH_EXPORT __declspec(dllexport)
#  define CUDF_STREAMING_PREFETCH_IMPORT __declspec(dllimport)
#else
#  define CUDF_STREAMING_PREFETCH_EXPORT __attribute__((visibility("default")))
#  define CUDF_STREAMING_PREFETCH_IMPORT __attribute__((visibility("default")))
#endif
#ifdef CUDF_STREAMING_PREFETCH_BUILDING
#  define CUDF_STREAMING_PREFETCH_API CUDF_STREAMING_PREFETCH_EXPORT
#else
#  define CUDF_STREAMING_PREFETCH_API CUDF_STREAMING_PREFETCH_IMPORT
#endif
