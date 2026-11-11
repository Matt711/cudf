# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Native pipelines for the streaming executor (experimental).

A native region is a subgraph that runs per chunk on native threads instead of as one
actor per operator: a remote parquet scan, the projections, column expressions and inner
joins it is probed through, and the group-by they feed. A lowering pass
(:func:`rewrite_native_regions`) replaces each matching subgraph with a
:class:`NativeRegion` node whose children are the join build sides; everything outside the
supported shape keeps the regular path. ``CUDF_POLARS_NATIVE_REGION=1`` turns it on.
"""

from __future__ import annotations

from cudf_polars.streaming.native.region import (
    NativeProgram,
    NativeRegion,
    enabled,
    rewrite_native_regions,
)

__all__ = ["NativeProgram", "NativeRegion", "enabled", "rewrite_native_regions"]
