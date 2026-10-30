# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Optional cross-query byte cache for remote (``s3://``) parquet reads (prototype).

Enabled by setting ``CUDF_POLARS_BYTECACHE_GB`` (float, GB of host memory; ``0`` routes reads
through the cache-capable source with caching off). The compiled extension is found through
``CUDF_POLARS_BYTECACHE_PATH`` (directory holding ``_bytecache*.so``). Block size and fused GET size
can be set with ``CUDF_POLARS_BYTECACHE_BLOCK_MB`` (default 1) and ``CUDF_POLARS_BYTECACHE_GET_MB`` (default 16).
When the variable is unset every function here is a no-op and the standard read path is used.
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["enabled", "stats", "wrap_sources"]

_lock = threading.Lock()
_mod: Any = None
_checked = False


def _load() -> Any:
    global _mod, _checked  # noqa: PLW0603
    if _checked:
        return _mod
    with _lock:
        if _checked:
            return _mod
        gb = os.environ.get("CUDF_POLARS_BYTECACHE_GB")
        if gb:
            path = os.environ.get("CUDF_POLARS_BYTECACHE_PATH")
            if path and path not in sys.path:
                sys.path.insert(0, path)
            mod = importlib.import_module("_bytecache")
            block = int(float(os.environ.get("CUDF_POLARS_BYTECACHE_BLOCK_MB", "1")) * (1 << 20))
            get = int(float(os.environ.get("CUDF_POLARS_BYTECACHE_GET_MB", "16")) * (1 << 20))
            mod.configure(int(float(gb) * 1e9), block, get)
            _mod = mod
        _checked = True
    return _mod


def enabled() -> bool:
    """Whether the byte cache is switched on for this process."""
    return _load() is not None


def stats() -> dict[str, int] | None:
    """Cache counters, or None when disabled."""
    mod = _load()
    return None if mod is None else mod.get_stats()


def _is_remote(path: str) -> bool:
    return "://" in str(path)


def wrap_sources(
    paths: Sequence[str], sizes: Sequence[int | None] | None = None
) -> list[Any] | None:
    """
    Return cache-backed data sources for ``paths``, or None to use the standard path.

    The caller must keep the returned list alive for as long as the sources are read.
    """
    mod = _load()
    if mod is None or not paths or not all(_is_remote(p) for p in paths):
        return None
    sizes = sizes if sizes is not None else [None] * len(paths)
    return [mod.CachedDatasource(str(p), int(s or 0)) for p, s in zip(paths, sizes, strict=True)]
