# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent, process-local cache of parquet file metadata."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from cudf_polars.dsl.ir import IR
    from cudf_polars.dsl.utils.io import CachedParquetInfo
    from cudf_polars.utils.config import ParquetOptions

# Remote footer fetches are network-latency-bound, so a larger pool than
# core count pays off; local footer fetches are dominated by CPU-bound
# footer parsing, not waiting, so they don't benefit the same way.
_DEFAULT_REMOTE_NUM_THREADS = 32


def default_num_threads(*, remote: bool) -> int:
    """Return the default size of the dedicated metadata-fetch pool."""
    if remote:
        return _DEFAULT_REMOTE_NUM_THREADS
    return max(1, (os.cpu_count() or 4) // 4)


class FetchStats(NamedTuple):
    """A point-in-time snapshot of a :class:`MetadataCache`'s cumulative fetch cost."""

    count: int
    """Number of completed fetches (HEAD, if any, plus footer read) since the cache opened."""
    seconds: float
    """
    Sum of each fetch's own wall-clock duration. Fetches run concurrently on the
    pool, so this is cumulative thread-time, not elapsed time for the phase as a
    whole — it can exceed the wall-clock span it was measured over.
    """


def _any_remote_parquet_path(ir: IR) -> bool:
    """Return whether any parquet ``Scan`` node in ``ir`` touches a remote path."""
    import pylibcudf as plc

    from cudf_polars.dsl.ir import Scan
    from cudf_polars.dsl.traversal import traversal

    return any(
        plc.io.SourceInfo._is_remote_uri(path)
        for node in traversal([ir])
        if isinstance(node, Scan) and node.typ == "parquet"
        for path in node.paths
    )


class MetadataCache:
    """
    One engine's cached parquet file metadata on a process, keyed by path.

    Entries are never evicted. Concurrent requests for the same path are
    coalesced onto one future.

    Parameters
    ----------
    num_threads
        Number of threads in the dedicated fetch pool.
    validate_etag
        Whether to check a local file's cached entry against a fresh
        ``stat()`` on every lookup, treating a mismatch as a miss. Remote
        files are never checked this way.
    track_estimates
        Whether to record every column-size estimate and, once a real read
        observes the same ``(path, column)``, validate the estimate against
        it.
    """

    def __init__(
        self,
        num_threads: int,
        *,
        validate_etag: bool = False,
        track_estimates: bool = False,
    ) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, Future[CachedParquetInfo]] = {}
        self._observed_column_bytes: dict[str, dict[str, float]] = {}
        self._validate_etag = validate_etag
        self._track_estimates = track_estimates
        self._pending_estimates: dict[str, dict[str, tuple[float, str]]] = {}
        self._estimate_validations: list[dict[str, Any]] = []
        self._fetch_count = 0
        self._fetch_seconds = 0.0
        self._executor = ThreadPoolExecutor(
            max_workers=num_threads, thread_name_prefix="cudf-polars-metadata-fetch"
        )

    def _is_stale(self, path: str, future: Future[CachedParquetInfo]) -> bool:
        """Return whether a resolved future's cached info no longer matches ``path``."""
        if (
            not self._validate_etag
            or not future.done()
            or future.exception() is not None
        ):
            return False
        from cudf_polars.dsl.utils.io import _local_file_etag

        current = _local_file_etag(path)
        cached = future.result().etag
        return current is not None and cached is not None and current != cached

    def _timed_fetch(self, fetch: Callable[[], CachedParquetInfo]) -> CachedParquetInfo:
        """Run ``fetch``, recording its wall-clock duration regardless of outcome."""
        start = time.monotonic()
        try:
            return fetch()
        finally:
            elapsed = time.monotonic() - start
            with self._lock:
                self._fetch_count += 1
                self._fetch_seconds += elapsed

    def get_or_submit(
        self, path: str, fetch: Callable[[], CachedParquetInfo]
    ) -> Future[CachedParquetInfo]:
        """Return the future for ``path``, submitting ``fetch`` if not already cached."""
        with self._lock:
            future = self._entries.get(path)
            if future is not None and self._is_stale(path, future):
                future = None
            if future is None:
                future = self._executor.submit(self._timed_fetch, fetch)
                self._entries[path] = future
        return future

    def fetch_stats(self) -> FetchStats:
        """Return a snapshot of cumulative fetch count and thread-time so far."""
        with self._lock:
            return FetchStats(self._fetch_count, self._fetch_seconds)

    def peek(self, path: str) -> CachedParquetInfo | None:
        """Return ``path``'s cached metadata if already resolved, without fetching."""
        with self._lock:
            future = self._entries.get(path)
            if future is None or self._is_stale(path, future):
                return None
        if not future.done():
            return None
        try:
            return future.result()
        except Exception:
            return None

    def record_estimate(
        self, path: str, column: str, bytes_per_row: float, step: str
    ) -> None:
        """Record a column-size estimate, pending validation against a real read."""
        if not self._track_estimates:
            return
        with self._lock:
            self._pending_estimates.setdefault(path, {})[column] = (
                bytes_per_row,
                step,
            )

    def record_observed_column_bytes(
        self, path: str, column: str, bytes_per_row: float, *, validates: bool = True
    ) -> None:
        """
        Record a column's real decoded bytes-per-row, observed from an actual read.

        ``validates=False`` for an observation that seeds its own estimate
        (for example row-group sampling bootstrapping the cache with its
        own result), since that would otherwise validate an estimate
        against itself.
        """
        with self._lock:
            self._observed_column_bytes.setdefault(path, {})[column] = bytes_per_row
            if not (validates and self._track_estimates):
                return
            pending = self._pending_estimates.get(path, {}).pop(column, None)
            if pending is not None:
                estimated, step = pending
                self._estimate_validations.append(
                    {
                        "path": path,
                        "column": column,
                        "step": step,
                        "estimated_bytes_per_row": estimated,
                        "observed_bytes_per_row": bytes_per_row,
                        "ratio": bytes_per_row / estimated if estimated else None,
                    }
                )

    def estimate_validations(self) -> list[dict[str, Any]]:
        """Return a copy of every recorded estimate-vs-observation comparison so far."""
        with self._lock:
            return list(self._estimate_validations)

    def mean_observed_column_bytes(
        self, paths: Iterable[str], column: str
    ) -> float | None:
        """
        Return the mean observed bytes-per-row for ``column`` among ``paths``.

        Only ``paths`` themselves are considered: column-name identity is
        only safe to assume within one scan's own paths, since two
        unrelated tables can share a column name with unrelated size
        characteristics.
        """
        with self._lock:
            values = [
                self._observed_column_bytes[path][column]
                for path in paths
                if column in self._observed_column_bytes.get(path, {})
            ]
        if not values:
            return None
        return sum(values) / len(values)

    def clear(self) -> None:
        """Drop all cached entries (idempotent)."""
        with self._lock:
            self._entries.clear()
            self._observed_column_bytes.clear()
            self._pending_estimates.clear()

    def close(self) -> None:
        """Shut down the dedicated pool and drop all entries (idempotent)."""
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.clear()


_caches: dict[str, MetadataCache] = {}


def open_cache(
    uid: str,
    *,
    num_threads: int,
    validate_etag: bool = False,
    track_estimates: bool = False,
) -> MetadataCache:
    """Return this engine's metadata cache on the current process, creating it if absent."""
    if uid not in _caches:
        _caches[uid] = MetadataCache(
            num_threads, validate_etag=validate_etag, track_estimates=track_estimates
        )
    return _caches[uid]


def require_cache(uid: str) -> MetadataCache:
    """Return this engine's metadata cache; raise :class:`KeyError` if it has been closed."""
    return _caches[uid]


def close_cache(uid: str) -> None:
    """Close and drop this engine's metadata cache on the current process (idempotent)."""
    cache = _caches.pop(uid, None)
    if cache is not None:
        cache.close()


def close_all() -> None:
    """Close and drop every metadata cache on the current process (idempotent)."""
    for cache in _caches.values():
        cache.close()
    _caches.clear()


def clear_cache(uid: str) -> None:
    """Drop all entries from this engine's metadata cache, if it still exists (idempotent)."""
    cache = _caches.get(uid)
    if cache is not None:
        cache.clear()


def clear_all() -> None:
    """Drop all entries from every metadata cache on the current process (idempotent)."""
    for cache in _caches.values():
        cache.clear()


def resolve_cache(
    uid: str, parquet_options: ParquetOptions, ir: IR
) -> MetadataCache | None:
    """
    Return this engine's metadata cache per ``parquet_options``, or ``None`` if disabled.

    ``ir`` is only inspected the first time a given ``uid``'s cache is
    created, to size its dedicated pool for local versus remote paths;
    later queries reuse the same cache regardless of their own paths.
    """
    if not parquet_options.persistent_metadata_cache:
        return None
    if uid in _caches:
        return _caches[uid]
    num_threads = parquet_options.metadata_fetch_pool_size
    if num_threads is None:
        num_threads = default_num_threads(remote=_any_remote_parquet_path(ir))
    return open_cache(
        uid,
        num_threads=num_threads,
        validate_etag=parquet_options.validate_cached_metadata_etag,
        track_estimates=parquet_options._track_column_size_estimates,
    )
