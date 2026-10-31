# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Optional cross-query byte cache for remote (``s3://``) parquet reads (prototype).

Read-ahead is off unless ``CUDF_POLARS_BYTECACHE_READAHEAD=1`` (it did not help in measurements, see DESIGN.md).
Enabled by setting ``CUDF_POLARS_BYTECACHE_GB`` (float, GB of host memory; ``0`` routes reads
through the cache-capable source with caching off). The compiled extension is found through
``CUDF_POLARS_BYTECACHE_PATH`` (directory holding ``_bytecache*.so``). Block size and fused GET size
can be set with ``CUDF_POLARS_BYTECACHE_BLOCK_MB`` (default 0.25) and ``CUDF_POLARS_BYTECACHE_GET_MB`` (default 16).
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

__all__ = [
    "enabled",
    "register_layouts",
    "register_layouts_from_sources",
    "start_scan_prefetch",
    "stats",
    "wrap_sources",
]

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
            block = int(float(os.environ.get("CUDF_POLARS_BYTECACHE_BLOCK_MB", "0.25")) * (1 << 20))
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


_registered: set[str] = set()
_warmed: set[str] = set()
_layout_errors: list[str] = []


def _layout(info: Any) -> tuple[list[str], list[int], list[int], list[int]]:
    """Per-column-chunk byte ranges of one parquet file, from the footer pylibcudf has already parsed."""
    import pylibcudf as plc  # noqa: PLC0415

    fm = info.file_metadata
    source = plc.io.SourceInfo([plc.io.types.FilepathSource(info.path, info.size)])

    def options(cols: list[str] | None) -> Any:
        opts = plc.io.parquet.ParquetReaderOptions.builder(source).build()
        if cols is not None:
            opts.set_column_names(cols)
        return opts

    base = options(None)
    row_groups = plc.io.experimental.HybridScanReader.from_metadata(
        plc.io.experimental.HybridScanMetadata.from_parquet_metadata(fm, base)
    ).all_row_groups(base)
    columns = list(fm.columnchunk_metadata.keys())
    col_index: list[int] = []
    offsets: list[int] = []
    sizes: list[int] = []
    # One call with every column selected returns the chunks row-group-major (row group 0: every column, then row group 1, ...).
    all_opts = options(columns)
    ranges = plc.io.experimental.HybridScanReader.from_metadata(
        plc.io.experimental.HybridScanMetadata.from_parquet_metadata(fm, all_opts)
    ).all_column_chunks_byte_ranges(row_groups, all_opts)
    if len(ranges) == len(row_groups) * len(columns):
        for k, rng in enumerate(ranges):
            col_index.append(k % len(columns))
            offsets.append(rng.offset)
            sizes.append(rng.size)
    else:  # nested or unusual schema: one call per column
        for i, name in enumerate(columns):
            opts = options([name])
            reader = plc.io.experimental.HybridScanReader.from_metadata(
                plc.io.experimental.HybridScanMetadata.from_parquet_metadata(fm, opts)
            )
            for rng in reader.all_column_chunks_byte_ranges(row_groups, opts):
                col_index.append(i)
                offsets.append(rng.offset)
                sizes.append(rng.size)
    return columns, col_index, offsets, sizes


def register_layouts(infos: Sequence[Any]) -> None:
    """
    Tell the byte cache where each column chunk of the given files lives (the metadata store).

    ``infos`` are ``CachedParquetInfo`` objects whose sources were created with :func:`wrap_sources`.
    The layout only drives per-column accounting and, later, eviction and read-ahead; a failure here
    never affects the query.
    """
    mod = _load()
    if mod is None:
        return
    if os.environ.get("CUDF_POLARS_BYTECACHE_VERBOSE"):
        print(f"bytecache: register_layouts called with {len(infos)} files", file=sys.stderr)
    for info in infos:
        path = str(info.path)
        if not _is_remote(path):
            continue
        with _lock:
            if path in _registered:
                continue
            _registered.add(path)
        try:
            columns, col_index, offsets, sizes = _layout(info)
            mod.register_layout(path, path.rsplit("/", 1)[0], columns, col_index, offsets, sizes)
        except Exception as err:  # noqa: BLE001  (best effort, see docstring)
            _layout_errors.append(repr(err))
            if len(_layout_errors) <= 3:
                print(f"bytecache: layout registration failed for {path}: {err!r}", file=sys.stderr)


def register_layouts_from_sources(paths: Sequence[str], sources: Sequence[Any]) -> None:
    """
    Register the layout of files whose footers the engine did not prefetch.

    Reads each unregistered file's footer through its cache-backed source (the footer blocks stay cached,
    so this costs one small read per file per process), then registers the column layout.
    """
    mod = _load()
    if mod is None:
        return
    import types  # noqa: PLC0415

    import pylibcudf as plc  # noqa: PLC0415

    with _lock:
        todo = [(p, s) for p, s in zip(paths, sources, strict=True) if str(p) not in _registered and _is_remote(str(p))]
    if not todo:
        return
    try:
        footers = plc.io.parquet_metadata.read_parquet_footers(
            plc.io.types.SourceInfo([s for _, s in todo])
        )
    except Exception as err:  # noqa: BLE001  (best effort)
        _layout_errors.append(repr(err))
        return
    register_layouts(
        [
            types.SimpleNamespace(path=str(p), size=s.nbytes(), file_metadata=fm)
            for (p, s), fm in zip(todo, footers, strict=True)
        ]
    )


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
    # Open all not-yet-seen files and read their footers into the cache concurrently (GIL released), instead of one
    # HEAD and one footer round trip at a time.
    with _lock:
        cold = [str(p) for p in paths if str(p) not in _warmed]
        _warmed.update(cold)
    if cold:
        mod.warm_footers(cold)
    return [mod.CachedDatasource(str(p), int(s or 0)) for p, s in zip(paths, sizes, strict=True)]


class ScanPrefetcher:
    """
    Read-ahead for one scan node.

    A background thread fetches the projected columns of upcoming scan tasks into the byte cache while the
    scan producers work on earlier tasks, staying at most ``ahead_tasks`` tasks and about ``ahead_bytes`` bytes
    ahead of the newest task a producer has started. The data waits in host memory, so a task holds device
    memory only for the copy and decode, not while waiting for the network. The thread ends by itself after
    submitting every task (or after 60 s without progress).
    """

    def __init__(
        self,
        mod: Any,
        tasks: Sequence[Any],
        columns: list[str],
        ahead_tasks: int,
        ahead_bytes: int,
        manage: bool = False,
    ) -> None:
        self._mod = mod
        self._manage = manage
        self._tasks = tasks
        self._columns = columns
        self._ahead_tasks = ahead_tasks
        self._ahead_bytes = ahead_bytes
        self._cond = threading.Condition()
        self._frontier = -1
        self._enqueued: list[int] = []  # bytes enqueued for each submitted task
        self._groups: dict[int, int] = {}  # task index -> prefetch group (scan manager mode)
        self._submitted = threading.Condition()
        self._done_submitting = False
        self._wait_pool: Any = None
        self._thread = threading.Thread(target=self._run, name="bytecache-readahead", daemon=True)
        self._thread.start()

    def advance(self, task_idx: int) -> None:
        """Tell the prefetcher that a producer is starting task ``task_idx``."""
        with self._cond:
            if task_idx > self._frontier:
                self._frontier = task_idx
                self._cond.notify_all()

    def wait_ready(self, task_idx: int, timeout_s: float = 60.0) -> bool:
        """
        Block until every range of task ``task_idx`` is host-resident (scan manager mode).

        Declares demand for the task first (so the window moves up to it), then waits for the task to be submitted
        and for its prefetch group to finish. Falls back to returning False after ``timeout_s`` so that a stuck
        fetch only costs time: the task then reads whatever is missing on demand.
        """
        import time  # noqa: PLC0415

        self.advance(task_idx)
        deadline = time.monotonic() + timeout_s
        with self._submitted:
            while task_idx not in self._groups and not self._done_submitting:
                left = deadline - time.monotonic()
                if left <= 0 or not self._submitted.wait(timeout=left):
                    return False
            group = self._groups.get(task_idx)
        if group is None:
            return True
        left = max(deadline - time.monotonic(), 0.0)
        return bool(self._mod.wait_prefetch_group(group, int(left * 1000)))

    async def wait_ready_async(self, task_idx: int) -> bool:
        """Async wrapper around :meth:`wait_ready`; the wait runs on a private thread pool, not the event loop."""
        import asyncio  # noqa: PLC0415
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        if self._wait_pool is None:
            self._wait_pool = ThreadPoolExecutor(max_workers=64, thread_name_prefix="bytecache-wait")
        return await asyncio.get_running_loop().run_in_executor(self._wait_pool, self.wait_ready, task_idx)

    def _may_submit(self, nxt: int) -> bool:
        ahead = self._enqueued[max(self._frontier + 1, 0) : nxt]
        return nxt <= self._frontier + self._ahead_tasks and sum(ahead) < self._ahead_bytes

    def _register_all(self) -> None:
        """Open every remote file of the node and register its column layout, in parallel (one HEAD and a footer read each)."""
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        paths = sorted({str(p) for t in self._tasks for p in t.paths if _is_remote(str(p))} - _registered)

        def one(path: str) -> None:
            try:
                src = self._mod.CachedDatasource(path, 0)
                register_layouts_from_sources([path], [src])
            except Exception as err:  # noqa: BLE001  (best effort)
                _layout_errors.append(repr(err))

        if paths:
            self._mod.warm_footers(paths)  # all footers into the cache at once, GIL released
            with ThreadPoolExecutor(max_workers=min(32, len(paths))) as pool:
                list(pool.map(one, paths))

    def _run(self) -> None:
        try:
            self._run_inner()
        finally:
            with self._submitted:
                self._done_submitting = True
                self._submitted.notify_all()

    def _run_inner(self) -> None:
        self._register_all()
        nxt = 0
        while nxt < len(self._tasks):
            with self._cond:
                if not self._cond.wait_for(lambda: self._may_submit(nxt), timeout=60):
                    return
            total = 0
            group = self._mod.new_prefetch_group() if self._manage else 0
            for path in self._tasks[nxt].paths:
                path = str(path)
                if not _is_remote(path):
                    continue
                try:
                    if path not in _registered:
                        src = self._mod.CachedDatasource(path, 0)
                        register_layouts_from_sources([path], [src])
                    total += self._mod.prefetch_columns(path, path.rsplit("/", 1)[0], self._columns, group)
                except Exception as err:  # noqa: BLE001  (read-ahead is best effort)
                    _layout_errors.append(repr(err))
            self._enqueued.append(total)
            if self._manage:
                with self._submitted:
                    self._groups[nxt] = group
                    self._submitted.notify_all()
            nxt += 1


def scan_manager_enabled() -> bool:
    """True when ``CUDF_POLARS_SCAN_MANAGER=1`` (default off): scan tasks wait until their data is host-resident."""
    return os.environ.get("CUDF_POLARS_SCAN_MANAGER", "0") == "1" and os.environ.get("CUDF_POLARS_BYTECACHE_GB") is not None


def scan_manager_producers(default: int) -> int:
    """Producer count in scan manager mode (``CUDF_POLARS_SCAN_MANAGER_PRODUCERS``); tasks hold device memory only to decode."""
    return int(os.environ.get("CUDF_POLARS_SCAN_MANAGER_PRODUCERS", str(default))) if scan_manager_enabled() else default


def start_scan_prefetch(tasks: Sequence[Any]) -> ScanPrefetcher | None:
    """Start read-ahead for a scan node's tasks, or return None when the byte cache or read-ahead is off."""
    mod = _load()
    manage = scan_manager_enabled()
    if mod is None or not tasks or not (manage or os.environ.get("CUDF_POLARS_BYTECACHE_READAHEAD", "0") == "1"):
        return None
    if not any(_is_remote(str(p)) for p in tasks[0].paths):
        return None
    columns = list(tasks[0].base_scan.schema.keys())
    ahead_tasks = int(os.environ.get("CUDF_POLARS_BYTECACHE_AHEAD_TASKS", "24"))
    ahead_bytes = int(float(os.environ.get("CUDF_POLARS_BYTECACHE_AHEAD_GB", "12")) * 1e9)
    return ScanPrefetcher(mod, tasks, columns, ahead_tasks, ahead_bytes, manage)
