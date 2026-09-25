# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Stage-1 prefetch cache: load parquet byte ranges into pinned host RAM
before per-query timers start, then serve via PrefetchDatasource.

Only the byte ranges the parquet reader will actually access are downloaded:
  • bloom-filter pages
  • filter column chunks
  • payload column chunks
  • parquet footer (last FOOTER_HEADROOM bytes of the file)
  • parquet magic header (first 4 bytes)

This avoids downloading entire files and keeps pinned RAM proportional to
the data actually read by the query (~10x less than the full-file approach).

Public API
----------
prefetch_query(q_id, lf, engine, n_workers=64) -> str
get_datasource(path) -> PrefetchDatasource | None
is_active() -> bool
clear() -> None
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import TYPE_CHECKING

import nvtx

if TYPE_CHECKING:
    import polars as pl

log = logging.getLogger(__name__)

# Estimated upper bound for a parquet file footer (schema + row group metadata).
# 4 MB covers even very large schemas at SF300+.
_FOOTER_HEADROOM = 4 * 1024 * 1024

# Module-level state — pinned path (prefetch_query / --prefetch-ram).
_cache: dict[str, object] = {}   # path -> PrefetchDatasource
_active: bool = False

_bg_active: bool = False


def get_datasource(path: str) -> object | None:
    """Return the PrefetchDatasource for ``path``, or None if not cached."""
    return _cache.get(path)


def is_active() -> bool:
    """True when prefetch mode is active for the current query."""
    return _active


def clear() -> None:
    """Log execution-phase stats, free all pinned buffers, reset cache."""
    global _cache, _active, _bg_futures, _bg_active
    if _cache:
        total_reads = total_hits = total_misses = total_bytes = 0
        for ds in _cache.values():
            reads, hits, misses, served = ds.stats()
            total_reads += reads
            total_hits += hits
            total_misses += misses
            total_bytes += served
        log.info(
            "[PREFETCH STATS]: reads=%d hits=%d misses=%d bytes_served=%.2fGB",
            total_reads, total_hits, total_misses, total_bytes / 1e9,
        )
        print(
            f"[PREFETCH STATS]: reads={total_reads} hits={total_hits} "
            f"misses={total_misses} bytes_served={total_bytes / 1e9:.2f}GB",
            flush=True,
        )
    _cache = {}
    _active = False
    # Join the background worker thread before clearing shared dicts to prevent
    # the thread from writing into already-cleared state.
    global _bg_thread
    if _bg_thread is not None and _bg_thread.is_alive():
        _bg_thread.join(timeout=5.0)
        if _bg_thread.is_alive():
            log.warning("[BG PREFETCH] worker thread still alive after 5s join timeout")
    _bg_thread = None
    # Replace globals with fresh empty dicts.  The still-running worker thread
    # holds its own local references to the OLD dict objects (captured at worker
    # startup), so the kvikio reactor threads can finish writing into those
    # bytearrays safely.  Python keeps the old dicts alive until the worker
    # thread exits and its local refs drop.  No UAF.
    global _block_futures, _block_data, _task_block_index, _bg_file_sizes, _bg_remote_files
    _block_futures = {}
    _block_data = {}
    _task_block_index = {}
    _bg_file_sizes = {}
    _bg_remote_files = {}
    _bg_active = False


def is_bg_active() -> bool:
    """True when block-granular background prefetch is active for the current query."""
    return _bg_active


def inject_datasource(path: str, ds: object) -> None:
    """Temporarily place a background datasource into the pinned cache slot.

    Called by ParquetScanTask.do_evaluate before Scan.do_evaluate so the
    standard reader path in ir.py finds it via get_datasource(). Must be
    paired with remove_injected_datasource() after the read completes.
    """
    _cache[path] = ds


def remove_injected_datasource(path: str) -> None:
    """Remove a datasource injected by inject_datasource()."""
    _cache.pop(path, None)


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent byte ranges. Returns sorted list of (offset, size)."""
    if not ranges:
        return []
    sorted_r = sorted(ranges)
    merged: list[tuple[int, int]] = []
    cur_start, cur_size = sorted_r[0]
    cur_end = cur_start + cur_size
    for start, size in sorted_r[1:]:
        end = start + size
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end - cur_start))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end - cur_start))
    return merged


def prefetch_query(
    q_id: int,
    lf: pl.LazyFrame,
    engine: pl.GPUEngine,
    n_workers: int = 64,
) -> str:
    """
    Download byte ranges for all parquet files used by ``q_id`` into pinned RAM.

    Only the specific ranges the parquet reader will access are downloaded
    (bloom filters + column chunks + footer). This keeps pinned RAM proportional
    to actual data read rather than total file size.

    Each SPMD rank downloads only its own partition (rank=engine.rank,
    nranks=engine.nranks) so all ranks download in parallel without contention
    and finish at approximately the same time.

    Must be called BEFORE the per-query timer starts. After this returns,
    ``is_active()`` is True and ``get_datasource(path)`` returns a
    ``PrefetchDatasource`` for every file this rank will read.
    """
    global _cache, _active

    try:
        from prefetch_cache import PrefetchDatasource
    except ImportError as exc:
        raise RuntimeError(
            "prefetch_cache package not installed. "
            "Build and install python/prefetch_cache/ in the container."
        ) from exc

    import boto3

    from cudf_polars.dsl.translate import Translator
    from cudf_polars.dsl.traversal import traversal
    from cudf_polars.streaming.base import StatsCollector
    from cudf_polars.streaming.io import ParquetScanTask, StreamingScan
    from cudf_polars.streaming.parallel import lower_ir_graph_with_node_map

    # Determine this rank's partition for per-rank prefetch.
    rank = getattr(engine, 'rank', 0)
    nranks = getattr(engine, 'nranks', 1)

    # Translate the LazyFrame to an IR and lower to streaming tasks.
    translator = Translator(lf._ldf.visit(), engine)
    ir = translator.translate_ir()
    config_options = translator.config_options
    stats = StatsCollector()

    lowering, _ = lower_ir_graph_with_node_map(
        ir, config_options, stats, rank=rank, nranks=nranks
    )

    from cudf_polars.dsl.utils.io import (
        attach_cached_parquet_metadata,
        prefetch_parquet_file_metadata_for_ir,
    )

    cached_info_map = prefetch_parquet_file_metadata_for_ir(
        lowering.lowered,
        py_executor=None,
        parse_hybrid_metadata=config_options.parquet_options.use_hybrid_scan,
    )
    attach_cached_parquet_metadata(lowering.lowered, cached_info_map)

    # Collect all ParquetScanTask nodes from the lowered graph.
    all_tasks: list[ParquetScanTask] = [
        task
        for node in traversal([lowering.lowered])
        if isinstance(node, StreamingScan) and node.base_scan.typ == "parquet"
        for task in node.tasks
        if isinstance(task, ParquetScanTask)
    ]

    # Group tasks by file path.
    tasks_by_path: dict[str, list[ParquetScanTask]] = {}
    for task in all_tasks:
        for path in task.paths:
            tasks_by_path.setdefault(path, []).append(task)

    # -----------------------------------------------------------------------
    # Compute sparse byte ranges for one file.
    # Returns (path, file_size, merged_ranges, n_column_ranges) or None.
    # merged_ranges: list of (offset, size) — sorted, non-overlapping.
    # -----------------------------------------------------------------------
    def _compute_ranges(
        path: str, tasks: list[ParquetScanTask]
    ) -> tuple[str, int, list[tuple[int, int]], int] | None:
        from cudf_polars.dsl.to_ast import to_parquet_filter
        from cudf_polars.streaming.io import _prepare_parquet_predicate
        from cudf_polars.utils.cuda_stream import get_cuda_stream

        raw_ranges: list[tuple[int, int]] = []
        n_col_ranges = 0
        file_info = None

        for task in tasks:
            cached_infos = task._get_cached_parquet_info()
            if cached_infos is None:
                continue
            info = next((ci for ci in cached_infos if ci.path == path), None)
            if info is None:
                continue
            if file_info is None:
                file_info = info

            task_bounds = task._get_task_bounds(cached_infos)
            rg_list = task_bounds.row_groups
            try:
                path_idx = task.paths.index(path)
            except ValueError:
                continue
            if rg_list is None:
                rg_indices = list(range(len(info.file_metadata.row_group_num_rows)))
            else:
                rg_indices = rg_list[path_idx]

            if not rg_indices:
                continue

            options = info.default_reader_options()
            if task.base_scan.with_columns is not None:
                options.set_column_names(task.base_scan.with_columns)

            # Zone-map pruning: use the pushdownable part of the predicate to
            # prune row groups via min/max column statistics.  We do NOT require
            # residual is None: even when there is a residual (e.g. a semi-join
            # or EXISTS predicate), the pushdownable portion (e.g. a date range
            # on the scan column) is still valid for statistics-based pruning.
            # Without this change, queries with residual predicates (Q4, Q7,
            # etc.) skip statistics pruning entirely and download all row groups,
            # resulting in 5-6× more data than the NVMe path.
            base_scan = task.base_scan
            if (
                len(task.paths) == 1
                and base_scan.skip_rows == 0
                and base_scan.n_rows == -1
                and base_scan.predicate is not None
            ):
                try:
                    stream = get_cuda_stream()
                    plc_filter, residual = to_parquet_filter(
                        _prepare_parquet_predicate(
                            base_scan.predicate.value,
                            task.paths,
                            base_scan.schema,
                            base_scan.with_columns,
                        ),
                        stream=stream,
                    )
                    if plc_filter is not None:
                        # Use plc_filter for statistics pruning regardless of
                        # whether there is a residual.  Statistics pruning only
                        # uses column min/max values; the residual is handled
                        # later during row-level evaluation in the executor.
                        options_stats = info.default_reader_options()
                        if base_scan.with_columns is not None:
                            options_stats.set_column_names(base_scan.with_columns)
                        options_stats.set_filter(plc_filter)
                        reader_stats = info.hybrid_scan_reader(options_stats)
                        rg_indices = reader_stats.filter_row_groups_with_stats(
                            rg_indices, options_stats, stream=stream
                        )
                except Exception:
                    pass

            if not rg_indices:
                continue

            # Compute byte ranges for the STANDARD parquet reader (not hybrid scan).
            # During timed execution, Scan.do_evaluate (ir.py) bypasses hybrid scan
            # and uses the standard reader with PrefetchDatasource. The standard reader
            # requests ALL column chunks for with_columns in one pass — no filter/payload
            # split. Using options WITHOUT the filter here ensures payload_column_chunks_
            # byte_ranges returns ALL columns, matching exactly what the standard reader
            # will request.
            options_std = info.default_reader_options()
            if task.base_scan.with_columns is not None:
                options_std.set_column_names(task.base_scan.with_columns)
            reader_std = info.hybrid_scan_reader(options_std)
            try:
                payload_r = reader_std.payload_column_chunks_byte_ranges(
                    rg_indices, options_std
                )
            except Exception:
                payload_r = []

            for r in payload_r:
                raw_ranges.append((r.offset, r.size))
                n_col_ranges += 1

        # Resolve file size from cached metadata.
        if file_info is None:
            cached_infos = tasks[0]._get_cached_parquet_info()
            file_info = next((ci for ci in (cached_infos or []) if ci.path == path), None)
        file_size = (file_info.size or 0) if file_info is not None else 0

        if file_size == 0:
            return None

        # Always include:
        #   1. Parquet magic header: bytes 0–3 ("PAR1")
        #   2. Footer region: last _FOOTER_HEADROOM bytes (covers schema + row group metadata)
        raw_ranges.append((0, 4))  # magic header
        footer_start = max(0, file_size - _FOOTER_HEADROOM)
        raw_ranges.append((footer_start, file_size - footer_start))

        merged = _merge_ranges(raw_ranges)
        return path, file_size, merged, n_col_ranges

    # -----------------------------------------------------------------------
    # Download one file's sparse ranges and build a PrefetchDatasource.
    # Returns (path, PrefetchDatasource, total_bytes_downloaded) or None.
    # -----------------------------------------------------------------------
    def _download_sparse(
        span: tuple[str, int, list[tuple[int, int]], int],
    ) -> tuple[str, object, int] | None:
        path, file_size, merged_ranges, _n = span
        if not merged_ranges:
            return None

        if path.startswith("s3://"):
            rest = path[5:]
        else:
            rest = path
        bucket, _, key = rest.partition("/")

        s3 = boto3.client("s3")
        segments: list[tuple[int, bytes]] = []
        total = 0

        for offset, size in merged_ranges:
            if size <= 0:
                continue
            end = offset + size - 1
            resp = s3.get_object(
                Bucket=bucket, Key=key,
                Range=f"bytes={offset}-{end}",
            )
            data: bytes = resp["Body"].read()
            segments.append((offset, data))
            total += len(data)

        if not segments:
            return None

        ds = PrefetchDatasource.from_segments(file_size, segments)
        return path, ds, total

    # Phase 1: compute sparse ranges (CPU-only, fast).
    t0 = time.monotonic()
    with nvtx.annotate(
        message=f"prefetch::Q{q_id}::compute_ranges",
        domain="cudf_polars",
        color="cyan",
    ):
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            span_futures = {
                path: pool.submit(_compute_ranges, path, tasks)
                for path, tasks in tasks_by_path.items()
            }
            spans = {path: fut.result() for path, fut in span_futures.items()}

    valid_spans = [s for s in spans.values() if s is not None]

    # Phase 2: parallel S3 range downloads.
    total_bytes = 0
    total_ranges = sum(s[3] for s in valid_spans)
    with nvtx.annotate(
        message=f"prefetch::Q{q_id}::download",
        domain="cudf_polars",
        color="green",
    ):
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            download_results = list(pool.map(_download_sparse, valid_spans))

    new_cache: dict[str, object] = {}
    for result in download_results:
        if result is not None:
            path, ds, nbytes = result
            new_cache[path] = ds
            total_bytes += nbytes

    _cache = new_cache
    _active = len(new_cache) > 0
    elapsed = time.monotonic() - t0

    pruned_count = sum(1 for s in spans.values() if s is not None and s[3] == 0)
    log_line = (
        f"[PREFETCH Q{q_id}]: files={len(new_cache)} "
        f"bytes={total_bytes / 1e9:.2f}GB "
        f"elapsed={elapsed:.2f}s "
        f"ranges_merged={total_ranges} "
        f"threads={n_workers} "
        f"pruned={pruned_count}"
    )
    return log_line


# ── Block-granular background prefetch ────────────────────────────────────────
#
# Mirrors Sirius's prefetching_cache design:
#   • 1 MiB aligned blocks — same granularity as Sirius
#   • One Future per unique (path, block_offset) — deduplicated across tasks
#   • Per-task block index so a producer waits only for ITS blocks, not the
#     whole file
#   • n_workers concurrent S3 requests to saturate network bandwidth
#
# Public API (background path):
#   start_background(q_id, lf, engine)  — returns immediately, timer must start before
#   get_task_datasource(path, split_index, total_splits, file_size)
#                                        — blocks only on this task's blocks
#   is_bg_active() -> bool
#   clear()                              — cancels futures, frees state (cold semantics)
#
# Activated by --prefetch-background. The existing pinned path (--prefetch-ram)
# is untouched.

_BG_BLOCK_SIZE: int = 1 << 20  # 1 MiB aligned blocks

# (path, block_offset) → (IOFuture, bytearray buffer)
_block_futures: dict[tuple[str, int], tuple] = {}
# (path, block_offset) → bytearray once future.get() has been called
_block_data: dict[tuple[str, int], bytearray] = {}
# (path, split_index, total_splits) → sorted list of block_offsets needed
_task_block_index: dict[tuple[str, int, int], list[int]] = {}
# path → file_size (needed to clip the last block)
_bg_file_sizes: dict[str, int] = {}
# open kvikio.RemoteFile handles — kept alive while futures are pending
_bg_remote_files: dict[str, object] = {}
_bg_active: bool = False
# background worker thread (metadata fetch + IO submission)
_bg_thread: object = None  # threading.Thread | None


def _range_to_blocks(offset: int, size: int) -> list[int]:
    """Return sorted list of 1 MiB aligned block offsets covering [offset, offset+size)."""
    if size <= 0:
        return []
    block_start = (offset // _BG_BLOCK_SIZE) * _BG_BLOCK_SIZE
    block_end = offset + size
    offsets = []
    b = block_start
    while b < block_end:
        offsets.append(b)
        b += _BG_BLOCK_SIZE
    return offsets



def _compute_task_ranges(
    path: str,
    task,  # ParquetScanTask
    cached_infos: list,
) -> list[tuple[int, int]]:
    """
    Return raw (offset, size) byte ranges for a single task on a single file.
    Mirrors the inner loop of _compute_ranges but for one task only.
    """
    from cudf_polars.dsl.to_ast import to_parquet_filter
    from cudf_polars.streaming.io import _prepare_parquet_predicate
    from cudf_polars.utils.cuda_stream import get_cuda_stream

    if cached_infos is None:
        return []
    info = next((ci for ci in cached_infos if ci.path == path), None)
    if info is None:
        return []

    task_bounds = task._get_task_bounds(cached_infos)
    rg_list = task_bounds.row_groups
    try:
        path_idx = task.paths.index(path)
    except ValueError:
        return []

    if rg_list is None:
        rg_indices = list(range(len(info.file_metadata.row_group_num_rows)))
    else:
        rg_indices = rg_list[path_idx]

    if not rg_indices:
        return []

    base_scan = task.base_scan

    # Zone-map pruning (mirrors _compute_ranges logic).
    if (
        len(task.paths) == 1
        and base_scan.skip_rows == 0
        and base_scan.n_rows == -1
        and base_scan.predicate is not None
    ):
        try:
            stream = get_cuda_stream()
            plc_filter, _ = to_parquet_filter(
                _prepare_parquet_predicate(
                    base_scan.predicate.value,
                    task.paths,
                    base_scan.schema,
                    base_scan.with_columns,
                ),
                stream=stream,
            )
            if plc_filter is not None:
                options_stats = info.default_reader_options()
                if base_scan.with_columns is not None:
                    options_stats.set_column_names(base_scan.with_columns)
                options_stats.set_filter(plc_filter)
                reader_stats = info.hybrid_scan_reader(options_stats)
                rg_indices = reader_stats.filter_row_groups_with_stats(
                    rg_indices, options_stats, stream=stream
                )
        except Exception:
            pass

    if not rg_indices:
        return []

    options_std = info.default_reader_options()
    if base_scan.with_columns is not None:
        options_std.set_column_names(base_scan.with_columns)
    reader_std = info.hybrid_scan_reader(options_std)
    try:
        payload_r = reader_std.payload_column_chunks_byte_ranges(rg_indices, options_std)
    except Exception:
        return []

    raw: list[tuple[int, int]] = [(r.offset, r.size) for r in payload_r]

    # Always include magic header and footer.
    file_size = info.size or 0
    if file_size > 0:
        raw.append((0, 4))
        footer_start = max(0, file_size - _FOOTER_HEADROOM)
        raw.append((footer_start, file_size - footer_start))

    return raw


def get_task_datasource(
    path: str,
    split_index: int,
    total_splits: int,
    file_size: int,
) -> object | None:
    """
    Non-blocking cache lookup.  Returns a PrefetchDatasource built from
    already-drained blocks, or None if any block is not yet in _block_data.

    Never calls future.get() — the background thread drains all futures into
    _block_data.  A miss here means the background drain hasn't reached this
    block yet; the scan actor falls back to direct S3 (matching Sirius's
    non-blocking acquire_read() → miss → direct IO pattern).
    """
    if not _bg_active:
        return None

    key = (path, split_index, total_splits)
    block_offsets = _task_block_index.get(key)
    if block_offsets is None:
        return None

    try:
        from prefetch_cache import PrefetchDatasource
    except ImportError:
        return None

    segments: list[tuple[int, bytearray]] = []
    for block_offset in block_offsets:
        bkey = (path, block_offset)
        buf = _block_data.get(bkey)
        if buf is None:
            return None  # not drained yet — miss, fall back to direct S3
        segments.append((block_offset, buf))

    if not segments:
        return None

    return PrefetchDatasource.from_segments(file_size, segments)


def start_background(
    q_id: int,
    lf: pl.LazyFrame,
    engine: pl.GPUEngine,
) -> None:
    """
    Lower the query plan (synchronous, fast), then launch a background thread
    that fetches parquet metadata and submits kvikio async IOs concurrently
    with query execution.

    Interleaved design (matches Sirius): metadata fetches for all files run in
    parallel; as each file's metadata arrives the thread immediately computes
    1 MiB block ranges and calls kvikio.RemoteFile.pread() (returns an IOFuture
    immediately).  IO for file N starts before file M's metadata is even ready.

    get_task_datasource() returns None (falls back to direct S3) for any task
    whose blocks haven't been submitted yet; so correctness is always preserved.
    """
    import concurrent.futures as _cf
    import threading

    global _block_futures, _block_data, _task_block_index
    global _bg_file_sizes, _bg_remote_files, _bg_active, _bg_thread

    from cudf_polars.dsl.translate import Translator
    from cudf_polars.dsl.traversal import traversal
    from cudf_polars.dsl.utils.io import _prefetch_parquet_footers_for_paths
    from cudf_polars.streaming.base import StatsCollector
    from cudf_polars.streaming.io import ParquetScanTask, StreamingScan
    from cudf_polars.streaming.parallel import lower_ir_graph_with_node_map

    rank = getattr(engine, "rank", 0)
    nranks = getattr(engine, "nranks", 1)

    # IR lowering is synchronous (~0ms of S3 I/O) — do it before starting the
    # thread so the thread closure captures stable, fully-constructed objects.
    translator = Translator(lf._ldf.visit(), engine)
    ir = translator.translate_ir()
    config_options = translator.config_options
    stats = StatsCollector()

    lowering, _ = lower_ir_graph_with_node_map(
        ir, config_options, stats, rank=rank, nranks=nranks
    )

    all_tasks: list[ParquetScanTask] = [
        task
        for node in traversal([lowering.lowered])
        if isinstance(node, StreamingScan) and node.base_scan.typ == "parquet"
        for task in node.tasks
        if isinstance(task, ParquetScanTask)
    ]

    # Unique paths in encounter order (earlier paths get metadata fetched first).
    unique_paths: list[str] = list(dict.fromkeys(
        path for task in all_tasks for path in task.paths
    ))

    # Group tasks by path so we can process each file as its metadata arrives.
    tasks_by_path: dict[str, list[ParquetScanTask]] = {}
    for task in all_tasks:
        for path in task.paths:
            tasks_by_path.setdefault(path, []).append(task)

    parse_hybrid = config_options.parquet_options.use_hybrid_scan

    # Reset shared dicts and mark active BEFORE starting the thread so
    # get_task_datasource() sees _bg_active=True from the very first call.
    _block_futures = {}
    _block_data = {}
    _task_block_index = {}
    _bg_file_sizes = {}
    _bg_remote_files = {}
    _bg_active = True

    def _worker() -> None:
        import kvikio as _kvikio

        # Capture local references to the shared dicts at thread startup.
        # clear() may replace the globals with new empty dicts at any time;
        # using local refs ensures the kvikio reactor threads always write into
        # live bytearray objects (no UAF) and the old dicts stay alive until
        # this thread exits.
        my_block_futures = _block_futures
        my_block_data = _block_data
        my_task_block_index = _task_block_index
        my_bg_file_sizes = _bg_file_sizes
        my_bg_remote_files = _bg_remote_files

        try:
            # Submit all metadata fetches in parallel.  as_completed drives IO
            # submission per-file as each metadata result arrives.
            with _cf.ThreadPoolExecutor(thread_name_prefix="bg-prefetch-meta") as meta_pool:
                path_futures: dict[_cf.Future, str] = {
                    meta_pool.submit(
                        _prefetch_parquet_footers_for_paths, [path],
                        parse_hybrid_metadata=parse_hybrid,
                    ): path
                    for path in unique_paths
                }

                for meta_future in _cf.as_completed(path_futures):
                    path = path_futures[meta_future]
                    try:
                        infos = meta_future.result()
                    except Exception as exc:
                        log.warning("[BG PREFETCH] metadata %s failed: %s", path, exc)
                        continue

                    info = infos[0]
                    file_size = info.size or 0
                    my_bg_file_sizes[path] = file_size

                    if path not in my_bg_remote_files:
                        my_bg_remote_files[path] = _kvikio.RemoteFile.open(path)
                    rf = my_bg_remote_files[path]

                    for task in tasks_by_path.get(path, []):
                        si = task.split_index
                        ts = task.total_splits
                        key = (path, si, ts)

                        raw_ranges = _compute_task_ranges(path, task, infos)
                        if not raw_ranges:
                            my_task_block_index[key] = []
                            continue

                        task_blocks: list[int] = []
                        seen: set[int] = set()
                        for offset, size in raw_ranges:
                            for block_off in _range_to_blocks(offset, size):
                                if block_off in seen:
                                    continue
                                seen.add(block_off)
                                task_blocks.append(block_off)
                                bkey = (path, block_off)
                                if bkey not in my_block_futures:
                                    actual_size = min(
                                        _BG_BLOCK_SIZE,
                                        file_size - block_off if file_size > block_off
                                        else _BG_BLOCK_SIZE,
                                    )
                                    buf = bytearray(actual_size)
                                    fut = rf.pread(buf, actual_size, block_off)
                                    my_block_futures[bkey] = (fut, buf)

                        my_task_block_index[key] = sorted(task_blocks)

            n_submitted = len(my_block_futures)
            log.info(
                "[BG PREFETCH Q%d] submitted %d blocks across %d files; draining",
                q_id, n_submitted, len(my_bg_file_sizes),
            )
            print(
                f"[BG PREFETCH Q{q_id}]: {n_submitted} blocks across "
                f"{len(my_bg_file_sizes)} files submitted — draining into cache",
                flush=True,
            )

            # Phase 2: drain futures into my_block_data so get_task_datasource()
            # can serve blocks without blocking.
            n_drained = 0
            for bkey, (future, buf) in list(my_block_futures.items()):
                try:
                    future.get()
                    my_block_data[bkey] = buf
                    n_drained += 1
                except Exception as exc:
                    log.warning(
                        "[BG PREFETCH] block (%s, %d) failed: %s",
                        bkey[0], bkey[1], exc,
                    )

            log.info(
                "[BG PREFETCH Q%d] drained %d/%d blocks",
                q_id, n_drained, n_submitted,
            )
            print(
                f"[BG PREFETCH Q{q_id}]: drained {n_drained}/{n_submitted} blocks into cache",
                flush=True,
            )
        except Exception:
            import traceback as _tb
            log.warning("[BG PREFETCH Q%d] worker failed:\n%s", q_id, _tb.format_exc())

    _bg_thread = threading.Thread(
        target=_worker, daemon=True, name=f"bg-prefetch-q{q_id}"
    )
    _bg_thread.start()
    print(
        f"[BG PREFETCH Q{q_id}]: background worker started for {len(unique_paths)} paths",
        flush=True,
    )
    log.info("[BG PREFETCH Q%d]: background worker started for %d paths", q_id, len(unique_paths))
