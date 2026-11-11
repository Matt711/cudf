# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Remote parquet scans read with cuCascade's REST engine, fetched ahead of their decode.

A process-wide :class:`cudf_polars_cucascade.scan_io.ScanIO` (the "service") does all object-store
I/O of a remote parquet scan: one footer request per file (a suffix-range GET that returns the
size, the version tag and the footer, as Sirius does), and the column chunks each task decodes,
read into a pinned host arena ahead of the decode. This module plans a task from its footers
(row-group pruning with footer statistics, byte ranges of the projected columns, neighbouring
ranges merged into fewer requests) and decodes a task from its staged bytes with the hybrid scan
reader. Nothing here touches KvikIO or a ``SourceInfo`` of a remote path.
"""

from __future__ import annotations

import atexit
import dataclasses
import functools
import os
import threading
import time
from typing import TYPE_CHECKING, Any, NamedTuple

import pylibcudf as plc

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import _cast_literals_to_physical_types
from cudf_polars.dsl.to_ast import to_parquet_filter
from cudf_polars.streaming.io import (
    _dataframe_from_hybrid_scan_table,
    _dataframe_with_source_index_from_hybrid_scan_table,
    _empty_hybrid_scan_dataframe,
)
from cudf_polars.utils.config import HybridScanPassMode

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.expr import Expr
    from cudf_polars.streaming.io import ParquetScanTask
    from cudf_polars.typing import Schema

__all__ = [
    "RemotePlan",
    "begin_query",
    "configure_scan_io",
    "materialize",
    "plan",
    "scan_io",
]

# Neighbouring column chunks closer than this are read with one request (cuCascade's merge gap).
MERGE_GAP_BYTES = 512 << 10

_service: Any = None
_service_lock = threading.Lock()


def configure_scan_io(
    *,
    arena_bytes: int,
    reactors: int = 8,
    connections: int = 32,
    footer_threads: int = 32,
    inflight_bytes: int = 0,
    endpoint: str | None = None,
) -> Any:
    """Create the process-wide service (once), with credentials from the environment."""
    global _service  # noqa: PLW0603
    with _service_lock:
        if _service is None:
            from cudf_polars_cucascade.scan_io import ScanIO  # noqa: PLC0415

            region = os.environ.get("AWS_REGION") or os.environ.get(
                "AWS_DEFAULT_REGION", "us-east-1"
            )
            _service = ScanIO(
                endpoint or os.environ.get("AWS_ENDPOINT_URL", f"https://s3.{region}.amazonaws.com"),
                region,
                os.environ.get("AWS_ACCESS_KEY_ID", ""),
                os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
                os.environ.get("AWS_SESSION_TOKEN", ""),
                arena_bytes=arena_bytes,
                reactors=reactors,
                connections=connections,
                footer_threads=footer_threads,
                inflight_bytes=inflight_bytes,
            )
        return _service


def scan_io() -> Any:
    """The service, or None when remote scans read the regular way."""
    return _service


class Footer(NamedTuple):
    """A file's footer as the scan task bounds expect it."""

    path: str
    size: int
    file_metadata: plc.io.parquet_metadata.FileMetaData


def eligible(task: ParquetScanTask) -> bool:
    """Whether a task is read through the service."""
    base_scan = task.base_scan
    return (
        _service is not None
        and base_scan.typ == "parquet"
        and task.hive_parts is None
        and task.lake_options is None
        and base_scan.skip_rows == 0
        and base_scan.n_rows == -1
        and base_scan.row_index is None
        and base_scan.include_file_paths is None
        and all(plc.io.SourceInfo._is_remote_uri(p) for p in task.paths)
    )


@functools.lru_cache(maxsize=256)
def _physical_types(path: str) -> dict[str, plc.DataType]:
    # From the footer the service already holds: a minimal in-memory parquet file, no request.
    metadata = plc.io.parquet_metadata.read_parquet_metadata(
        plc.io.SourceInfo([scan_io().footer_file(path)])
    )
    return metadata.schema().column_types()


def _predicate(task: ParquetScanTask, columns: list[str] | None) -> Expr:
    import polars as pl  # noqa: PLC0415

    base_scan = task.base_scan
    predicate = base_scan.predicate.value
    cols = columns or list(base_scan.schema.keys())
    if any(
        isinstance(base_scan.schema[c].polars_type, pl.Decimal)
        for c in cols
        if c in base_scan.schema
    ):
        types = _physical_types(task.paths[0])
        predicate = _cast_literals_to_physical_types(
            predicate, {name: types[name] for name in cols if name in types}
        )
    return predicate


@dataclasses.dataclass
class RemotePlan:
    """What a task reads, before any data I/O: its reader and the byte ranges of each pass."""

    reader: Any
    row_groups: Any
    options: Any
    paths: list[str]
    residual: Any
    empty: bool
    two_pass: bool
    # pass name ("all", "filter", "payload") -> (byte ranges, source index of each range)
    passes: dict[str, tuple[list[Any], list[int]]]
    # estimated peak device bytes of the decode
    decoded_bytes: int = 0
    # requests: merged ranges (source, offset, size) placed back to back in the task's block
    requests: list[tuple[int, int, int]] = dataclasses.field(default_factory=list)
    # pass name -> block position of each of its ranges
    positions: dict[str, list[int]] = dataclasses.field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        return sum(size for _, _, size in self.requests)


def _merge(plan: RemotePlan) -> None:
    """Merge the ranges of every pass into fewer requests and place them back to back."""
    items: list[tuple[int, int, int, str, int]] = []  # (source, offset, size, pass, index)
    for name, (ranges, sources) in plan.passes.items():
        for i, (r, s) in enumerate(zip(ranges, sources, strict=True)):
            items.append((s, r.offset, r.size, name, i))
    items.sort()
    plan.positions = {name: [0] * len(ranges) for name, (ranges, _) in plan.passes.items()}
    requests: list[list[int]] = []  # [source, offset, end, block position]
    block = 0
    for source, offset, size, name, i in items:
        end = offset + size
        last = requests[-1] if requests else None
        if last is not None and last[0] == source and offset <= last[2] + MERGE_GAP_BYTES:
            if end > last[2]:
                block += end - last[2]
                last[2] = end
        else:
            requests.append([source, offset, end, block])
            block += size
            last = requests[-1]
        plan.positions[name][i] = last[3] + (offset - last[1])
    plan.requests = [(s, o, e - o) for s, o, e, _ in requests]


def plan(task: ParquetScanTask, stream: Stream) -> RemotePlan:
    """Plan a task from its footers (waits for their requests)."""
    service = scan_io()
    base_scan = task.base_scan
    parquet_options = task.parquet_options
    footers = []
    for path in task.paths:
        size, metadata = service.footer(path)
        footers.append(Footer(path, size, metadata))
    bounds = task._get_task_bounds(footers)
    with_columns = None if base_scan.with_columns is None else list(base_scan.with_columns)
    plc_filter = residual = None
    if base_scan.predicate is not None:
        plc_filter, residual = to_parquet_filter(
            _predicate(task, with_columns),
            stream=stream,
            unreadable_columns=frozenset(),
        )
        if plc_filter is None:
            residual = base_scan.predicate.value
    source_info = plc.io.SourceInfo(
        [plc.io.types.FilepathSource(f.path, f.size) for f in footers]
    )
    options = (
        plc.io.parquet.ParquetReaderOptions.builder(source_info)
        .decimal_width(plc.TypeId.DECIMAL128)
        .use_jit_filter(use_jit_filter=parquet_options.use_jit_filter)
        .build()
    )
    if with_columns is not None:
        options.set_column_names(with_columns)
    if plc_filter is not None:
        options.set_filter(plc_filter)
    reader = plc.io.experimental.HybridScanMultiFile.from_parquet_metadatas(
        [f.file_metadata for f in footers], options
    )
    row_groups = plc.io.experimental.RowGroupIndices.from_lists(bounds.row_groups)
    if plc_filter is not None and parquet_options._hybrid_scan_stats_pruning:
        row_groups = reader.filter_row_groups_with_stats(row_groups, options, stream=stream)
    empty = not reader.total_rows_in_row_groups(row_groups)
    two_pass = (
        plc_filter is not None and parquet_options.pass_mode is not HybridScanPassMode.SINGLE_PASS
    )
    passes: dict[str, tuple[list[Any], list[int]]] = {}
    if not empty:
        if two_pass:
            passes["filter"] = reader.filter_column_chunks_byte_ranges(row_groups, options)
            passes["payload"] = reader.payload_column_chunks_byte_ranges(row_groups, options)
        else:
            passes["all"] = reader.all_column_chunks_byte_ranges(row_groups, options)
    # Estimated decoded bytes: per column, the larger of rows x its fixed width and its
    # uncompressed size in the footer (which counts the characters of string columns).
    decoded = 0
    columns = with_columns if with_columns is not None else list(base_scan.schema)
    for footer, rgs in zip(footers, bounds.row_groups, strict=True):
        counts = footer.file_metadata.row_group_num_rows
        rows = sum(counts[rg] for rg in rgs if rg < len(counts))
        uncompressed = footer.file_metadata.columnchunk_metadata
        for name in columns:
            try:
                fixed = rows * plc.types.size_of(base_scan.schema[name].plc_type)
            except Exception:  # noqa: BLE001  (strings and nested types: offsets plus data)
                fixed = rows * 16
            sizes = uncompressed.get(name, ())
            decoded += max(fixed, sum(sizes[rg] for rg in rgs if rg < len(sizes)))
    result = RemotePlan(
        reader,
        row_groups,
        options,
        [f.path for f in footers],
        residual,
        empty,
        two_pass,
        passes,
        decoded * (2 if plc_filter is not None or residual is not None else 1),
    )
    _merge(result)
    return result


@dataclasses.dataclass
class NativePlan:
    """A task planned and queued by the service (its reader stays in the service)."""

    task_id: int
    empty: bool
    nbytes: int
    decoded_bytes: int
    residual: Any
    options: Any  # the scan's reader options; must outlive the decode
    reader: Any = None


def scan_options(task: ParquetScanTask, stream: Stream) -> tuple[Any, Any]:
    """The reader options (columns, filter) of a task's scan and the residual predicate."""
    base_scan = task.base_scan
    parquet_options = task.parquet_options
    with_columns = None if base_scan.with_columns is None else list(base_scan.with_columns)
    plc_filter = residual = None
    if base_scan.predicate is not None:
        plc_filter, residual = to_parquet_filter(
            _predicate(task, with_columns),
            stream=stream,
            unreadable_columns=frozenset(),
        )
        if plc_filter is None:
            residual = base_scan.predicate.value
    source_info = plc.io.SourceInfo([task.paths[0]])
    options = (
        plc.io.parquet.ParquetReaderOptions.builder(source_info)
        .decimal_width(plc.TypeId.DECIMAL128)
        .use_jit_filter(use_jit_filter=parquet_options.use_jit_filter)
        .build()
    )
    if with_columns is not None:
        options.set_column_names(with_columns)
    if plc_filter is not None:
        options.set_filter(plc_filter)
    return options, residual


def plan_native(
    task: ParquetScanTask, stream_id: int, options: Any, residual: Any, stream: Stream
) -> NativePlan | None:
    """Plan and queue a task in the service (no GIL); None if it must be planned in Python."""
    task_id, empty, unsupported, too_big, staged, decoded = scan_io().plan_task(
        stream_id,
        list(task.paths),
        task.split_index,
        task.total_splits,
        options,
        task.parquet_options._hybrid_scan_stats_pruning,
        MERGE_GAP_BYTES,
        stream,
    )
    if unsupported or too_big:
        return None
    return NativePlan(task_id, empty, staged, decoded, residual, options)


def materialize_native(task: ParquetScanTask, plan: NativePlan, stream: Stream) -> DataFrame:
    """Decode a natively planned task (in the service, without the GIL)."""
    from cudf_polars.dsl.expr import NamedExpr  # noqa: PLC0415
    from cudf_polars.dsl.ir import apply_predicate  # noqa: PLC0415

    base_scan = task.base_scan
    schema = base_scan.schema
    with_columns = None if base_scan.with_columns is None else list(base_scan.with_columns)
    if plan.empty:
        df = _empty_hybrid_scan_dataframe(schema, with_columns, stream)
    else:
        _, df = _dataframe_with_source_index_from_hybrid_scan_table(
            scan_io().materialize(plan.task_id, stream),
            schema,
            stream,
            prepended_source_index=False,
        )
        df = df.select(list(schema.keys()))
    return apply_predicate(
        df,
        NamedExpr(base_scan.predicate.name, plan.residual) if plan.residual is not None else None,
    )


class _DeviceView:
    """A byte range of a device buffer, with the CUDA array interface (keeps the buffer alive)."""

    def __init__(self, owner: Any, offset: int, size: int) -> None:
        self._owner = owner
        self.__cuda_array_interface__ = {
            "shape": (size,),
            "typestr": "|u1",
            "data": (owner.ptr + offset, False),
            "version": 3,
        }


def materialize(
    task: ParquetScanTask, plan: RemotePlan, block: Any, stream: Stream
) -> DataFrame:
    """Decode a task from its staged bytes, copied to the device buffer ``block``."""
    from cudf_polars.dsl.expr import NamedExpr  # noqa: PLC0415
    from cudf_polars.dsl.ir import apply_predicate  # noqa: PLC0415

    base_scan = task.base_scan
    schema = base_scan.schema
    with_columns = None if base_scan.with_columns is None else list(base_scan.with_columns)

    def spans(name: str) -> list[Any]:
        ranges, _ = plan.passes[name]
        return [
            plc.gpumemoryview(_DeviceView(block, pos, r.size))
            for r, pos in zip(ranges, plan.positions[name], strict=True)
        ]

    reader, row_groups, options = plan.reader, plan.row_groups, plan.options
    if plan.empty:
        df = _empty_hybrid_scan_dataframe(schema, with_columns, stream)
    elif not plan.two_pass:
        _, df = _dataframe_with_source_index_from_hybrid_scan_table(
            reader.materialize_all_columns(row_groups, spans("all"), options, stream=stream),
            schema,
            stream,
            prepended_source_index=False,
        )
        df = df.select(list(schema.keys()))
    else:
        row_mask = reader.build_all_true_row_mask(row_groups, stream=stream)
        _, filter_df = _dataframe_with_source_index_from_hybrid_scan_table(
            reader.materialize_filter_columns(
                row_groups,
                spans("filter"),
                row_mask,
                plc.io.experimental.UseDataPageMask.YES,
                options,
                stream=stream,
            ),
            schema,
            stream,
            prepended_source_index=False,
        )
        columns = filter_df.columns
        requested = with_columns if with_columns is not None else list(schema)
        if set(requested) - filter_df.column_names_set:
            payload_df = _dataframe_from_hybrid_scan_table(
                reader.materialize_payload_columns(
                    row_groups,
                    spans("payload"),
                    row_mask,
                    plc.io.experimental.UseDataPageMask.YES,
                    options,
                    stream=stream,
                ),
                schema,
                stream,
            )
            columns = [*columns, *payload_df.columns]
        df = DataFrame(columns, stream=stream).select(list(schema.keys()))
    return apply_predicate(
        df,
        NamedExpr(base_scan.predicate.name, plan.residual) if plan.residual is not None else None,
    )


_query = {"index": -1, "t0": 0.0}


def begin_query() -> None:
    """
    Start of a query: forget the previous query's footers.

    Cold reads fetch every byte of a query from the object store, footers included.
    """
    if _service is not None:
        _service.clear_footers()
        _physical_types.cache_clear()
        _query["index"] += 1
        _query["t0"] = time.perf_counter()
        trace("query", -1, -1, 0)


_trace: list[tuple] = []


def trace(event: str, scan: int, task: int, nbytes: int) -> None:
    """Record an I/O event (seconds since the query started) when CUDF_POLARS_IO_TRACE is set."""
    if _TRACE_PATH:
        _trace.append(
            (_query["index"], round(time.perf_counter() - _query["t0"], 4), event, scan, task, nbytes)
        )


def _dump_trace() -> None:
    if _TRACE_PATH and _trace:
        with open(_TRACE_PATH, "w") as f:
            f.write("query\tt\tevent\tscan\ttask\tbytes\n")
            for row in _trace:
                f.write("\t".join(map(str, row)) + "\n")


_TRACE_PATH = os.environ.get("CUDF_POLARS_IO_TRACE")
atexit.register(_dump_trace)
