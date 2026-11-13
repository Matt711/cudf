# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Infer ordering from Parquet scan-task metadata."""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import os
from typing import TYPE_CHECKING

import pylibcudf as plc
from cudf_streaming.channel_metadata import (
    OrderKey,
    OrderScheme,
    Ordering,
    Partitioning,
)
from cudf_streaming.table_chunk import TableChunk

from cudf_polars.dsl.tracing import nvtx_annotate_cudf_polars
from cudf_polars.dsl.utils.naming import names_to_indices
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.collectives.ordering import (
    _extract_boundaries_from_endpoint_rows as _extract_ordering_boundaries,
)
from cudf_polars.streaming.io import ParquetScanTask
from cudf_polars.streaming.partitioning_requests import (
    OrderPartitioningRequest,
    StrictPartitioningRequest,
)
from cudf_polars.utils.dtypes import is_order_preserving_cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import CachedParquetInfo, IRExecutionContext, Scan
    from cudf_polars.streaming.io import StreamingScan
    from cudf_polars.streaming.partitioning_requests import PartitioningRequest


def _null_column(ir: StreamingScan, name: str, size: int, stream: Stream) -> plc.Column:
    return plc.Column.from_scalar(
        plc.Scalar.from_py(None, ir.schema[name].plc_type, stream=stream),
        size,
        stream=stream,
    )


def _gather_rows(table: plc.Table, rows: list[int], stream: Stream) -> plc.Table:
    return plc.copying.gather(
        table,
        plc.Column.from_iterable_of_py(
            rows, plc.DataType(plc.TypeId.INT32), stream=stream
        ),
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        stream=stream,
    )


def _get_ordering_candidates(
    ir: StreamingScan,
    requests: tuple[PartitioningRequest, ...],
) -> list[tuple[str, OrderKey]]:
    """
    Return distinct leading keys from explicit ordering hints.

    With ``CUDF_POLARS_INFER_KEY_ORDER=1``, the leading key of a group-by (a strict
    partitioning request) is a candidate too, ascending: when the footer statistics
    prove the scan is ordered on it with strict boundaries, the group-by runs
    chunkwise, and joins on the key downstream may run ordered.
    """
    candidates: list[tuple[str, OrderKey]] = []
    for request in requests:
        if isinstance(request, StrictPartitioningRequest):
            # Only a group-by's key: it then runs chunkwise instead of keeping partials of every
            # key to the end. A join's ordered strategy did not beat broadcast or shuffle (Q3,
            # Q5, Q7, Q12 at SF1000), and an ordered scan starts later and decodes in order.
            if not (_infer_key_order() and request.keys and request.from_groupby):
                continue
            name = request.keys[0]
            if name not in ir.schema:
                continue
            (column_index,) = names_to_indices((name,), ir.schema)
            candidate = (
                name,
                OrderKey(
                    column_index, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE
                ),
            )
            if candidate not in candidates:
                candidates.append(candidate)
            continue
        if (
            not isinstance(request, OrderPartitioningRequest)
            or request.source != "hint"
        ):
            continue
        assert request.keys, "Order partitioning requests must have at least one key."
        key = request.keys[0]
        assert key.name in ir.schema, (
            f"Ordering request key {key.name!r} must be present in scan schema."
        )
        (column_index,) = names_to_indices((key.name,), ir.schema)
        candidate = (key.name, OrderKey(column_index, key.order, key.null_order))
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


@functools.cache
def _inference_executor() -> concurrent.futures.ThreadPoolExecutor:
    return concurrent.futures.ThreadPoolExecutor(8, thread_name_prefix="scan-ordering")


def _infer_key_order() -> bool:
    return os.environ.get("CUDF_POLARS_INFER_KEY_ORDER") == "1"


def _get_rank_parquet_info_map(
    base_scan: Scan,
    paths: list[str],
) -> dict[str, CachedParquetInfo]:
    path_set = set(paths)
    cached_parquet_info_map = {
        info.path: info
        for info in (base_scan.cached_parquet_info or ())
        if info.path in path_set
    }
    if set(cached_parquet_info_map) == path_set:
        return cached_parquet_info_map

    from cudf_polars.streaming import remote_io

    if remote_io.scan_io() is not None and all(
        plc.io.SourceInfo._is_remote_uri(p) for p in paths
    ):
        # The scan service fetches every footer of a remote scan anyway: request them now and
        # parse them from the service, without a request of our own.
        from cudf_polars.dsl.utils.io import CachedParquetInfo

        service = remote_io.scan_io()
        service.request_footers(paths)
        infos = {}
        for p in paths:
            size, metadata = service.footer(p)
            infos[p] = CachedParquetInfo(p, size, metadata)
        return infos

    from cudf_polars.dsl.utils.io import _prefetch_parquet_footers_for_paths

    fetched = _prefetch_parquet_footers_for_paths(paths)
    rank_parquet_info_map = {info.path: info for info in fetched}
    assert all(path in rank_parquet_info_map for path in paths), (
        "Ordering footer metadata must contain all rank-local scan paths."
    )
    return rank_parquet_info_map


def _stats_are_safe(
    rank_row_group_metadata: Sequence[plc.io.parquet_metadata.RowGroup],
    name: str,
    indices: list[int],
) -> bool:
    for index in indices:
        row_group = rank_row_group_metadata[index]
        column_chunk = next(
            (
                column_chunk
                for column_chunk in row_group.columns
                if ".".join(column_chunk.meta_data.path_in_schema) == name
            ),
            None,
        )
        stats = None if column_chunk is None else column_chunk.meta_data.statistics
        # Nulls are safe when they all belong in the first or last task,
        # depending on the requested null order. Bail for now to keep the
        # inference logic simple.
        if (
            stats is None
            or stats.null_count is None
            or stats.null_count != 0
            or stats.is_min_value_exact is False
            or stats.is_max_value_exact is False
        ):
            return False
    return True


def _candidate_task_bounds(
    file_metadata: list[plc.io.parquet_metadata.FileMetaData],
    rank_row_group_metadata: Sequence[plc.io.parquet_metadata.RowGroup],
    task_row_group_indices: Sequence[list[int] | None],
    name: str,
    key: OrderKey,
    dtype: plc.DataType,
    stream: Stream,
) -> plc.Column | None:
    """
    Return start and end bounds for each local task.

    Bounds alternate: ``[task0_start, task0_end, task1_start, task1_end, ...]``.
    Parquet stores min/max statistics as encoded values. libcudf decodes them
    into typed device columns used by libcudf sorting operations.
    """
    # Ordering keys can name synthetic columns, such as hive partitions
    # or include_file_paths, that are not Parquet leaf columns.
    if not file_metadata or any(
        name not in metadata.columnchunk_metadata for metadata in file_metadata
    ):
        return None
    bounds = plc.io.parquet_metadata.read_parquet_column_chunk_bounds(
        file_metadata, columns=[name], stream=stream
    )

    columns = bounds.columns()[2:]
    assert len(columns) == 2, "Single-column parquet bounds must have min/max columns."
    min_col, max_col = columns
    assert min_col.size() == max_col.size(), "Parquet min/max columns must align."
    bounds_type = min_col.type()
    assert max_col.type() == bounds_type, (
        "Parquet min/max columns must have matching types."
    )
    assert len(rank_row_group_metadata) == min_col.size(), (
        "Decoded parquet bounds must match footer row-group metadata."
    )

    if min_col.null_count() or max_col.null_count():
        return None

    valid_task_row_group_indices: list[list[int]] = []
    for row_group_indices in task_row_group_indices:
        if row_group_indices is None or not _stats_are_safe(
            rank_row_group_metadata, name, row_group_indices
        ):
            return None
        valid_task_row_group_indices.append(row_group_indices)

    if bounds_type != dtype:
        if not is_order_preserving_cast(bounds_type, dtype):
            return None
        min_col = plc.unary.cast(min_col, dtype, stream=stream)
        max_col = plc.unary.cast(max_col, dtype, stream=stream)

    start, end = (
        (max_col, min_col)
        if key.order == plc.types.Order.DESCENDING
        else (
            min_col,
            max_col,
        )
    )
    row_group_bounds = plc.concatenate.concatenate(
        [plc.Table([start]), plc.Table([end])], stream=stream
    )

    task_bounds: list[plc.Table] = []
    for row_group_indices in valid_task_row_group_indices:
        selected = _gather_rows(
            row_group_bounds,
            [
                i
                for group in row_group_indices
                for i in (group, len(rank_row_group_metadata) + group)
            ],
            stream,
        )
        if not plc.sorting.is_sorted(
            selected, [key.order], [key.null_order], stream=stream
        ):
            return plc.Column.all_null_like(
                min_col, 2 * len(task_row_group_indices), stream=stream
            )
        task_bounds.append(_gather_rows(selected, [0, selected.num_rows() - 1], stream))

    return plc.concatenate.concatenate(task_bounds, stream=stream).columns()[0]


def _extract_local_task_bounds(
    ir: StreamingScan,
    tasks: Sequence[ParquetScanTask],
    candidates: list[tuple[str, OrderKey]],
    rank_parquet_info_map: dict[str, CachedParquetInfo],
    stream: Stream,
) -> plc.Table:
    """Return two endpoint rows per task for each candidate."""
    paths = list(dict.fromkeys(path for task in tasks for path in task.paths))
    rank_row_group_offset_map: dict[str, int] = {}
    rank_row_group_metadata: list[plc.io.parquet_metadata.RowGroup] = []
    for path in paths:
        info = rank_parquet_info_map[path]
        rank_row_group_offset_map[path] = len(rank_row_group_metadata)
        rank_row_group_metadata.extend(info.file_metadata.row_groups)
    task_row_group_indices = [
        task.absolute_row_group_indices(
            rank_parquet_info_map, rank_row_group_offset_map
        )
        for task in tasks
    ]

    file_metadata = [rank_parquet_info_map[path].file_metadata for path in paths]
    bound_count = 2 * len(tasks)
    columns: list[plc.Column] = []
    for name, key in candidates:
        column = (
            _candidate_task_bounds(
                file_metadata,
                rank_row_group_metadata,
                task_row_group_indices,
                name,
                key,
                ir.schema[name].plc_type,
                stream,
            )
            if tasks
            else None
        )
        if column is None:
            column = _null_column(ir, name, bound_count, stream)
        columns.append(column)
    return plc.Table(columns)


def _key_statistics_usable(
    paths: list[str],
    file_metadata: list[plc.io.parquet_metadata.FileMetaData],
    name: str,
) -> bool:
    """Whether every row group's statistics of ``name`` can bound it (exact min/max, no nulls)."""
    from cudf_polars.streaming import remote_io

    service = remote_io.scan_io()
    if service is not None and hasattr(service, "key_statistics_usable"):
        # From the service's parsed footers, without a Python object per column chunk.
        return all(service.key_statistics_usable(paths, name))
    if any(name not in metadata.columnchunk_metadata for metadata in file_metadata):
        return False
    for metadata in file_metadata:
        row_groups = metadata.row_groups
        if not row_groups:
            continue
        index = next(
            (
                i
                for i, chunk in enumerate(row_groups[0].columns)
                if ".".join(chunk.meta_data.path_in_schema) == name
            ),
            None,
        )
        if index is None:
            return False
        for row_group in row_groups:
            stats = row_group.columns[index].meta_data.statistics
            if (
                stats is None
                or stats.null_count != 0
                or stats.is_min_value_exact is False
                or stats.is_max_value_exact is False
            ):
                return False
    return True


def _row_group_bounds(
    paths: list[str],
    file_metadata: list[plc.io.parquet_metadata.FileMetaData],
    row_group_count: int,
    name: str,
    key: OrderKey,
    dtype: plc.DataType,
    stream: Stream,
) -> plc.Column | None:
    """
    Start then end bound (in ``key``'s order) of every rank-local row group, as one column.

    Rows ``[0, n)`` are the starts and ``[n, 2n)`` the ends; None when the footer statistics
    cannot bound the key (missing, inexact, nulls, or a cast that does not keep the order).
    """
    if not file_metadata or not _key_statistics_usable(paths, file_metadata, name):
        return None
    bounds = plc.io.parquet_metadata.read_parquet_column_chunk_bounds(
        file_metadata, columns=[name], stream=stream
    )
    min_col, max_col = bounds.columns()[2:]
    if min_col.size() != row_group_count:
        return None
    if min_col.null_count() or max_col.null_count():
        return None
    if min_col.type() != dtype:
        if not is_order_preserving_cast(min_col.type(), dtype):
            return None
        min_col = plc.unary.cast(min_col, dtype, stream=stream)
        max_col = plc.unary.cast(max_col, dtype, stream=stream)
    start, end = (
        (max_col, min_col)
        if key.order == plc.types.Order.DESCENDING
        else (min_col, max_col)
    )
    return plc.concatenate.concatenate([start, end], stream=stream)


def _fast_local_task_bounds(
    ir: StreamingScan,
    tasks: list[ParquetScanTask],
    candidates: list[tuple[str, OrderKey]],
    rank_parquet_info_map: dict[str, CachedParquetInfo],
    stream: Stream,
) -> tuple[list[ParquetScanTask], plc.Table]:
    """
    Put the tasks in key order and return their bounds, in a few device calls per candidate.

    Like ``_extract_local_task_bounds`` after reordering (experiment, CUDF_POLARS_INFER_KEY_ORDER).

    A candidate's bounds are null unless every row group's ``[start, end]``, in task order, is
    sorted: each task is then ordered inside and the tasks are ordered among themselves.
    """
    paths = list(dict.fromkeys(path for task in tasks for path in task.paths))
    offsets: dict[str, int] = {}
    n = 0
    for path in paths:
        offsets[path] = n
        n += len(rank_parquet_info_map[path].file_metadata.row_group_num_rows)
    groups = [
        task.absolute_row_group_indices(rank_parquet_info_map, offsets)
        for task in tasks
    ]
    file_metadata = [rank_parquet_info_map[path].file_metadata for path in paths]
    bounds = [
        None
        if any(g is None or not g for g in groups)
        else _row_group_bounds(
            paths,
            file_metadata,
            n,
            name,
            key,
            ir.schema[name].plc_type,
            stream,
        )
        for name, key in candidates
    ]
    if len(tasks) > 1 and bounds[0] is not None:
        # Files are listed in name order (part.10 before part.2): read them in key order.
        (first_starts,) = _gather_rows(
            plc.Table([bounds[0]]), [g[0] for g in groups], stream
        ).columns()
        values = first_starts.to_arrow().to_pylist()
        order = sorted(range(len(tasks)), key=values.__getitem__)
        if candidates[0][1].order == plc.types.Order.DESCENDING:
            order.reverse()
        tasks = [tasks[i] for i in order]
        groups = [groups[i] for i in order]
    columns: list[plc.Column] = []
    for (name, key), column in zip(candidates, bounds, strict=True):
        if column is not None:
            (sequence,) = _gather_rows(
                plc.Table([column]),
                [i for g in groups for r in g for i in (r, n + r)],
                stream,
            ).columns()
            if plc.sorting.is_sorted(
                plc.Table([sequence]), [key.order], [key.null_order], stream=stream
            ):
                (task_bounds,) = _gather_rows(
                    plc.Table([column]),
                    [i for g in groups for i in (g[0], n + g[-1])],
                    stream,
                ).columns()
                columns.append(task_bounds)
                continue
        columns.append(_null_column(ir, name, 2 * len(tasks), stream))
    return tasks, plc.Table(columns)


def _partitioning_from_task_bounds(
    context: Context,
    candidates: list[tuple[str, OrderKey]],
    bounds: plc.Table,
    global_chunk_count: int,
    stream: Stream,
) -> Partitioning | None:
    """Infer global partitioning from scan-task bounds in rank order."""
    for i, (_, key) in enumerate(candidates):
        column = bounds.columns()[i]
        if column.null_count():
            continue

        candidate_bounds = plc.Table([column])
        if not plc.sorting.is_sorted(
            candidate_bounds, [key.order], [key.null_order], stream=stream
        ):
            continue

        if global_chunk_count < 2:
            ordering_boundaries = plc.Table(
                [plc.Column.from_iterable_of_py([], column.type(), stream=stream)]
            )
            strict = True
        else:
            ordering_boundaries, strict = _extract_ordering_boundaries(
                candidate_bounds, global_chunk_count, stream
            )
        return Partitioning(
            inter_rank=OrderScheme(
                [
                    Ordering(
                        [key],
                        TableChunk.from_pylibcudf_table(
                            ordering_boundaries,
                            stream,
                            exclusive_view=True,
                            br=context.br(),
                        ),
                        strict_boundaries=strict,
                    )
                ]
            ),
            local="inherit",
        )
    return None


@nvtx_annotate_cudf_polars(message="parquet_metadata_ordering")
async def parquet_metadata_ordering(
    context: Context,
    comm: Communicator,
    ir: StreamingScan,
    global_chunk_count: int,
    requests: tuple[PartitioningRequest, ...],
    ir_context: IRExecutionContext,
    collective_id: int,
) -> tuple[Partitioning | None, list[ParquetScanTask]]:
    """
    Return partitioning inferred from Parquet footer statistics, and the scan's tasks.

    Only the leading key of each ordering request is inspected. For a request
    on ``[a, b]``, this inspects ``a`` but not ``b``, because single-column
    statistics cannot prove ordering on ``[a, b]``. Inference succeeds only
    when task bounds are globally ordered.

    With ``CUDF_POLARS_INFER_KEY_ORDER=1`` the tasks are first put in the order of
    the leading candidate's footer bounds (files are often listed in an order that
    is not the key's, e.g. ``part.10`` before ``part.2``); the returned tasks are
    the order the scan must read them in. (Experimental: assumes the scan's row
    order is not otherwise observed.)
    """
    assert ir.base_scan.typ == "parquet", (
        f"Expected parquet Scan, got {ir.base_scan.typ}."
    )
    assert global_chunk_count > 0, "Scan partition count must be positive."

    tasks: list[ParquetScanTask] = []
    for task in ir.tasks:
        assert isinstance(task, ParquetScanTask)
        tasks.append(task)

    if not (candidates := _get_ordering_candidates(ir, requests)):
        return None, tasks

    from cudf_polars.streaming import remote_io

    paths = list(dict.fromkeys(path for task in tasks for path in task.paths))
    remote_io.trace("ord_begin", -1, len(paths), 0)
    if _infer_key_order():
        # Its own threads: the shared executor is busy decoding the scans already started.
        loop = asyncio.get_running_loop()
        rank_parquet_info_map = await loop.run_in_executor(
            _inference_executor(), _get_rank_parquet_info_map, ir.base_scan, paths
        )
    else:
        rank_parquet_info_map = await ir_context.to_thread(
            _get_rank_parquet_info_map, ir.base_scan, paths
        )
    remote_io.trace("ord_footers", -1, len(paths), 0)
    stream = ir_context.get_cuda_stream()
    if _infer_key_order() and comm.nranks == 1:
        tasks, local_task_bounds = await loop.run_in_executor(
            _inference_executor(),
            _fast_local_task_bounds,
            ir,
            tasks,
            candidates,
            rank_parquet_info_map,
            stream,
        )
    else:
        local_task_bounds = _extract_local_task_bounds(
            ir, tasks, candidates, rank_parquet_info_map, stream
        )
    remote_io.trace("ord_bounds", -1, len(paths), 0)
    global_task_bounds = local_task_bounds
    if comm.nranks > 1:
        local_chunk = TableChunk.from_pylibcudf_table(
            local_task_bounds, stream, exclusive_view=True, br=context.br()
        )
        allgather = AllGatherManager(context, comm, collective_id)
        with allgather.inserting() as inserter:
            await inserter.insert(comm.rank, local_chunk)
        global_task_bounds = await allgather.extract_concatenated(
            stream, ordered=True, ir_context=ir_context
        )

    # StreamingScan emits one output chunk per scan task, so task-bound rows
    # also line up with the global chunk count used by Ordering metadata.
    assert global_task_bounds.num_rows() == 2 * global_chunk_count, (
        "Ordering task bounds must contain first/last rows for every scan task."
    )
    return _partitioning_from_task_bounds(
        context, candidates, global_task_bounds, global_chunk_count, stream
    ), tasks
