# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid scan prefetch pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pylibcudf as plc
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.core.memory_reserve_or_wait import reserve_memory

from cudf_polars.dsl.ir import _prepare_parquet_predicate
from cudf_polars.dsl.to_ast import to_parquet_filter
from cudf_polars.streaming.io import (
    PrefetchedByteRanges,
    _decide_pass_mode,
    _PinnedBatch,
    _split_row_group_indices,
)
from cudf_polars.utils.config import (
    HybridScanPassMode,
    HybridScanPrefetchMemoryMode,
)
from cudf_polars.utils.cuda_stream import get_cuda_stream

if TYPE_CHECKING:
    import asyncio

    import pylibcudf.expressions as plc_expr
    from kvikio.cufile import CuFile, IOFuture
    from kvikio.remote_file import RemoteFile
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.io import SplitScan
    from cudf_polars.utils.config import ParquetOptions


def _issue_pread_calls(
    handle: CuFile | RemoteFile, ranges: list[Any], host: memoryview
) -> list[IOFuture]:
    """
    Issue one ``pread`` call per run of consecutive, file-adjacent ranges.

    ``ranges`` stays in the order the caller needs it back in (positional,
    matching a ``HybridScanReader`` byte-range call), so only *runs* of
    already-adjacent entries get merged, not the whole list resorted by file
    offset. Column chunk ranges come back roughly file-ordered already, so
    this still catches most of the coalescing benefit without touching the
    order the caller depends on.
    """
    futures = []
    offset = 0
    i = 0
    n = len(ranges)
    while i < n:
        group_start = offset
        group_file_start = ranges[i].offset
        group_file_end = group_file_start + ranges[i].size
        offset += ranges[i].size
        j = i + 1
        while j < n and ranges[j].offset == group_file_end:
            group_file_end += ranges[j].size
            offset += ranges[j].size
            j += 1
        futures.append(
            handle.pread(
                host[group_start:offset],
                size=group_file_end - group_file_start,
                file_offset=group_file_start,
            )
        )
        i = j
    return futures


async def _reserve_pinned_batch(
    context: Context,
    parquet_options: ParquetOptions,
    handle: CuFile | RemoteFile,
    ranges: list[Any],
) -> _PinnedBatch | None:
    """
    Reserve pinned host memory and issue reads for one batch of byte ranges.

    Returns ``None`` when ``ranges`` is empty, or when
    ``HybridScanPrefetchMemoryMode.FAIL_FAST`` couldn't reserve memory. A
    ``None`` batch is handled the same as a full prefetch miss by the
    consumer, it just reads that batch synchronously later.
    """
    if not ranges:
        return None
    total = sum(r.size for r in ranges)
    br = context.br()
    if parquet_options.prefetch_memory_mode is HybridScanPrefetchMemoryMode.WAIT:
        # TODO: WAIT has no equivalent to FAIL_FAST's demotion. A reservation
        # here can queue behind other pinned memory contention (e.g. shuffle
        # spill) with no way to proactively free up our own holdings.
        reservation = await reserve_memory(
            context,
            size=total,
            net_memory_delta=total,
            mem_type=MemoryType.PINNED_HOST,
        )
    else:
        mem_types = (
            [MemoryType.PINNED_HOST, MemoryType.HOST]
            if parquet_options.prefetch_allow_host_fallback
            else [MemoryType.PINNED_HOST]
        )
        try:
            # TODO: consider demoting our own oldest not-yet-consumed pinned
            # entries to pageable host here instead of just missing.
            reservation = br.reserve_or_fail(total, mem_types)
        except RuntimeError:
            return None
    buf = br.make_buffer(total, br.stream_pool.get_stream(), reservation)
    with buf.host_view() as host:
        futures = _issue_pread_calls(handle, ranges, host)
    return _PinnedBatch(ranges=ranges, host=host, futures=futures, buf=buf)


def _prepare_prefetch(
    scan: SplitScan,
) -> tuple[
    plc_expr.Expression, list[int], HybridScanPassMode, list[Any], list[Any] | None
] | None:
    """
    Prune row groups for one SplitScan and compute its byte ranges.

    Returns ``None`` when the predicate can't be expressed as a parquet
    filter. Otherwise returns ``(plc_filter, row_group_indices, pass_mode,
    primary_ranges, payload_ranges)``. ``payload_ranges`` is ``None`` under
    ``SINGLE_PASS`` (``primary_ranges`` already covers every column), and a
    second list under ``TWO_PASS`` (``primary_ranges`` is the filter
    columns' ranges).
    """
    cached_info = scan.cached_parquet_info
    assert cached_info is not None
    predicate = scan.base_scan.predicate
    assert predicate is not None
    stream = get_cuda_stream()

    plc_filter, residual = to_parquet_filter(
        _prepare_parquet_predicate(
            predicate.value, scan.paths, scan.schema, scan.base_scan.with_columns
        ),
        stream=stream,
    )
    if plc_filter is None or residual is not None:
        return None

    row_group_indices = _split_row_group_indices(
        len(cached_info[0].file_metadata.row_group_num_rows),
        scan.total_splits,
        scan.split_index,
    )
    row_group_count_before_pruning = len(row_group_indices)

    options = cached_info[0].default_reader_options()
    if scan.base_scan.with_columns is not None:
        options.set_column_names(scan.base_scan.with_columns)
    options.set_filter(plc_filter)
    reader = cached_info[0].hybrid_scan_reader(options)

    parquet_options = scan.parquet_options
    if parquet_options._hybrid_scan_stats_pruning:
        row_group_indices = reader.filter_row_groups_with_stats(
            row_group_indices, options, stream=stream
        )
        if row_group_indices:
            bloom_ranges, _ = reader.secondary_filters_byte_ranges(
                row_group_indices, options
            )
            if bloom_ranges:
                source_info = plc.io.SourceInfo(
                    [
                        plc.io.types.FilepathSource(
                            cached_info[0].path, cached_info[0].size
                        )
                    ]
                )
                bloom_chunks = plc.io.parquet_io_utils.fetch_byte_ranges_to_device(
                    source_info, bloom_ranges, stream=stream
                )
                row_group_indices = reader.filter_row_groups_with_bloom_filters(
                    bloom_chunks, row_group_indices, options, stream=stream
                )

    if not row_group_indices:
        return plc_filter, [], HybridScanPassMode.SINGLE_PASS, [], None

    pass_mode = _decide_pass_mode(
        parquet_options.pass_mode, row_group_indices, row_group_count_before_pruning
    )
    if pass_mode is HybridScanPassMode.SINGLE_PASS:
        ranges = reader.all_column_chunks_byte_ranges(row_group_indices, options)
        return plc_filter, row_group_indices, pass_mode, ranges, None
    filter_ranges = reader.filter_column_chunks_byte_ranges(row_group_indices, options)
    payload_ranges = reader.payload_column_chunks_byte_ranges(row_group_indices, options)
    return plc_filter, row_group_indices, pass_mode, filter_ranges, payload_ranges


async def prefetch_scan_byte_ranges(
    scan: SplitScan,
    context: Context,
    ir_context: IRExecutionContext,
    *,
    wait_for: asyncio.Event | None,
    own_turn: asyncio.Event,
) -> PrefetchedByteRanges | None:
    """
    Prune row groups for one SplitScan and prefetch its byte ranges.

    Pruning and byte-range computation are offloaded to ``ir_context``'s
    thread pool and run freely, out of order across splits. Claiming pinned
    memory and issuing reads waits for ``wait_for`` first (the previous
    split's own attempt, within the same producer), so a split due for
    consumption soon can't lose its reservation to one that isn't, then
    signals ``own_turn`` before returning, whether or not a reservation
    succeeded.

    Returns ``None`` when the predicate can't be expressed as a parquet
    filter, the caller falls back to ``SplitScan.do_evaluate`` in that case.
    """
    prepared = await ir_context.to_thread(_prepare_prefetch, scan)
    if wait_for is not None:
        await wait_for.wait()
    try:
        if prepared is None:
            return None
        plc_filter, row_group_indices, pass_mode, primary_ranges, payload_ranges = (
            prepared
        )
        if not row_group_indices:
            return PrefetchedByteRanges.empty(plc_filter)

        assert scan.cached_parquet_info is not None
        handle = scan.cached_parquet_info[0].remote_handle()
        parquet_options = scan.parquet_options

        if pass_mode is HybridScanPassMode.SINGLE_PASS:
            all_columns = await _reserve_pinned_batch(
                context, parquet_options, handle, primary_ranges
            )
            return PrefetchedByteRanges(
                row_group_indices=row_group_indices,
                pass_mode=pass_mode,
                plc_filter=plc_filter,
                all_columns=all_columns,
            )

        filter_batch = await _reserve_pinned_batch(
            context, parquet_options, handle, primary_ranges
        )
        payload_batch = await _reserve_pinned_batch(
            context, parquet_options, handle, payload_ranges or []
        )
        return PrefetchedByteRanges(
            row_group_indices=row_group_indices,
            pass_mode=pass_mode,
            plc_filter=plc_filter,
            filter=filter_batch,
            payload=payload_batch,
        )
    finally:
        own_turn.set()
