# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid scan prefetch pipeline."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, NamedTuple

import nvtx

import pylibcudf as plc
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.core.memory_reserve_or_wait import reserve_memory

from cudf_polars.dsl.ir import _prepare_parquet_predicate
from cudf_polars.dsl.to_ast import to_parquet_filter
from cudf_polars.dsl.tracing import CUDF_POLARS_NVTX_DOMAIN, nvtx_annotate_cudf_polars
from cudf_polars.streaming.io import (
    PinnedBatch,
    PrefetchedByteRanges,
    decide_pass_mode,
    split_row_group_indices,
)
from cudf_polars.utils.config import HybridScanPassMode
from cudf_polars.utils.cuda_stream import get_cuda_stream

if TYPE_CHECKING:
    import asyncio

    from kvikio.cufile import CuFile, IOFuture
    from kvikio.remote_file import RemoteFile

    import pylibcudf.expressions as plc_expr
    from rapidsmpf.memory.buffer import Buffer, BufferHostView
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.io import SplitScan


# TODO: kvikio is adding a remote batch API that coalesces and splits ranges
# itself. Once that lands, coalesce_adjacent_ranges and issue_pread_calls
# below should likely be replaced by calls into it instead of maintaining our
# own coalescing logic.
class CoalescedRange(NamedTuple):
    """One run of consecutive, file-adjacent ranges, merged into a single span."""

    host_start: int
    """Start offset into the destination host buffer."""
    host_end: int
    """End offset into the destination host buffer."""
    file_offset: int
    """Start offset into the source file."""
    file_size: int
    """Number of bytes to read from the source file."""


@nvtx_annotate_cudf_polars(message="coalesce_adjacent_ranges")
def coalesce_adjacent_ranges(ranges: list[Any]) -> list[CoalescedRange]:
    """
    Merge runs of consecutive, file-adjacent ranges into single spans.

    Pure endpoint-adjustment logic, no I/O: computes where each merged span
    starts and ends in both the destination host buffer and the source
    file, for a caller to issue reads against.

    Parameters
    ----------
    ranges
        Byte ranges to coalesce, in the order the caller needs them back in
        (positional, matching a ``HybridScanReader`` byte-range call). Only
        *runs* of already-adjacent entries get merged, not the whole list
        resorted by file offset. Column chunk ranges come back roughly
        file-ordered already, so this still catches most of the coalescing
        benefit without touching the order the caller depends on.

    Returns
    -------
    One :class:`CoalescedRange` per coalesced run, in file order.
    """
    groups = []
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
        groups.append(
            CoalescedRange(
                group_start, offset, group_file_start, group_file_end - group_file_start
            )
        )
        i = j
    return groups


@nvtx_annotate_cudf_polars(message="issue_pread_calls")
def issue_pread_calls(
    handle: CuFile | RemoteFile, ranges: list[Any], host: memoryview
) -> list[IOFuture]:
    """
    Issue one ``pread`` call per run of consecutive, file-adjacent ranges.

    Parameters
    ----------
    handle
        Open kvikio handle to read from.
    ranges
        Byte ranges to read; coalesced via :func:`coalesce_adjacent_ranges`
        before issuing reads.
    host
        Pinned host buffer to read into, sized to fit every range in
        ``ranges`` back to back.

    Returns
    -------
    One ``pread`` future per coalesced run, in file order.
    """
    return [
        handle.pread(
            host[coalesced.host_start : coalesced.host_end],
            size=coalesced.file_size,
            file_offset=coalesced.file_offset,
        )
        for coalesced in coalesce_adjacent_ranges(ranges)
    ]


@nvtx_annotate_cudf_polars(message="issue_reads_into_pinned_buffer")
def issue_reads_into_pinned_buffer(
    buf: Buffer, handle: CuFile | RemoteFile, ranges: list[Any]
) -> tuple[BufferHostView, memoryview, list[IOFuture]]:
    """
    Take a pinned host view of a buffer and issue reads into it.

    A single unit of work so it can be offloaded to a thread together:
    kvikio's ``pread`` submission releases the GIL, but that only lets other
    *threads* make progress. It's still a synchronous call from the event
    loop's perspective (no ``await``), so a slow submission, e.g. contention
    on kvikio's reactor from many concurrent prefetch tasks submitting
    around the same time, would otherwise stall the event loop and every
    other task on it.

    Deliberately doesn't use ``with buf.host_view() as host:`` here: the
    view's exclusive write lock is meant to stay held until the writes
    it's guarding are actually done, but the ``pread`` futures issued
    below are still in flight when this function returns, they're only
    awaited much later (see ``copy_pinned_batch_to_device``). Exiting the
    context manager here would release that lock before the writes it's
    protecting have finished. The caller is responsible for exiting the
    returned view once the returned futures actually complete.

    Parameters
    ----------
    buf
        The pinned buffer to view and read into.
    handle
        Open kvikio handle to read from.
    ranges
        Byte ranges to read.

    Returns
    -------
    The still-open view, the pinned host view read into, and one ``pread``
    future per coalesced run issued against it, in file order.
    """
    view = buf.host_view()
    host = view.__enter__()
    try:
        futures = issue_pread_calls(handle, ranges, host)
    except BaseException:
        view.__exit__(*sys.exc_info())
        raise
    return view, host, futures


async def reserve_pinned_batch(
    context: Context,
    ir_context: IRExecutionContext,
    handle: CuFile | RemoteFile,
    ranges: list[Any],
) -> PinnedBatch | None:
    """
    Reserve pinned host memory and issue reads for one batch of byte ranges.

    Parameters
    ----------
    context
        The rapidsmpf context to reserve memory through.
    ir_context
        The execution context to offload the buffer allocation and reads to.
    handle
        Open kvikio handle to read from.
    ranges
        Byte ranges to reserve for and read.

    Returns
    -------
    The reserved, in-flight batch, or ``None`` when ``ranges`` is empty.
    """
    if not ranges:
        return None
    total = sum(r.size for r in ranges)
    br = context.br()
    # `nvtx.start_range`/`end_range` (not the push/pop `nvtx_annotate_cudf_polars`
    # context manager) since this span crosses `await` points; many prefetch
    # tasks interleave on the same event-loop thread, which would corrupt a
    # thread-local push/pop stack.
    batch_range = nvtx.start_range(
        message="reserve_pinned_batch", domain=CUDF_POLARS_NVTX_DOMAIN, payload=total
    )
    try:
        # TODO: a reservation here can queue behind other pinned memory
        # contention (e.g. shuffle spill) with no way to proactively free up
        # our own holdings.
        wait_range = nvtx.start_range(
            message="reserve_memory_wait", domain=CUDF_POLARS_NVTX_DOMAIN, payload=total
        )
        try:
            reservation = await reserve_memory(
                context,
                size=total,
                net_memory_delta=total,
                mem_type=MemoryType.PINNED_HOST,
            )
        finally:
            nvtx.end_range(wait_range)
        buf = await ir_context.to_prefetch_thread(
            br.make_buffer, total, br.stream_pool.get_stream(), reservation
        )
        view, host, futures = await ir_context.to_prefetch_thread(
            issue_reads_into_pinned_buffer, buf, handle, ranges
        )
    finally:
        nvtx.end_range(batch_range)
    return PinnedBatch(ranges=ranges, host=host, futures=futures, buf=buf, view=view)


@nvtx_annotate_cudf_polars(message="prepare_prefetch")
def prepare_prefetch(
    scan: SplitScan,
) -> (
    tuple[
        plc_expr.Expression, list[int], HybridScanPassMode, list[Any], list[Any] | None
    ]
    | None
):
    """
    Prune row groups for one scan task and compute its byte ranges.

    Parameters
    ----------
    scan
        The scan task to prune and compute byte ranges for.

    Returns
    -------
    ``None`` when the predicate can't be expressed as a parquet filter.
    Otherwise ``(plc_filter, row_group_indices, pass_mode, primary_ranges,
    payload_ranges)``. ``payload_ranges`` is ``None`` under ``SINGLE_PASS``
    (``primary_ranges`` already covers every column), and a second list
    under ``TWO_PASS`` (``primary_ranges`` is the filter columns' ranges).
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

    row_group_indices = split_row_group_indices(
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
            bloom_ranges = reader.bloom_filters_byte_ranges(row_group_indices, options)
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

    pass_mode = decide_pass_mode(
        parquet_options.pass_mode, row_group_indices, row_group_count_before_pruning
    )
    if pass_mode is HybridScanPassMode.SINGLE_PASS:
        ranges = reader.all_column_chunks_byte_ranges(row_group_indices, options)
        return plc_filter, row_group_indices, pass_mode, ranges, None
    filter_ranges = reader.filter_column_chunks_byte_ranges(row_group_indices, options)
    payload_ranges = reader.payload_column_chunks_byte_ranges(
        row_group_indices, options
    )
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
    Prune row groups for one scan task and prefetch its byte ranges.

    Pruning and byte-range computation are offloaded to ``ir_context``'s
    main thread pool and run freely, out of order across tasks. Claiming
    pinned memory and issuing reads is offloaded to the dedicated prefetch
    thread pool instead, and waits for ``wait_for`` first (the previous
    task's own attempt, within the same producer), so a task due for
    consumption soon can't lose its reservation to one that isn't, then
    signals ``own_turn`` before returning, whether or not a reservation
    succeeded.

    Parameters
    ----------
    scan
        The scan task to prefetch.
    context
        The rapidsmpf context to reserve memory through.
    ir_context
        The execution context to offload pruning and byte-range computation to.
    wait_for
        Event to wait on before claiming pinned memory and issuing reads, or
        ``None`` for the first task in a producer's chain.
    own_turn
        Event this task sets once it's done claiming resources, whether or
        not a reservation succeeded, so the next task in the chain can proceed.

    Returns
    -------
    The prefetched byte ranges, or ``None`` when the predicate can't be
    expressed as a parquet filter, the caller falls back to
    ``SplitScan.do_evaluate`` in that case.
    """
    task_range = nvtx.start_range(
        message="prefetch_scan_byte_ranges", domain=CUDF_POLARS_NVTX_DOMAIN
    )
    try:
        prepared = await ir_context.to_thread(prepare_prefetch, scan)
        if wait_for is not None:
            wait_range = nvtx.start_range(
                message="prefetch_wait_for_turn", domain=CUDF_POLARS_NVTX_DOMAIN
            )
            try:
                await wait_for.wait()
            finally:
                nvtx.end_range(wait_range)
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

            if pass_mode is HybridScanPassMode.SINGLE_PASS:
                all_columns = await reserve_pinned_batch(
                    context, ir_context, handle, primary_ranges
                )
                return PrefetchedByteRanges(
                    row_group_indices=row_group_indices,
                    pass_mode=pass_mode,
                    plc_filter=plc_filter,
                    all_columns=all_columns,
                )

            filter_batch = await reserve_pinned_batch(
                context, ir_context, handle, primary_ranges
            )
            payload_batch = await reserve_pinned_batch(
                context, ir_context, handle, payload_ranges or []
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
    finally:
        nvtx.end_range(task_range)
