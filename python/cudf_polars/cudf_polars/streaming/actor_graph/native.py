# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Streaming network nodes for native regions."""

from __future__ import annotations

import asyncio
import os
import threading
from typing import TYPE_CHECKING, Any

import pylibcudf as plc
from cudf_streaming.channel_metadata import ChannelMetadata
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.streaming.core.actor import define_actor

from cudf_polars.containers import DataFrame
from cudf_polars.dsl import expr as E
from cudf_polars.dsl.ir import GroupBy
from cudf_polars.streaming import remote_io
from cudf_polars.streaming.actor_graph import remote_fetch
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
    ir_context_for_node,
)
from cudf_polars.streaming.actor_graph.partitioning import maybe_remap_partitioning
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    chunk_to_frame,
    empty_table_chunk,
    gather_in_task_group,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.streaming.native import NativeRegion
from cudf_polars.streaming.native.expressions import NoMatch, compute
from cudf_polars.streaming.native.region import (
    translate_chunkwise,
    translate_join,
)
from cudf_polars.streaming.utils import _concat

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR, IRExecutionContext, Join
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator


class _Planner:
    """
    Plans and queues the probe scan's tasks in the scan service, on a thread.

    Until the pipeline runs nothing releases the scan's arena blocks, and the build sides
    (scanned through the same service) need blocks to finish: so only about
    ``CUDF_POLARS_NATIVE_PREFETCH_MB`` of reads are queued before :meth:`attach`. After it,
    every task goes to the pipeline as soon as it is planned.
    """

    def __init__(self, region: NativeRegion, ir_context: IRExecutionContext) -> None:
        self._region = region
        self._ir_context = ir_context
        self._budget = (
            int(os.environ.get("CUDF_POLARS_NATIVE_PREFETCH_MB", "20480")) << 20
        )
        self._lock = threading.Lock()
        self._ahead: list[int] = []
        self._pipeline: Any = None
        self._attached = threading.Event()
        self._aborted = False
        self.stream_id = 0
        # The reader options hold the scan's filter, which every decode uses.
        self.options: Any = None

    def run(self) -> None:
        service = remote_io.scan_io()
        tasks = self._region.scan.tasks
        service.request_footers([p for t in tasks for p in t.paths])
        self.stream_id = service.open_stream()
        stream = self._ir_context.get_cuda_stream()
        options, residual = remote_io.scan_options(tasks[0], stream)
        self.options = options
        if residual is not None:
            raise RuntimeError(
                "native region: the scan's filter is not fully pushed down"
            )
        queued = 0
        for task in tasks:
            if queued >= self._budget:
                self._attached.wait()
            if self._aborted:
                return
            plan = remote_io.plan_native(
                task, self.stream_id, options, residual, stream
            )
            if plan is None:
                raise RuntimeError(
                    "native region: a task the native planner cannot plan"
                )
            if plan.empty:
                continue
            queued += plan.nbytes
            with self._lock:
                if self._pipeline is None:
                    self._ahead.append(plan.task_id)
                    continue
            self._pipeline.submit(plan.task_id)

    def attach(self, pipeline: Any) -> None:
        """The pipeline runs: hand it the tasks queued so far and every later one."""
        with self._lock:
            self._pipeline = pipeline
            ahead, self._ahead = self._ahead, []
        for task_id in ahead:
            pipeline.submit(task_id)
        self._attached.set()

    def abort(self) -> None:
        """Stop planning (the actor failed before the pipeline ran)."""
        self._aborted = True
        self._attached.set()


async def _collect_build(
    context: Context, ir_context: IRExecutionContext, ch: Channel[TableChunk], build: IR
) -> tuple[plc.Table, list[TableChunk]]:
    """
    A build side, collected into one table.

    The table may view the chunks' memory (a single chunk is not copied): the chunks are
    returned too, to be kept alive with it.
    """
    await recv_metadata(ch, context)
    chunks: list[TableChunk] = []
    while (msg := await ch.recv(context)) is not None:
        chunks.append(TableChunk.from_message(msg, br=context.br()))
    if not chunks:
        chunks = [empty_table_chunk(build, context, ir_context.get_cuda_stream())]
    chunks, _ = await make_table_chunks_available_or_wait(
        context, chunks, reserve_extra=0, net_memory_delta=0
    )
    df = _concat(*[chunk_to_frame(c, build) for c in chunks], context=ir_context)
    df.stream.synchronize()
    return df.table, chunks


def _fold_branches(results: list[plc.Table], n_keys: int, stream: Any) -> plc.Table:
    """Union branches hold the same keys: one more sum per key folds them."""
    if len(results) == 1:
        return results[0]
    table = plc.concatenate.concatenate(results, stream=stream)
    gb = plc.groupby.GroupBy(
        plc.Table(list(table.columns()[:n_keys])), plc.types.NullPolicy.INCLUDE
    )
    keys, aggs = gb.aggregate(
        [
            plc.groupby.GroupByRequest(c, [plc.aggregation.sum()])
            for c in table.columns()[n_keys:]
        ],
        stream=stream,
    )
    return plc.Table([*keys.columns(), *(r.columns()[0] for r in aggs)])


def _cast_to(table: plc.Table, dtypes: list[Any], stream: Any) -> plc.Table:
    """``table`` with each column cast to its dtype in ``dtypes`` where they differ."""
    return plc.Table(
        [
            c
            if c.type() == dtype.plc_type
            else plc.unary.cast(c, dtype.plc_type, stream=stream)
            for c, dtype in zip(table.columns(), dtypes, strict=True)
        ]
    )


@define_actor()
async def native_region_actor(
    context: Context,
    ir: NativeRegion,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    chs_build: list[Channel[TableChunk]],
) -> None:
    """Collect the build sides, run the probe chains natively, send the result."""
    from cudf_polars_cucascade.scan_io import ProbePipeline

    async with shutdown_on_error(
        context,
        chs_in=tuple(chs_build),
        chs_out=(ch_out,),
        trace_ir=ir,
        ir_context=ir_context,
    ) as actor_scope:
        ir_context = actor_scope.require_ir_context()
        program = ir.program
        # The probe scan's reads start now, while the build sides arrive.
        planner = _Planner(ir, ir_context)
        planning = asyncio.ensure_future(ir_context.to_thread(planner.run))
        try:
            # Concurrently: build sides may share upstream fanouts.
            collected = await gather_in_task_group(
                *(
                    _collect_build(context, ir_context, ch, build)
                    for ch, build in zip(chs_build, ir.children, strict=True)
                )
            )
        except BaseException:
            planner.abort()
            raise
        pipeline = await ir_context.to_thread(
            ProbePipeline,
            remote_io.scan_io(),
            program.branches,
            [table for table, _ in collected],
            program.build_keys,
            int(os.environ.get("CUDF_POLARS_NATIVE_THREADS", "4")),
            program.input_columns,
        )
        planner.attach(pipeline)
        await planning
        try:
            results = await ir_context.to_thread(pipeline.finish)
        finally:
            remote_io.scan_io().close_stream(planner.stream_id)
        for k, v in pipeline.stats().items():
            remote_fetch._count(f"native_{k}", v)

        stream = ir_context.get_cuda_stream()
        table = _fold_branches(results, program.n_keys, stream)
        df = DataFrame.from_table(
            _cast_to(table, list(ir.schema.values()), stream),
            list(ir.schema),
            list(ir.schema.values()),
            stream,
        )
        await send_metadata(ch_out, context, ChannelMetadata(local_count=1))
        await send_chunk(
            context,
            ch_out,
            TableChunk.from_pylibcudf_table(
                df.table, df.stream, exclusive_view=True, br=context.br()
            ),
            0,
            tracer=actor_scope.tracer,
        )
        await ch_out.drain(context)


# ---------------------------------------------------------------- C++ actors


def _actors_mode() -> str:
    """``CUDF_POLARS_NATIVE_ACTORS``: ``fused`` (default), ``composed`` or ``python``."""
    return os.environ.get("CUDF_POLARS_NATIVE_ACTORS", "fused")


def _stage_groups(stages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """One group per operator: a join, a compute or a group-by, with the selects after it."""
    groups: list[list[dict[str, Any]]] = []
    for s in stages:
        if s["op"] == "select" and groups:
            groups[-1].append(s)
        else:
            groups.append([s])
    return groups


@define_actor()
async def _fold_branches_actor(
    context: Context,
    ir: NativeRegion,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    chs_in: list[Channel[TableChunk]],
) -> None:
    """Union branches folded separately (composed mode): fold their sums into one chunk."""
    async with shutdown_on_error(
        context,
        chs_in=tuple(chs_in),
        chs_out=(ch_out,),
        trace_ir=ir,
        ir_context=ir_context,
    ) as actor_scope:
        ir_context = actor_scope.require_ir_context()
        tables: list[plc.Table] = []
        chunks: list[TableChunk] = []
        for ch in chs_in:
            await recv_metadata(ch, context)
            while (msg := await ch.recv(context)) is not None:
                chunks.append(TableChunk.from_message(msg, br=context.br()))
        chunks, _ = await make_table_chunks_available_or_wait(
            context, chunks, reserve_extra=0, net_memory_delta=0
        )
        for chunk in chunks:
            chunk.stream.synchronize()
            tables.append(chunk_to_frame(chunk, ir).table)
        stream = ir_context.get_cuda_stream()
        table = _fold_branches(tables, ir.program.n_keys, stream)
        df = DataFrame.from_table(
            _cast_to(table, list(ir.schema.values()), stream),
            list(ir.schema),
            list(ir.schema.values()),
            stream,
        )
        await send_metadata(ch_out, context, ChannelMetadata(local_count=1))
        await send_chunk(
            context,
            ch_out,
            TableChunk.from_pylibcudf_table(
                df.table, df.stream, exclusive_view=True, br=context.br()
            ),
            0,
            tracer=actor_scope.tracer,
        )
        await ch_out.drain(context)


def _cpp_sub_network(
    ir: NativeRegion, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    from cudf_polars_cucascade.scan_io import (
        broadcast_actor,
        chain_actor,
        scan_source_actor,
    )

    context = rec.state["context"]
    actors, channels = process_children(ir, rec)
    chs_build = [channels[child].reserve_output_slot() for child in ir.children]
    channels[ir] = ChannelManager(context)
    ch_out = channels[ir].reserve_input_slot()
    program = ir.program
    scan = ir.scan
    ir_context = ir_context_for_node(rec, ir)
    options, residual = remote_io.scan_options(
        scan.tasks[0], ir_context.get_cuda_stream()
    )
    if residual is not None:
        raise RuntimeError("native region: the scan's filter is not fully pushed down")
    type_id = lambda dtype: int(dtype.plc_type.id())  # noqa: E731
    build_types = [[type_id(t) for t in child.schema.values()] for child in ir.children]
    out_types = [type_id(t) for t in ir.schema.values()]
    concurrency = int(os.environ.get("CUDF_POLARS_NATIVE_THREADS", "4"))
    nodes: list[Any] = []
    ch_scan = context.create_channel()
    nodes.append(
        scan_source_actor(
            context,
            ch_scan,
            remote_io.scan_io(),
            [(list(t.paths), t.split_index, t.total_splits) for t in scan.tasks],
            options,
            scan.tasks[0].parquet_options._hybrid_scan_stats_pruning,
            remote_io.MERGE_GAP_BYTES,
            concurrency,
            program.input_columns,
            [type_id(scan.schema[c]) for c in program.input_columns],
            int(os.environ.get("CUDF_POLARS_NATIVE_PREFETCH_MB", "20480")) << 20,
        )
    )
    if _actors_mode() == "fused":
        nodes.append(
            chain_actor(
                context,
                chs_build,
                program.build_keys,
                build_types,
                ch_scan,
                ch_out,
                program.branches,
                concurrency,
                program.n_keys,
                out_types,
            )
        )
    else:
        # One C++ actor per operator: a fanout to the branches, then per branch a chain of
        # single-operator actors, each join reading its own build side.
        if len(program.branches) == 1:
            branch_inputs = [ch_scan]
        else:
            branch_inputs = [context.create_channel() for _ in program.branches]
            nodes.append(broadcast_actor(context, ch_scan, branch_inputs))
        branch_outputs = []
        for stages, ch_branch in zip(program.branches, branch_inputs, strict=True):
            groups = _stage_groups(stages)
            ch_in = ch_branch
            for i, group in enumerate(groups):
                last = i == len(groups) - 1
                builds = [s["build"] for s in group if s["op"] == "join"]
                # Each actor reads only its own build sides: renumber them from 0.
                local = [
                    {**s, "build": builds.index(s["build"])} if s["op"] == "join" else s
                    for s in group
                ]
                ch_next = context.create_channel()
                nodes.append(
                    chain_actor(
                        context,
                        [chs_build[b] for b in builds],
                        [program.build_keys[b] for b in builds],
                        [build_types[b] for b in builds],
                        ch_in,
                        ch_next,
                        [local],
                        concurrency,
                        program.n_keys if last else -1,
                        out_types if last else [],
                    )
                )
                ch_in = ch_next
            branch_outputs.append(ch_in)
        nodes.append(
            _fold_branches_actor(context, ir, ir_context, ch_out, branch_outputs)
        )
    actors[ir] = nodes
    return actors, channels


@generate_ir_sub_network.register(NativeRegion)
def _(
    ir: NativeRegion, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    if _actors_mode() != "python":
        return _cpp_sub_network(ir, rec)
    # One output slot per build side, so a build feeding two joins is read twice.
    actors, channels = process_children(ir, rec)
    chs_build = [channels[child].reserve_output_slot() for child in ir.children]
    channels[ir] = ChannelManager(rec.state["context"])
    actors[ir] = [
        native_region_actor(
            rec.state["context"],
            ir,
            ir_context_for_node(rec, ir),
            channels[ir].reserve_input_slot(),
            chs_build,
        )
    ]
    return actors, channels


# ---------------------------------------------------------------- native operators


def native_join_enabled() -> bool:
    """Whether broadcast joins probe natively (``CUDF_POLARS_NATIVE_JOIN=1``)."""
    return os.environ.get("CUDF_POLARS_NATIVE_JOIN") == "1"


def native_probe_plan(
    ir: Join, broadcast_side: str
) -> tuple[list[dict[str, Any]], list[int]] | None:
    """The native stages probing ``ir``'s broadcast side, and its key columns, or None."""
    probe_left = broadcast_side == "right"
    names = list(ir.children[0 if probe_left else 1].schema)
    stages: list[dict[str, Any]] = []
    builds: list[IR] = []
    build_keys: list[list[int]] = []
    try:
        translate_join(
            ir,
            probe_left=probe_left,
            names=names,
            stages=stages,
            builds=builds,
            build_keys=build_keys,
        )
    except NoMatch:
        return None
    return stages, build_keys[0]


async def native_broadcast_probe(
    context: Context,
    small: DataFrame,
    stages: list[dict[str, Any]],
    build_keys: list[int],
    ch_large: Channel[TableChunk],
    ch_out: Channel[TableChunk],
) -> None:
    """
    Probe every chunk of ``ch_large`` against ``small`` in a C++ actor, in arrival order.

    The caller has received both inputs' metadata and sent the output's; the actor drains
    ``ch_out``.
    """
    from cudf_polars_cucascade.scan_io import chain_actor

    from rapidsmpf.streaming.core.actor import when_all

    ch_build: Channel[TableChunk] = context.create_channel()
    actor = chain_actor(
        context,
        [ch_build],
        [build_keys],
        [[int(c.dtype.plc_type.id()) for c in small.columns]],
        ch_large,
        ch_out,
        [stages],
        int(os.environ.get("CUDF_POLARS_NATIVE_THREADS", "4")),
        -1,
        [],
        forward_metadata=False,
        ordered=True,
    )

    async def feed() -> None:
        await send_metadata(ch_build, context, ChannelMetadata(local_count=1))
        await send_chunk(
            context,
            ch_build,
            TableChunk.from_pylibcudf_table(
                small.table, small.stream, exclusive_view=False, br=context.br()
            ),
            0,
            tracer=None,
        )
        await ch_build.drain(context)

    await gather_in_task_group(feed(), when_all(context, [actor]))


def native_chunkwise_enabled() -> bool:
    """Whether chunkwise chains run natively (``CUDF_POLARS_NATIVE_CHUNKWISE=1``)."""
    return os.environ.get("CUDF_POLARS_NATIVE_CHUNKWISE") == "1"


def native_chunkwise_plan(nodes: tuple[IR, ...]) -> list[dict[str, Any]] | None:
    """The native stages of a chain of chunkwise nodes, or None when unsupported."""
    try:
        return translate_chunkwise(nodes)
    except NoMatch:
        return None


async def native_chain(
    context: Context,
    stages: list[dict[str, Any]],
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
) -> None:
    """
    Run ``stages`` on every chunk of ``ch_in`` in a C++ actor, in arrival order.

    The caller has received the input's metadata and sent the output's; the actor drains
    ``ch_out``.
    """
    from cudf_polars_cucascade.scan_io import chain_actor

    from rapidsmpf.streaming.core.actor import when_all

    actor = chain_actor(
        context,
        [],
        [],
        [],
        ch_in,
        ch_out,
        [stages],
        int(os.environ.get("CUDF_POLARS_NATIVE_THREADS", "4")),
        -1,
        [],
        forward_metadata=False,
        ordered=True,
    )
    await when_all(context, [actor])


def native_groupby_enabled() -> bool:
    """Whether group-by partials are computed natively (``CUDF_POLARS_NATIVE_GROUPBY=1``)."""
    return os.environ.get("CUDF_POLARS_NATIVE_GROUPBY") == "1"


def native_piecewise_plan(
    ir: GroupBy,
) -> tuple[GroupBy, list[dict[str, Any]]] | None:
    """
    The piecewise group-by of ``ir`` and its native stages, or None.

    Only group-bys whose aggregations are all sums: re-aggregating a sum's partials gives the
    same sums, so every strategy the group-by actor picks stays correct.
    """
    from cudf_polars.streaming.actor_graph.groupby import (
        DecomposedGroupBy,
    )

    if ir.zlice is not None or not all(
        isinstance(req.value, E.Agg) and req.value.name == "sum"
        for req in ir.agg_requests
    ):
        return None
    piecewise = DecomposedGroupBy.from_ir(ir).piecewise_ir
    assert isinstance(piecewise, GroupBy)
    if not all(isinstance(k.value, E.Col) for k in piecewise.keys) or not all(
        isinstance(req.value, E.Agg) and req.value.name == "sum"
        for req in piecewise.agg_requests
    ):
        return None
    names = list(piecewise.children[0].schema)
    stages: list[dict[str, Any]] = []
    try:
        values = compute(
            tuple(
                E.NamedExpr(req.name, req.value.children[0])
                for req in piecewise.agg_requests
            ),
            names,
            stages,
        )
    except NoMatch:
        return None
    stages.append(
        {
            "op": "groupby",
            "keys": [names.index(k.value.name) for k in piecewise.keys],
            "sums": [names.index(v) for v in values],
        }
    )
    return piecewise, stages


@define_actor()
async def native_piecewise_actor(
    context: Context,
    piecewise: GroupBy,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    stages: list[dict[str, Any]],
) -> None:
    """Compute each chunk's piecewise group-by natively, one output chunk per input chunk."""
    async with shutdown_on_error(
        context,
        chs_in=(ch_in,),
        chs_out=(ch_out,),
        trace_ir=piecewise,
        ir_context=ir_context,
    ):
        metadata_in = await recv_metadata(ch_in, context)
        partitioning = maybe_remap_partitioning(
            piecewise, metadata_in.partitioning, context=context
        )
        await send_metadata(
            ch_out,
            context,
            ChannelMetadata(
                local_count=metadata_in.local_count,
                partitioning=partitioning,
                duplicated=metadata_in.duplicated,
            ),
        )
        await native_chain(context, stages, ch_in, ch_out)
