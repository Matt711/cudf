# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Experiment: run a remote scan's probe chains up to a group-by natively.

``CUDF_POLARS_NATIVE_REGION=1``: a GroupBy whose input is, per union branch, a chain of
projections, column expressions and inner joins in which one remote parquet scan is the probe
side, is executed by the extension's ProbePipeline. The other side of each join (its build
side) runs as usual and is collected into one table; the scan's tasks are planned and fetched
by the scan service and decoded, joined, computed and aggregated on native threads, with no
Python per chunk. The actor sends the group-by's result as one chunk. Anything outside the
supported shape (single rank, sums only, the expression subset below) is left alone.
"""

from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cudf_streaming.channel_metadata import ChannelMetadata
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.streaming.core.actor import define_actor

import pylibcudf as plc

from cudf_polars.containers import DataFrame
from cudf_polars.dsl import expr as E
from cudf_polars.dsl.ir import IR, GroupBy, HStack, Join, Projection, Select, Union
from cudf_polars.dsl.traversal import traversal
from cudf_polars.streaming import remote_io
from cudf_polars.streaming.actor_graph import remote_fetch
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    chunk_to_frame,
    empty_table_chunk,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.streaming.io import StreamingScan
from cudf_polars.streaming.utils import _concat

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IRExecutionContext

__all__ = ["Region", "enabled", "match", "native_region_actor"]


def enabled() -> bool:
    """Whether native regions are on."""
    return os.environ.get("CUDF_POLARS_NATIVE_REGION") == "1"


_BINOPS = {
    plc.binaryop.BinaryOperator.ADD,
    plc.binaryop.BinaryOperator.SUB,
    plc.binaryop.BinaryOperator.MUL,
    plc.binaryop.BinaryOperator.TRUE_DIV,
}
_F64 = plc.TypeId.FLOAT64
_NUMERIC = {
    plc.TypeId.INT16,
    plc.TypeId.INT32,
    plc.TypeId.INT64,
    plc.TypeId.UINT32,
    plc.TypeId.FLOAT64,
}


class _NoMatch(Exception):
    pass


@dataclass
class Region:
    """A matched region: the probe scan, the build inputs and the native stages."""

    scan: StreamingScan
    input_columns: list[str]
    builds: list[IR] = field(default_factory=list)
    build_keys: list[list[int]] = field(default_factory=list)
    branches: list[list[dict]] = field(default_factory=list)


def _expr(e: E.Expr, names: list[str], nodes: list[dict]) -> int:
    """Append ``e``'s nodes; returns the index of its result node."""
    if isinstance(e, E.Col):
        nodes.append({"op": "column", "column": names.index(e.name)})
    elif isinstance(e, E.Literal):
        if e.dtype.plc_type.id() not in (_F64, plc.TypeId.INT32, plc.TypeId.INT64):
            raise _NoMatch(f"literal of {e.dtype}")
        nodes.append({"op": "literal", "value": float(e.value)})
    elif isinstance(e, E.BinOp) and e.op in _BINOPS:
        if e.dtype.plc_type.id() != _F64:
            raise _NoMatch(f"binary op of {e.dtype}")
        lhs = _expr(e.children[0], names, nodes)
        rhs = _expr(e.children[1], names, nodes)
        nodes.append({"op": "binary", "binop": int(e.op), "lhs": lhs, "rhs": rhs})
    elif (
        isinstance(e, E.TemporalFunction)
        and e.name == E.TemporalFunction.Name.Year
    ):
        nodes.append({"op": "year", "lhs": _expr(e.children[0], names, nodes)})
    elif isinstance(e, E.Cast) and e.dtype.plc_type.id() in _NUMERIC:
        lhs = _expr(e.children[0], names, nodes)
        nodes.append({"op": "cast", "binop": int(e.dtype.plc_type.id()), "lhs": lhs})
    else:
        raise _NoMatch(f"expression {type(e).__name__}")
    return len(nodes) - 1


def _compute(
    named: tuple[E.NamedExpr, ...], names: list[str], stages: list[dict]
) -> list[str]:
    """Columns for each named expression (computed ones appended); returns their names."""
    exprs: list[list[dict]] = []
    out: list[str] = []
    appended: list[str] = []
    for ne in named:
        if isinstance(ne.value, E.Col):
            out.append(ne.value.name)
            continue
        nodes: list[dict] = []
        _expr(ne.value, names, nodes)
        exprs.append(nodes)
        tmp = f"__native_{len(names) + len(appended)}"
        appended.append(tmp)
        out.append(tmp)
    if exprs:
        stages.append({"op": "compute", "exprs": exprs})
        names.extend(appended)
    return out


def _select(source: list[str], target: list[str], names: list[str], stages: list[dict]) -> None:
    """Select ``source`` columns (by name) and call them ``target``."""
    if source != names:
        stages.append({"op": "select", "columns": [names.index(n) for n in source]})
    names[:] = target


def _apply(node: IR, names: list[str], region: Region, stages: list[dict], scan: IR) -> None:
    """Translate one node on the probe path (its probe input already in ``names``)."""
    if isinstance(node, Projection):
        cols = list(node.schema)
        _select(cols, cols, names, stages)
    elif isinstance(node, Select):
        src = _compute(node.exprs, names, stages)
        _select(src, [ne.name for ne in node.exprs], names, stages)
    elif isinstance(node, HStack):
        computed = _compute(node.columns, names, stages)
        by_name = dict(zip((ne.name for ne in node.columns), computed, strict=True))
        target = list(node.schema)
        _select([by_name.get(n, n) for n in target], target, names, stages)
    elif isinstance(node, Join):
        how, _nulls_equal, zlice, suffix, coalesce, _order = node.options
        if how != "Inner" or zlice is not None or node.predicate is not None:
            raise _NoMatch(f"join {how}")
        left, right = node.children
        probe_left = any(n == scan for n in traversal([left]))
        build = right if probe_left else left
        if any(n == scan for n in traversal([build])):
            raise _NoMatch("the scan feeds both sides of a join")
        probe_on, build_on = (
            (node.left_on, node.right_on) if probe_left else (node.right_on, node.left_on)
        )
        if not all(isinstance(ne.value, E.Col) for ne in (*probe_on, *build_on)):
            raise _NoMatch("join on expressions")
        build_names = list(build.schema)
        # Output: the left columns, then the right ones (right keys dropped when coalesced),
        # suffixed on collision.
        left_names = list(left.schema)
        right_keys = {ne.value.name for ne in node.right_on}
        right_kept = [n for n in right.schema if not (coalesce and n in right_keys)]
        expected = left_names + [n if n not in left_names else n + suffix for n in right_kept]
        if expected != list(node.schema):
            raise _NoMatch("join output columns")
        if set(left_names) & set(right_kept):
            raise _NoMatch("join suffix")
        probe_kept, build_kept = (
            (left_names, right_kept) if probe_left else (right_kept, left_names)
        )
        region.builds.append(build)
        region.build_keys.append([build_names.index(ne.value.name) for ne in build_on])
        stages.append(
            {
                "op": "join",
                "build": len(region.builds) - 1,
                "probe_keys": [names.index(ne.value.name) for ne in probe_on],
                "keep_probe": [names.index(n) for n in probe_kept],
                "keep_build": [build_names.index(n) for n in build_kept],
            }
        )
        # Native output: kept probe columns, then kept build columns; reorder to the join's.
        names[:] = [*probe_kept, *build_kept]
        out = list(node.schema)
        _select(out, out, names, stages)
    else:
        raise _NoMatch(type(node).__name__)


def _branch_path(root: IR, scan: IR) -> list[IR]:
    """The nodes from ``root`` down to (not including) ``scan``, bottom-up."""
    path: list[IR] = []
    node = root
    while node != scan:
        path.append(node)
        if isinstance(node, Join):
            node = next(c for c in node.children if any(n == scan for n in traversal([c])))
        elif len(node.children) == 1:
            node = node.children[0]
        else:
            raise _NoMatch(f"{type(node).__name__} on the probe path")
    return path[::-1]


def match(
    ir: GroupBy, partition_info: Any, nranks: int, dep_count: Any = None
) -> Region | None:
    """The native region rooted at ``ir``, or None."""
    if not enabled() or nranks != 1 or ir.zlice is not None or dep_count is None:
        return None
    try:
        scans = [
            n
            for n in traversal([ir])
            if isinstance(n, StreamingScan) and remote_fetch.enabled(n.tasks)
        ]
        if not scans:
            return None
        scan = max(scans, key=lambda n: partition_info[n].count)
        # Nodes between the group-by and the union (or the single branch), top-down.
        post: list[IR] = []
        node = ir.children[0]
        while not isinstance(node, Union) and not isinstance(node, Join):
            if len(node.children) != 1 or node == scan:
                raise _NoMatch("no join below the group-by")
            post.append(node)
            node = node.children[0]
        roots = list(node.children) if isinstance(node, Union) else [node]
        if isinstance(node, Union) and node.zlice is not None:
            raise _NoMatch("union slice")
        region = Region(scan=scan, input_columns=list(scan.schema))
        # Every node the region replaces must be used only inside it.
        inside = [ir, *post, *([node] if isinstance(node, Union) else [])]
        for root in roots:
            inside.extend(_branch_path(root, scan))
        inside.append(scan)
        parents: dict[IR, int] = {}
        for n in inside:
            for c in n.children:
                if any(c is m for m in inside):
                    parents[c] = parents.get(c, 0) + 1
        for n in inside[1:]:
            if parents.get(n, 0) != dep_count[n]:
                raise _NoMatch("a replaced node is used outside the region")
        for root in roots:
            names = list(scan.schema)
            stages: list[dict] = []
            for n in [*_branch_path(root, scan), *post[::-1]]:
                _apply(n, names, region, stages, scan)
            # The group-by: column keys, sums.
            if not all(isinstance(k.value, E.Col) for k in ir.keys):
                raise _NoMatch("group-by keys")
            sums: list[E.NamedExpr] = []
            for req in ir.agg_requests:
                if not (isinstance(req.value, E.Agg) and req.value.name == "sum"):
                    raise _NoMatch("aggregation")
                sums.append(E.NamedExpr(req.name, req.value.children[0]))
            values = _compute(tuple(sums), names, stages)
            stages.append(
                {
                    "op": "groupby",
                    "keys": [names.index(k.value.name) for k in ir.keys],
                    "sums": [names.index(v) for v in values],
                }
            )
            region.branches.append(stages)
    except _NoMatch as e:
        remote_fetch._count(f"native_nomatch:{e}")
        return None
    return region


class _Planner:
    """
    Plans and queues the scan's tasks in the service, on a thread.

    Until the pipeline runs nothing releases the scan's arena blocks, and the build sides
    (scanned through the same service) need blocks to finish: so only about
    ``CUDF_POLARS_NATIVE_PREFETCH_MB`` of reads are queued ahead of :meth:`attach`. After
    it, every task goes to the pipeline as soon as it is planned.
    """

    def __init__(self, region: Region, ir_context: IRExecutionContext) -> None:
        self._region = region
        self._ir_context = ir_context
        self._budget = int(os.environ.get("CUDF_POLARS_NATIVE_PREFETCH_MB", "8192")) << 20
        self._lock = threading.Lock()
        self._ahead: list[int] = []
        self._pipeline: Any = None
        self._attached = threading.Event()
        self._aborted = False
        self.stream_id = 0
        # The reader options hold the scan's filter, which the decodes use: keep them alive.
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
            raise RuntimeError("native region: the scan's predicate is not fully pushed down")
        queued = 0
        for task in tasks:
            if queued >= self._budget:
                self._attached.wait()
            if self._aborted:
                return
            plan = remote_io.plan_native(task, self.stream_id, options, residual, stream)
            if plan is None:
                raise RuntimeError("native region: a task the native planner cannot plan")
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
    context: Context,
    ir_context: IRExecutionContext,
    ch: Channel[TableChunk],
    build: IR,
) -> tuple[plc.Table, list[TableChunk]]:
    """
    A build side, collected into one table.

    The table may view the chunks' memory (a single chunk is not copied): keep them too.
    """
    await recv_metadata(ch, context)
    chunks: list[TableChunk] = []
    while (msg := await ch.recv(context)) is not None:
        chunks.append(TableChunk.from_message(msg, br=context.br()))
    if not chunks:
        chunks = [empty_table_chunk(build, context, ir_context.get_cuda_stream())]
    chunks, _extra = await make_table_chunks_available_or_wait(
        context, chunks, reserve_extra=0, net_memory_delta=0
    )
    df = _concat(*[chunk_to_frame(c, build) for c in chunks], context=ir_context)
    df.stream.synchronize()
    return df.table, chunks


async def _collect_builds(
    context: Context,
    ir_context: IRExecutionContext,
    chs_build: list[Channel[TableChunk]],
    region: Region,
) -> list[tuple[plc.Table, list[TableChunk]]]:
    """Every build side, collected concurrently (they may share upstream fanouts)."""
    return list(
        await asyncio.gather(
            *(
                _collect_build(context, ir_context, ch, build)
                for ch, build in zip(chs_build, region.builds, strict=True)
            )
        )
    )


@define_actor()
async def native_region_actor(
    context: Context,
    ir: GroupBy,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    chs_build: list[Channel[TableChunk]],
    region: Region,
) -> None:
    """Collect the build sides, run the probe chains natively, send the group-by result."""
    from cudf_polars_cucascade.scan_io import ProbePipeline  # noqa: PLC0415

    async with shutdown_on_error(
        context,
        chs_in=tuple(chs_build),
        chs_out=(ch_out,),
        trace_ir=ir,
        ir_context=ir_context,
    ) as actor_scope:
        ir_context = actor_scope.require_ir_context()
        # The probe scan's reads start now, while the build sides arrive.
        planner = _Planner(region, ir_context)
        planning = asyncio.ensure_future(asyncio.to_thread(planner.run))
        try:
            collected = await _collect_builds(context, ir_context, chs_build, region)
        except BaseException:
            planner.abort()
            raise
        pipeline = await asyncio.to_thread(
            ProbePipeline,
            remote_io.scan_io(),
            region.branches,
            [table for table, _ in collected],
            region.build_keys,
            int(os.environ.get("CUDF_POLARS_NATIVE_THREADS", "4")),
            region.input_columns,
        )
        planner.attach(pipeline)
        await planning

        def _run() -> list[plc.Table]:
            results = pipeline.finish()
            for k, v in pipeline.stats().items():
                remote_fetch._count(f"native_{k}", v)
            return results

        try:
            results = await asyncio.to_thread(_run)
        finally:
            remote_io.scan_io().close_stream(planner.stream_id)

        # Branch results hold the same keys: fold them with one more sum per key.
        stream = ir_context.get_cuda_stream()
        n_keys = len(ir.keys)
        table = results[0]
        if len(results) > 1:
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
            table = plc.Table([*keys.columns(), *(r.columns()[0] for r in aggs)])
        columns = [
            c
            if c.type() == dtype.plc_type
            else plc.unary.cast(c, dtype.plc_type, stream=stream)
            for c, dtype in zip(table.columns(), ir.schema.values(), strict=True)
        ]
        df = DataFrame.from_table(
            plc.Table(columns), list(ir.schema), list(ir.schema.values()), stream
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
