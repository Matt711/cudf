# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Matching native regions and rewriting the lowered graph to use them."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from cudf_polars.dsl import expr as E
from cudf_polars.dsl.ir import (
    IR,
    Filter,
    GroupBy,
    HStack,
    Join,
    Projection,
    Select,
    Union,
)
from cudf_polars.dsl.traversal import CachingVisitor, reuse_if_unchanged, traversal
from cudf_polars.streaming import remote_io
from cudf_polars.streaming.base import PartitionInfo
from cudf_polars.streaming.io import ParquetScanTask, StreamingScan
from cudf_polars.streaming.native.expressions import NoMatch, compute, select

if TYPE_CHECKING:
    from collections.abc import Hashable, MutableMapping

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.typing import Schema

__all__ = [
    "NativeProgram",
    "NativeRegion",
    "enabled",
    "rewrite_native_regions",
    "translate_chunkwise",
    "translate_join",
]


def enabled() -> bool:
    """Whether native regions are on (``CUDF_POLARS_NATIVE_REGION=1``)."""
    return os.environ.get("CUDF_POLARS_NATIVE_REGION") == "1"


class NativeProgram:
    """
    What a native region runs.

    Parameters
    ----------
    input_columns
        The probe scan's columns, in the order the stages address them.
    build_keys
        Per build side (the region's children, in order), its key column indices.
    branches
        Per union branch, its stages; each branch ends in a group-by.
    n_keys
        Number of group-by key columns (the result's leading columns).
    """

    __slots__ = ("branches", "build_keys", "input_columns", "key", "n_keys")

    def __init__(
        self,
        input_columns: list[str],
        build_keys: list[list[int]],
        branches: list[list[dict[str, Any]]],
        n_keys: int,
    ) -> None:
        self.input_columns = input_columns
        self.build_keys = build_keys
        self.branches = branches
        self.n_keys = n_keys
        self.key = json.dumps(
            [input_columns, build_keys, branches, n_keys], sort_keys=True
        )


class NativeRegion(IR):
    """
    A subgraph run by the native pipeline.

    Its children are the build sides of the region's joins (one per join, in program
    order); the probe scan is an attribute, since the region reads it itself.
    """

    __slots__ = ("program", "scan")
    _non_child = ("schema", "program", "scan")
    _n_non_child_args = 0
    program: NativeProgram
    scan: StreamingScan

    def __init__(
        self,
        schema: Schema,
        program: NativeProgram,
        scan: StreamingScan,
        *builds: IR,
    ) -> None:
        self.schema = schema
        self.program = program
        self.scan = scan
        self._non_child_args = ()
        self.children = builds

    def get_hashable(self) -> Hashable:
        """Hashable representation of the node."""
        schema = tuple(self.schema.items())
        return (type(self), schema, self.program.key, self.scan, self.children)

    @classmethod
    def do_evaluate(cls, *, context: IRExecutionContext) -> Any:
        """Raises: a native region only runs in the streaming network."""
        raise NotImplementedError("NativeRegion runs only in the streaming network")


def _contains(root: IR, node: IR) -> bool:
    return any(n == node for n in traversal([root]))


def translate_join(
    node: Join,
    *,
    probe_left: bool,
    names: list[str],
    stages: list[dict[str, Any]],
    builds: list[IR],
    build_keys: list[list[int]],
) -> None:
    """
    Append the stages of an inner join probed by one side (its input is ``names``).

    The other side is appended to ``builds``; the stages probe it by that index and leave the
    join's output columns, in the join's order, in ``names``.
    """
    how, _nulls_equal, zlice, _suffix, coalesce, _order = node.options
    if how != "Inner" or zlice is not None or node.predicate is not None:
        raise NoMatch(f"join {how}")
    left, right = node.children
    build = right if probe_left else left
    probe_on, build_on = (
        (node.left_on, node.right_on) if probe_left else (node.right_on, node.left_on)
    )
    if not all(isinstance(ne.value, E.Col) for ne in (*probe_on, *build_on)):
        raise NoMatch("join on expressions")
    build_names = list(build.schema)
    # The join's output: the left columns, then the right ones (right keys dropped when
    # coalesced). Name collisions (suffixes) are not supported.
    left_names = list(left.schema)
    right_keys = {ne.value.name for ne in node.right_on}
    right_kept = [n for n in right.schema if not (coalesce and n in right_keys)]
    if set(left_names) & set(right_kept):
        raise NoMatch("join suffix")
    if left_names + right_kept != list(node.schema):
        raise NoMatch("join output columns")
    probe_kept, build_kept = (
        (left_names, right_kept) if probe_left else (right_kept, left_names)
    )
    builds.append(build)
    build_keys.append([build_names.index(ne.value.name) for ne in build_on])
    stages.append(
        {
            "op": "join",
            "build": len(builds) - 1,
            "probe_keys": [names.index(ne.value.name) for ne in probe_on],
            "keep_probe": [names.index(n) for n in probe_kept],
            "keep_build": [build_names.index(n) for n in build_kept],
        }
    )
    # The native output is the kept probe columns, then the kept build columns.
    names[:] = [*probe_kept, *build_kept]
    out = list(node.schema)
    select(out, out, names, stages)


def _translate(
    node: IR,
    names: list[str],
    stages: list[dict[str, Any]],
    builds: list[IR],
    build_keys: list[list[int]],
    scan: StreamingScan,
) -> None:
    """Append the stages of one node on the probe path (its input is ``names``)."""
    if isinstance(node, Projection):
        cols = list(node.schema)
        select(cols, cols, names, stages)
    elif isinstance(node, Select):
        src = compute(node.exprs, names, stages)
        select(src, [ne.name for ne in node.exprs], names, stages)
    elif isinstance(node, Filter):
        (mask,) = compute((node.mask,), names, stages)
        stages.append({"op": "filter", "columns": [names.index(mask)]})
        cols = list(node.schema)
        select(cols, cols, names, stages)
    elif isinstance(node, HStack):
        computed = compute(node.columns, names, stages)
        by_name = dict(zip((ne.name for ne in node.columns), computed, strict=True))
        target = list(node.schema)
        select([by_name.get(n, n) for n in target], target, names, stages)
    elif isinstance(node, Join):
        left, right = node.children
        probe_left = _contains(left, scan)
        build = right if probe_left else left
        if _contains(build, scan):
            raise NoMatch("the scan feeds both sides of a join")
        # The probe scan must be the largest scan around: no build side reads more tasks.
        if any(
            isinstance(n, StreamingScan) and len(n.tasks) >= len(scan.tasks)
            for n in traversal([build])
        ):
            raise NoMatch("a build side reads as much as the probe")
        translate_join(
            node,
            probe_left=probe_left,
            names=names,
            stages=stages,
            builds=builds,
            build_keys=build_keys,
        )
    else:
        raise NoMatch(type(node).__name__)


def translate_chunkwise(nodes: tuple[IR, ...]) -> list[dict[str, Any]]:
    """
    The native stages of a chain of projections, selections, filters and column expressions.

    ``nodes`` run in order, each on the previous one's output, the first on its child's.

    Raises
    ------
    NoMatch
        When a node or expression is outside what native stages support.
    """
    if not nodes:
        raise NoMatch("empty chain")
    names = list(nodes[0].children[0].schema)
    stages: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, (Projection, Select, HStack, Filter)) or (
            isinstance(node, Filter)
            and os.environ.get("CUDF_POLARS_NATIVE_FILTER") != "1"
        ):
            raise NoMatch(type(node).__name__)
        _translate(node, names, stages, [], [], scan=None)  # type: ignore[arg-type]
    if names != list(nodes[-1].schema):
        raise NoMatch("output columns")
    return stages


def _probe_path(root: IR, scan: StreamingScan) -> list[IR]:
    """The nodes from ``root`` down to ``scan`` (excluded), bottom-up."""
    path: list[IR] = []
    node = root
    while node != scan:
        path.append(node)
        if isinstance(node, Join):
            node = next(c for c in node.children if _contains(c, scan))
        elif len(node.children) == 1:
            node = node.children[0]
        else:
            raise NoMatch(f"{type(node).__name__} on the probe path")
    return path[::-1]


def match_groupby(
    ir: GroupBy, dep_count: MutableMapping[IR, int]
) -> tuple[NativeProgram, StreamingScan, list[IR]]:
    """
    Match a native region rooted at a group-by.

    The group-by's input must be, per union branch, a chain of projections, selections,
    column expressions and inner joins down to one remote parquet scan, which is the probe
    side of every join. Every node the region replaces must be used only inside it.

    Returns
    -------
    The program, the probe scan and the build side of each join, in program order.

    Raises
    ------
    NoMatch
        When the subgraph is outside the supported shape.
    """
    if ir.zlice is not None:
        raise NoMatch("group-by slice")
    if not all(isinstance(k.value, E.Col) for k in ir.keys):
        raise NoMatch("group-by keys")
    if not all(
        isinstance(req.value, E.Agg) and req.value.name == "sum"
        for req in ir.agg_requests
    ):
        raise NoMatch("aggregation")
    scans = [
        n
        for n in traversal([ir])
        if isinstance(n, StreamingScan)
        and n.tasks
        and all(
            isinstance(t, ParquetScanTask) and remote_io.eligible(t) for t in n.tasks
        )
    ]
    if not scans:
        raise NoMatch("no remote scan")
    scan = max(scans, key=lambda n: len(n.tasks))
    # The nodes between the group-by and the union (or the one branch), top-down.
    post: list[IR] = []
    node = ir.children[0]
    while not isinstance(node, (Union, Join)):
        if len(node.children) != 1 or node == scan:
            raise NoMatch("no join below the group-by")
        post.append(node)
        node = node.children[0]
    if isinstance(node, Union) and node.zlice is not None:
        raise NoMatch("union slice")
    roots = list(node.children) if isinstance(node, Union) else [node]
    paths = [_probe_path(root, scan) for root in roots]
    # Every replaced node must be used only inside the region.
    inside = [ir, *post, *([node] if isinstance(node, Union) else [])]
    for path in paths:
        inside.extend(path)
    inside.append(scan)
    # Branches may share nodes (one join feeding both): count each node once.
    unique = list({id(n): n for n in inside}.values())
    parents: defaultdict[int, int] = defaultdict(int)
    for n in unique:
        for c in n.children:
            parents[id(c)] += 1
    for n in unique[1:]:
        if parents[id(n)] != dep_count[n]:
            raise NoMatch(f"a replaced {type(n).__name__} is used outside the region")

    builds: list[IR] = []
    build_keys: list[list[int]] = []
    branches: list[list[dict[str, Any]]] = []
    for path in paths:
        names = list(scan.schema)
        stages: list[dict[str, Any]] = []
        for n in [*path, *post[::-1]]:
            _translate(n, names, stages, builds, build_keys, scan)
        sums = tuple(
            E.NamedExpr(req.name, req.value.children[0]) for req in ir.agg_requests
        )
        values = compute(sums, names, stages)
        stages.append(
            {
                "op": "groupby",
                "keys": [names.index(k.value.name) for k in ir.keys],
                "sums": [names.index(v) for v in values],
            }
        )
        branches.append(stages)
    program = NativeProgram(list(scan.schema), build_keys, branches, len(ir.keys))
    return program, scan, builds


def rewrite_native_regions(
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    *,
    nranks: int,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """
    Replace every matching subgraph of a lowered graph with a :class:`NativeRegion`.

    Parameters
    ----------
    ir
        Root of the lowered graph.
    partition_info
        Its partition information.
    nranks
        Number of ranks (native regions run on a single rank only).

    Returns
    -------
    The rewritten graph and its partition information.
    """
    if not enabled() or nranks != 1:
        return ir, partition_info
    dep_count: defaultdict[IR, int] = defaultdict(int)
    for n in traversal([ir]):
        for c in n.children:
            dep_count[c] += 1
    info: dict[IR, PartitionInfo] = dict(partition_info)

    def visit(node: IR, rec: Any) -> IR:
        if isinstance(node, GroupBy):
            try:
                program, scan, builds = match_groupby(node, dep_count)
            except NoMatch:
                pass
            else:
                region = NativeRegion(
                    node.schema, program, scan, *(rec(b) for b in builds)
                )
                info[region] = PartitionInfo(count=1)
                return region
        new = reuse_if_unchanged(node, rec)
        if new is not node:
            info[new] = partition_info[node]
        return new

    mapper: Any = CachingVisitor(visit, state=None)
    return mapper(ir), info
