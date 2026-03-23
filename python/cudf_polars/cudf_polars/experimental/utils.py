# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition utilities."""

from __future__ import annotations

import operator
import warnings
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING

from cudf_polars.dsl.expr import (
    Agg,
    BinOp,
    BooleanFunction,
    Cast,
    Col,
    Expr,
    Filter,
    Gather,
    GroupedRollingWindow,
    Len,
    Literal,
    LiteralColumn,
    RollingWindow,
    Slice,
    Sort,
    SortBy,
    StringFunction,
    StructFunction,
    TemporalFunction,
    Ternary,
    UnaryFunction,
)
from cudf_polars.dsl.ir import Union
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.base import ColumnStat, PartitionInfo

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.base import ColumnStats
    from cudf_polars.experimental.dispatch import LowerIRTransformer
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


def _concat(*dfs: DataFrame, context: IRExecutionContext) -> DataFrame:
    # Concatenate a sequence of DataFrames vertically
    return dfs[0] if len(dfs) == 1 else Union.do_evaluate(None, *dfs, context=context)


def _debug_expr_str(e: Expr, depth: int = 0, max_depth: int = 6) -> str:
    """Get a concise, human-readable representation of an expression tree."""
    if depth > max_depth:
        return "..."

    def _children() -> str:
        return ", ".join(_debug_expr_str(c, depth + 1, max_depth) for c in e.children)

    if isinstance(e, Col):
        return f"Col({e.name!r}, dtype={e.dtype})"
    elif isinstance(e, Literal):
        return f"Literal({e.value!r}, dtype={e.dtype})"
    elif isinstance(e, LiteralColumn):
        return f"LiteralColumn(dtype={e.dtype})"
    elif isinstance(e, Len):
        return f"Len(dtype={e.dtype})"
    elif isinstance(e, Agg):
        return f"Agg({e.name!r}, {_children()}, dtype={e.dtype})"
    elif isinstance(e, UnaryFunction):
        return f"UnaryFunction({e.name!r}, {_children()}, dtype={e.dtype})"
    elif isinstance(e, Cast):
        return f"Cast(strict={e.strict}, {_children()}, dtype={e.dtype})"
    elif isinstance(e, BinOp):
        return f"BinOp({e.op.name}, {_children()}, dtype={e.dtype})"
    elif isinstance(e, BooleanFunction):
        return f"BooleanFunction({e.name.name}, {_children()}, dtype={e.dtype})"
    elif isinstance(e, TemporalFunction):
        return f"TemporalFunction({e.name.name}, {_children()}, dtype={e.dtype})"
    elif isinstance(e, StringFunction):
        return f"StringFunction({e.name.name}, {_children()}, dtype={e.dtype})"
    elif isinstance(e, StructFunction):
        return f"StructFunction({e.name.name}, {_children()}, dtype={e.dtype})"
    elif isinstance(e, Ternary):
        return f"Ternary({_children()}, dtype={e.dtype})"
    elif isinstance(e, Filter):
        return f"Filter({_children()}, dtype={e.dtype})"
    elif isinstance(e, Gather):
        return f"Gather({_children()}, dtype={e.dtype})"
    elif isinstance(e, Slice):
        return f"Slice(offset={e.offset}, length={e.length}, {_children()}, dtype={e.dtype})"
    elif isinstance(e, Sort):
        return f"Sort(options={e.options}, {_children()}, dtype={e.dtype})"
    elif isinstance(e, SortBy):
        return f"SortBy(options={e.options}, {_children()}, dtype={e.dtype})"
    elif isinstance(e, RollingWindow):
        return f"RollingWindow(orderby={e.orderby!r}, {_children()}, dtype={e.dtype})"
    elif isinstance(e, GroupedRollingWindow):
        return f"GroupedRollingWindow({_children()}, dtype={e.dtype})"
    elif e.children:
        return f"{type(e).__name__}({_children()}, dtype={e.dtype})"
    else:
        return f"{type(e).__name__}(dtype={e.dtype})"


def _debug_ir_str(ir: IR) -> str:
    """Get a concise debug summary of an IR node for fallback messages."""
    type_name = type(ir).__name__
    schema_items = list(ir.schema.items())
    schema_str = (
        "{"
        + ", ".join(f"{k}: {v}" for k, v in schema_items[:5])
        + (f", ... ({len(schema_items)} cols total)" if len(schema_items) > 5 else "")
        + "}"
    )
    children_str = " -> ".join(type(c).__name__ for c in ir.children)
    lines = [f"  IR: {type_name}", f"  schema: {schema_str}"]
    if children_str:
        lines.append(f"  children: {children_str}")
    # Join-specific: show join type and key columns
    if hasattr(ir, "left_on") and hasattr(ir, "right_on") and hasattr(ir, "options"):
        lines.append(f"  join_type: {ir.options[0]}")
        left_keys = [ne.name for ne in ir.left_on]
        right_keys = [ne.name for ne in ir.right_on]
        if left_keys or right_keys:
            lines.append(f"  left_on:  {left_keys}")
            lines.append(f"  right_on: {right_keys}")
    # ConditionalJoin-specific: show predicate
    elif hasattr(ir, "predicate"):
        lines.append(f"  predicate: {_debug_expr_str(ir.predicate)}")
    return "\n".join(lines)


def _fallback_inform(
    msg: str, config_options: ConfigOptions[StreamingExecutor]
) -> None:
    """Inform the user of single-partition fallback."""
    match fallback_mode := config_options.executor.fallback_mode:
        case "warn" | "debug":
            warnings.warn(msg, stacklevel=2)
        case "raise":
            raise NotImplementedError(msg)
        case "silent":
            pass
        case _:  # pragma: no cover; Should never get here.
            raise ValueError(
                f"{fallback_mode} is not a supported 'fallback_mode' "
                "option. Please use 'warn', 'raise', 'silent', or 'debug'."
            )


def _dynamic_planning_on(config_options: ConfigOptions[StreamingExecutor]) -> bool:
    """Check if dynamic planning is enabled for rapidsmpf runtime."""
    return (
        config_options.executor.runtime == "rapidsmpf"
        and config_options.executor.dynamic_planning is not None
    )


def _lower_ir_fallback(
    ir: IR,
    rec: LowerIRTransformer,
    *,
    msg: str | None = None,
    debug_info: str | None = None,
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Catch-all single-partition lowering logic.
    # If any children contain multiple partitions,
    # those children will be collapsed with `Repartition`.
    from cudf_polars.experimental.repartition import Repartition

    config_options = rec.state["config_options"]
    rapidsmpf_engine = config_options.executor.runtime == "rapidsmpf"

    # Lower children
    lowered_children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
    partition_info = reduce(operator.or_, _partition_info)

    # Ensure all children are single-partitioned
    children = []
    inform = False
    for c in lowered_children:
        child = c
        if multi_partitioned := partition_info[c].count > 1:
            inform = True
        if multi_partitioned or rapidsmpf_engine:
            # Fall-back logic
            child = Repartition(child.schema, child)
            partition_info[child] = PartitionInfo(count=1)
        children.append(child)

    if inform and msg:
        # Warn/raise the user if any children were collapsed
        # and the "fallback_mode" configuration is not "silent"
        if config_options.executor.fallback_mode == "debug":
            parts = [msg]
            if debug_info:
                parts.append(debug_info)
            parts.append(_debug_ir_str(ir))
            child_counts = " -> ".join(
                f"{type(c).__name__}(count={partition_info[c].count})"
                for c in lowered_children
            )
            parts.append(f"  child_partitions: {child_counts}")
            msg = "\n".join(parts)
        _fallback_inform(msg, config_options)

    # Reconstruct and return
    new_node = ir.reconstruct(children)
    partition_info[new_node] = PartitionInfo(count=1)
    return new_node, partition_info


def _leaf_column_names(expr: Expr) -> tuple[str, ...]:
    """Find the leaf column names of an expression."""
    if expr.children:
        return tuple(
            chain.from_iterable(_leaf_column_names(child) for child in expr.children)
        )
    elif isinstance(expr, Col):
        return (expr.name,)
    else:
        return ()


def _get_unique_fractions(
    column_names: Sequence[str],
    user_unique_fractions: dict[str, float],
    *,
    row_count: ColumnStat[int] | None = None,
    column_stats: dict[str, ColumnStats] | None = None,
) -> dict[str, float]:
    """
    Return unique-fraction statistics subset.

    Parameters
    ----------
    column_names
        The column names to get unique-fractions for.
    user_unique_fractions
        The user-provided unique-fraction dictionary.
    row_count
        Row-count statistics. This will be None if
        statistics planning is not enabled.
    column_stats
        The column statistics. This will be None if
        statistics planning is not enabled.

    Returns
    -------
    unique_fractions
        The final unique-fraction dictionary.
    """
    unique_fractions: dict[str, float] = {}
    column_stats = column_stats or {}
    row_count = row_count or ColumnStat[int](None)
    if isinstance(row_count.value, int) and row_count.value > 0:
        for c in set(column_names).intersection(column_stats):
            if (unique_count := column_stats[c].unique_count.value) is not None:
                # Use unique_count_estimate (if available)
                unique_fractions[c] = max(
                    min(1.0, unique_count / row_count.value),
                    0.00001,
                )

    # Update with user-provided unique-fractions
    unique_fractions.update(
        {
            c: max(min(f, 1.0), 0.00001)
            for c, f in user_unique_fractions.items()
            if c in column_names
        }
    )
    return unique_fractions



def _contains_over(exprs: Sequence[Expr]) -> bool:
    """Return True if any expression in 'exprs' contains an over(...) (ie. GroupedRollingWindow)."""
    return any(isinstance(e, GroupedRollingWindow) for e in traversal(exprs))


def _contains_unsupported_fill_strategy(exprs: Sequence[Expr]) -> bool:
    for e in traversal(exprs):
        if (
            isinstance(e, UnaryFunction)
            and e.name == "fill_null_with_strategy"
            and e.options[0] not in ("zero", "one")
        ):
            return True
    return False
