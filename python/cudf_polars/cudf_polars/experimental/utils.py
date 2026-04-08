# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition utilities."""

from __future__ import annotations

import operator
import warnings
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING

from cudf_polars.dsl.expr import Col, Expr, GroupedRollingWindow, UnaryFunction
from cudf_polars.dsl.ir import Union
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.base import PartitionInfo

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.expr import Expr
    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.dispatch import LowerIRTransformer
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


def _concat(*dfs: DataFrame, context: IRExecutionContext) -> DataFrame:
    # Concatenate a sequence of DataFrames vertically
    return dfs[0] if len(dfs) == 1 else Union.do_evaluate(None, *dfs, context=context)


def _fallback_inform(
    msg: str, config_options: ConfigOptions[StreamingExecutor]
) -> None:
    """Inform the user of single-partition fallback."""
    match fallback_mode := config_options.executor.fallback_mode:
        case "warn":
            warnings.warn(msg, stacklevel=2)
        case "raise":
            raise NotImplementedError(msg)
        case "silent":
            pass
        case _:  # pragma: no cover; Should never get here.
            raise ValueError(
                f"{fallback_mode} is not a supported 'fallback_mode' "
                "option. Please use 'warn', 'raise', or 'silent'."
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
        _fallback_inform(msg, rec.state["config_options"])

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
) -> dict[str, float]:
    """
    Return unique-fraction statistics subset.

    Parameters
    ----------
    column_names
        The column names to get unique-fractions for.
    user_unique_fractions
        The user-provided unique-fraction dictionary.

    Returns
    -------
    unique_fractions
        The final unique-fraction dictionary filtered to column_names.
    """
    return {
        c: max(min(f, 1.0), 0.00001)
        for c, f in user_unique_fractions.items()
        if c in column_names
    }


def _contains_over(exprs: Sequence[Expr]) -> bool:
    """Return True if any expression in 'exprs' contains an over(...) (ie. GroupedRollingWindow)."""
    return any(isinstance(e, GroupedRollingWindow) for e in traversal(exprs))


def _extract_over_shuffle_keys(
    col_exprs: Sequence[Expr],
) -> tuple[str, ...] | None:
    """
    Extract common partition keys from all ``over()`` expressions.

    Traverses *col_exprs* without descending into ``GroupedRollingWindow``
    children (those contain internal aggregation nodes that are handled by
    the window expression itself and are not "blocking" for the HStack).

    Returns a tuple of column-name strings if every non-pointwise node in
    the tree is a ``GroupedRollingWindow`` whose ``by`` keys are all simple
    ``Col`` references **and** every such node uses the *same* set of keys.
    Returns ``None`` if any non-pointwise node is not a
    ``GroupedRollingWindow``, or if the ``over()`` keys differ across nodes,
    or if any key is not a plain column reference.
    """
    common_keys: tuple[str, ...] | None = None
    found_grw = False

    seen: set[int] = set()
    stack: list[Expr] = list(reversed(list(col_exprs)))

    while stack:
        e = stack.pop()
        eid = id(e)
        if eid in seen:
            continue
        seen.add(eid)

        if isinstance(e, GroupedRollingWindow):
            found_grw = True
            by_exprs = e.children[: e.by_count]
            # All partition-key expressions must be plain column references
            if not all(isinstance(b, Col) for b in by_exprs):
                return None
            these_keys: tuple[str, ...] = tuple(b.name for b in by_exprs)  # type: ignore[union-attr]
            if common_keys is None:
                common_keys = these_keys
            elif common_keys != these_keys:
                return None
            # Do NOT recurse into GroupedRollingWindow's children —
            # they contain internal aggregation nodes that are non-pointwise
            # but handled by the window expression itself.
            continue

        if not e.is_pointwise:
            # Non-pointwise node that is not a GroupedRollingWindow — unsupported
            return None

        for child in reversed(e.children):
            if id(child) not in seen:
                stack.append(child)

    return common_keys if found_grw else None


def _contains_unsupported_fill_strategy(exprs: Sequence[Expr]) -> bool:
    for e in traversal(exprs):
        if (
            isinstance(e, UnaryFunction)
            and e.name == "fill_null_with_strategy"
            and e.options[0] not in ("zero", "one")
        ):
            return True
    return False
