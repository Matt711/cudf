# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""SQL compatibility helpers for naive TPC-DS Polars implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    pass

# Controlled by the runner wrapper in PDSDSUnoptimizedPolarsQueries.
# When True, sql_sum returns null for all-null groups (matching SQL SUM semantics).
# This flag is set at polars_impl() call time, before the LazyFrame is constructed,
# so the expression tree reflects the correct behavior for validation runs.
_VALIDATION_MODE: bool = False


def enable_validation_mode() -> None:
    """Enable SQL-compatible null handling (call before polars_impl during validation)."""
    global _VALIDATION_MODE
    _VALIDATION_MODE = True


def disable_validation_mode() -> None:
    """Disable SQL-compatible null handling."""
    global _VALIDATION_MODE
    _VALIDATION_MODE = False


def sql_sum(expr: pl.Expr) -> pl.Expr:
    """SQL-compatible SUM: returns null when all inputs are null.

    In standard SQL, SUM over an all-null group returns NULL.
    Polars returns 0. This wrapper corrects that discrepancy, but only
    when _VALIDATION_MODE is True (i.e. when the validation pipeline
    is evaluating the result), to avoid overhead during pure benchmarking.
    """
    if _VALIDATION_MODE:
        return pl.when(expr.count() > 0).then(expr.sum()).otherwise(None)
    return expr.sum()


def sql_n_unique(expr: pl.Expr) -> pl.Expr:
    """SQL-compatible COUNT(DISTINCT x): excludes NULLs from the distinct count.

    Polars n_unique() counts NULL as a distinct value; SQL COUNT(DISTINCT x) does not.
    Uses .filter() to exclude nulls before counting to avoid arithmetic on two
    aggregation expressions, which can fail in Polars streaming mode.
    """
    return expr.filter(expr.is_not_null()).n_unique()


def sql_rollup(
    lf: pl.LazyFrame,
    group_cols: list[str],
    agg_exprs: list[pl.Expr],
) -> pl.LazyFrame:
    """Reproduce SQL ROLLUP behaviour.

    For ``GROUP BY ROLLUP(col1, col2, ..., colN)`` generates N+1 aggregation
    levels:

    * Level N  : GROUP BY col1, col2, ..., colN  (finest granularity)
    * Level N-1: GROUP BY col1, ..., col(N-1);  colN = NULL
    * ...
    * Level 0  : no grouping; all cols = NULL   (grand total)

    Output columns: ``group_cols + [agg output names] + ["_rollup_lochierarchy"]``.

    ``_rollup_lochierarchy`` is the sum of SQL GROUPING() values for all group_cols
    at each level, i.e. the count of rollup-induced NULLs.  Callers must use this
    column (not ``col.is_null()``) to compute GROUPING()-based expressions, because
    the data may already contain real NULLs in the group columns.
    """
    schema = lf.collect_schema()
    agg_names = [e.meta.output_name() for e in agg_exprs]
    all_cols = group_cols + agg_names + ["_rollup_lochierarchy"]
    levels: list[pl.LazyFrame] = []

    for i in range(len(group_cols), -1, -1):
        active = group_cols[:i]
        nulled = group_cols[i:]

        if active:
            grouped = lf.group_by(active).agg(agg_exprs)
        else:
            # Grand-total row: aggregate over entire frame (no grouping key)
            grouped = lf.select(agg_exprs)

        null_exprs = [pl.lit(None, dtype=schema[col]).alias(col) for col in nulled]
        if null_exprs:
            grouped = grouped.with_columns(null_exprs)

        grouped = grouped.with_columns(
            pl.lit(len(nulled), dtype=pl.Int64).alias("_rollup_lochierarchy")
        )
        levels.append(grouped.select(all_cols))

    return pl.concat(levels, how="vertical_relaxed")
