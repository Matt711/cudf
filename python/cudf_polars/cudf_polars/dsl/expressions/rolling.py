# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""Rolling DSL nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame, DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.utils.reshape import broadcast
from cudf_polars.dsl.utils.windows import offsets_to_windows, range_window_bounds

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cudf_polars.typing import ClosedInterval, Duration

__all__ = ["GroupedRollingWindow", "RollingWindow", "to_request"]


def to_request(
    value: expr.Expr, orderby: Column, df: DataFrame
) -> plc.rolling.RollingRequest:
    """
    Produce a rolling request for evaluation with pylibcudf.

    Parameters
    ----------
    value
        The expression to perform the rolling aggregation on.
    orderby
        Orderby column, used as input to the request when the aggregation is Len.
    df
        DataFrame used to evaluate the inputs to the aggregation.
    """
    min_periods = 1
    if isinstance(value, expr.Len):
        # A count aggregation, we need a column so use the orderby column
        col = orderby
    elif isinstance(value, expr.Agg):
        child = value.children[0]
        col = child.evaluate(df, context=ExecutionContext.ROLLING)
        if value.name == "var":
            # Polars variance produces null if nvalues <= ddof
            # libcudf produces NaN. However, we can get the polars
            # behaviour by setting the minimum window size to ddof +
            # 1.
            min_periods = value.options + 1
    else:
        col = value.evaluate(
            df, context=ExecutionContext.ROLLING
        )  # pragma: no cover; raise before we get here because we
        # don't do correct handling of empty groups
    return plc.rolling.RollingRequest(col.obj, min_periods, value.agg_request)


class RollingWindow(Expr):
    __slots__ = (
        "closed_window",
        "following",
        "offset",
        "orderby",
        "orderby_dtype",
        "period",
        "preceding",
    )
    _non_child = (
        "dtype",
        "orderby_dtype",
        "offset",
        "period",
        "closed_window",
        "orderby",
    )

    def __init__(
        self,
        dtype: DataType,
        orderby_dtype: DataType,
        offset: Duration,
        period: Duration,
        closed_window: ClosedInterval,
        orderby: str,
        agg: Expr,
    ) -> None:
        self.dtype = dtype
        self.orderby_dtype = orderby_dtype
        # NOTE: Save original `offset` and `period` args,
        # because the `preceding` and `following` attributes
        # cannot be serialized (and must be reconstructed
        # within `__init__`).
        self.offset = offset
        self.period = period
        self.preceding, self.following = offsets_to_windows(
            orderby_dtype, offset, period
        )
        self.closed_window = closed_window
        self.orderby = orderby
        self.children = (agg,)
        self.is_pointwise = False
        if agg.agg_request.kind() == plc.aggregation.Kind.COLLECT_LIST:
            raise NotImplementedError(
                "Incorrect handling of empty groups for list collection"
            )
        if not plc.rolling.is_valid_rolling_aggregation(agg.dtype.plc, agg.agg_request):
            raise NotImplementedError(f"Unsupported rolling aggregation {agg}")

    def do_evaluate(  # noqa: D102
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        if context != ExecutionContext.FRAME:
            raise RuntimeError(
                "Rolling aggregation inside groupby/over/rolling"
            )  # pragma: no cover; translation raises first
        (agg,) = self.children
        orderby = df.column_map[self.orderby]
        # Polars casts integral orderby to int64, but only for calculating window bounds
        if (
            plc.traits.is_integral(orderby.obj.type())
            and orderby.obj.type().id() != plc.TypeId.INT64
        ):
            orderby_obj = plc.unary.cast(orderby.obj, plc.DataType(plc.TypeId.INT64))
        else:
            orderby_obj = orderby.obj
        preceding, following = range_window_bounds(
            self.preceding, self.following, self.closed_window
        )
        if orderby.obj.null_count() != 0:
            raise RuntimeError(
                f"Index column '{self.orderby}' in rolling may not contain nulls"
            )
        if not orderby.check_sorted(
            order=plc.types.Order.ASCENDING, null_order=plc.types.NullOrder.BEFORE
        ):
            raise RuntimeError(
                f"Index column '{self.orderby}' in rolling is not sorted, please sort first"
            )
        (result,) = plc.rolling.grouped_range_rolling_window(
            plc.Table([]),
            orderby_obj,
            plc.types.Order.ASCENDING,
            plc.types.NullOrder.BEFORE,
            preceding,
            following,
            [to_request(agg, orderby, df)],
        ).columns()
        return Column(result, dtype=self.dtype)


class GroupedRollingWindow(Expr):
    """
    Compute a window ``.over(...)`` aggregation and broadcast to rows.

    Notes
    -----
    - This expression node currently implements **grouped window mapping**
      (aggregate once per group, then broadcast back), not rolling windows.
    - It can be extended later to support `rolling(...).over(...)`
      when polars supports that expression.
    """

    __slots__ = ("by_count", "named_aggs", "options", "post")
    _non_child = ("dtype", "options", "named_aggs", "post", "by_count")

    def __init__(
        self,
        dtype: DataType,
        options: Any,
        named_aggs: Sequence[expr.NamedExpr],
        post: expr.NamedExpr,
        *by: Expr,
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.named_aggs = tuple(named_aggs)
        self.post = post
        self.is_pointwise = False

        unsupported = [
            type(named_expr.value).__name__
            for named_expr in self.named_aggs
            if not (
                isinstance(named_expr.value, (expr.Len, expr.Agg))
                or (
                    isinstance(named_expr.value, expr.UnaryFunction)
                    and named_expr.value.name == "rank"
                )
            )
        ]
        if unsupported:
            kinds = ", ".join(sorted(set(unsupported)))
            raise NotImplementedError(
                f"Unsupported over(...) only expression: {kinds}="
            )

        # Ensures every partition-by is an Expr
        # Fixes over(1) cases with the streaming
        # executor and a small blocksize
        by_expr = [
            (b if isinstance(b, Expr) else expr.Literal(DataType(pl.Int64()), b))
            for b in by
        ]

        # Expose agg dependencies as children so the streaming
        # executor retains required source columns
        child_deps = [
            v.children[0]
            for ne in self.named_aggs
            for v in (ne.value,)
            if isinstance(v, expr.Agg)
            or (isinstance(v, expr.UnaryFunction) and v.name == "rank")
        ]
        self.by_count = len(by_expr)
        self.children = tuple(by_expr) + tuple(child_deps)

    @staticmethod
    def _rank_over_groups(
        values: Column,
        group_ids: plc.Column,
        *,
        method: str,
        descending: bool,
        out_dtype: DataType,
        num_groups: int,
    ) -> Column:
        """
        Compute per-row group-wise ranks (ordinal/dense/min/max/average).

        Implementation notes:
        - Exclude nulls from the ranking domain (Polars semantics) by compacting
        out null rows first, computing ranks on the compacted rows, then
        scattering results back into an all-null output of length n.
        - Within the compacted rows, we compute a stable sorted order over
        (group_ids, values) with nulls AFTER (there are no nulls in values_nn).
        - Ordinal positions are then adjusted to per-group 1-based ranks; for ties
        we build run starts/ends and derive dense/min/max/average as needed.
        """
        order = plc.types.Order.DESCENDING if descending else plc.types.Order.ASCENDING

        n = values.size
        size_t = plc.types.SIZE_TYPE
        zero = plc.Scalar.from_py(0, size_t)
        one = plc.Scalar.from_py(1, size_t)

        # Compact out rows where `values` is null; carry original row indices
        row_idx = plc.filling.sequence(n, zero, one)  # 0..n-1
        full_tbl = plc.Table([values.obj, group_ids, row_idx])
        # keep rows where column-0 (values) is non-null
        nn_tbl = plc.stream_compaction.drop_nulls(full_tbl, [0], 1)
        values_nn, group_ids_nn, row_idx_nn = nn_tbl.columns()
        n_nn = nn_tbl.num_rows()

        # If everything is null, return an all-null column of requested dtype
        if n_nn == 0:
            null_scalar = plc.Scalar.from_py(None, out_dtype.plc)
            return Column(plc.Column.from_scalar(null_scalar, n), dtype=out_dtype)

        # Sorted order of compacted rows by (group_id ASC, value ASC/DESC, nulls AFTER)
        order_idx = plc.sorting.stable_sorted_order(
            plc.Table([group_ids_nn, values_nn]),
            [plc.types.Order.ASCENDING, order],
            [plc.types.NullOrder.AFTER, plc.types.NullOrder.AFTER],
        )

        # Invert permutation over compacted rows:
        pos_global = plc.filling.sequence(n_nn, zero, one)  # 0..n_nn-1
        inv_pos_tbl = plc.copying.scatter(
            plc.Table([pos_global]),
            order_idx,
            plc.Table([plc.Column.from_scalar(zero, n_nn)]),
        )
        inv_pos = inv_pos_tbl.columns()[0]

        # For each group, find its first position in the sorted order.
        rg_sorted = plc.copying.gather(
            plc.Table([group_ids_nn]),
            order_idx,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]
        first_idx = plc.stream_compaction.distinct_indices(
            plc.Table([rg_sorted]),
            plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
            plc.types.NullEquality.EQUAL,
            plc.types.NanEquality.ALL_EQUAL,
        )
        first_rg = plc.copying.gather(
            plc.Table([rg_sorted]), first_idx, plc.copying.OutOfBoundsPolicy.DONT_CHECK
        ).columns()[0]
        first_pos_lookup = plc.copying.scatter(
            plc.Table([first_idx]),
            first_rg,
            plc.Table([plc.Column.from_scalar(zero, num_groups)]),
        ).columns()[0]

        # Map each compacted row's group_id -> that group's first position
        first_pos_per_row = plc.copying.gather(
            plc.Table([first_pos_lookup]),
            group_ids_nn,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]

        # Ordinal rank (compacted): (inv_pos - first_pos_of_group)  1
        pos_in_group = plc.binaryop.binary_operation(
            inv_pos, first_pos_per_row, plc.binaryop.BinaryOperator.SUB, inv_pos.type()
        )
        rank_1based = plc.binaryop.binary_operation(
            pos_in_group, one, plc.binaryop.BinaryOperator.ADD, pos_in_group.type()
        )

        def _scatter_back(ranks_compacted: plc.Column) -> Column:
            # Cast to requested dtype
            out_plc = ranks_compacted
            if out_plc.type().id() != out_dtype.plc.id():
                out_plc = plc.unary.cast(out_plc, out_dtype.plc)
            # Build all-null output of length n, then scatter compacted results by original row indices
            null_scalar = plc.Scalar.from_py(None, out_dtype.plc)
            out_full = plc.Column.from_scalar(null_scalar, n)
            out_full = plc.copying.scatter(
                plc.Table([out_plc]), row_idx_nn, plc.Table([out_full])
            ).columns()[0]
            return Column(out_full, dtype=out_dtype)

        if method == "ordinal":
            return _scatter_back(rank_1based)

        # Shared prep for dense / min / max / average (on compacted rows)

        # Same positions but in the (group,value)-sorted order
        # pos_in_group_sorted = (0..n_nn-1) - first_pos_of_group_for_that_row
        first_pos_for_sorted = plc.copying.gather(
            plc.Table([first_pos_lookup]),
            rg_sorted,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]
        pos_in_group_sorted = plc.binaryop.binary_operation(
            pos_global,
            first_pos_for_sorted,
            plc.binaryop.BinaryOperator.SUB,
            pos_global.type(),
        )

        # Compute "runs" of equal values within each group:
        val_sorted = plc.copying.gather(
            plc.Table([values_nn]),
            order_idx,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]
        run_starts_idx = plc.stream_compaction.distinct_indices(
            plc.Table([rg_sorted, val_sorted]),
            plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
            plc.types.NullEquality.EQUAL,
            plc.types.NanEquality.ALL_EQUAL,
        )

        # Boolean markers at run starts (in sorted order), then inclusive scan to get run_id.
        run_marker_dtype = plc.types.SIZE_TYPE
        one_size = plc.Scalar.from_py(1, run_marker_dtype)
        zero_size = plc.Scalar.from_py(0, run_marker_dtype)
        run_start_markers = plc.copying.scatter(
            plc.Table([plc.Column.from_scalar(one_size, run_starts_idx.size())]),
            run_starts_idx,
            plc.Table([plc.Column.from_scalar(zero_size, n_nn)]),
        ).columns()[0]
        run_id_sorted = plc.reduce.scan(
            run_start_markers, plc.aggregation.sum(), plc.reduce.ScanType.INCLUSIVE
        )
        # Map run_id back to compacted row order
        run_id = plc.copying.gather(
            plc.Table([run_id_sorted]),
            inv_pos,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]

        if method == "dense":
            # dense rank = (run_id - first_run_id_of_group)  1
            run_id_at_group_start = plc.copying.gather(
                plc.Table([run_id_sorted]),
                first_idx,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            ).columns()[0]
            run_base_lookup = plc.copying.scatter(
                plc.Table([run_id_at_group_start]),
                first_rg,
                plc.Table(
                    [
                        plc.Column.from_scalar(
                            plc.Scalar.from_py(0, run_id_sorted.type()), num_groups
                        )
                    ]
                ),
            ).columns()[0]
            run_base_per_row = plc.copying.gather(
                plc.Table([run_base_lookup]),
                group_ids_nn,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            ).columns()[0]

            dense_rank = plc.binaryop.binary_operation(
                run_id,
                run_base_per_row,
                plc.binaryop.BinaryOperator.SUB,
                run_id.type(),
            )
            dense_rank = plc.binaryop.binary_operation(
                dense_rank, one, plc.binaryop.BinaryOperator.ADD, dense_rank.type()
            )
            return _scatter_back(dense_rank)

        # Build per-run MIN/MAX ranks (based on ordinal positions within group)
        num_runs = run_starts_idx.size()

        # Build run_end indices:
        end_default = plc.Scalar.from_py(n_nn - 1, size_t)
        end_positions = plc.Column.from_scalar(end_default, num_runs)
        if num_runs > 1:
            upto_last = plc.filling.sequence(num_runs - 1, zero, one)
            one_to_last = plc.filling.sequence(
                num_runs - 1, plc.Scalar.from_py(1, size_t), one
            )
            starts_tail = plc.copying.gather(
                plc.Table([run_starts_idx]),
                one_to_last,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            ).columns()[0]
            starts_tail_minus1 = plc.binaryop.binary_operation(
                starts_tail, one, plc.binaryop.BinaryOperator.SUB, starts_tail.type()
            )
            end_positions = plc.copying.scatter(
                plc.Table([starts_tail_minus1]),
                upto_last,
                plc.Table([end_positions]),
            ).columns()[0]

        # pos_in_group at run starts/ends (in sorted order)
        min_pos_per_run = plc.copying.gather(
            plc.Table([pos_in_group_sorted]),
            run_starts_idx,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]
        max_pos_per_run = plc.copying.gather(
            plc.Table([pos_in_group_sorted]),
            end_positions,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]

        # Convert to 1-based ranks
        min_rank_per_run = plc.binaryop.binary_operation(
            min_pos_per_run,
            one,
            plc.binaryop.BinaryOperator.ADD,
            min_pos_per_run.type(),
        )
        max_rank_per_run = plc.binaryop.binary_operation(
            max_pos_per_run,
            one,
            plc.binaryop.BinaryOperator.ADD,
            max_pos_per_run.type(),
        )

        # Map run_id -> min/max ranks via lookups (index runs by (run_id - 1))
        run_ids_at_starts = plc.copying.gather(
            plc.Table([run_id_sorted]),
            run_starts_idx,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]
        run_ids_zero_based = plc.binaryop.binary_operation(
            run_id_sorted, one, plc.binaryop.BinaryOperator.SUB, run_id_sorted.type()
        )
        run_ids_zero_based_at_starts = plc.binaryop.binary_operation(
            run_ids_at_starts,
            one,
            plc.binaryop.BinaryOperator.SUB,
            run_ids_at_starts.type(),
        )

        min_lookup = plc.Column.from_scalar(
            plc.Scalar.from_py(0, min_rank_per_run.type()), num_runs
        )
        max_lookup = plc.Column.from_scalar(
            plc.Scalar.from_py(0, max_rank_per_run.type()), num_runs
        )
        min_lookup = plc.copying.scatter(
            plc.Table([min_rank_per_run]),
            run_ids_zero_based_at_starts,
            plc.Table([min_lookup]),
        ).columns()[0]
        max_lookup = plc.copying.scatter(
            plc.Table([max_rank_per_run]),
            run_ids_zero_based_at_starts,
            plc.Table([max_lookup]),
        ).columns()[0]

        # Per-row (in sorted order) min/max ranks via run_id
        min_rank_sorted_rows = plc.copying.gather(
            plc.Table([min_lookup]),
            run_ids_zero_based,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]
        max_rank_sorted_rows = plc.copying.gather(
            plc.Table([max_lookup]),
            run_ids_zero_based,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]

        # Map back to compacted row order
        min_rank_rows = plc.copying.gather(
            plc.Table([min_rank_sorted_rows]),
            inv_pos,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]
        max_rank_rows = plc.copying.gather(
            plc.Table([max_rank_sorted_rows]),
            inv_pos,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]

        if method == "min":
            return _scatter_back(min_rank_rows)

        if method == "max":
            return _scatter_back(max_rank_rows)

        if method == "average":
            # compute in f64 to match Polars, then cast if needed
            f64 = plc.DataType(plc.TypeId.FLOAT64)
            a = plc.unary.cast(min_rank_rows, f64)
            b = plc.unary.cast(max_rank_rows, f64)
            sum_min_max = plc.binaryop.binary_operation(
                a, b, plc.binaryop.BinaryOperator.ADD, f64
            )
            two = plc.Scalar.from_py(2.0, f64)
            avg_rank = plc.binaryop.binary_operation(
                sum_min_max, two, plc.binaryop.BinaryOperator.DIV, f64
            )
            return _scatter_back(
                avg_rank
                if out_dtype.plc.id() == f64.id()
                else plc.unary.cast(avg_rank, out_dtype.plc)
            )

        raise NotImplementedError(f"rank(over=...): unsupported method {method!r}")

    def do_evaluate(  # noqa: D102
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        if context != ExecutionContext.FRAME:
            raise RuntimeError(
                "Window mapping (.over) can only be evaluated at the frame level"
            )  # pragma: no cover; translation raises first

        by_exprs = self.children[: self.by_count]
        by_cols = list(
            broadcast(
                *(b.evaluate(df, context=ExecutionContext.FRAME) for b in by_exprs),
                target_length=df.num_rows,
            )
        )
        by_tbl = plc.Table([c.obj for c in by_cols])

        sorted_flag = (
            plc.types.Sorted.YES
            if all(k.is_sorted for k in by_cols)
            else plc.types.Sorted.NO
        )
        grouper = plc.groupby.GroupBy(
            by_tbl,
            null_handling=plc.types.NullPolicy.INCLUDE,
            keys_are_sorted=sorted_flag,
            column_order=[k.order for k in by_cols],
            null_precedence=[k.null_order for k in by_cols],
        )

        # Split expressions: scalar aggs (Len/Agg) vs per-row ops (rank)
        scalar_named: list[expr.NamedExpr] = []
        rank_named: list[expr.NamedExpr] = []
        for ne in self.named_aggs:
            if isinstance(ne.value, expr.UnaryFunction) and ne.value.name == "rank":
                rank_named.append(ne)
            else:
                scalar_named.append(ne)

        gb_requests: list[plc.groupby.GroupByRequest] = []
        out_names: list[str] = []
        out_dtypes: list[DataType] = []
        # We might need group keys (and counts) even if there are only ranks.
        need_counts_for_rank = bool(rank_named)

        # Build scalar agg requests first (preserving your logic)
        for ne in scalar_named:
            val = ne.value
            out_names.append(ne.name)
            out_dtypes.append(val.dtype)

            if isinstance(val, expr.Len):
                # Count rows per group via sum(1).
                ones = plc.Column.from_scalar(
                    plc.Scalar.from_py(1, plc.DataType(plc.TypeId.INT8)), df.num_rows
                )
                gb_requests.append(
                    plc.groupby.GroupByRequest(ones, [plc.aggregation.sum()])
                )
            elif isinstance(val, expr.Agg):
                (child,) = (
                    val.children if val.name != "quantile" else (val.children[0],)
                )
                col = child.evaluate(df, context=ExecutionContext.FRAME).obj
                gb_requests.append(plc.groupby.GroupByRequest(col, [val.agg_request]))

        # If ranks are present, push a leading "counts" request to obtain group_keys_tbl and group sizes.
        if need_counts_for_rank:
            ones = plc.Column.from_scalar(
                plc.Scalar.from_py(1, plc.DataType(plc.TypeId.INT8)), df.num_rows
            )
            gb_requests.insert(
                0, plc.groupby.GroupByRequest(ones, [plc.aggregation.sum()])
            )

        group_keys_tbl, value_tables = grouper.aggregate(gb_requests)
        out_cols = iter(t.columns()[0] for t in value_tables)

        # Build gather maps to broadcast per-group scalar results to all rows.
        # Also left-join input keys to group-keys so every input row appears exactly once.
        lg, rg = plc.join.left_join(
            by_tbl, group_keys_tbl, plc.types.NullEquality.EQUAL
        )

        # Reorder the gather maps to preserve left/input order
        left_rows, right_rows = by_tbl.num_rows(), group_keys_tbl.num_rows()
        init = plc.Scalar.from_py(0, plc.types.SIZE_TYPE)
        step = plc.Scalar.from_py(1, plc.types.SIZE_TYPE)
        left_order = plc.copying.gather(
            plc.Table([plc.filling.sequence(left_rows, init, step)]),
            lg,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        )
        right_order = plc.copying.gather(
            plc.Table([plc.filling.sequence(right_rows, init, step)]),
            rg,
            plc.copying.OutOfBoundsPolicy.NULLIFY,
        )
        # Sort both maps by (left_order, right_order), then use the reordered right map
        # to gather group aggregates in the original row order.
        _, rg = plc.sorting.stable_sort_by_key(
            plc.Table([lg, rg]),
            plc.Table([*left_order.columns(), *right_order.columns()]),
            [plc.types.Order.ASCENDING, plc.types.Order.ASCENDING],
            [plc.types.NullOrder.AFTER, plc.types.NullOrder.AFTER],
        ).columns()

        # Broadcast each aggregated result back to row-shape using the right map.
        broadcasted_cols = []
        # If we inserted the extra leading count (for ranks), pull it off the iterator first and ignore it.
        if need_counts_for_rank:
            _ = next(
                out_cols, None
            )  # discard counts column (we only needed keys/sizes)

        for named_expr, dtype in zip(scalar_named, out_dtypes, strict=True):
            col = next(out_cols)
            broadcasted_cols.append(
                Column(
                    plc.copying.gather(
                        plc.Table([col]),
                        rg,
                        plc.copying.OutOfBoundsPolicy.NULLIFY,
                    ).columns()[0],
                    name=named_expr.name,
                    dtype=dtype,
                )
            )

        # Per-group rank evaluation (per-row outputs)
        if rank_named:
            # Use the reordered right-map (rg) as per-row group ids.
            group_ids = rg
            num_groups = group_keys_tbl.num_rows()
            for ne in rank_named:
                rank_expr = ne.value
                assert isinstance(rank_expr, expr.UnaryFunction)
                (child_expr,) = rank_expr.children
                values = child_expr.evaluate(df, context=ExecutionContext.FRAME)

                method_str, descending, _ = rank_expr.options
                ranked = GroupedRollingWindow._rank_over_groups(
                    values,
                    group_ids,
                    method=method_str,
                    descending=bool(descending),
                    out_dtype=rank_expr.dtype,
                    num_groups=num_groups,
                )
                # Name the column with the placeholder from agg decomposition
                ranked.name = ne.name
                broadcasted_cols.append(ranked)

        # Create a temporary DataFrame with the broadcasted columns named by their
        # placeholder names from agg decomposition, then evaluate the post-expression.
        df = DataFrame(broadcasted_cols)
        return self.post.value.evaluate(df, context=ExecutionContext.FRAME)
