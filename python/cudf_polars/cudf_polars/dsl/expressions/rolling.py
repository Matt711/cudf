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

    __slots__ = ("by_count", "named_aggs", "options", "order_count", "post")
    _non_child = ("dtype", "options", "named_aggs", "post", "by_count", "order_count")

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

        # options layout (from translate): (mapping, has_order_by, ob_desc, ob_nulls_last, order_count?)
        order_count = 0
        if isinstance(self.options, tuple) and len(self.options) >= 5:
            order_count = int(self.options[4])

        # partition exprs first, then trailing order_by exprs
        part_expr = by_expr[:-order_count] if order_count else by_expr
        order_expr = by_expr[-order_count:] if order_count else []

        # Expose agg dependencies as children so the streaming
        # executor retains required source columns
        child_deps = [
            v.children[0]
            for ne in self.named_aggs
            for v in (ne.value,)
            if isinstance(v, expr.Agg)
            or (isinstance(v, expr.UnaryFunction) and v.name == "rank")
        ]

        # If order_by is present and a non-ordinal rank is requested, we still support it,
        # so do not raise here (we compute ties on order_by keys). Keep validation for unsupported kinds only.
        self.by_count = len(part_expr)
        self.order_count = len(order_expr)
        self.children = tuple(part_expr) + tuple(order_expr) + tuple(child_deps)

    @staticmethod
    def _rebuild_col_with_nulls(
        ranks: plc.Column, i: plc.Column, n: int, out_dtype: DataType
    ) -> Column:
        out_plc = (
            ranks
            if ranks.type().id() == out_dtype.plc.id()
            else plc.unary.cast(ranks, out_dtype.plc)
        )
        ranks_with_nulls = plc.Column.from_scalar(
            plc.Scalar.from_py(None, out_dtype.plc), n
        )
        ranks_with_nulls = plc.copying.scatter(
            plc.Table([out_plc]), i, plc.Table([ranks_with_nulls])
        ).columns()[0]
        return Column(ranks_with_nulls, dtype=out_dtype)

    @staticmethod
    def _segmented_rank_over_orderby(
        *,
        group_indices: plc.Column,
        order_cols: list[plc.Column],
        group_keys_rows: int,
        ob_desc: bool,
        ob_nulls_last: bool,
        method: str,
        out_dtype: DataType,
        n_rows: int,
    ) -> Column:
        """Rank within each partition using order_by keys (supports ordinal/dense/min/max/average)."""
        size_type = plc.types.SIZE_TYPE
        zero = plc.Scalar.from_py(0, size_type)
        one = plc.Scalar.from_py(1, size_type)

        # Stable sort by (group, *order_by, row_id) to break ties deterministically
        row_id = plc.filling.sequence(n_rows, zero, one)
        key_order = plc.types.Order.ASCENDING
        key_null = plc.types.NullOrder.AFTER
        ob_order = plc.types.Order.DESCENDING if ob_desc else plc.types.Order.ASCENDING
        ob_null = (
            plc.types.NullOrder.AFTER if ob_nulls_last else plc.types.NullOrder.BEFORE
        )

        grouped_order = plc.sorting.stable_sorted_order(
            plc.Table([group_indices, *order_cols, row_id]),
            [key_order, *([ob_order] * len(order_cols)), plc.types.Order.ASCENDING],
            [key_null, *([ob_null] * len(order_cols)), plc.types.NullOrder.AFTER],
        )

        # position of each original row in grouped order, and group id in grouped order
        k_seq = plc.filling.sequence(n_rows, zero, one)
        pos_in_grouped = plc.copying.scatter(
            plc.Table([k_seq]),
            grouped_order,
            plc.Table([plc.Column.from_scalar(zero, n_rows)]),
        ).columns()[0]
        gid_grouped = plc.copying.gather(
            plc.Table([group_indices]),
            grouped_order,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]

        # first position per group in grouped order
        first_pos_idx = plc.stream_compaction.distinct_indices(
            plc.Table([gid_grouped]),
            plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
            plc.types.NullEquality.EQUAL,
            plc.types.NanEquality.ALL_EQUAL,
        )
        groups_at_first = plc.copying.gather(
            plc.Table([gid_grouped]),
            first_pos_idx,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]
        first_pos_at_group = plc.copying.scatter(
            plc.Table([first_pos_idx]),
            groups_at_first,
            plc.Table([plc.Column.from_scalar(zero, group_keys_rows)]),
        ).columns()[0]
        first_pos_per_row = plc.copying.gather(
            plc.Table([first_pos_at_group]),
            group_indices,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]

        # ordinal rank = pos - first_pos + 1
        ordinal = plc.binaryop.binary_operation(
            plc.binaryop.binary_operation(
                pos_in_grouped,
                first_pos_per_row,
                plc.binaryop.BinaryOperator.SUB,
                size_type,
            ),
            one,
            plc.binaryop.BinaryOperator.ADD,
            size_type,
        )
        if method == "ordinal":
            return Column(
                ordinal
                if ordinal.type().id() == out_dtype.plc.id()
                else plc.unary.cast(ordinal, out_dtype.plc),
                dtype=out_dtype,
            )

        # build run ids for ties on (group, *order_by) in grouped order
        order_tbl = plc.Table(
            [
                gid_grouped,
                *[
                    plc.copying.gather(
                        plc.Table([c]),
                        grouped_order,
                        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                    ).columns()[0]
                    for c in order_cols
                ],
            ]
        )
        run_starts = plc.stream_compaction.distinct_indices(
            order_tbl,
            plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
            plc.types.NullEquality.EQUAL,
            plc.types.NanEquality.ALL_EQUAL,
        )
        num_runs = run_starts.size()

        run_markers = plc.copying.scatter(
            plc.Table([plc.Column.from_scalar(one, num_runs)]),
            run_starts,
            plc.Table([plc.Column.from_scalar(zero, n_rows)]),
        ).columns()[0]
        run_id_sorted = plc.reduce.scan(
            run_markers, plc.aggregation.sum(), plc.reduce.ScanType.INCLUSIVE
        )  # 1-based run ids in grouped order

        # dense rank: consecutive run ids per group
        if method == "dense":
            run_id_at_first = plc.copying.gather(
                plc.Table([run_id_sorted]),
                first_pos_idx,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            ).columns()[0]
            dense_base_map = plc.copying.scatter(
                plc.Table([run_id_at_first]),
                groups_at_first,
                plc.Table([plc.Column.from_scalar(zero, group_keys_rows)]),
            ).columns()[0]
            dense_base_per_row = plc.copying.gather(
                plc.Table([dense_base_map]),
                group_indices,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            ).columns()[0]
            dense = plc.binaryop.binary_operation(
                plc.binaryop.binary_operation(
                    # map run_id_sorted back to row order via inverse of 'grouped_order'
                    plc.copying.scatter(
                        plc.Table([run_id_sorted]),
                        grouped_order,
                        plc.Table([plc.Column.from_scalar(zero, n_rows)]),
                    ).columns()[0],
                    dense_base_per_row,
                    plc.binaryop.BinaryOperator.SUB,
                    size_type,
                ),
                one,
                plc.binaryop.BinaryOperator.ADD,
                size_type,
            )
            return Column(
                dense
                if dense.type().id() == out_dtype.plc.id()
                else plc.unary.cast(dense, out_dtype.plc),
                dtype=out_dtype,
            )

        # min/max/average based on per-run min/max of pos_in_group
        # pos_in_group (0-based) in grouped order: pos_in_grouped - first_pos(group_of_that_row_in_grouped)
        first_pos_for_row_grouped = plc.copying.gather(
            plc.Table([first_pos_per_row]),
            grouped_order,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]
        pos_in_group_sorted = plc.binaryop.binary_operation(
            plc.copying.gather(
                plc.Table([pos_in_grouped]),
                grouped_order,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            ).columns()[0],
            first_pos_for_row_grouped,
            plc.binaryop.BinaryOperator.SUB,
            size_type,
        )

        run_ends = plc.Column.from_scalar(
            plc.Scalar.from_py(n_rows - 1, size_type), num_runs
        )
        if num_runs > 1:
            tail = plc.copying.gather(
                plc.Table([run_starts]),
                plc.filling.sequence(num_runs - 1, one, one),
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            ).columns()[0]
            tail = plc.binaryop.binary_operation(
                tail, one, plc.binaryop.BinaryOperator.SUB, size_type
            )
            run_ends = plc.copying.scatter(
                plc.Table([tail]),
                plc.filling.sequence(num_runs - 1, zero, one),
                plc.Table([run_ends]),
            ).columns()[0]

        min_pos = plc.copying.gather(
            plc.Table([pos_in_group_sorted]),
            run_starts,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]
        max_pos = plc.copying.gather(
            plc.Table([pos_in_group_sorted]),
            run_ends,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]
        min_pos = plc.binaryop.binary_operation(
            min_pos, one, plc.binaryop.BinaryOperator.ADD, size_type
        )
        max_pos = plc.binaryop.binary_operation(
            max_pos, one, plc.binaryop.BinaryOperator.ADD, size_type
        )

        # map per-run min/max back to per-row via run_id_sorted
        run_ids_at_starts = plc.copying.gather(
            plc.Table([run_id_sorted]),
            run_starts,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]
        run_index_per_row_sorted = plc.binaryop.binary_operation(
            run_id_sorted, one, plc.binaryop.BinaryOperator.SUB, size_type
        )
        run_index_per_run = plc.binaryop.binary_operation(
            run_ids_at_starts, one, plc.binaryop.BinaryOperator.SUB, size_type
        )

        def _per_row_rank_from_run_positions(positions: plc.Column) -> plc.Column:
            mapping = plc.copying.scatter(
                plc.Table([positions]),
                run_index_per_run,
                plc.Table(
                    [plc.Column.from_scalar(plc.Scalar.from_py(0, size_type), num_runs)]
                ),
            ).columns()[0]
            ranks_sorted = plc.copying.gather(
                plc.Table([mapping]),
                run_index_per_row_sorted,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            ).columns()[0]
            # back to input row order
            return plc.copying.scatter(
                plc.Table([ranks_sorted]),
                grouped_order,
                plc.Table(
                    [plc.Column.from_scalar(plc.Scalar.from_py(0, size_type), n_rows)]
                ),
            ).columns()[0]

        if method == "min":
            out = _per_row_rank_from_run_positions(min_pos)
            return Column(
                out
                if out.type().id() == out_dtype.plc.id()
                else plc.unary.cast(out, out_dtype.plc),
                dtype=out_dtype,
            )
        if method == "max":
            out = _per_row_rank_from_run_positions(max_pos)
            return Column(
                out
                if out.type().id() == out_dtype.plc.id()
                else plc.unary.cast(out, out_dtype.plc),
                dtype=out_dtype,
            )
        if method == "average":
            f64 = plc.DataType(plc.TypeId.FLOAT64)
            min_r = _per_row_rank_from_run_positions(min_pos)
            max_r = _per_row_rank_from_run_positions(max_pos)
            min_f = plc.unary.cast(min_r, f64)
            max_f = plc.unary.cast(max_r, f64)
            s = plc.binaryop.binary_operation(
                min_f, max_f, plc.binaryop.BinaryOperator.ADD, f64
            )
            two = plc.Scalar.from_py(2.0, f64)
            avg = plc.binaryop.binary_operation(
                s, two, plc.binaryop.BinaryOperator.DIV, f64
            )
            return Column(
                avg
                if out_dtype.plc.id() == f64.id()
                else plc.unary.cast(avg, out_dtype.plc),
                dtype=out_dtype,
            )

        raise NotImplementedError(f"rank({method=}).over(..)")  # pragma: no cover

    def do_evaluate(  # noqa: D102
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        if context != ExecutionContext.FRAME:
            raise RuntimeError(
                "Window mapping (.over) can only be evaluated at the frame level"
            )  # pragma: no cover; translation raises first

        by_exprs = self.children[: self.by_count]
        by_cols = broadcast(
            *(b.evaluate(df) for b in by_exprs),
            target_length=df.num_rows,
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

        # Split up expressions into scalar aggs (eg. Len) vs per-row (eg. rank)
        scalar_named: list[expr.NamedExpr] = []
        rank_named: list[expr.NamedExpr] = []
        for ne in self.named_aggs:
            v = ne.value
            if isinstance(v, expr.UnaryFunction) and v.name == "rank":
                rank_named.append(ne)
            else:
                scalar_named.append(ne)

        # Build GroupByRequests for scalar aggregations
        gb_requests: list[plc.groupby.GroupByRequest] = []
        out_names: list[str] = []
        out_dtypes: list[DataType] = []
        for ne in scalar_named:
            val = ne.value
            out_names.append(ne.name)
            out_dtypes.append(val.dtype)

            if isinstance(val, expr.Len):
                # A count aggregation, we need a column so use a key column
                col = by_cols[0].obj
                gb_requests.append(plc.groupby.GroupByRequest(col, [val.agg_request]))
            elif isinstance(val, expr.Agg):
                (child,) = (
                    val.children if val.name != "quantile" else (val.children[0],)
                )
                col = child.evaluate(df, context=ExecutionContext.FRAME).obj
                gb_requests.append(plc.groupby.GroupByRequest(col, [val.agg_request]))

        group_keys_tbl, value_tables = grouper.aggregate(gb_requests)
        out_cols = (t.columns()[0] for t in value_tables)

        # We do a left-join between the input keys to group-keys
        # so every input row appears exactly once. left_order is
        # returned un-ordered by libcudf.
        left_order, right_order = plc.join.left_join(
            by_tbl, group_keys_tbl, plc.types.NullEquality.EQUAL
        )

        # Scatter the right order indices into an all-null table
        # and at the position of the index in left order. Now we
        # have the map between rows and groups with the correct ordering.
        left_rows = left_order.size()
        target = plc.Column.from_scalar(
            plc.Scalar.from_py(None, plc.types.SIZE_TYPE), left_rows
        )
        aligned_map = plc.copying.scatter(
            plc.Table([right_order]),
            left_order,
            plc.Table([target]),
        ).columns()[0]

        # Broadcast each scalar aggregated result back to row-shape using
        # the aligned mapping between row indices and group indices.
        broadcasted_cols = [
            Column(
                plc.copying.gather(
                    plc.Table([col]), aligned_map, plc.copying.OutOfBoundsPolicy.NULLIFY
                ).columns()[0],
                name=named_expr.name,
                dtype=dtype,
            )
            for named_expr, dtype, col in zip(
                scalar_named, out_dtypes, out_cols, strict=True
            )
        ]

        # Rank results
        if rank_named:
            # options: (mapping, has_order_by, ob_desc, ob_nulls_last, order_count?)
            _, has_order_by, ob_desc, ob_nulls_last, order_count = (
                (*self.options, 0) if len(self.options) < 5 else self.options
            )

            if has_order_by and order_count > 0:
                # Evaluate order_by exprs (they follow the partition_by children)
                order_exprs = self.children[self.by_count : self.by_count + order_count]
                order_cols = broadcast(
                    *(e.evaluate(df) for e in order_exprs),
                    target_length=df.num_rows,
                )
                order_plc_cols = [c.obj for c in order_cols]

                # segmented ranks based on order_by keys (support ties)
                for ne in rank_named:
                    rank_expr = ne.value
                    (child_expr,) = rank_expr.children  # unused for order_by path
                    assert isinstance(rank_expr, expr.UnaryFunction)
                    method_str, _descending_ignored, _ = rank_expr.options

                    ranked_col = self._segmented_rank_over_orderby(
                        group_indices=aligned_map,
                        order_cols=order_plc_cols,
                        group_keys_rows=group_keys_tbl.num_rows(),
                        ob_desc=bool(ob_desc),
                        ob_nulls_last=bool(ob_nulls_last),
                        method=method_str,
                        out_dtype=rank_expr.dtype,
                        n_rows=df.num_rows,
                    ).obj

                    # Min/Max/Dense/Ordinal -> IDX_DTYPE
                    # See https://github.com/pola-rs/polars/blob/main/crates/polars-ops/src/series/ops/rank.rs
                    dtype = rank_expr.dtype
                    if method_str in {"min", "max", "dense", "ordinal"}:
                        dest = dtype.plc.id()
                        src = ranked_col.type().id()
                        if dest == plc.TypeId.UINT32 and src != plc.TypeId.UINT32:
                            ranked_col = plc.unary.cast(
                                ranked_col, plc.DataType(plc.TypeId.UINT32)
                            )
                        elif (
                            dest == plc.TypeId.UINT64 and src != plc.TypeId.UINT64
                        ):  # pragma: no cover
                            ranked_col = plc.unary.cast(
                                ranked_col, plc.DataType(plc.TypeId.UINT64)
                            )
                    broadcasted_cols.append(
                        Column(ranked_col, name=ne.name, dtype=rank_expr.dtype)
                    )
            else:
                # Fast path: single GroupBy.scan over the value column using libcudf rank aggregation.
                rank_requests: list[plc.groupby.GroupByRequest] = []
                rank_out_names: list[str] = []
                rank_out_dtypes: list[DataType] = []
                method_for_cast: list[str] = []

                for ne in rank_named:
                    rank_expr = ne.value
                    (child_expr,) = rank_expr.children
                    val_col = child_expr.evaluate(
                        df, context=ExecutionContext.FRAME
                    ).obj
                    assert isinstance(rank_expr, expr.UnaryFunction)
                    method_str, descending, _ = rank_expr.options

                    rank_method = {
                        "average": plc.aggregation.RankMethod.AVERAGE,
                        "min": plc.aggregation.RankMethod.MIN,
                        "max": plc.aggregation.RankMethod.MAX,
                        "dense": plc.aggregation.RankMethod.DENSE,
                        "ordinal": plc.aggregation.RankMethod.FIRST,
                    }[method_str]

                    order = (
                        plc.types.Order.DESCENDING
                        if descending
                        else plc.types.Order.ASCENDING
                    )
                    # Polars semantics: exclude nulls from domain; nulls get null ranks.
                    null_precedence = (
                        plc.types.NullOrder.BEFORE
                        if descending
                        else plc.types.NullOrder.AFTER
                    )
                    agg = plc.aggregation.rank(
                        rank_method,
                        column_order=order,
                        null_handling=plc.types.NullPolicy.EXCLUDE,
                        null_precedence=null_precedence,
                        percentage=plc.aggregation.RankPercentage.NONE,
                    )

                    rank_requests.append(plc.groupby.GroupByRequest(val_col, [agg]))
                    rank_out_names.append(ne.name)
                    rank_out_dtypes.append(rank_expr.dtype)
                    method_for_cast.append(method_str)

                _, rank_tables = grouper.scan(rank_requests)

                # Reorder scan results from grouped-order back to input row order
                n_rows = df.num_rows
                zero = plc.Scalar.from_py(0, plc.types.SIZE_TYPE)
                one = plc.Scalar.from_py(1, plc.types.SIZE_TYPE)
                row_id = plc.filling.sequence(n_rows, zero, one)

                key_orders = [k.order for k in by_cols]
                key_nulls = [k.null_order for k in by_cols]
                grouped_order = plc.sorting.stable_sorted_order(
                    plc.Table([*(c.obj for c in by_cols), row_id]),
                    [*key_orders, plc.types.Order.ASCENDING],
                    [*key_nulls, plc.types.NullOrder.AFTER],
                )

                for name, dtype, tbl, method_str in zip(
                    rank_out_names,
                    rank_out_dtypes,
                    rank_tables,
                    method_for_cast,
                    strict=True,
                ):
                    col_grouped = tbl.columns()[0]
                    target = plc.Column.from_scalar(
                        plc.Scalar.from_py(None, col_grouped.type()), n_rows
                    )
                    col_input = plc.copying.scatter(
                        plc.Table([col_grouped]),
                        grouped_order,
                        plc.Table([target]),
                    ).columns()[0]
                    # Min/Max/Dense/Ordinal -> IDX_DTYPE
                    # See https://github.com/pola-rs/polars/blob/main/crates/polars-ops/src/series/ops/rank.rs
                    if method_str in {"min", "max", "dense", "ordinal"}:
                        dest = dtype.plc.id()
                        src = col_input.type().id()
                        if dest == plc.TypeId.UINT32 and src != plc.TypeId.UINT32:
                            col_input = plc.unary.cast(
                                col_input, plc.DataType(plc.TypeId.UINT32)
                            )
                        elif (
                            dest == plc.TypeId.UINT64 and src != plc.TypeId.UINT64
                        ):  # pragma: no cover
                            col_input = plc.unary.cast(
                                col_input, plc.DataType(plc.TypeId.UINT64)
                            )
                    broadcasted_cols.append(Column(col_input, name=name, dtype=dtype))

        # Create a temporary DataFrame with the broadcasted columns named by their
        # placeholder names from agg decomposition, then evaluate the post-expression.
        df = DataFrame(broadcasted_cols)
        return self.post.value.evaluate(df, context=ExecutionContext.FRAME)
