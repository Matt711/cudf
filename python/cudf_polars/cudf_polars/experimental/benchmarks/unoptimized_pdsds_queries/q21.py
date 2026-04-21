# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q21 — naive one-for-one Polars translation of the SQL."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data
from cudf_polars.experimental.benchmarks.unoptimized_pdsds_queries.sql_helpers import (
    sql_sum,
)

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:  # type: ignore[return]
    raise NotImplementedError


def polars_impl(run_config: RunConfig) -> QueryResult:
    """TPC-DS q21 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=21, qualification=run_config.qualification
    )

    sales_date = params["sales_date"]

    inventory = get_data(run_config.dataset_path, "inventory", run_config.suffix)
    warehouse = get_data(run_config.dataset_path, "warehouse", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    sales_date_obj = date.fromisoformat(sales_date)
    start_date = sales_date_obj - timedelta(days=30)
    end_date = sales_date_obj + timedelta(days=30)

    sales_date_lit = pl.lit(sales_date_obj)
    start_date_lit = pl.lit(start_date)
    end_date_lit = pl.lit(end_date)

    # FROM order: inventory, warehouse, item, date_dim
    # All WHERE conditions after all joins
    # The subquery x is: group_by warehouse_name and item_id computing inv_before/inv_after
    # Then outer WHERE filters the ratio
    inner = (
        inventory
        .join(warehouse, how="inner",
              left_on="inv_warehouse_sk", right_on="w_warehouse_sk")
        .join(item, how="inner",
              left_on="inv_item_sk", right_on="i_item_sk")
        .join(date_dim, how="inner",
              left_on="inv_date_sk", right_on="d_date_sk")
        .filter(
            pl.col("i_current_price").is_between(0.99, 1.49)
            & pl.col("d_date").is_between(start_date_lit, end_date_lit)
        )
        .group_by(["w_warehouse_name", "i_item_id"])
        .agg([
            sql_sum(
                pl.when(pl.col("d_date") < sales_date_lit)
                .then(pl.col("inv_quantity_on_hand"))
                .otherwise(0)
            ).alias("inv_before"),
            sql_sum(
                pl.when(pl.col("d_date") >= sales_date_lit)
                .then(pl.col("inv_quantity_on_hand"))
                .otherwise(0)
            ).alias("inv_after"),
        ])
        .filter(
            pl.when(pl.col("inv_before") > 0)
            .then(pl.col("inv_after") / pl.col("inv_before"))
            .otherwise(None)
            .is_between(2.0 / 3.0, 3.0 / 2.0)
        )
    )

    inner = inner.sort(['w_warehouse_name', 'i_item_id'], descending=[False, False], nulls_last=True).limit(100)

    sort_by = [("w_warehouse_name", False), ("i_item_id", False)]
    limit = 100

    return QueryResult(frame=inner, sort_by=sort_by, limit=limit)
