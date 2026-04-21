# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q40 — naive one-for-one Polars translation of the SQL."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data
from cudf_polars.experimental.benchmarks.unoptimized_pdsds_queries.sql_helpers import sql_sum

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:  # type: ignore[return]
    raise NotImplementedError


def polars_impl(run_config: RunConfig) -> QueryResult:
    params = load_parameters(
        int(run_config.scale_factor), query_id=40, qualification=run_config.qualification
    )
    sales_date = params["sales_date"]

    sales_date_obj = datetime.strptime(sales_date, "%Y-%m-%d")
    start_date_obj = sales_date_obj - timedelta(days=30)
    end_date_obj = sales_date_obj + timedelta(days=30)

    target_date = pl.date(sales_date_obj.year, sales_date_obj.month, sales_date_obj.day)
    start_date = pl.date(start_date_obj.year, start_date_obj.month, start_date_obj.day)
    end_date = pl.date(end_date_obj.year, end_date_obj.month, end_date_obj.day)

    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    catalog_returns = get_data(
        run_config.dataset_path, "catalog_returns", run_config.suffix
    )
    warehouse = get_data(run_config.dataset_path, "warehouse", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # FROM catalog_sales LEFT OUTER JOIN catalog_returns
    #   ON (cs_order_number = cr_order_number AND cs_item_sk = cr_item_sk),
    #   warehouse, item, date_dim
    # WHERE i_current_price BETWEEN 0.99 AND 1.49
    #   AND i_item_sk = cs_item_sk AND cs_warehouse_sk = w_warehouse_sk
    #   AND cs_sold_date_sk = d_date_sk
    #   AND d_date BETWEEN sales_date-30days AND sales_date+30days
    # All WHERE conditions after all joins (naive rule 2)
    result = (
        catalog_sales.join(
            catalog_returns,
            left_on=["cs_order_number", "cs_item_sk"],
            right_on=["cr_order_number", "cr_item_sk"],
            how="left",
        )
        .join(warehouse, left_on="cs_warehouse_sk", right_on="w_warehouse_sk")
        .join(item, left_on="cs_item_sk", right_on="i_item_sk")
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .filter(
            (pl.col("i_current_price").is_between(0.99, 1.49))
            & (pl.col("d_date").is_between(start_date, end_date))
        )
        .with_columns(
            [
                pl.when(pl.col("d_date") < target_date)
                .then(pl.col("cs_sales_price") - pl.col("cr_refunded_cash").fill_null(0))
                .otherwise(0)
                .alias("_sales_before"),
                pl.when(pl.col("d_date") >= target_date)
                .then(pl.col("cs_sales_price") - pl.col("cr_refunded_cash").fill_null(0))
                .otherwise(0)
                .alias("_sales_after"),
            ]
        )
        .group_by(["w_state", "i_item_id"])
        .agg(
            [
                sql_sum(pl.col("_sales_before")).alias("sales_before"),
                sql_sum(pl.col("_sales_after")).alias("sales_after"),
            ]
        )
        .select(["w_state", "i_item_id", "sales_before", "sales_after"])
    )

    result = result.sort(['w_state', 'i_item_id'], descending=[False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("w_state", False), ("i_item_id", False)],
        limit=100,
    )
