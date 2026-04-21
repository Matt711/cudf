# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q53 — naive one-for-one Polars translation of the SQL."""

from __future__ import annotations

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
    """TPC-DS q53 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=53, qualification=run_config.qualification
    )
    dms = params["dms"]
    categories1 = params["categories1"]
    classes1 = params["classes1"]
    brands1 = params["brands1"]
    categories2 = params["categories2"]
    classes2 = params["classes2"]
    brands2 = params["brands2"]

    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)

    month_seq_list = list(range(dms, dms + 12))

    # Inner query:
    # FROM item, store_sales, date_dim, store
    # WHERE ss_item_sk = i_item_sk AND ss_sold_date_sk = d_date_sk
    #   AND ss_store_sk = s_store_sk
    #   AND d_month_seq IN (dms..dms+11)
    #   AND (brand/class/cat condition 1 OR condition 2)
    # GROUP BY i_manufact_id, d_qoy
    # WINDOW: Avg(Sum(ss_sales_price)) OVER (PARTITION BY i_manufact_id)
    #
    # Rule 13: AVG(SUM(col)) OVER (PARTITION BY x)
    #   => group_by (x, d_qoy) to get sum_sales, then .mean().over("x") for avg
    inner = (
        item.join(store_sales, left_on="i_item_sk", right_on="ss_item_sk", how="inner")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk", how="inner")
        .filter(
            pl.col("d_month_seq").is_in(month_seq_list)
            & (
                (
                    pl.col("i_category").is_in(categories1)
                    & pl.col("i_class").is_in(classes1)
                    & pl.col("i_brand").is_in(brands1)
                )
                | (
                    pl.col("i_category").is_in(categories2)
                    & pl.col("i_class").is_in(classes2)
                    & pl.col("i_brand").is_in(brands2)
                )
            )
        )
        .group_by(["i_manufact_id", "d_qoy"])
        .agg(sql_sum(pl.col("ss_sales_price")).alias("sum_sales"))
        .with_columns(
            pl.col("sum_sales").mean().over("i_manufact_id").alias("avg_quarterly_sales")
        )
        .select(["i_manufact_id", "sum_sales", "avg_quarterly_sales"])
    )

    # Outer WHERE:
    # CASE WHEN avg_quarterly_sales > 0
    #   THEN Abs(sum_sales - avg_quarterly_sales) / avg_quarterly_sales
    #   ELSE NULL END > 0.1
    result = inner.filter(
        pl.when(pl.col("avg_quarterly_sales") > 0)
        .then(
            (pl.col("sum_sales") - pl.col("avg_quarterly_sales")).abs()
            / pl.col("avg_quarterly_sales")
        )
        .otherwise(None)
        > 0.1
    )

    result = result.sort(['avg_quarterly_sales', 'sum_sales', 'i_manufact_id'], descending=[False, False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("avg_quarterly_sales", False), ("sum_sales", False), ("i_manufact_id", False)],
        limit=100,
    )
