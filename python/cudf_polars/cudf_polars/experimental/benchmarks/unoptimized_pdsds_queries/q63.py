# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q63 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q63 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=63, qualification=run_config.qualification
    )
    dms = params["dms"]

    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)

    # Inner query:
    # FROM item, store_sales, date_dim, store
    # WHERE ss_item_sk=i_item_sk AND ss_sold_date_sk=d_date_sk AND ss_store_sk=s_store_sk
    #   AND d_month_seq IN (dms..dms+11)
    #   AND (brand/class/cat condition)
    # GROUP BY i_manager_id, d_moy
    # sum_sales = Sum(ss_sales_price)
    # avg_monthly_sales = Avg(Sum(...)) OVER (PARTITION BY i_manager_id)
    # Rule 13: group_by first, then .mean().over("i_manager_id")
    inner = (
        item.join(store_sales, left_on="i_item_sk", right_on="ss_item_sk", how="inner")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk", how="inner")
        .filter(
            pl.col("d_month_seq").is_in(list(range(dms, dms + 12)))
            & (
                (
                    pl.col("i_category").is_in(["Books", "Children", "Electronics"])
                    & pl.col("i_class").is_in(["personal", "portable", "reference", "self-help"])
                    & pl.col("i_brand").is_in([
                        "scholaramalgamalg #14",
                        "scholaramalgamalg #7",
                        "exportiunivamalg #9",
                        "scholaramalgamalg #9",
                    ])
                )
                | (
                    pl.col("i_category").is_in(["Women", "Music", "Men"])
                    & pl.col("i_class").is_in(["accessories", "classical", "fragrances", "pants"])
                    & pl.col("i_brand").is_in([
                        "amalgimporto #1",
                        "edu packscholar #1",
                        "exportiimporto #1",
                        "importoamalg #1",
                    ])
                )
            )
        )
        .group_by(["i_manager_id", "d_moy"])
        .agg(sql_sum(pl.col("ss_sales_price")).alias("sum_sales"))
        .with_columns(
            pl.col("sum_sales").mean().over("i_manager_id").alias("avg_monthly_sales")
        )
        .select(["i_manager_id", "sum_sales", "avg_monthly_sales"])
    )

    # Outer WHERE:
    # CASE WHEN avg_monthly_sales > 0
    #   THEN Abs(sum_sales - avg_monthly_sales) / avg_monthly_sales
    #   ELSE NULL END > 0.1
    result = inner.filter(
        pl.when(pl.col("avg_monthly_sales") > 0)
        .then(
            (pl.col("sum_sales") - pl.col("avg_monthly_sales")).abs()
            / pl.col("avg_monthly_sales")
        )
        .otherwise(None)
        > 0.1
    )

    result = result.sort(['i_manager_id', 'avg_monthly_sales', 'sum_sales'], descending=[False, False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("i_manager_id", False), ("avg_monthly_sales", False), ("sum_sales", False)],
        limit=100,
    )
