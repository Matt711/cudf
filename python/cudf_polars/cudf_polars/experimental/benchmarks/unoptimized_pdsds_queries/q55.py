# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q55 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q55 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=55, qualification=run_config.qualification
    )
    year = params["year"]
    month = params["month"]
    manager_id = params["manager_id"]

    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # FROM date_dim, store_sales, item
    # WHERE d_date_sk = ss_sold_date_sk AND ss_item_sk = i_item_sk
    #   AND i_manager_id = manager_id AND d_moy = month AND d_year = year
    # GROUP BY i_brand, i_brand_id
    # SELECT i_brand_id brand_id, i_brand brand, Sum(ss_ext_sales_price) ext_price
    # ORDER BY ext_price DESC, i_brand_id LIMIT 100
    result = (
        date_dim.join(store_sales, left_on="d_date_sk", right_on="ss_sold_date_sk", how="inner")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk", how="inner")
        .filter(
            (pl.col("i_manager_id") == manager_id)
            & (pl.col("d_moy") == month)
            & (pl.col("d_year") == year)
        )
        .group_by(["i_brand", "i_brand_id"])
        .agg(sql_sum(pl.col("ss_ext_sales_price")).alias("ext_price"))
        .select(
            pl.col("i_brand_id").alias("brand_id"),
            pl.col("i_brand").alias("brand"),
            pl.col("ext_price"),
        )
    )

    result = result.sort(['ext_price', 'brand_id'], descending=[True, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("ext_price", True), ("brand_id", False)],
        limit=100,
    )
