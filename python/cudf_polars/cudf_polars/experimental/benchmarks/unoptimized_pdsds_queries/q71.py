# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q71 — naive one-for-one Polars translation of the SQL."""
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
    params = load_parameters(
        int(run_config.scale_factor), query_id=71, qualification=run_config.qualification
    )
    year = params["year"]
    month = params["month"]
    manager = params["manager"]

    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    time_dim = get_data(run_config.dataset_path, "time_dim", run_config.suffix)

    # UNION ALL of three subqueries, each joining with date_dim (comma = inner join)
    # WHERE d_date_sk = ws/cs/ss_sold_date_sk AND d_moy=month AND d_year=year
    # No .select() early (naive rule 3) — but subquery has explicit column list
    ws_part = (
        web_sales
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter((pl.col("d_moy") == month) & (pl.col("d_year") == year))
        .select([
            pl.col("ws_ext_sales_price").alias("ext_price"),
            pl.col("ws_sold_date_sk").alias("sold_date_sk"),
            pl.col("ws_item_sk").alias("sold_item_sk"),
            pl.col("ws_sold_time_sk").alias("time_sk"),
        ])
    )

    cs_part = (
        catalog_sales
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter((pl.col("d_moy") == month) & (pl.col("d_year") == year))
        .select([
            pl.col("cs_ext_sales_price").alias("ext_price"),
            pl.col("cs_sold_date_sk").alias("sold_date_sk"),
            pl.col("cs_item_sk").alias("sold_item_sk"),
            pl.col("cs_sold_time_sk").alias("time_sk"),
        ])
    )

    ss_part = (
        store_sales
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter((pl.col("d_moy") == month) & (pl.col("d_year") == year))
        .select([
            pl.col("ss_ext_sales_price").alias("ext_price"),
            pl.col("ss_sold_date_sk").alias("sold_date_sk"),
            pl.col("ss_item_sk").alias("sold_item_sk"),
            pl.col("ss_sold_time_sk").alias("time_sk"),
        ])
    )

    tmp = pl.concat([ws_part, cs_part, ss_part], how="diagonal_relaxed")

    # Outer query: FROM item, tmp, time_dim
    # WHERE sold_item_sk = i_item_sk
    #   AND i_manager_id = manager
    #   AND time_sk = t_time_sk
    #   AND (t_meal_time = 'breakfast' OR t_meal_time = 'dinner')
    # GROUP BY i_brand, i_brand_id, t_hour, t_minute
    result = (
        item
        .join(tmp, left_on="i_item_sk", right_on="sold_item_sk", how="inner")
        .join(time_dim, left_on="time_sk", right_on="t_time_sk", how="inner")
        .filter(
            (pl.col("i_manager_id") == manager)
            & pl.col("t_meal_time").is_in(["breakfast", "dinner"])
        )
        .group_by(["i_brand", "i_brand_id", "t_hour", "t_minute"])
        .agg(sql_sum(pl.col("ext_price")).alias("ext_price"))
        .select([
            pl.col("i_brand_id").alias("brand_id"),
            pl.col("i_brand").alias("brand"),
            "t_hour",
            "t_minute",
            "ext_price",
        ])
    )

    result = result.sort(["ext_price", "brand_id", "t_hour"], descending=[True, False, False], nulls_last=False)
    return QueryResult(
        frame=result,
        sort_by=[("ext_price", True), ("brand_id", False), ("t_hour", False)],
        limit=None,
        nulls_last=False,
    )
