# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q66 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q66 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=66, qualification=run_config.qualification
    )
    year = params["year"]
    time_one = params["time_one"]
    smc = params["smc"]
    sales_one = params["sales_one"]
    sales_two = params["sales_two"]
    net_one = params["net_one"]
    net_two = params["net_two"]

    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    warehouse = get_data(run_config.dataset_path, "warehouse", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    time_dim = get_data(run_config.dataset_path, "time_dim", run_config.suffix)
    ship_mode = get_data(run_config.dataset_path, "ship_mode", run_config.suffix)

    month_names = ["jan", "feb", "mar", "apr", "may", "jun",
                   "jul", "aug", "sep", "oct", "nov", "dec"]

    # Inner UNION ALL branch 1: web_sales
    # FROM web_sales, warehouse, date_dim, time_dim, ship_mode
    # WHERE ws_warehouse_sk=w_warehouse_sk AND ws_sold_date_sk=d_date_sk
    #   AND ws_sold_time_sk=t_time_sk AND ws_ship_mode_sk=sm_ship_mode_sk
    #   AND d_year=year AND t_time BETWEEN time_one AND time_one+28800
    #   AND sm_carrier IN (smc)
    # GROUP BY warehouse columns + d_year
    web_sales_aggs = [
        sql_sum(
            pl.when(pl.col("d_moy") == (i + 1))
            .then(pl.col(sales_one) * pl.col("ws_quantity"))
            .otherwise(0)
        ).alias(f"{month_names[i]}_sales")
        for i in range(12)
    ]
    web_net_aggs = [
        sql_sum(
            pl.when(pl.col("d_moy") == (i + 1))
            .then(pl.col(net_one) * pl.col("ws_quantity"))
            .otherwise(0)
        ).alias(f"{month_names[i]}_net")
        for i in range(12)
    ]

    web_part = (
        web_sales.join(warehouse, left_on="ws_warehouse_sk", right_on="w_warehouse_sk", how="inner")
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(time_dim, left_on="ws_sold_time_sk", right_on="t_time_sk", how="inner")
        .join(ship_mode, left_on="ws_ship_mode_sk", right_on="sm_ship_mode_sk", how="inner")
        .filter(
            (pl.col("d_year") == year)
            & pl.col("t_time").is_between(time_one, time_one + 28800)
            & pl.col("sm_carrier").is_in(smc)
        )
        .group_by([
            "w_warehouse_name",
            "w_warehouse_sq_ft",
            "w_city",
            "w_county",
            "w_state",
            "w_country",
            "d_year",
        ])
        .agg(web_sales_aggs + web_net_aggs)
        .with_columns(
            pl.lit(smc[0] + "," + smc[1]).alias("ship_carriers"),
            pl.col("d_year").alias("year1"),
        )
    )

    # Inner UNION ALL branch 2: catalog_sales
    # FROM catalog_sales, warehouse, date_dim, time_dim, ship_mode
    # WHERE cs_warehouse_sk=w_warehouse_sk AND cs_sold_date_sk=d_date_sk
    #   AND cs_sold_time_sk=t_time_sk AND cs_ship_mode_sk=sm_ship_mode_sk
    #   AND d_year=year AND t_time BETWEEN time_one AND time_one+28800
    #   AND sm_carrier IN (smc)
    # GROUP BY warehouse columns + d_year
    cat_sales_aggs = [
        sql_sum(
            pl.when(pl.col("d_moy") == (i + 1))
            .then(pl.col(sales_two) * pl.col("cs_quantity"))
            .otherwise(0)
        ).alias(f"{month_names[i]}_sales")
        for i in range(12)
    ]
    cat_net_aggs = [
        sql_sum(
            pl.when(pl.col("d_moy") == (i + 1))
            .then(pl.col(net_two) * pl.col("cs_quantity"))
            .otherwise(0)
        ).alias(f"{month_names[i]}_net")
        for i in range(12)
    ]

    cat_part = (
        catalog_sales.join(warehouse, left_on="cs_warehouse_sk", right_on="w_warehouse_sk", how="inner")
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(time_dim, left_on="cs_sold_time_sk", right_on="t_time_sk", how="inner")
        .join(ship_mode, left_on="cs_ship_mode_sk", right_on="sm_ship_mode_sk", how="inner")
        .filter(
            (pl.col("d_year") == year)
            & pl.col("t_time").is_between(time_one, time_one + 28800)
            & pl.col("sm_carrier").is_in(smc)
        )
        .group_by([
            "w_warehouse_name",
            "w_warehouse_sq_ft",
            "w_city",
            "w_county",
            "w_state",
            "w_country",
            "d_year",
        ])
        .agg(cat_sales_aggs + cat_net_aggs)
        .with_columns(
            pl.lit(smc[0] + "," + smc[1]).alias("ship_carriers"),
            pl.col("d_year").alias("year1"),
        )
    )

    # UNION ALL then outer GROUP BY + SUM per month column
    # SELECT ... Sum(jan_sales) AS jan_sales, ... Sum(jan_sales/w_warehouse_sq_ft) AS jan_sales_per_sq_foot ...
    unioned = pl.concat([web_part, cat_part], how="diagonal_relaxed")

    per_sqft_exprs = [
        sql_sum(pl.col(f"{m}_sales") / pl.col("w_warehouse_sq_ft")).alias(f"{m}_sales_per_sq_foot")
        for m in month_names
    ]

    result = (
        unioned.group_by([
            "w_warehouse_name",
            "w_warehouse_sq_ft",
            "w_city",
            "w_county",
            "w_state",
            "w_country",
            "ship_carriers",
            "year1",
        ])
        .agg(
            [sql_sum(pl.col(f"{m}_sales")).alias(f"{m}_sales") for m in month_names]
            + per_sqft_exprs
            + [sql_sum(pl.col(f"{m}_net")).alias(f"{m}_net") for m in month_names]
        )
        .select(
            ["w_warehouse_name", "w_warehouse_sq_ft", "w_city", "w_county",
             "w_state", "w_country", "ship_carriers", "year1"]
            + [f"{m}_sales" for m in month_names]
            + [f"{m}_sales_per_sq_foot" for m in month_names]
            + [f"{m}_net" for m in month_names]
        )
    )

    result = result.sort('w_warehouse_name', descending=False, nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("w_warehouse_name", False)],
        limit=100,
    )
