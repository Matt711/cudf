# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q62 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q62 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=62, qualification=run_config.qualification
    )
    dms = params["dms"]

    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    warehouse = get_data(run_config.dataset_path, "warehouse", run_config.suffix)
    ship_mode = get_data(run_config.dataset_path, "ship_mode", run_config.suffix)
    web_site = get_data(run_config.dataset_path, "web_site", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # FROM web_sales, warehouse, ship_mode, web_site, date_dim
    # WHERE d_month_seq BETWEEN dms AND dms+11
    #   AND ws_ship_date_sk = d_date_sk
    #   AND ws_warehouse_sk = w_warehouse_sk
    #   AND ws_ship_mode_sk = sm_ship_mode_sk
    #   AND ws_web_site_sk = web_site_sk
    # GROUP BY Substr(w_warehouse_name, 1, 20), sm_type, web_name
    # SELECT bucket counts
    result = (
        web_sales.join(date_dim, left_on="ws_ship_date_sk", right_on="d_date_sk", how="inner")
        .join(warehouse, left_on="ws_warehouse_sk", right_on="w_warehouse_sk", how="inner")
        .join(ship_mode, left_on="ws_ship_mode_sk", right_on="sm_ship_mode_sk", how="inner")
        .join(web_site, left_on="ws_web_site_sk", right_on="web_site_sk", how="inner")
        .filter(pl.col("d_month_seq").is_between(dms, dms + 11))
        .with_columns(
            (pl.col("ws_ship_date_sk") - pl.col("ws_sold_date_sk")).alias("shipping_delay"),
            pl.col("w_warehouse_name").str.slice(0, 20).alias("warehouse_substr"),
        )
        .group_by(["warehouse_substr", "sm_type", "web_name"])
        .agg(
            sql_sum(
                pl.when(pl.col("shipping_delay") <= 30).then(pl.lit(1)).otherwise(pl.lit(0))
            ).alias("30 days"),
            sql_sum(
                pl.when(
                    (pl.col("shipping_delay") > 30) & (pl.col("shipping_delay") <= 60)
                ).then(pl.lit(1)).otherwise(pl.lit(0))
            ).alias("31-60 days"),
            sql_sum(
                pl.when(
                    (pl.col("shipping_delay") > 60) & (pl.col("shipping_delay") <= 90)
                ).then(pl.lit(1)).otherwise(pl.lit(0))
            ).alias("61-90 days"),
            sql_sum(
                pl.when(
                    (pl.col("shipping_delay") > 90) & (pl.col("shipping_delay") <= 120)
                ).then(pl.lit(1)).otherwise(pl.lit(0))
            ).alias("91-120 days"),
            sql_sum(
                pl.when(pl.col("shipping_delay") > 120).then(pl.lit(1)).otherwise(pl.lit(0))
            ).alias(">120 days"),
        )
        .select(
            pl.col("warehouse_substr").alias("substr(w_warehouse_name, 1, 20)"),
            "sm_type",
            "web_name",
            "30 days",
            "31-60 days",
            "61-90 days",
            "91-120 days",
            ">120 days",
        )
    )

    result = result.sort(['substr(w_warehouse_name, 1, 20)', 'sm_type', 'web_name'], descending=[False, False, False]).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[
            ("substr(w_warehouse_name, 1, 20)", False),
            ("sm_type", False),
            ("web_name", False),
        ],
        limit=100,
    )
