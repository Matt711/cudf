# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q99 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q99 naive Polars implementation.

    SELECT Substr(w_warehouse_name, 1, 20), sm_type, cc_name,
           Sum(CASE WHEN cs_ship_date_sk - cs_sold_date_sk <= 30 THEN 1 ELSE 0 END) AS '30 days',
           Sum(CASE WHEN ... > 30 AND ... <= 60 THEN 1 ELSE 0 END) AS '31-60 days',
           Sum(CASE WHEN ... > 60 AND ... <= 90 THEN 1 ELSE 0 END) AS '61-90 days',
           Sum(CASE WHEN ... > 90 AND ... <= 120 THEN 1 ELSE 0 END) AS '91-120 days',
           Sum(CASE WHEN ... > 120 THEN 1 ELSE 0 END) AS '>120 days'
    FROM catalog_sales, warehouse, ship_mode, call_center, date_dim
    WHERE d_month_seq BETWEEN d_month_seq AND d_month_seq+11
      AND cs_ship_date_sk = d_date_sk
      AND cs_warehouse_sk = w_warehouse_sk
      AND cs_ship_mode_sk = sm_ship_mode_sk
      AND cs_call_center_sk = cc_call_center_sk
    GROUP BY Substr(w_warehouse_name, 1, 20), sm_type, cc_name
    ORDER BY Substr(w_warehouse_name, 1, 20), sm_type, cc_name
    LIMIT 100;
    """
    params = load_parameters(
        int(run_config.scale_factor), query_id=99, qualification=run_config.qualification
    )
    d_month_seq = params["d_month_seq"]

    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    warehouse = get_data(run_config.dataset_path, "warehouse", run_config.suffix)
    ship_mode = get_data(run_config.dataset_path, "ship_mode", run_config.suffix)
    call_center = get_data(run_config.dataset_path, "call_center", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # Join in FROM order: catalog_sales, warehouse, ship_mode, call_center, date_dim
    # ALL WHERE conditions after joins (naive rule 2)
    result = (
        catalog_sales
        .join(warehouse, left_on="cs_warehouse_sk", right_on="w_warehouse_sk", how="inner")
        .join(ship_mode, left_on="cs_ship_mode_sk", right_on="sm_ship_mode_sk", how="inner")
        .join(call_center, left_on="cs_call_center_sk", right_on="cc_call_center_sk", how="inner")
        .join(date_dim, left_on="cs_ship_date_sk", right_on="d_date_sk", how="inner")
        .filter(pl.col("d_month_seq").is_between(d_month_seq, d_month_seq + 11))
        .with_columns([
            (pl.col("cs_ship_date_sk") - pl.col("cs_sold_date_sk")).alias("ship_days"),
            pl.col("w_warehouse_name").str.slice(0, 20).alias("substr(w_warehouse_name, 1, 20)"),
        ])
        .group_by(["substr(w_warehouse_name, 1, 20)", "sm_type", "cc_name"])
        .agg([
            sql_sum(
                pl.when(pl.col("ship_days") <= 30).then(1).otherwise(0)
            ).alias("30 days"),
            sql_sum(
                pl.when((pl.col("ship_days") > 30) & (pl.col("ship_days") <= 60)).then(1).otherwise(0)
            ).alias("31-60 days"),
            sql_sum(
                pl.when((pl.col("ship_days") > 60) & (pl.col("ship_days") <= 90)).then(1).otherwise(0)
            ).alias("61-90 days"),
            sql_sum(
                pl.when((pl.col("ship_days") > 90) & (pl.col("ship_days") <= 120)).then(1).otherwise(0)
            ).alias("91-120 days"),
            sql_sum(
                pl.when(pl.col("ship_days") > 120).then(1).otherwise(0)
            ).alias(">120 days"),
        ])
        .select([
            "substr(w_warehouse_name, 1, 20)",
            "sm_type",
            "cc_name",
            "30 days",
            "31-60 days",
            "61-90 days",
            "91-120 days",
            ">120 days",
        ])
    )

    result = result.sort(['substr(w_warehouse_name, 1, 20)', 'sm_type', 'cc_name'], descending=[False, False, False]).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[
            ("substr(w_warehouse_name, 1, 20)", False),
            ("sm_type", False),
            ("cc_name", False),
        ],
        limit=100,
    )
