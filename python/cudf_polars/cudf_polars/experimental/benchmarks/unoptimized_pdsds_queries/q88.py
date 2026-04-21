# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q88 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q88 naive Polars implementation.

    SELECT *
    FROM (SELECT count(*) h8_30_to_9 ...) s1,
         (SELECT count(*) h9_to_9_30 ...) s2,
         ...
         (SELECT count(*) h12_to_12_30 ...) s8;

    Each subquery counts store_sales rows for a specific hour window,
    cross-joined (comma-separated in FROM = cross join of scalar subqueries).
    All share the same household_demographics filter.
    """
    params = load_parameters(
        int(run_config.scale_factor), query_id=88, qualification=run_config.qualification
    )
    s_store_name = params["s_store_name"]
    hd1 = params["hd_dep_count1"]
    hd2 = params["hd_dep_count2"]
    hd3 = params["hd_dep_count3"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    household_demographics = get_data(run_config.dataset_path, "household_demographics", run_config.suffix)
    time_dim = get_data(run_config.dataset_path, "time_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)

    # Common household_demographics filter
    hd_filter = (
        ((pl.col("hd_dep_count") == hd1) & (pl.col("hd_vehicle_count") <= hd1 + 2))
        | ((pl.col("hd_dep_count") == hd2) & (pl.col("hd_vehicle_count") <= hd2 + 2))
        | ((pl.col("hd_dep_count") == hd3) & (pl.col("hd_vehicle_count") <= hd3 + 2))
    )

    # Helper: build one count subquery for a given hour window condition
    def _count_window(hour_filter: pl.Expr) -> pl.LazyFrame:
        return (
            store_sales
            .join(time_dim, left_on="ss_sold_time_sk", right_on="t_time_sk", how="inner")
            .join(household_demographics, left_on="ss_hdemo_sk", right_on="hd_demo_sk", how="inner")
            .join(store, left_on="ss_store_sk", right_on="s_store_sk", how="inner")
            .filter(hd_filter & (pl.col("s_store_name") == s_store_name) & hour_filter)
            .select([pl.len()])
        )

    s1 = _count_window((pl.col("t_hour") == 8) & (pl.col("t_minute") >= 30))
    s2 = _count_window((pl.col("t_hour") == 9) & (pl.col("t_minute") < 30))
    s3 = _count_window((pl.col("t_hour") == 9) & (pl.col("t_minute") >= 30))
    s4 = _count_window((pl.col("t_hour") == 10) & (pl.col("t_minute") < 30))
    s5 = _count_window((pl.col("t_hour") == 10) & (pl.col("t_minute") >= 30))
    s6 = _count_window((pl.col("t_hour") == 11) & (pl.col("t_minute") < 30))
    s7 = _count_window((pl.col("t_hour") == 11) & (pl.col("t_minute") >= 30))
    s8 = _count_window((pl.col("t_hour") == 12) & (pl.col("t_minute") < 30))

    # Scalar subqueries cross-joined: collect each, then assemble single row
    # (comma-sep subqueries in SQL FROM produce a row cross-product of scalars)
    result = (
        s1.rename({"len": "h8_30_to_9"})
        .with_columns(pl.lit(s2.collect()["len"][0]).alias("h9_to_9_30"))
        .with_columns(pl.lit(s3.collect()["len"][0]).alias("h9_30_to_10"))
        .with_columns(pl.lit(s4.collect()["len"][0]).alias("h10_to_10_30"))
        .with_columns(pl.lit(s5.collect()["len"][0]).alias("h10_30_to_11"))
        .with_columns(pl.lit(s6.collect()["len"][0]).alias("h11_to_11_30"))
        .with_columns(pl.lit(s7.collect()["len"][0]).alias("h11_30_to_12"))
        .with_columns(pl.lit(s8.collect()["len"][0]).alias("h12_to_12_30"))
        .select([
            pl.col("h8_30_to_9").cast(pl.Int64),
            pl.col("h9_to_9_30").cast(pl.Int64),
            pl.col("h9_30_to_10").cast(pl.Int64),
            pl.col("h10_to_10_30").cast(pl.Int64),
            pl.col("h10_30_to_11").cast(pl.Int64),
            pl.col("h11_to_11_30").cast(pl.Int64),
            pl.col("h11_30_to_12").cast(pl.Int64),
            pl.col("h12_to_12_30").cast(pl.Int64),
        ])
    )

    return QueryResult(
        frame=result,
        sort_by=[],
        limit=None,
    )
