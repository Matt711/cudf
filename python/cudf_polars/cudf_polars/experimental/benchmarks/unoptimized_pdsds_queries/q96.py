# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q96 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q96 naive Polars implementation.

    SELECT Count(*)
    FROM store_sales, household_demographics, time_dim, store
    WHERE ss_sold_time_sk = time_dim.t_time_sk
      AND ss_hdemo_sk = household_demographics.hd_demo_sk
      AND ss_store_sk = s_store_sk
      AND time_dim.t_hour = t_hour
      AND time_dim.t_minute >= t_minute
      AND household_demographics.hd_dep_count = hd_dep_count
      AND store.s_store_name = s_store_name
    ORDER BY Count(*) LIMIT 100;
    """
    params = load_parameters(
        int(run_config.scale_factor), query_id=96, qualification=run_config.qualification
    )
    t_hour = params["t_hour"]
    t_minute = params["t_minute"]
    hd_dep_count = params["hd_dep_count"]
    s_store_name = params["s_store_name"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    household_demographics = get_data(run_config.dataset_path, "household_demographics", run_config.suffix)
    time_dim = get_data(run_config.dataset_path, "time_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)

    # Join in FROM order: store_sales, household_demographics, time_dim, store
    # ALL WHERE conditions after joins (naive rule 2)
    result = (
        store_sales
        .join(household_demographics, left_on="ss_hdemo_sk", right_on="hd_demo_sk", how="inner")
        .join(time_dim, left_on="ss_sold_time_sk", right_on="t_time_sk", how="inner")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk", how="inner")
        .filter(
            (pl.col("t_hour") == t_hour)
            & (pl.col("t_minute") >= t_minute)
            & (pl.col("hd_dep_count") == hd_dep_count)
            & (pl.col("s_store_name") == s_store_name)
        )
        .select([pl.len().alias("count_star()")])
    )

    result = result.sort('count_star()', descending=False).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("count_star()", False)],
        limit=100,
    )
