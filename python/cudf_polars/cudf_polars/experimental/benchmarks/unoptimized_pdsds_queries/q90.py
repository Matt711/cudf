# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q90 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q90 naive Polars implementation.

    SELECT Cast(amc AS DECIMAL(15,4)) / Cast(pmc AS DECIMAL(15,4)) am_pm_ratio
    FROM (SELECT Count(*) amc FROM web_sales, household_demographics, time_dim, web_page
          WHERE ws_sold_time_sk = t_time_sk AND ws_ship_hdemo_sk = hd_demo_sk
            AND ws_web_page_sk = wp_web_page_sk
            AND t_hour BETWEEN am_hour AND am_hour+1
            AND hd_dep_count = hd_dep_count
            AND wp_char_count BETWEEN wp_char_count_min AND wp_char_count_max) at1,
         (SELECT Count(*) pmc FROM web_sales, household_demographics, time_dim, web_page
          WHERE ... AND t_hour BETWEEN pm_hour AND pm_hour+1 ...) pt
    ORDER BY am_pm_ratio LIMIT 100;

    Two scalar subqueries: compute each count, divide, return single row.
    """
    params = load_parameters(
        int(run_config.scale_factor), query_id=90, qualification=run_config.qualification
    )
    am_hour = params["am_hour"]
    pm_hour = params["pm_hour"]
    hd_dep_count = params["hd_dep_count"]
    wp_char_count_min = params["wp_char_count_min"]
    wp_char_count_max = params["wp_char_count_max"]

    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    household_demographics = get_data(run_config.dataset_path, "household_demographics", run_config.suffix)
    time_dim = get_data(run_config.dataset_path, "time_dim", run_config.suffix)
    web_page = get_data(run_config.dataset_path, "web_page", run_config.suffix)

    # Shared base: web_sales, household_demographics, time_dim, web_page
    # JOIN in FROM order, ALL WHERE conditions after joins
    base = (
        web_sales
        .join(time_dim, left_on="ws_sold_time_sk", right_on="t_time_sk", how="inner")
        .join(household_demographics, left_on="ws_ship_hdemo_sk", right_on="hd_demo_sk", how="inner")
        .join(web_page, left_on="ws_web_page_sk", right_on="wp_web_page_sk", how="inner")
        .filter(
            (pl.col("hd_dep_count") == hd_dep_count)
            & pl.col("wp_char_count").is_between(wp_char_count_min, wp_char_count_max)
        )
    )

    # amc subquery: Count(*) WHERE t_hour BETWEEN am_hour AND am_hour+1
    amc = (
        base
        .filter(pl.col("t_hour").is_between(am_hour, am_hour + 1))
        .select([pl.len()])
        .collect()["len"][0]
    )

    # pmc subquery: Count(*) WHERE t_hour BETWEEN pm_hour AND pm_hour+1
    pmc = (
        base
        .filter(pl.col("t_hour").is_between(pm_hour, pm_hour + 1))
        .select([pl.len()])
        .collect()["len"][0]
    )

    # CAST(amc AS DECIMAL(15,4)) / CAST(pmc AS DECIMAL(15,4))
    am_pm_ratio = float(amc) / float(pmc) if pmc != 0 else None

    result = pl.LazyFrame({"am_pm_ratio": [am_pm_ratio]})

    result = result.sort('am_pm_ratio', descending=False, nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("am_pm_ratio", False)],
        limit=100,
    )
