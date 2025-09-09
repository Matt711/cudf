# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 70."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 70."""
    return """
    SELECT Sum(ss_net_profit)                     AS total_sum,
                   s_state,
                   s_county,
                   Grouping(s_state) + Grouping(s_county) AS lochierarchy,
                   Rank()
                     OVER (
                       partition BY Grouping(s_state)+Grouping(s_county), CASE WHEN
                     Grouping(
                     s_county) = 0 THEN s_state END
                       ORDER BY Sum(ss_net_profit) DESC)  AS rank_within_parent
    FROM   store_sales,
           date_dim d1,
           store
    WHERE  d1.d_month_seq BETWEEN 1200 AND 1200 + 11
           AND d1.d_date_sk = ss_sold_date_sk
           AND s_store_sk = ss_store_sk
           AND s_state IN (SELECT s_state
                           FROM   (SELECT s_state                               AS
                                          s_state,
                                          Rank()
                                            OVER (
                                              partition BY s_state
                                              ORDER BY Sum(ss_net_profit) DESC) AS
                                          ranking
                                   FROM   store_sales,
                                          store,
                                          date_dim
                                   WHERE  d_month_seq BETWEEN 1200 AND 1200 + 11
                                          AND d_date_sk = ss_sold_date_sk
                                          AND s_store_sk = ss_store_sk
                                   GROUP  BY s_state) tmp1
                           WHERE  ranking <= 5)
    GROUP  BY rollup( s_state, s_county )
    ORDER  BY lochierarchy DESC,
              CASE
                WHEN lochierarchy = 0 THEN s_state
              END,
              rank_within_parent
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 70."""
    # Load required tables using scan_parquet for lazy evaluation
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    # Filter date_dim for the target period (d_month_seq between 1200 and 1211)
    target_dates = date_dim.filter(pl.col("d_month_seq").is_between(1200, 1211)).select(
        "d_date_sk"
    )
    # First, find the top 5 states by profit (subquery)
    top_states = (
        store_sales.join(target_dates, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .group_by("s_state")
        .agg(
            [
                pl.col("ss_net_profit").count().alias("count_profit"),
                pl.col("ss_net_profit").sum().alias("sum_profit"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("count_profit") == 0)
                .then(None)
                .otherwise(pl.col("sum_profit"))
                .alias("state_profit")
            ]
        )
        .with_columns(
            [
                pl.col("state_profit")
                .rank(method="min", descending=True)
                .over("s_state")
                .alias("ranking")
            ]
        )
        .filter(pl.col("ranking") <= 5)
        .select("s_state")
    )
    # Main query: join sales data and filter by top states
    base_data = (
        store_sales.join(target_dates, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(top_states, on="s_state")
    )
    # Create ROLLUP equivalent: three levels of aggregation
    # Level 1: Group by state and county (lochierarchy = 0)
    level1 = (
        base_data.group_by(["s_state", "s_county"])
        .agg(
            [
                pl.col("ss_net_profit").count().alias("count_profit"),
                pl.col("ss_net_profit").sum().alias("sum_profit"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("count_profit") == 0)
                .then(None)
                .otherwise(pl.col("sum_profit"))
                .alias("total_sum"),
                pl.lit(0, dtype=pl.Int64).alias("lochierarchy"),
            ]
        )
        .select(["s_state", "s_county", "total_sum", "lochierarchy"])
    )
    # Level 2: Group by state only (lochierarchy = 1)
    level2 = (
        base_data.group_by("s_state")
        .agg(
            [
                pl.col("ss_net_profit").count().alias("count_profit"),
                pl.col("ss_net_profit").sum().alias("sum_profit"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("count_profit") == 0)
                .then(None)
                .otherwise(pl.col("sum_profit"))
                .alias("total_sum"),
                pl.lit(None).cast(pl.String).alias("s_county"),
                pl.lit(1, dtype=pl.Int64).alias("lochierarchy"),
            ]
        )
        .select(["s_state", "s_county", "total_sum", "lochierarchy"])
    )
    # Level 3: Grand total (lochierarchy = 2)
    level3 = (
        base_data.select(
            [
                pl.col("ss_net_profit").count().alias("count_profit"),
                pl.col("ss_net_profit").sum().alias("sum_profit"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("count_profit") == 0)
                .then(None)
                .otherwise(pl.col("sum_profit"))
                .alias("total_sum"),
                pl.lit(None).cast(pl.String).alias("s_state"),
                pl.lit(None).cast(pl.String).alias("s_county"),
                pl.lit(2, dtype=pl.Int64).alias("lochierarchy"),
            ]
        )
        .select(["s_state", "s_county", "total_sum", "lochierarchy"])
    )
    # Combine all levels
    rollup_result = pl.concat([level1, level2, level3])
    # Add window function: rank within parent
    return (
        rollup_result.with_columns(
            [
                pl.col("total_sum")
                .rank(method="dense", descending=True)
                .over(
                    [
                        "lochierarchy",
                        pl.when(pl.col("lochierarchy") == 0)
                        .then(pl.col("s_state"))
                        .otherwise(None),
                    ]
                )
                # Cast -> Int64 to match DuckDB
                .cast(pl.Int64)
                .alias("rank_within_parent")
            ]
        )
        .select(
            ["total_sum", "s_state", "s_county", "lochierarchy", "rank_within_parent"]
        )
        .sort(
            [
                "lochierarchy",
                pl.when(pl.col("lochierarchy") == 0)
                .then(pl.col("s_state"))
                .otherwise(None),
                "rank_within_parent",
            ],
            descending=[True, False, False],
            nulls_last=True,
        )
        .limit(100)
    )
