# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 86."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 86."""
    return """
    SELECT Sum(ws_net_paid)                         AS total_sum,
                   i_category,
                   i_class,
                   Grouping(i_category) + Grouping(i_class) AS lochierarchy,
                   Rank()
                     OVER (
                       partition BY Grouping(i_category)+Grouping(i_class), CASE
                     WHEN Grouping(
                     i_class) = 0 THEN i_category END
                       ORDER BY Sum(ws_net_paid) DESC)      AS rank_within_parent
    FROM   web_sales,
           date_dim d1,
           item
    WHERE  d1.d_month_seq BETWEEN 1183 AND 1183 + 11
           AND d1.d_date_sk = ws_sold_date_sk
           AND i_item_sk = ws_item_sk
    GROUP  BY rollup( i_category, i_class )
    ORDER  BY lochierarchy DESC,
              CASE
                WHEN lochierarchy = 0 THEN i_category
              END,
              rank_within_parent
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 86."""
    # Load tables
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    # Base query with joins and filters
    base_query = (
        web_sales
        # Join with date_dim (d1.d_date_sk = ws_sold_date_sk AND d_month_seq BETWEEN 1183 AND 1194)
        .join(
            date_dim.filter(pl.col("d_month_seq").is_between(1183, 1183 + 11)),
            left_on="ws_sold_date_sk",
            right_on="d_date_sk",
            how="inner",
        )
        # Join with item (i_item_sk = ws_item_sk)
        .join(item, left_on="ws_item_sk", right_on="i_item_sk", how="inner")
    )
    # ROLLUP simulation: Create three levels of aggregation
    # Level 1: Group by both i_category and i_class (lochierarchy = 0)
    level1 = (
        base_query.group_by(["i_category", "i_class"])
        .agg([pl.col("ws_net_paid").sum().alias("total_sum")])
        .with_columns([pl.lit(0, dtype=pl.Int64).alias("lochierarchy")])
        .select(
            ["total_sum", "i_category", pl.col("i_class").cast(pl.Utf8), "lochierarchy"]
        )
    )
    # Level 2: Group by i_category only (lochierarchy = 1)
    level2 = (
        base_query.group_by("i_category")
        .agg([pl.col("ws_net_paid").sum().alias("total_sum")])
        .with_columns([pl.lit(1, dtype=pl.Int64).alias("lochierarchy")])
        .select(
            [
                "total_sum",
                "i_category",
                pl.lit(None).cast(pl.Utf8).alias("i_class"),
                "lochierarchy",
            ]
        )
    )
    # Level 3: Grand total (lochierarchy = 2)
    level3 = (
        base_query.select([pl.col("ws_net_paid").sum().alias("total_sum")])
        .with_columns([pl.lit(2, dtype=pl.Int64).alias("lochierarchy")])
        .select(
            [
                "total_sum",
                pl.lit(None).cast(pl.Utf8).alias("i_category"),
                pl.lit(None).cast(pl.Utf8).alias("i_class"),
                "lochierarchy",
            ]
        )
    )
    # Combine all levels
    combined = pl.concat([level1, level2, level3])
    # Add window function for ranking
    return (
        combined.with_columns(
            [
                # Create partition key for window function
                pl.when(pl.col("lochierarchy") == 0)
                .then(pl.col("i_category"))
                .otherwise(None)
                .alias("partition_category")
            ]
        )
        .with_columns(
            [
                # Add rank within parent using window function
                pl.col("total_sum")
                .rank(method="ordinal", descending=True)
                .over([pl.col("lochierarchy"), pl.col("partition_category")])
                .cast(pl.Int64)
                .alias("rank_within_parent")
            ]
        )
        # Select final columns
        .select(
            ["total_sum", "i_category", "i_class", "lochierarchy", "rank_within_parent"]
        )
        # Order by lochierarchy DESC, then by i_category when lochierarchy = 0, then by rank
        .sort(
            [
                "lochierarchy",
                pl.when(pl.col("lochierarchy") == 0)
                .then(pl.col("i_category"))
                .otherwise(None),
                "rank_within_parent",
            ],
            descending=[True, False, False],
            nulls_last=True,
        )
        # Limit to 100 rows
        .limit(100)
    )
