# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 27."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 27."""
    return """
    SELECT i_item_id,
                   s_state,
                   Grouping(s_state)   g_state,
                   Avg(ss_quantity)    agg1,
                   Avg(ss_list_price)  agg2,
                   Avg(ss_coupon_amt)  agg3,
                   Avg(ss_sales_price) agg4
    FROM   store_sales,
           customer_demographics,
           date_dim,
           store,
           item
    WHERE  ss_sold_date_sk = d_date_sk
           AND ss_item_sk = i_item_sk
           AND ss_store_sk = s_store_sk
           AND ss_cdemo_sk = cd_demo_sk
           AND cd_gender = 'M'
           AND cd_marital_status = 'D'
           AND cd_education_status = 'College'
           AND d_year = 2000
           AND s_state IN ( 'TN', 'TN', 'TN', 'TN',
                            'TN', 'TN' )
    GROUP  BY rollup ( i_item_id, s_state )
    ORDER  BY i_item_id,
              s_state
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 27."""
    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    customer_demographics = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    base_data = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(customer_demographics, left_on="ss_cdemo_sk", right_on="cd_demo_sk")
        # TODO: Bug in cudf_polars with FILTER on Column with NULLs
        # .filter(
        #     (pl.col("cd_gender") == "M") &
        #     (pl.col("cd_marital_status") == "D") &
        #     (pl.col("cd_education_status") == "College") &
        #     (pl.col("d_year") == 2000) &
        #     (pl.col("s_state").is_in(["TN"]))
        # )
        .filter(
            (pl.col("cd_gender").is_not_null() & (pl.col("cd_gender") == "M"))
            & (
                pl.col("cd_marital_status").is_not_null()
                & (pl.col("cd_marital_status") == "D")
            )
            & (
                pl.col("cd_education_status").is_not_null()
                & (pl.col("cd_education_status") == "College")
            )
            & (pl.col("d_year").is_not_null() & (pl.col("d_year") == 2000))
            & (pl.col("s_state").is_not_null() & pl.col("s_state").is_in(["TN"]))
        )
    )
    # ROLLUP simulation: Create both grouping levels
    # Level 1: GROUP BY i_item_id, s_state (detailed level)
    level1 = (
        base_data.group_by(["i_item_id", "s_state"])
        .agg(
            [
                pl.col("ss_quantity").mean().alias("agg1"),
                pl.col("ss_list_price").mean().alias("agg2"),
                pl.col("ss_coupon_amt").mean().alias("agg3"),
                pl.col("ss_sales_price").mean().alias("agg4"),
            ]
        )
        .select(
            [
                "i_item_id",
                "s_state",
                # Dtype == Int64 to match DuckDB
                pl.lit(0, dtype=pl.Int64).alias("g_state"),  # GROUPING(s_state) = 0
                "agg1",
                "agg2",
                "agg3",
                "agg4",
            ]
        )
    )
    # Level 2: GROUP BY i_item_id (rolled up s_state)
    level2 = (
        base_data.group_by(["i_item_id"])
        .agg(
            [
                pl.col("ss_quantity").mean().alias("agg1"),
                pl.col("ss_list_price").mean().alias("agg2"),
                pl.col("ss_coupon_amt").mean().alias("agg3"),
                pl.col("ss_sales_price").mean().alias("agg4"),
            ]
        )
        .select(
            [
                "i_item_id",
                pl.lit(None, dtype=pl.String).alias("s_state"),
                # Dtype == Int64 to match DuckDB
                pl.lit(1, dtype=pl.Int64).alias("g_state"),  # GROUPING(s_state) = 1
                "agg1",
                "agg2",
                "agg3",
                "agg4",
            ]
        )
    )
    return (
        pl.concat([level1, level2])
        .sort(["i_item_id", "s_state"], nulls_last=True)
        .limit(100)
    )
