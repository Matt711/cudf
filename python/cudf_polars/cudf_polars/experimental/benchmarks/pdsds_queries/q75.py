# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 75."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 75."""
    return """
    WITH all_sales
         AS (SELECT d_year,
                    i_brand_id,
                    i_class_id,
                    i_category_id,
                    i_manufact_id,
                    Sum(sales_cnt) AS sales_cnt,
                    Sum(sales_amt) AS sales_amt
             FROM   (SELECT d_year,
                            i_brand_id,
                            i_class_id,
                            i_category_id,
                            i_manufact_id,
                            cs_quantity - COALESCE(cr_return_quantity, 0)        AS
                            sales_cnt,
                            cs_ext_sales_price - COALESCE(cr_return_amount, 0.0) AS
                            sales_amt
                     FROM   catalog_sales
                            JOIN item
                              ON i_item_sk = cs_item_sk
                            JOIN date_dim
                              ON d_date_sk = cs_sold_date_sk
                            LEFT JOIN catalog_returns
                                   ON ( cs_order_number = cr_order_number
                                        AND cs_item_sk = cr_item_sk )
                     WHERE  i_category = 'Men'
                     UNION
                     SELECT d_year,
                            i_brand_id,
                            i_class_id,
                            i_category_id,
                            i_manufact_id,
                            ss_quantity - COALESCE(sr_return_quantity, 0)     AS
                            sales_cnt,
                            ss_ext_sales_price - COALESCE(sr_return_amt, 0.0) AS
                            sales_amt
                     FROM   store_sales
                            JOIN item
                              ON i_item_sk = ss_item_sk
                            JOIN date_dim
                              ON d_date_sk = ss_sold_date_sk
                            LEFT JOIN store_returns
                                   ON ( ss_ticket_number = sr_ticket_number
                                        AND ss_item_sk = sr_item_sk )
                     WHERE  i_category = 'Men'
                     UNION
                     SELECT d_year,
                            i_brand_id,
                            i_class_id,
                            i_category_id,
                            i_manufact_id,
                            ws_quantity - COALESCE(wr_return_quantity, 0)     AS
                            sales_cnt,
                            ws_ext_sales_price - COALESCE(wr_return_amt, 0.0) AS
                            sales_amt
                     FROM   web_sales
                            JOIN item
                              ON i_item_sk = ws_item_sk
                            JOIN date_dim
                              ON d_date_sk = ws_sold_date_sk
                            LEFT JOIN web_returns
                                   ON ( ws_order_number = wr_order_number
                                        AND ws_item_sk = wr_item_sk )
                     WHERE  i_category = 'Men') sales_detail
             GROUP  BY d_year,
                       i_brand_id,
                       i_class_id,
                       i_category_id,
                       i_manufact_id)
    SELECT prev_yr.d_year                        AS prev_year,
                   curr_yr.d_year                        AS year1,
                   curr_yr.i_brand_id,
                   curr_yr.i_class_id,
                   curr_yr.i_category_id,
                   curr_yr.i_manufact_id,
                   prev_yr.sales_cnt                     AS prev_yr_cnt,
                   curr_yr.sales_cnt                     AS curr_yr_cnt,
                   curr_yr.sales_cnt - prev_yr.sales_cnt AS sales_cnt_diff,
                   curr_yr.sales_amt - prev_yr.sales_amt AS sales_amt_diff
    FROM   all_sales curr_yr,
           all_sales prev_yr
    WHERE  curr_yr.i_brand_id = prev_yr.i_brand_id
           AND curr_yr.i_class_id = prev_yr.i_class_id
           AND curr_yr.i_category_id = prev_yr.i_category_id
           AND curr_yr.i_manufact_id = prev_yr.i_manufact_id
           AND curr_yr.d_year = 2002
           AND prev_yr.d_year = 2002 - 1
           AND Cast(curr_yr.sales_cnt AS DECIMAL(17, 2)) / Cast(prev_yr.sales_cnt AS
                                                                    DECIMAL(17, 2))
               < 0.9
    ORDER  BY sales_cnt_diff,
              curr_yr.i_brand_id, -- added for deterministic ordering
              curr_yr.i_class_id, -- added for deterministic ordering
              curr_yr.i_category_id, -- added for deterministic ordering
              curr_yr.i_manufact_id -- added for deterministic ordering
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 75."""
    # Load required tables using scan_parquet for lazy evaluation
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    catalog_returns = get_data(
        run_config.dataset_path, "catalog_returns", run_config.suffix
    )
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(
        run_config.dataset_path, "store_returns", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    web_returns = get_data(run_config.dataset_path, "web_returns", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    # Filter items for Men's category
    men_items = item.filter(pl.col("i_category") == "Men").select(
        ["i_item_sk", "i_brand_id", "i_class_id", "i_category_id", "i_manufact_id"]
    )
    # Create all_sales CTE equivalent - Catalog sales component
    catalog_component = (
        catalog_sales.join(men_items, left_on="cs_item_sk", right_on="i_item_sk")
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(
            catalog_returns,
            left_on=["cs_order_number", "cs_item_sk"],
            right_on=["cr_order_number", "cr_item_sk"],
            how="left",
        )
        .with_columns(
            [
                # TODO: Support coalesce expression
                # (pl.col("cs_quantity") - pl.coalesce([pl.col("cr_return_quantity"), pl.lit(0)])).alias("sales_cnt"),
                # (pl.col("cs_ext_sales_price") - pl.coalesce([pl.col("cr_return_amount"), pl.lit(0.0)])).alias("sales_amt")
                (
                    pl.col("cs_quantity") - pl.col("cr_return_quantity").fill_null(0)
                ).alias("sales_cnt"),
                (
                    pl.col("cs_ext_sales_price")
                    - pl.col("cr_return_amount").fill_null(0.0)
                ).alias("sales_amt"),
            ]
        )
        .select(
            [
                "d_year",
                "i_brand_id",
                "i_class_id",
                "i_category_id",
                "i_manufact_id",
                "sales_cnt",
                "sales_amt",
            ]
        )
    )
    # Create all_sales CTE equivalent - Store sales component
    store_component = (
        store_sales.join(men_items, left_on="ss_item_sk", right_on="i_item_sk")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(
            store_returns,
            left_on=["ss_ticket_number", "ss_item_sk"],
            right_on=["sr_ticket_number", "sr_item_sk"],
            how="left",
        )
        .with_columns(
            [
                # TODO: Support coalesce expression
                # (pl.col("ss_quantity") - pl.coalesce([pl.col("sr_return_quantity"), pl.lit(0)])).alias("sales_cnt"),
                # (pl.col("ss_ext_sales_price") - pl.coalesce([pl.col("sr_return_amt"), pl.lit(0.0)])).alias("sales_amt")
                (
                    pl.col("ss_quantity") - pl.col("sr_return_quantity").fill_null(0)
                ).alias("sales_cnt"),
                (
                    pl.col("ss_ext_sales_price")
                    - pl.col("sr_return_amt").fill_null(0.0)
                ).alias("sales_amt"),
            ]
        )
        .select(
            [
                "d_year",
                "i_brand_id",
                "i_class_id",
                "i_category_id",
                "i_manufact_id",
                "sales_cnt",
                "sales_amt",
            ]
        )
    )
    # Create all_sales CTE equivalent - Web sales component
    web_component = (
        web_sales.join(men_items, left_on="ws_item_sk", right_on="i_item_sk")
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(
            web_returns,
            left_on=["ws_order_number", "ws_item_sk"],
            right_on=["wr_order_number", "wr_item_sk"],
            how="left",
        )
        .with_columns(
            [
                # TODO: Support coalesce expression
                # (pl.col("ws_quantity") - pl.coalesce([pl.col("wr_return_quantity"), pl.lit(0)])).alias("sales_cnt"),
                # (pl.col("ws_ext_sales_price") - pl.coalesce([pl.col("wr_return_amt"), pl.lit(0.0)])).alias("sales_amt")
                (
                    pl.col("ws_quantity") - pl.col("wr_return_quantity").fill_null(0)
                ).alias("sales_cnt"),
                (
                    pl.col("ws_ext_sales_price")
                    - pl.col("wr_return_amt").fill_null(0.0)
                ).alias("sales_amt"),
            ]
        )
        .select(
            [
                "d_year",
                "i_brand_id",
                "i_class_id",
                "i_category_id",
                "i_manufact_id",
                "sales_cnt",
                "sales_amt",
            ]
        )
    )
    # Combine all three components (UNION) and aggregate
    all_sales = (
        pl.concat([catalog_component, store_component, web_component])
        .unique()
        .group_by(
            ["d_year", "i_brand_id", "i_class_id", "i_category_id", "i_manufact_id"]
        )
        .agg(
            [
                pl.col("sales_cnt").count().alias("count_sales_cnt"),
                pl.col("sales_cnt").sum().alias("sum_sales_cnt"),
                pl.col("sales_amt").count().alias("count_sales_amt"),
                pl.col("sales_amt").sum().alias("sum_sales_amt"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("count_sales_cnt") == 0)
                .then(None)
                .otherwise(pl.col("sum_sales_cnt"))
                .alias("sales_cnt"),
                pl.when(pl.col("count_sales_amt") == 0)
                .then(None)
                .otherwise(pl.col("sum_sales_amt"))
                .alias("sales_amt"),
            ]
        )
        .select(
            [
                "d_year",
                "i_brand_id",
                "i_class_id",
                "i_category_id",
                "i_manufact_id",
                "sales_cnt",
                "sales_amt",
            ]
        )
    )
    # Create current year (2002) and previous year (2001) views
    curr_yr = all_sales.filter(pl.col("d_year") == 2002).select(
        [
            pl.col("d_year").alias("curr_d_year"),
            pl.col("i_brand_id").alias("curr_brand_id"),
            pl.col("i_class_id").alias("curr_class_id"),
            pl.col("i_category_id").alias("curr_category_id"),
            pl.col("i_manufact_id").alias("curr_manufact_id"),
            pl.col("sales_cnt").alias("curr_yr_cnt"),
            pl.col("sales_amt").alias("curr_yr_amt"),
        ]
    )
    prev_yr = all_sales.filter(pl.col("d_year") == 2001).select(
        [
            pl.col("d_year").alias("prev_d_year"),
            pl.col("i_brand_id").alias("prev_brand_id"),
            pl.col("i_class_id").alias("prev_class_id"),
            pl.col("i_category_id").alias("prev_category_id"),
            pl.col("i_manufact_id").alias("prev_manufact_id"),
            pl.col("sales_cnt").alias("prev_yr_cnt"),
            pl.col("sales_amt").alias("prev_yr_amt"),
        ]
    )
    # Main query: join current and previous year data
    return (
        curr_yr.join(
            prev_yr,
            left_on=[
                "curr_brand_id",
                "curr_class_id",
                "curr_category_id",
                "curr_manufact_id",
            ],
            right_on=[
                "prev_brand_id",
                "prev_class_id",
                "prev_category_id",
                "prev_manufact_id",
            ],
        )
        # Apply decline filter: current/previous < 0.9
        .filter(
            (pl.col("prev_yr_cnt") > 0)
            & (
                pl.col("curr_yr_cnt").cast(pl.Float64)
                / pl.col("prev_yr_cnt").cast(pl.Float64)
                < 0.9
            )
        )
        # Calculate differences
        .with_columns(
            [
                (pl.col("curr_yr_cnt") - pl.col("prev_yr_cnt")).alias("sales_cnt_diff"),
                (pl.col("curr_yr_amt") - pl.col("prev_yr_amt")).alias("sales_amt_diff"),
            ]
        )
        # Select final columns
        .select(
            [
                pl.col("prev_d_year").alias("prev_year"),
                pl.col("curr_d_year").alias("year1"),
                pl.col("curr_brand_id").alias("i_brand_id"),
                pl.col("curr_class_id").alias("i_class_id"),
                pl.col("curr_category_id").alias("i_category_id"),
                pl.col("curr_manufact_id").alias("i_manufact_id"),
                # Cast -> Decimal to match DuckDB
                pl.col("prev_yr_cnt"),
                pl.col("curr_yr_cnt"),
                pl.col("sales_cnt_diff"),
                "sales_amt_diff",
            ]
        )
        # Sort by sales count difference with tie-breakers for deterministic ordering
        .sort(
            [
                "sales_cnt_diff",
                "i_brand_id",
                "i_class_id",
                "i_category_id",
                "i_manufact_id",
            ],
            nulls_last=True,
        )
        .limit(100)
    )
