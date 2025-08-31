# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 71."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 71."""
    return """
    SELECT i_brand_id     brand_id,
           i_brand        brand,
           t_hour,
           t_minute,
           Sum(ext_price) ext_price
    FROM   item,
           (SELECT ws_ext_sales_price AS ext_price,
                   ws_sold_date_sk    AS sold_date_sk,
                   ws_item_sk         AS sold_item_sk,
                   ws_sold_time_sk    AS time_sk
            FROM   web_sales,
                   date_dim
            WHERE  d_date_sk = ws_sold_date_sk
                   AND d_moy = 11
                   AND d_year = 2001
            UNION ALL
            SELECT cs_ext_sales_price AS ext_price,
                   cs_sold_date_sk    AS sold_date_sk,
                   cs_item_sk         AS sold_item_sk,
                   cs_sold_time_sk    AS time_sk
            FROM   catalog_sales,
                   date_dim
            WHERE  d_date_sk = cs_sold_date_sk
                   AND d_moy = 11
                   AND d_year = 2001
            UNION ALL
            SELECT ss_ext_sales_price AS ext_price,
                   ss_sold_date_sk    AS sold_date_sk,
                   ss_item_sk         AS sold_item_sk,
                   ss_sold_time_sk    AS time_sk
            FROM   store_sales,
                   date_dim
            WHERE  d_date_sk = ss_sold_date_sk
                   AND d_moy = 11
                   AND d_year = 2001) AS tmp,
           time_dim
    WHERE  sold_item_sk = i_item_sk
           AND i_manager_id = 1
           AND time_sk = t_time_sk
           AND ( t_meal_time = 'breakfast'
                  OR t_meal_time = 'dinner' )
    GROUP  BY i_brand,
              i_brand_id,
              t_hour,
              t_minute
    ORDER  BY --ext_price DESC,
              i_brand_id,
              i_brand, t_hour, t_minute; --added for extra sort determinism
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 71."""
    # Load required tables using scan_parquet for lazy evaluation
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    time_dim = get_data(run_config.dataset_path, "time_dim", run_config.suffix)
    # Filter date_dim for November 2001
    target_dates = (
        date_dim
        # TODO: There's some bug in cudf_polars with FILTER on Column with NULLs
        # .filter(
        #     (pl.col("d_moy") == 11) &
        #     (pl.col("d_year") == 2001)
        # )
        .filter(
            pl.col("d_moy").is_not_null()
            & (pl.col("d_moy") == 11)
            & pl.col("d_year").is_not_null()
            & (pl.col("d_year") == 2001)
        ).select("d_date_sk")
    )
    # Create UNION ALL equivalent: combine all three sales channels
    # Web sales component
    web_component = web_sales.join(
        target_dates, left_on="ws_sold_date_sk", right_on="d_date_sk"
    ).select(
        [
            pl.col("ws_ext_sales_price").alias("ext_price"),
            pl.col("ws_sold_date_sk").alias("sold_date_sk"),
            pl.col("ws_item_sk").alias("sold_item_sk"),
            pl.col("ws_sold_time_sk").alias("time_sk"),
        ]
    )
    # Catalog sales component
    catalog_component = catalog_sales.join(
        target_dates, left_on="cs_sold_date_sk", right_on="d_date_sk"
    ).select(
        [
            pl.col("cs_ext_sales_price").alias("ext_price"),
            pl.col("cs_sold_date_sk").alias("sold_date_sk"),
            pl.col("cs_item_sk").alias("sold_item_sk"),
            pl.col("cs_sold_time_sk").alias("time_sk"),
        ]
    )
    # Store sales component
    store_component = store_sales.join(
        target_dates, left_on="ss_sold_date_sk", right_on="d_date_sk"
    ).select(
        [
            pl.col("ss_ext_sales_price").alias("ext_price"),
            pl.col("ss_sold_date_sk").alias("sold_date_sk"),
            pl.col("ss_item_sk").alias("sold_item_sk"),
            pl.col("ss_sold_time_sk").alias("time_sk"),
        ]
    )
    # Combine all three components (UNION ALL)
    combined_sales = pl.concat([web_component, catalog_component, store_component])
    # Filter items by manager_id = 1
    filtered_items = (
        item
        # TODO: There's some bug in cudf_polars with FILTER on Column with NULLs
        # .filter(pl.col("i_manager_id") == 1)
        .filter(
            pl.col("i_manager_id").is_not_null() & (pl.col("i_manager_id") == 1)
        ).select(["i_item_sk", "i_brand_id", "i_brand"])
    )
    # Filter time_dim for breakfast or dinner
    filtered_time = (
        time_dim
        # TODO: There's some bug in cudf_polars with FILTER on Column with NULLs
        # .filter(pl.col("t_meal_time").is_in(["breakfast", "dinner"]))
        .filter(
            pl.col("t_meal_time").is_not_null()
            & pl.col("t_meal_time").is_in(["breakfast", "dinner"])
        ).select(["t_time_sk", "t_hour", "t_minute"])
    )
    # Main query: join all components and aggregate
    return (
        combined_sales
        # Join with filtered items
        .join(filtered_items, left_on="sold_item_sk", right_on="i_item_sk")
        # Join with filtered time
        .join(filtered_time, left_on="time_sk", right_on="t_time_sk")
        # Group by brand, brand_id, hour, minute
        .group_by(["i_brand", "i_brand_id", "t_hour", "t_minute"])
        .agg(
            [
                pl.col("ext_price").count().alias("count_price"),
                pl.col("ext_price").sum().alias("sum_price"),
                pl.col("i_brand_id").first().cast(pl.Int32).alias("brand_id_int"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("count_price") == 0)
                .then(None)
                .otherwise(pl.col("sum_price"))
                .alias("ext_price")
            ]
        )
        # Select and order columns to match SQL output
        .select(
            [
                pl.col("brand_id_int").alias("brand_id"),
                pl.col("i_brand").alias("brand"),
                "t_hour",
                "t_minute",
                "ext_price",
            ]
        )
        # .sort(["ext_price", "brand_id"], descending=[True, False], nulls_last=True)
        .sort(
            ["brand_id", "brand", "t_hour", "t_minute"],
            descending=[False, False, False, False],
            nulls_last=True,
        )
    )
