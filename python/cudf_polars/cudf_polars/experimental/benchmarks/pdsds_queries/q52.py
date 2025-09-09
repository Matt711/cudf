# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 52."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 52."""
    return """
    SELECT dt.d_year,
                   item.i_brand_id         brand_id,
                   item.i_brand            brand,
                   Sum(ss_ext_sales_price) ext_price
    FROM   date_dim dt,
           store_sales,
           item
    WHERE  dt.d_date_sk = store_sales.ss_sold_date_sk
           AND store_sales.ss_item_sk = item.i_item_sk
           AND item.i_manager_id = 1
           AND dt.d_moy = 11
           AND dt.d_year = 1999
    GROUP  BY dt.d_year,
              item.i_brand,
              item.i_brand_id
    ORDER  BY dt.d_year,
              ext_price DESC,
              brand_id
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 52."""
    # Load tables
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    return (
        store_sales
        # Join with date_dim
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        # Join with item
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        # Apply filters
        .filter(
            (pl.col("i_manager_id") == 1)
            & (pl.col("d_moy") == 11)
            & (pl.col("d_year") == 1999)
        )
        # Group by year, brand, brand_id
        .group_by(["d_year", "i_brand", "i_brand_id"])
        .agg(
            [
                pl.col("ss_ext_sales_price").sum().alias("ext_price_sum"),
                pl.col("ss_ext_sales_price")
                .drop_nulls()
                .count()
                .alias("ext_price_count"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("ext_price_count") > 0)
                .then(pl.col("ext_price_sum"))
                .otherwise(None)
                .alias("sum(ss_ext_sales_price)")
            ]
        )
        .drop(["ext_price_sum", "ext_price_count"])
        # Sort: year ASC, ext_price DESC, brand_id ASC
        .sort(
            ["d_year", "sum(ss_ext_sales_price)", "i_brand_id"],
            nulls_last=True,
            descending=[False, True, False],  # year ASC, ext_price DESC, brand_id ASC
        )
        .limit(100)
        # Final column selection and renaming to match SQL output
        .select(
            [
                pl.col("d_year"),
                pl.col("i_brand_id").alias("brand_id"),
                pl.col("i_brand").alias("brand"),
                pl.col("sum(ss_ext_sales_price)").alias("ext_price"),
            ]
        )
    )
