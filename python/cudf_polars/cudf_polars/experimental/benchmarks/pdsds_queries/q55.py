# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 55."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 55."""
    return """
    SELECT i_brand_id              brand_id,
                   i_brand                 brand,
                   Sum(ss_ext_sales_price) ext_price
    FROM   date_dim,
           store_sales,
           item
    WHERE  d_date_sk = ss_sold_date_sk
           AND ss_item_sk = i_item_sk
           AND i_manager_id = 33
           AND d_moy = 12
           AND d_year = 1998
    GROUP  BY i_brand,
              i_brand_id
    ORDER  BY ext_price DESC,
              i_brand_id
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 55."""
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
        # TODO: There's some bug in cudf_polars with FILTER on Column with NULLs
        # .filter(
        #     (pl.col("i_manager_id") == 33) &
        #     (pl.col("d_moy") == 12) &
        #     (pl.col("d_year") == 1998)
        # )
        .filter(
            (pl.col("i_manager_id").is_not_null() & (pl.col("i_manager_id") == 33))
            & (pl.col("d_moy").is_not_null() & (pl.col("d_moy") == 12))
            & (pl.col("d_year").is_not_null() & (pl.col("d_year") == 1998))
        )
        # Group by brand and brand_id
        .group_by(["i_brand", "i_brand_id"])
        .agg(
            [
                # Simple sum of extended sales price
                pl.col("ss_ext_sales_price").sum().alias("sum_ext_sales_raw")
            ]
        )
        # Apply ternary logic after groupby
        .with_columns(
            [
                # Null-safe sum using post-ternary approach
                pl.when(pl.col("sum_ext_sales_raw").is_not_null())
                .then(pl.col("sum_ext_sales_raw"))
                .otherwise(None)
                .alias("sum(ss_ext_sales_price)")
            ]
        )
        # Sort: ext_price DESC, brand_id ASC
        .sort(
            ["sum(ss_ext_sales_price)", "i_brand_id"],
            nulls_last=True,
            descending=[True, False],  # ext_price DESC, brand_id ASC
        )
        .limit(100)
        # Final column selection and renaming to match SQL output
        .select(
            [
                pl.col("i_brand_id").alias("brand_id"),
                pl.col("i_brand").alias("brand"),
                pl.col("sum(ss_ext_sales_price)").alias("ext_price"),
            ]
        )
    )
