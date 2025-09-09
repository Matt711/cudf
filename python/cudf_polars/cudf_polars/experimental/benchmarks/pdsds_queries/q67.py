# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 67."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 67."""
    return """
    select *
    from (select i_category
                ,i_class
                ,i_brand
                ,i_product_name
                ,d_year
                ,d_qoy
                ,d_moy
                ,s_store_id
                ,sumsales
                ,rank() over (partition by i_category order by sumsales desc) rk
          from (select i_category
                      ,i_class
                      ,i_brand
                      ,i_product_name
                      ,d_year
                      ,d_qoy
                      ,d_moy
                      ,s_store_id
                      ,sum(coalesce(ss_sales_price*ss_quantity,0)) sumsales
                from store_sales
                    ,date_dim
                    ,store
                    ,item
           where  ss_sold_date_sk=d_date_sk
              and ss_item_sk=i_item_sk
              and ss_store_sk = s_store_sk
              and d_month_seq between 1181 and 1181+11
           group by  rollup(i_category, i_class, i_brand, i_product_name, d_year, d_qoy, d_moy,s_store_id))dw1) dw2
    where rk <= 100
    order by i_category
            ,i_class
            ,i_brand
            ,i_product_name
            ,d_year
            ,d_qoy
            ,d_moy
            ,s_store_id
            ,sumsales
            ,rk
    limit 100
    ;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 67."""
    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    # Base data with joins and filters
    base_data = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .filter(pl.col("d_month_seq").is_between(1181, 1181 + 11))
        .with_columns(
            [
                # Calculate sales amount, handling nulls like COALESCE
                pl.when(
                    pl.col("ss_sales_price").is_null() | pl.col("ss_quantity").is_null()
                )
                .then(0)
                .otherwise(pl.col("ss_sales_price") * pl.col("ss_quantity"))
                .alias("sales_amount")
            ]
        )
    )
    # Simulate ROLLUP by creating multiple aggregation levels
    # Level 1: Full detail
    level1 = (
        base_data.group_by(
            [
                "i_category",
                "i_class",
                "i_brand",
                "i_product_name",
                "d_year",
                "d_qoy",
                "d_moy",
                "s_store_id",
            ]
        )
        .agg([pl.col("sales_amount").sum().alias("sumsales")])
        .select(
            [
                "i_category",
                "i_class",
                "i_brand",
                "i_product_name",
                "d_year",
                "d_qoy",
                "d_moy",
                "s_store_id",
                "sumsales",
            ]
        )
    )
    # Level 2: Roll up s_store_id (set to NULL)
    level2 = (
        base_data.group_by(
            [
                "i_category",
                "i_class",
                "i_brand",
                "i_product_name",
                "d_year",
                "d_qoy",
                "d_moy",
            ]
        )
        .agg([pl.col("sales_amount").sum().alias("sumsales")])
        .with_columns([pl.lit(None, dtype=pl.String).alias("s_store_id")])
        .select(
            [
                "i_category",
                "i_class",
                "i_brand",
                "i_product_name",
                "d_year",
                "d_qoy",
                "d_moy",
                "s_store_id",
                "sumsales",
            ]
        )
    )
    # Level 3: Roll up d_moy as well
    level3 = (
        base_data.group_by(
            ["i_category", "i_class", "i_brand", "i_product_name", "d_year", "d_qoy"]
        )
        .agg([pl.col("sales_amount").sum().alias("sumsales")])
        .with_columns(
            [
                pl.lit(None, dtype=pl.Int32).alias("d_moy"),
                pl.lit(None, dtype=pl.String).alias("s_store_id"),
            ]
        )
        .select(
            [
                "i_category",
                "i_class",
                "i_brand",
                "i_product_name",
                "d_year",
                "d_qoy",
                "d_moy",
                "s_store_id",
                "sumsales",
            ]
        )
    )
    # Level 4: Roll up d_qoy as well
    level4 = (
        base_data.group_by(
            ["i_category", "i_class", "i_brand", "i_product_name", "d_year"]
        )
        .agg([pl.col("sales_amount").sum().alias("sumsales")])
        .with_columns(
            [
                pl.lit(None, dtype=pl.Int32).alias("d_qoy"),
                pl.lit(None, dtype=pl.Int32).alias("d_moy"),
                pl.lit(None, dtype=pl.String).alias("s_store_id"),
            ]
        )
        .select(
            [
                "i_category",
                "i_class",
                "i_brand",
                "i_product_name",
                "d_year",
                "d_qoy",
                "d_moy",
                "s_store_id",
                "sumsales",
            ]
        )
    )
    # Level 5: Roll up d_year as well
    level5 = (
        base_data.group_by(["i_category", "i_class", "i_brand", "i_product_name"])
        .agg([pl.col("sales_amount").sum().alias("sumsales")])
        .with_columns(
            [
                pl.lit(None, dtype=pl.Int32).alias("d_year"),
                pl.lit(None, dtype=pl.Int32).alias("d_qoy"),
                pl.lit(None, dtype=pl.Int32).alias("d_moy"),
                pl.lit(None, dtype=pl.String).alias("s_store_id"),
            ]
        )
        .select(
            [
                "i_category",
                "i_class",
                "i_brand",
                "i_product_name",
                "d_year",
                "d_qoy",
                "d_moy",
                "s_store_id",
                "sumsales",
            ]
        )
    )
    # Level 6: Roll up i_product_name as well
    level6 = (
        base_data.group_by(["i_category", "i_class", "i_brand"])
        .agg([pl.col("sales_amount").sum().alias("sumsales")])
        .with_columns(
            [
                pl.lit(None, dtype=pl.String).alias("i_product_name"),
                pl.lit(None, dtype=pl.Int32).alias("d_year"),
                pl.lit(None, dtype=pl.Int32).alias("d_qoy"),
                pl.lit(None, dtype=pl.Int32).alias("d_moy"),
                pl.lit(None, dtype=pl.String).alias("s_store_id"),
            ]
        )
        .select(
            [
                "i_category",
                "i_class",
                "i_brand",
                "i_product_name",
                "d_year",
                "d_qoy",
                "d_moy",
                "s_store_id",
                "sumsales",
            ]
        )
    )
    # Level 7: Roll up i_brand as well
    level7 = (
        base_data.group_by(["i_category", "i_class"])
        .agg([pl.col("sales_amount").sum().alias("sumsales")])
        .with_columns(
            [
                pl.lit(None, dtype=pl.String).alias("i_brand"),
                pl.lit(None, dtype=pl.String).alias("i_product_name"),
                pl.lit(None, dtype=pl.Int32).alias("d_year"),
                pl.lit(None, dtype=pl.Int32).alias("d_qoy"),
                pl.lit(None, dtype=pl.Int32).alias("d_moy"),
                pl.lit(None, dtype=pl.String).alias("s_store_id"),
            ]
        )
        .select(
            [
                "i_category",
                "i_class",
                "i_brand",
                "i_product_name",
                "d_year",
                "d_qoy",
                "d_moy",
                "s_store_id",
                "sumsales",
            ]
        )
    )
    # Level 8: Roll up i_class as well
    level8 = (
        base_data.group_by(["i_category"])
        .agg([pl.col("sales_amount").sum().alias("sumsales")])
        .with_columns(
            [
                pl.lit(None, dtype=pl.String).alias("i_class"),
                pl.lit(None, dtype=pl.String).alias("i_brand"),
                pl.lit(None, dtype=pl.String).alias("i_product_name"),
                pl.lit(None, dtype=pl.Int32).alias("d_year"),
                pl.lit(None, dtype=pl.Int32).alias("d_qoy"),
                pl.lit(None, dtype=pl.Int32).alias("d_moy"),
                pl.lit(None, dtype=pl.String).alias("s_store_id"),
            ]
        )
        .select(
            [
                "i_category",
                "i_class",
                "i_brand",
                "i_product_name",
                "d_year",
                "d_qoy",
                "d_moy",
                "s_store_id",
                "sumsales",
            ]
        )
    )
    # Level 9: Grand total (roll up everything)
    level9 = (
        base_data.select([pl.col("sales_amount").sum().alias("sumsales")])
        .with_columns(
            [
                pl.lit(None, dtype=pl.String).alias("i_category"),
                pl.lit(None, dtype=pl.String).alias("i_class"),
                pl.lit(None, dtype=pl.String).alias("i_brand"),
                pl.lit(None, dtype=pl.String).alias("i_product_name"),
                pl.lit(None, dtype=pl.Int32).alias("d_year"),
                pl.lit(None, dtype=pl.Int32).alias("d_qoy"),
                pl.lit(None, dtype=pl.Int32).alias("d_moy"),
                pl.lit(None, dtype=pl.String).alias("s_store_id"),
            ]
        )
        .select(
            [
                "i_category",
                "i_class",
                "i_brand",
                "i_product_name",
                "d_year",
                "d_qoy",
                "d_moy",
                "s_store_id",
                "sumsales",
            ]
        )
    )
    # Combine all rollup levels
    rollup_data = pl.concat(
        [level1, level2, level3, level4, level5, level6, level7, level8, level9]
    )
    # Add ranking within each category
    ranked_data = rollup_data.with_columns(
        [
            pl.col("sumsales")
            .rank(method="dense", descending=True)
            .over("i_category")
            # Cast -> Int64 to match DuckDB
            .cast(pl.Int64)
            .alias("rk")
        ]
    )
    # Filter for top 100 ranks and apply final sorting
    return (
        ranked_data.filter(pl.col("rk") <= 100)
        .sort(
            [
                "i_category",
                "i_class",
                "i_brand",
                "i_product_name",
                "d_year",
                "d_qoy",
                "d_moy",
                "s_store_id",
                "sumsales",
                "rk",
            ],
            nulls_last=True,
            descending=[False] * 10,
        )
        .limit(100)
    )
