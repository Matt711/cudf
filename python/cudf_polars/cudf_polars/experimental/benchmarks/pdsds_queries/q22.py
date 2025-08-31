# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 22."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 22."""
    return """
    SELECT i_product_name,
                   i_brand,
                   i_class,
                   i_category,
                   Avg(inv_quantity_on_hand) qoh
    FROM   inventory,
           date_dim,
           item,
           warehouse
    WHERE  inv_date_sk = d_date_sk
           AND inv_item_sk = i_item_sk
           AND inv_warehouse_sk = w_warehouse_sk
           AND d_month_seq BETWEEN 1205 AND 1205 + 11
    GROUP  BY rollup( i_product_name, i_brand, i_class, i_category )
    ORDER  BY qoh,
              i_product_name,
              i_brand,
              i_class,
              i_category
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 22."""
    # Load tables
    inventory = get_data(run_config.dataset_path, "inventory", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    warehouse = get_data(run_config.dataset_path, "warehouse", run_config.suffix)
    base_data = (
        inventory.join(date_dim, left_on="inv_date_sk", right_on="d_date_sk")
        .join(item, left_on="inv_item_sk", right_on="i_item_sk")
        .join(warehouse, left_on="inv_warehouse_sk", right_on="w_warehouse_sk")
        .filter(pl.col("d_month_seq").is_between(1205, 1205 + 11))
    )
    # ROLLUP simulation: Create all grouping levels
    # Level 1: All columns
    level1 = (
        base_data.group_by(["i_product_name", "i_brand", "i_class", "i_category"])
        .agg([pl.col("inv_quantity_on_hand").mean().alias("qoh")])
        .select(["i_product_name", "i_brand", "i_class", "i_category", "qoh"])
    )
    # Level 2: Drop i_category
    level2 = (
        base_data.group_by(["i_product_name", "i_brand", "i_class"])
        .agg([pl.col("inv_quantity_on_hand").mean().alias("qoh")])
        .select(
            [
                "i_product_name",
                "i_brand",
                "i_class",
                pl.lit(None, dtype=pl.String).alias("i_category"),
                "qoh",
            ]
        )
    )
    # Level 3: Drop i_class
    level3 = (
        base_data.group_by(["i_product_name", "i_brand"])
        .agg([pl.col("inv_quantity_on_hand").mean().alias("qoh")])
        .select(
            [
                "i_product_name",
                "i_brand",
                pl.lit(None, dtype=pl.String).alias("i_class"),
                pl.lit(None, dtype=pl.String).alias("i_category"),
                "qoh",
            ]
        )
    )
    # Level 4: Drop i_brand
    level4 = (
        base_data.group_by(["i_product_name"])
        .agg([pl.col("inv_quantity_on_hand").mean().alias("qoh")])
        .select(
            [
                "i_product_name",
                pl.lit(None, dtype=pl.String).alias("i_brand"),
                pl.lit(None, dtype=pl.String).alias("i_class"),
                pl.lit(None, dtype=pl.String).alias("i_category"),
                "qoh",
            ]
        )
    )
    # Level 5: Drop all (grand total)
    level5 = base_data.select(
        [pl.col("inv_quantity_on_hand").mean().alias("qoh")]
    ).select(
        [
            pl.lit(None, dtype=pl.String).alias("i_product_name"),
            pl.lit(None, dtype=pl.String).alias("i_brand"),
            pl.lit(None, dtype=pl.String).alias("i_class"),
            pl.lit(None, dtype=pl.String).alias("i_category"),
            "qoh",
        ]
    )
    return (
        pl.concat([level1, level2, level3, level4, level5])
        .sort(
            ["qoh", "i_product_name", "i_brand", "i_class", "i_category"],
            nulls_last=True,
        )
        .limit(100)
    )
