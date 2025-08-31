# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 39."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 39."""
    return """
    WITH inv
         AS (SELECT w_warehouse_name,
                    w_warehouse_sk,
                    i_item_sk,
                    d_moy,
                    stdev,
                    mean,
                    CASE mean
                      WHEN 0 THEN NULL
                      ELSE stdev / mean
                    END cov
             FROM  (SELECT w_warehouse_name,
                           w_warehouse_sk,
                           i_item_sk,
                           d_moy,
                           Stddev_samp(inv_quantity_on_hand) stdev,
                           Avg(inv_quantity_on_hand)         mean
                    FROM   inventory,
                           item,
                           warehouse,
                           date_dim
                    WHERE  inv_item_sk = i_item_sk
                           AND inv_warehouse_sk = w_warehouse_sk
                           AND inv_date_sk = d_date_sk
                           AND d_year = 2002
                    GROUP  BY w_warehouse_name,
                              w_warehouse_sk,
                              i_item_sk,
                              d_moy) foo
             WHERE  CASE mean
                      WHEN 0 THEN 0
                      ELSE stdev / mean
                    END > 1)
    SELECT inv1.w_warehouse_sk,
           inv1.i_item_sk,
           inv1.d_moy,
           inv1.mean,
           inv1.cov,
           inv2.w_warehouse_sk,
           inv2.i_item_sk,
           inv2.d_moy,
           inv2.mean,
           inv2.cov
    FROM   inv inv1,
           inv inv2
    WHERE  inv1.i_item_sk = inv2.i_item_sk
           AND inv1.w_warehouse_sk = inv2.w_warehouse_sk
           AND inv1.d_moy = 1
           AND inv2.d_moy = 1 + 1
    ORDER  BY inv1.w_warehouse_sk,
              inv1.i_item_sk,
              inv1.d_moy,
              inv1.mean,
              inv1.cov,
              inv2.d_moy,
              inv2.mean,
              inv2.cov;

    WITH inv
         AS (SELECT w_warehouse_name,
                    w_warehouse_sk,
                    i_item_sk,
                    d_moy,
                    stdev,
                    mean,
                    CASE mean
                      WHEN 0 THEN NULL
                      ELSE stdev / mean
                    END cov
             FROM  (SELECT w_warehouse_name,
                           w_warehouse_sk,
                           i_item_sk,
                           d_moy,
                           Stddev_samp(inv_quantity_on_hand) stdev,
                           Avg(inv_quantity_on_hand)         mean
                    FROM   inventory,
                           item,
                           warehouse,
                           date_dim
                    WHERE  inv_item_sk = i_item_sk
                           AND inv_warehouse_sk = w_warehouse_sk
                           AND inv_date_sk = d_date_sk
                           AND d_year = 2002
                    GROUP  BY w_warehouse_name,
                              w_warehouse_sk,
                              i_item_sk,
                              d_moy) foo
             WHERE  CASE mean
                      WHEN 0 THEN 0
                      ELSE stdev / mean
                    END > 1)
    SELECT inv1.w_warehouse_sk,
           inv1.i_item_sk,
           inv1.d_moy,
           inv1.mean,
           inv1.cov,
           inv2.w_warehouse_sk,
           inv2.i_item_sk,
           inv2.d_moy,
           inv2.mean,
           inv2.cov
    FROM   inv inv1,
           inv inv2
    WHERE  inv1.i_item_sk = inv2.i_item_sk
           AND inv1.w_warehouse_sk = inv2.w_warehouse_sk
           AND inv1.d_moy = 1
           AND inv2.d_moy = 1 + 1
           AND inv1.cov > 1.5
    ORDER  BY inv1.w_warehouse_sk,
              inv1.i_item_sk,
              inv1.d_moy,
              inv1.mean,
              inv1.cov,
              inv2.d_moy,
              inv2.mean,
              inv2.cov;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 39."""
    # Load tables
    inventory = get_data(run_config.dataset_path, "inventory", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    warehouse = get_data(run_config.dataset_path, "warehouse", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    # Create the equivalent of the 'inv' CTE
    # First, create the base aggregation (equivalent to the inner SELECT in the CTE)
    base_agg = (
        inventory.join(item, left_on="inv_item_sk", right_on="i_item_sk")
        .join(warehouse, left_on="inv_warehouse_sk", right_on="w_warehouse_sk")
        .join(date_dim, left_on="inv_date_sk", right_on="d_date_sk")
        # TODO: .filter(pl.col("d_year") == 2002) should work
        # There's some bug in cudf_polars with FILTER on Column with NULLs
        .filter(pl.col("d_year").is_not_null() & (pl.col("d_year") == 2002))
        .group_by(["w_warehouse_name", "inv_warehouse_sk", "inv_item_sk", "d_moy"])
        .agg(
            [
                pl.col("inv_quantity_on_hand").std().alias("stdev"),
                pl.col("inv_quantity_on_hand").mean().alias("mean"),
            ]
        )
    )
    # Create the 'inv' CTE equivalent with COV calculation and filtering
    inv_cte = base_agg.with_columns(
        [
            pl.when(pl.col("mean") == 0)
            .then(None)
            .otherwise(pl.col("stdev") / pl.col("mean"))
            .alias("cov")
        ]
    ).filter(
        pl.when(pl.col("mean") == 0)
        .then(False)  # noqa: FBT003
        .otherwise(pl.col("stdev") / pl.col("mean") > 1.0)
    )
    # Self-join to compare month 1 vs month 2 (equivalent to inv inv1, inv inv2)
    inv1 = inv_cte.filter(pl.col("d_moy") == 1).select(
        [
            pl.col("inv_warehouse_sk").alias("w_warehouse_sk"),
            pl.col("inv_item_sk").alias("i_item_sk"),
            "d_moy",
            "mean",
            "cov",
        ]
    )
    inv2 = inv_cte.filter(pl.col("d_moy") == 2).select(
        [
            pl.col("inv_warehouse_sk").alias("w_warehouse_sk_1"),
            pl.col("inv_item_sk").alias("i_item_sk_1"),
            pl.col("d_moy").alias("d_moy_1"),
            pl.col("mean").alias("mean_1"),
            pl.col("cov").alias("cov_1"),
        ]
    )
    # Join inv1 and inv2 on warehouse and item
    return (
        inv1.filter(pl.col("cov") > 1.5)
        .join(
            inv2,
            left_on=["w_warehouse_sk", "i_item_sk"],
            right_on=["w_warehouse_sk_1", "i_item_sk_1"],
            how="inner",
        )
        .with_columns(
            [
                pl.col("w_warehouse_sk").alias("w_warehouse_sk_1"),
                pl.col("i_item_sk").alias("i_item_sk_1"),
            ]
        )
        .select(
            [
                "w_warehouse_sk",
                "i_item_sk",
                "d_moy",
                "mean",
                "cov",
                "w_warehouse_sk_1",
                "i_item_sk_1",
                "d_moy_1",
                "mean_1",
                "cov_1",
            ]
        )
        .sort(
            [
                "w_warehouse_sk",
                "i_item_sk",
                "d_moy",
                "mean",
                "cov",
                "w_warehouse_sk_1",
                "i_item_sk_1",
                "d_moy_1",
                "mean_1",
                "cov_1",
            ]
        )
    )
