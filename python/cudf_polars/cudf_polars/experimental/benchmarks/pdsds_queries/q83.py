# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 83."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 83."""
    return """
    WITH sr_items
         AS (SELECT i_item_id               item_id,
                    Sum(sr_return_quantity) sr_item_qty
             FROM   store_returns,
                    item,
                    date_dim
             WHERE  sr_item_sk = i_item_sk
                    AND d_date IN (SELECT d_date
                                   FROM   date_dim
                                   WHERE  d_week_seq IN (SELECT d_week_seq
                                                         FROM   date_dim
                                                         WHERE
                                          d_date IN ( '1999-06-30',
                                                      '1999-08-28',
                                                      '1999-11-18'
                                                    )))
                    AND sr_returned_date_sk = d_date_sk
             GROUP  BY i_item_id),
         cr_items
         AS (SELECT i_item_id               item_id,
                    Sum(cr_return_quantity) cr_item_qty
             FROM   catalog_returns,
                    item,
                    date_dim
             WHERE  cr_item_sk = i_item_sk
                    AND d_date IN (SELECT d_date
                                   FROM   date_dim
                                   WHERE  d_week_seq IN (SELECT d_week_seq
                                                         FROM   date_dim
                                                         WHERE
                                          d_date IN ( '1999-06-30',
                                                      '1999-08-28',
                                                      '1999-11-18'
                                                    )))
                    AND cr_returned_date_sk = d_date_sk
             GROUP  BY i_item_id),
         wr_items
         AS (SELECT i_item_id               item_id,
                    Sum(wr_return_quantity) wr_item_qty
             FROM   web_returns,
                    item,
                    date_dim
             WHERE  wr_item_sk = i_item_sk
                    AND d_date IN (SELECT d_date
                                   FROM   date_dim
                                   WHERE  d_week_seq IN (SELECT d_week_seq
                                                         FROM   date_dim
                                                         WHERE
                                          d_date IN ( '1999-06-30',
                                                      '1999-08-28',
                                                      '1999-11-18'
                                                    )))
                    AND wr_returned_date_sk = d_date_sk
             GROUP  BY i_item_id)
    SELECT sr_items.item_id,
                   sr_item_qty,
                   sr_item_qty / ( sr_item_qty + cr_item_qty + wr_item_qty ) / 3.0 *
                   100 sr_dev,
                   cr_item_qty,
                   cr_item_qty / ( sr_item_qty + cr_item_qty + wr_item_qty ) / 3.0 *
                   100 cr_dev,
                   wr_item_qty,
                   wr_item_qty / ( sr_item_qty + cr_item_qty + wr_item_qty ) / 3.0 *
                   100 wr_dev,
                   ( sr_item_qty + cr_item_qty + wr_item_qty ) / 3.0
                   average
    FROM   sr_items,
           cr_items,
           wr_items
    WHERE  sr_items.item_id = cr_items.item_id
           AND sr_items.item_id = wr_items.item_id
    ORDER  BY sr_items.item_id,
              sr_item_qty
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 83."""
    # Load required tables using scan_parquet for lazy evaluation
    store_returns = get_data(
        run_config.dataset_path, "store_returns", run_config.suffix
    )
    catalog_returns = get_data(
        run_config.dataset_path, "catalog_returns", run_config.suffix
    )
    web_returns = get_data(run_config.dataset_path, "web_returns", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    # First, find the week sequences for the specific dates using individual comparisons
    week_sequences = (
        date_dim.filter(
            (pl.col("d_date") == pl.date(1999, 6, 30))
            | (pl.col("d_date") == pl.date(1999, 8, 28))
            | (pl.col("d_date") == pl.date(1999, 11, 18))
        )
        .select("d_week_seq")
        .unique()
    )
    # Get all dates in those week sequences
    valid_dates = (
        date_dim.join(week_sequences, on="d_week_seq")
        .select("d_date", "d_date_sk")
        .unique()
    )
    # CTE 1: sr_items (store returns)
    sr_items = (
        store_returns.join(
            item.select(["i_item_sk", "i_item_id"]),
            left_on="sr_item_sk",
            right_on="i_item_sk",
        )
        .join(valid_dates, left_on="sr_returned_date_sk", right_on="d_date_sk")
        .group_by("i_item_id")
        .agg(
            [
                pl.col("sr_return_quantity").count().alias("sr_item_qty_count"),
                pl.col("sr_return_quantity").sum().alias("sr_item_qty_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("sr_item_qty_count") == 0)
                .then(None)
                .otherwise(pl.col("sr_item_qty_sum"))
                .alias("sr_item_qty")
            ]
        )
        .select(["i_item_id", "sr_item_qty"])
    )
    # CTE 2: cr_items (catalog returns)
    cr_items = (
        catalog_returns.join(
            item.select(["i_item_sk", "i_item_id"]),
            left_on="cr_item_sk",
            right_on="i_item_sk",
        )
        .join(valid_dates, left_on="cr_returned_date_sk", right_on="d_date_sk")
        .group_by("i_item_id")
        .agg(
            [
                pl.col("cr_return_quantity").count().alias("cr_item_qty_count"),
                pl.col("cr_return_quantity").sum().alias("cr_item_qty_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("cr_item_qty_count") == 0)
                .then(None)
                .otherwise(pl.col("cr_item_qty_sum"))
                .alias("cr_item_qty")
            ]
        )
        .select(["i_item_id", "cr_item_qty"])
    )
    # CTE 3: wr_items (web returns)
    wr_items = (
        web_returns.join(
            item.select(["i_item_sk", "i_item_id"]),
            left_on="wr_item_sk",
            right_on="i_item_sk",
        )
        .join(valid_dates, left_on="wr_returned_date_sk", right_on="d_date_sk")
        .group_by("i_item_id")
        .agg(
            [
                pl.col("wr_return_quantity").count().alias("wr_item_qty_count"),
                pl.col("wr_return_quantity").sum().alias("wr_item_qty_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("wr_item_qty_count") == 0)
                .then(None)
                .otherwise(pl.col("wr_item_qty_sum"))
                .alias("wr_item_qty")
            ]
        )
        .select(["i_item_id", "wr_item_qty"])
    )
    # Main query: Join all three CTEs and calculate deviations
    return (
        sr_items.join(
            cr_items, on="i_item_id", suffix="_cr"
        )  # Inner join to match SQL behavior
        .join(
            wr_items, on="i_item_id", suffix="_wr"
        )  # Inner join to match SQL behavior
        .with_columns(
            [
                # If any quantity is null, make total and average null
                pl.when(
                    pl.col("sr_item_qty").is_null()
                    | pl.col("cr_item_qty").is_null()
                    | pl.col("wr_item_qty").is_null()
                )
                .then(None)
                .otherwise(
                    pl.col("sr_item_qty")
                    + pl.col("cr_item_qty")
                    + pl.col("wr_item_qty")
                )
                .alias("total_qty"),
                pl.when(
                    pl.col("sr_item_qty").is_null()
                    | pl.col("cr_item_qty").is_null()
                    | pl.col("wr_item_qty").is_null()
                )
                .then(None)
                .otherwise(
                    (
                        pl.col("sr_item_qty")
                        + pl.col("cr_item_qty")
                        + pl.col("wr_item_qty")
                    )
                    / 3.0
                )
                .cast(pl.Float64)
                .alias("average"),
            ]
        )
        .with_columns(
            [
                # Calculate deviations as percentages, all null if any quantity is null
                pl.when(
                    pl.col("sr_item_qty").is_null()
                    | pl.col("cr_item_qty").is_null()
                    | pl.col("wr_item_qty").is_null()
                )
                .then(None)
                .otherwise(pl.col("sr_item_qty") / pl.col("total_qty") / 3.0 * 100)
                # Cast -> Float64 to match DuckDB
                .cast(pl.Float64)
                .alias("sr_dev"),
                pl.when(
                    pl.col("sr_item_qty").is_null()
                    | pl.col("cr_item_qty").is_null()
                    | pl.col("wr_item_qty").is_null()
                )
                .then(None)
                .otherwise(pl.col("cr_item_qty") / pl.col("total_qty") / 3.0 * 100)
                .cast(pl.Float64)
                .alias("cr_dev"),
                pl.when(
                    pl.col("sr_item_qty").is_null()
                    | pl.col("cr_item_qty").is_null()
                    | pl.col("wr_item_qty").is_null()
                )
                .then(None)
                .otherwise(pl.col("wr_item_qty") / pl.col("total_qty") / 3.0 * 100)
                .cast(pl.Float64)
                .alias("wr_dev"),
            ]
        )
        .select(
            [
                pl.col("i_item_id").alias("item_id"),
                "sr_item_qty",
                "sr_dev",
                "cr_item_qty",
                "cr_dev",
                "wr_item_qty",
                "wr_dev",
                "average",
            ]
        )
        .sort(["item_id", "sr_item_qty"], nulls_last=True)
        .limit(100)
    )
