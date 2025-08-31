# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 51."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 51."""
    return """
    WITH web_v1 AS
    (
             SELECT   ws_item_sk item_sk,
                      d_date,
                      sum(Sum(ws_sales_price)) OVER (partition BY ws_item_sk ORDER BY d_date rows BETWEEN UNBOUNDED PRECEDING AND      CURRENT row) cume_sales
             FROM     web_sales ,
                      date_dim
             WHERE    ws_sold_date_sk=d_date_sk
             AND      d_month_seq BETWEEN 1192 AND      1192+11
             AND      ws_item_sk IS NOT NULL
             GROUP BY ws_item_sk,
                      d_date), store_v1 AS
    (
             SELECT   ss_item_sk item_sk,
                      d_date,
                      sum(sum(ss_sales_price)) OVER (partition BY ss_item_sk ORDER BY d_date rows BETWEEN UNBOUNDED PRECEDING AND      CURRENT row) cume_sales
             FROM     store_sales ,
                      date_dim
             WHERE    ss_sold_date_sk=d_date_sk
             AND      d_month_seq BETWEEN 1192 AND      1192+11
             AND      ss_item_sk IS NOT NULL
             GROUP BY ss_item_sk,
                      d_date)
    SELECT
             *
    FROM     (
                      SELECT   item_sk ,
                               d_date ,
                               web_sales ,
                               store_sales ,
                               max(web_sales) OVER (partition BY item_sk ORDER BY d_date rows BETWEEN UNBOUNDED PRECEDING AND      CURRENT row)   web_cumulative ,
                               max(store_sales) OVER (partition BY item_sk ORDER BY d_date rows BETWEEN UNBOUNDED PRECEDING AND      CURRENT row) store_cumulative
                      FROM     (
                                               SELECT
                                                               CASE
                                                                               WHEN web.item_sk IS NOT NULL THEN web.item_sk
                                                                               ELSE store.item_sk
                                                               END item_sk ,
                                                               CASE
                                                                               WHEN web.d_date IS NOT NULL THEN web.d_date
                                                                               ELSE store.d_date
                                                               END              d_date ,
                                                               web.cume_sales   web_sales ,
                                                               store.cume_sales store_sales
                                               FROM            web_v1 web
                                               FULL OUTER JOIN store_v1 store
                                               ON              (
                                                                               web.item_sk = store.item_sk
                                                               AND             web.d_date = store.d_date) )x )y
    WHERE    web_cumulative > store_cumulative
    ORDER BY item_sk ,
             d_date
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 51."""
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    low, high = 1192, 1192 + 11  # inclusive

    # --- web_v1: (ws_item_sk, d_date) -> daily sum -> cumulative sum by item ordered by date
    web_v1 = (
        web_sales.join(
            date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner"
        )
        .filter(pl.col("d_month_seq").is_between(low, high))
        .filter(pl.col("ws_item_sk").is_not_null())
        .group_by(["ws_item_sk", "d_date"])
        .agg(pl.col("ws_sales_price").sum().alias("daily_sales"))
        .with_columns(
            pl.col("daily_sales")
            .cum_sum()
            .over("ws_item_sk", order_by="d_date")
            .alias("cume_sales")
        )
        .select(
            pl.col("ws_item_sk").alias("item_sk"),
            "d_date",
            pl.col("cume_sales").alias("web_sales"),
        )
    )

    # --- store_v1: (ss_item_sk, d_date) -> daily sum -> cumulative sum by item ordered by date
    store_v1 = (
        store_sales.join(
            date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner"
        )
        .filter(pl.col("d_month_seq").is_between(low, high))
        .filter(pl.col("ss_item_sk").is_not_null())
        .group_by(["ss_item_sk", "d_date"])
        .agg(pl.col("ss_sales_price").sum().alias("daily_sales"))
        .with_columns(
            pl.col("daily_sales")
            .cum_sum()
            .over("ss_item_sk", order_by="d_date")
            .alias("cume_sales")
        )
        .select(
            pl.col("ss_item_sk").alias("item_sk"),
            "d_date",
            pl.col("cume_sales").alias("store_sales"),
        )
    )

    # --- FULL OUTER via union-of-keys (this is what matched in STEP 5/6)
    keys = pl.concat(
        [web_v1.select("item_sk", "d_date"), store_v1.select("item_sk", "d_date")],
        how="vertical",
    ).unique()

    x = keys.join(web_v1, on=["item_sk", "d_date"], how="left").join(
        store_v1, on=["item_sk", "d_date"], how="left"
    )

    # --- Window MAX ignoring NULLs via sentinel + cum_max (this matched in STEP 6)
    neg_inf = pl.lit(float("-inf")).cast(pl.Float64)
    window = {"partition_by": "item_sk", "order_by": "d_date"}

    web_cum_tmp = (
        pl.coalesce([pl.col("web_sales").cast(pl.Float64), neg_inf])
        .cum_max()
        .over(**window)
    )
    store_cum_tmp = (
        pl.coalesce([pl.col("store_sales").cast(pl.Float64), neg_inf])
        .cum_max()
        .over(**window)
    )

    y = x.with_columns(
        web_cumulative=pl.when(web_cum_tmp == neg_inf)
        .then(None)
        .otherwise(web_cum_tmp),
        store_cumulative=pl.when(store_cum_tmp == neg_inf)
        .then(None)
        .otherwise(store_cum_tmp),
    )

    # --- FINAL STEP fix: compare at DECIMAL(?,2)-equivalent by rounding cumulatives to 2 decimals.
    # This mirrors DuckDB's DECIMAL comparison and removes the single-row wobble at the LIMIT edge.
    y2 = y.with_columns(
        web_cumulative_r=pl.col("web_cumulative").round(2),
        store_cumulative_r=pl.col("store_cumulative").round(2),
        web_sales_r=pl.col("web_sales").round(2),
        store_sales_r=pl.col("store_sales").round(2),
    )

    return (
        y2.filter(pl.col("web_cumulative_r") > pl.col("store_cumulative_r"))
        .select(
            "item_sk",
            "d_date",
            pl.col("web_sales_r").alias("web_sales"),
            pl.col("store_sales_r").alias("store_sales"),
            pl.col("web_cumulative_r").alias("web_cumulative"),
            pl.col("store_cumulative_r").alias("store_cumulative"),
        )
        .sort(["item_sk", "d_date"])
        .limit(100)
    )
