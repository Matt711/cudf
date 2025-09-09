# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 77."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 77."""
    return """
    -- start query 77 in stream 0 using template query77.tpl
    WITH ss AS
    (
             SELECT   s_store_sk,
                      Sum(ss_ext_sales_price) AS sales,
                      Sum(ss_net_profit)      AS profit
             FROM     store_sales,
                      date_dim,
                      store
             WHERE    ss_sold_date_sk = d_date_sk
             AND      d_date BETWEEN Cast('2001-08-16' AS DATE) AND      (
                               Cast('2001-08-16' AS DATE) + INTERVAL '30' day)
             AND      ss_store_sk = s_store_sk
             GROUP BY s_store_sk) , sr AS
    (
             SELECT   s_store_sk,
                      sum(sr_return_amt) AS returns1,
                      sum(sr_net_loss)   AS profit_loss
             FROM     store_returns,
                      date_dim,
                      store
             WHERE    sr_returned_date_sk = d_date_sk
             AND      d_date BETWEEN cast('2001-08-16' AS date) AND      (
                               cast('2001-08-16' AS date) + INTERVAL '30' day)
             AND      sr_store_sk = s_store_sk
             GROUP BY s_store_sk), cs AS
    (
             SELECT   cs_call_center_sk,
                      sum(cs_ext_sales_price) AS sales,
                      sum(cs_net_profit)      AS profit
             FROM     catalog_sales,
                      date_dim
             WHERE    cs_sold_date_sk = d_date_sk
             AND      d_date BETWEEN cast('2001-08-16' AS date) AND      (
                               cast('2001-08-16' AS date) + INTERVAL '30' day)
             GROUP BY cs_call_center_sk ), cr AS
    (
             SELECT   cr_call_center_sk,
                      sum(cr_return_amount) AS returns1,
                      sum(cr_net_loss)      AS profit_loss
             FROM     catalog_returns,
                      date_dim
             WHERE    cr_returned_date_sk = d_date_sk
             AND      d_date BETWEEN cast('2001-08-16' AS date) AND      (
                               cast('2001-08-16' AS date) + INTERVAL '30' day)
             GROUP BY cr_call_center_sk ), ws AS
    (
             SELECT   wp_web_page_sk,
                      sum(ws_ext_sales_price) AS sales,
                      sum(ws_net_profit)      AS profit
             FROM     web_sales,
                      date_dim,
                      web_page
             WHERE    ws_sold_date_sk = d_date_sk
             AND      d_date BETWEEN cast('2001-08-16' AS date) AND      (
                               cast('2001-08-16' AS date) + INTERVAL '30' day)
             AND      ws_web_page_sk = wp_web_page_sk
             GROUP BY wp_web_page_sk), wr AS
    (
             SELECT   wp_web_page_sk,
                      sum(wr_return_amt) AS returns1,
                      sum(wr_net_loss)   AS profit_loss
             FROM     web_returns,
                      date_dim,
                      web_page
             WHERE    wr_returned_date_sk = d_date_sk
             AND      d_date BETWEEN cast('2001-08-16' AS date) AND      (
                               cast('2001-08-16' AS date) + INTERVAL '30' day)
             AND      wr_web_page_sk = wp_web_page_sk
             GROUP BY wp_web_page_sk)
    SELECT
             channel ,
             id ,
             sum(sales)   AS sales ,
             sum(returns1) AS returns1 ,
             sum(profit)  AS profit
    FROM     (
                       SELECT    'store channel' AS channel ,
                                 ss.s_store_sk   AS id ,
                                 sales ,
                                 COALESCE(returns1, 0)               AS returns1 ,
                                 (profit - COALESCE(profit_loss,0)) AS profit
                       FROM      ss
                       LEFT JOIN sr
                       ON        ss.s_store_sk = sr.s_store_sk
                       UNION ALL
                       SELECT 'catalog channel' AS channel ,
                              cs_call_center_sk AS id ,
                              sales ,
                              returns1 ,
                              (profit - profit_loss) AS profit
                       FROM   cs ,
                              cr
                       UNION ALL
                       SELECT    'web channel'     AS channel ,
                                 ws.wp_web_page_sk AS id ,
                                 sales ,
                                 COALESCE(returns1, 0)                  returns1 ,
                                 (profit - COALESCE(profit_loss,0)) AS profit
                       FROM      ws
                       LEFT JOIN wr
                       ON        ws.wp_web_page_sk = wr.wp_web_page_sk ) x
    GROUP BY rollup (channel, id)
    ORDER BY channel ,
             id ,
             sales --added to make deterministic
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 77."""
    # Load required tables using scan_parquet for lazy evaluation
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(
        run_config.dataset_path, "store_returns", run_config.suffix
    )
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    catalog_returns = get_data(
        run_config.dataset_path, "catalog_returns", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    web_returns = get_data(run_config.dataset_path, "web_returns", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    web_page = get_data(run_config.dataset_path, "web_page", run_config.suffix)
    # Filter date_dim for the 30-day period starting from 2001-08-16
    start_date = (pl.date(2001, 8, 16)).cast(pl.Datetime("us"))
    end_date = (start_date + pl.duration(days=30)).cast(pl.Datetime("us"))
    filtered_dates = date_dim.filter(
        (pl.col("d_date") >= start_date) & (pl.col("d_date") <= end_date)
    ).select("d_date_sk")
    # CTE: ss (store sales)
    ss = (
        store_sales.join(
            filtered_dates, left_on="ss_sold_date_sk", right_on="d_date_sk"
        )
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .group_by("ss_store_sk")
        .agg(
            [
                pl.col("ss_ext_sales_price").count().alias("sales_count"),
                pl.col("ss_ext_sales_price").sum().alias("sales_sum"),
                pl.col("ss_net_profit").count().alias("profit_count"),
                pl.col("ss_net_profit").sum().alias("profit_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("sales_count") == 0)
                .then(None)
                .otherwise(pl.col("sales_sum"))
                .alias("sales"),
                pl.when(pl.col("profit_count") == 0)
                .then(None)
                .otherwise(pl.col("profit_sum"))
                .alias("profit"),
            ]
        )
        .select(["ss_store_sk", "sales", "profit"])
    )
    # CTE: sr (store returns)
    sr = (
        store_returns.join(
            filtered_dates, left_on="sr_returned_date_sk", right_on="d_date_sk"
        )
        .join(store, left_on="sr_store_sk", right_on="s_store_sk")
        .group_by("sr_store_sk")
        .agg(
            [
                pl.col("sr_return_amt").count().alias("returns1_count"),
                pl.col("sr_return_amt").sum().alias("returns1_sum"),
                pl.col("sr_net_loss").count().alias("profit_loss_count"),
                pl.col("sr_net_loss").sum().alias("profit_loss_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("returns1_count") == 0)
                .then(None)
                .otherwise(pl.col("returns1_sum"))
                .alias("returns1"),
                pl.when(pl.col("profit_loss_count") == 0)
                .then(None)
                .otherwise(pl.col("profit_loss_sum"))
                .alias("profit_loss"),
            ]
        )
        .select(["sr_store_sk", "returns1", "profit_loss"])
    )
    # CTE: cs (catalog sales)
    cs = (
        catalog_sales.join(
            filtered_dates, left_on="cs_sold_date_sk", right_on="d_date_sk"
        )
        .group_by("cs_call_center_sk")
        .agg(
            [
                pl.col("cs_ext_sales_price").count().alias("sales_count"),
                pl.col("cs_ext_sales_price").sum().alias("sales_sum"),
                pl.col("cs_net_profit").count().alias("profit_count"),
                pl.col("cs_net_profit").sum().alias("profit_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("sales_count") == 0)
                .then(None)
                .otherwise(pl.col("sales_sum"))
                .alias("sales"),
                pl.when(pl.col("profit_count") == 0)
                .then(None)
                .otherwise(pl.col("profit_sum"))
                .alias("profit"),
            ]
        )
        .select(["cs_call_center_sk", "sales", "profit"])
    )
    # CTE: cr (catalog returns)
    cr = (
        catalog_returns.join(
            filtered_dates, left_on="cr_returned_date_sk", right_on="d_date_sk"
        )
        .group_by("cr_call_center_sk")
        .agg(
            [
                pl.col("cr_return_amount").count().alias("returns1_count"),
                pl.col("cr_return_amount").sum().alias("returns1_sum"),
                pl.col("cr_net_loss").count().alias("profit_loss_count"),
                pl.col("cr_net_loss").sum().alias("profit_loss_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("returns1_count") == 0)
                .then(None)
                .otherwise(pl.col("returns1_sum"))
                .alias("returns1"),
                pl.when(pl.col("profit_loss_count") == 0)
                .then(None)
                .otherwise(pl.col("profit_loss_sum"))
                .alias("profit_loss"),
            ]
        )
        .select(["cr_call_center_sk", "returns1", "profit_loss"])
    )
    # CTE: ws (web sales)
    ws = (
        web_sales.join(filtered_dates, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(web_page, left_on="ws_web_page_sk", right_on="wp_web_page_sk")
        .group_by("ws_web_page_sk")
        .agg(
            [
                pl.col("ws_ext_sales_price").count().alias("sales_count"),
                pl.col("ws_ext_sales_price").sum().alias("sales_sum"),
                pl.col("ws_net_profit").count().alias("profit_count"),
                pl.col("ws_net_profit").sum().alias("profit_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("sales_count") == 0)
                .then(None)
                .otherwise(pl.col("sales_sum"))
                .alias("sales"),
                pl.when(pl.col("profit_count") == 0)
                .then(None)
                .otherwise(pl.col("profit_sum"))
                .alias("profit"),
            ]
        )
        .select(["ws_web_page_sk", "sales", "profit"])
    )
    # CTE: wr (web returns)
    wr = (
        web_returns.join(
            filtered_dates, left_on="wr_returned_date_sk", right_on="d_date_sk"
        )
        .join(web_page, left_on="wr_web_page_sk", right_on="wp_web_page_sk")
        .group_by("wr_web_page_sk")
        .agg(
            [
                pl.col("wr_return_amt").count().alias("returns1_count"),
                pl.col("wr_return_amt").sum().alias("returns1_sum"),
                pl.col("wr_net_loss").count().alias("profit_loss_count"),
                pl.col("wr_net_loss").sum().alias("profit_loss_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("returns1_count") == 0)
                .then(None)
                .otherwise(pl.col("returns1_sum"))
                .alias("returns1"),
                pl.when(pl.col("profit_loss_count") == 0)
                .then(None)
                .otherwise(pl.col("profit_loss_sum"))
                .alias("profit_loss"),
            ]
        )
        .select(["wr_web_page_sk", "returns1", "profit_loss"])
    )
    # Create the three channel components
    # Store channel (LEFT JOIN)
    store_channel = ss.join(
        sr, left_on="ss_store_sk", right_on="sr_store_sk", how="left"
    ).select(
        [
            pl.lit("store channel").alias("channel"),
            pl.col("ss_store_sk").alias("id"),
            "sales",
            # pl.coalesce([pl.col("returns1"), pl.lit(0)]).alias("returns1"),
            # (pl.col("profit") - pl.coalesce([pl.col("profit_loss"), pl.lit(0)])).alias("profit")
            pl.col("returns1").fill_null(0).alias("returns1"),
            (pl.col("profit") - pl.col("profit_loss").fill_null(0)).alias("profit"),
        ]
    )
    # Catalog channel (Cartesian product as in SQL: FROM cs, cr)
    catalog_channel = cs.join(cr, how="cross").select(
        [
            pl.lit("catalog channel").alias("channel"),
            pl.col("cs_call_center_sk").alias("id"),
            "sales",
            "returns1",
            (pl.col("profit") - pl.col("profit_loss")).alias("profit"),
        ]
    )
    # Web channel (LEFT JOIN)
    web_channel = ws.join(
        wr, left_on="ws_web_page_sk", right_on="wr_web_page_sk", how="left"
    ).select(
        [
            pl.lit("web channel").alias("channel"),
            pl.col("ws_web_page_sk").alias("id"),
            "sales",
            # pl.coalesce([pl.col("returns1"), pl.lit(0)]).alias("returns1"),
            # (pl.col("profit") - pl.coalesce([pl.col("profit_loss"), pl.lit(0)])).alias("profit")
            pl.col("returns1").fill_null(0).alias("returns1"),
            (pl.col("profit") - pl.col("profit_loss").fill_null(0)).alias("profit"),
        ]
    )
    # Combine all channels (UNION ALL)
    combined_channels = pl.concat([store_channel, catalog_channel, web_channel])
    # Create ROLLUP equivalent: two levels of aggregation
    # Level 1: Group by channel and id
    level1 = (
        combined_channels.group_by(["channel", "id"])
        .agg(
            [
                pl.col("sales").count().alias("sales_count"),
                pl.col("sales").sum().alias("sales_sum"),
                pl.col("returns1").count().alias("returns1_count"),
                pl.col("returns1").sum().alias("returns1_sum"),
                pl.col("profit").count().alias("profit_count"),
                pl.col("profit").sum().alias("profit_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("sales_count") == 0)
                .then(None)
                .otherwise(pl.col("sales_sum"))
                .alias("sales"),
                pl.when(pl.col("returns1_count") == 0)
                .then(None)
                .otherwise(pl.col("returns1_sum"))
                .alias("returns1"),
                pl.when(pl.col("profit_count") == 0)
                .then(None)
                .otherwise(pl.col("profit_sum"))
                .alias("profit"),
            ]
        )
        .select(["channel", "id", "sales", "returns1", "profit"])
    )
    # Level 2: Group by channel only (id = NULL)
    level2 = (
        combined_channels.group_by("channel")
        .agg(
            [
                pl.col("sales").count().alias("sales_count"),
                pl.col("sales").sum().alias("sales_sum"),
                pl.col("returns1").count().alias("returns1_count"),
                pl.col("returns1").sum().alias("returns1_sum"),
                pl.col("profit").count().alias("profit_count"),
                pl.col("profit").sum().alias("profit_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("sales_count") == 0)
                .then(None)
                .otherwise(pl.col("sales_sum"))
                .alias("sales"),
                pl.when(pl.col("returns1_count") == 0)
                .then(None)
                .otherwise(pl.col("returns1_sum"))
                .alias("returns1"),
                pl.when(pl.col("profit_count") == 0)
                .then(None)
                .otherwise(pl.col("profit_sum"))
                .alias("profit"),
            ]
        )
        .select(["channel", "sales", "returns1", "profit"])
    )
    # Level 3: Grand total (both channel and id = NULL)
    level3 = (
        combined_channels.select(
            [
                pl.col("sales").count().alias("sales_count"),
                pl.col("sales").sum().alias("sales_sum"),
                pl.col("returns1").count().alias("returns1_count"),
                pl.col("returns1").sum().alias("returns1_sum"),
                pl.col("profit").count().alias("profit_count"),
                pl.col("profit").sum().alias("profit_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("sales_count") == 0)
                .then(None)
                .otherwise(pl.col("sales_sum"))
                .alias("sales"),
                pl.when(pl.col("returns1_count") == 0)
                .then(None)
                .otherwise(pl.col("returns1_sum"))
                .alias("returns1"),
                pl.when(pl.col("profit_count") == 0)
                .then(None)
                .otherwise(pl.col("profit_sum"))
                .alias("profit"),
            ]
        )
        .select(["sales", "returns1", "profit"])
    )
    # Combine all three levels and sort
    return (
        pl.concat([level1, level2, level3], how="diagonal")
        .sort(["channel", "id", "sales"], nulls_last=True)
        .limit(100)
    )  #
