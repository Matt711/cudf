# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 80."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 80."""
    return """
    WITH ssr AS
    (
                    SELECT          s_store_id                                    AS store_id,
                                    Sum(ss_ext_sales_price)                       AS sales,
                                    Sum(COALESCE(sr_return_amt, 0))               AS returns1,
                                    Sum(ss_net_profit - COALESCE(sr_net_loss, 0)) AS profit
                    FROM            store_sales
                    LEFT OUTER JOIN store_returns
                    ON              (
                                                    ss_item_sk = sr_item_sk
                                    AND             ss_ticket_number = sr_ticket_number),
                                    date_dim,
                                    store,
                                    item,
                                    promotion
                    WHERE           ss_sold_date_sk = d_date_sk
                    AND             d_date BETWEEN Cast('2000-08-26' AS DATE) AND             (
                                                    Cast('2000-08-26' AS DATE) + INTERVAL '30' day)
                    AND             ss_store_sk = s_store_sk
                    AND             ss_item_sk = i_item_sk
                    AND             i_current_price > 50
                    AND             ss_promo_sk = p_promo_sk
                    AND             p_channel_tv = 'N'
                    GROUP BY        s_store_id) , csr AS
    (
                    SELECT          cp_catalog_page_id                            AS catalog_page_id,
                                    sum(cs_ext_sales_price)                       AS sales,
                                    sum(COALESCE(cr_return_amount, 0))            AS returns1,
                                    sum(cs_net_profit - COALESCE(cr_net_loss, 0)) AS profit
                    FROM            catalog_sales
                    LEFT OUTER JOIN catalog_returns
                    ON              (
                                                    cs_item_sk = cr_item_sk
                                    AND             cs_order_number = cr_order_number),
                                    date_dim,
                                    catalog_page,
                                    item,
                                    promotion
                    WHERE           cs_sold_date_sk = d_date_sk
                    AND             d_date BETWEEN cast('2000-08-26' AS date) AND             (
                                                    cast('2000-08-26' AS date) + INTERVAL '30' day)
                    AND             cs_catalog_page_sk = cp_catalog_page_sk
                    AND             cs_item_sk = i_item_sk
                    AND             i_current_price > 50
                    AND             cs_promo_sk = p_promo_sk
                    AND             p_channel_tv = 'N'
                    GROUP BY        cp_catalog_page_id) , wsr AS
    (
                    SELECT          web_site_id,
                                    sum(ws_ext_sales_price)                       AS sales,
                                    sum(COALESCE(wr_return_amt, 0))               AS returns1,
                                    sum(ws_net_profit - COALESCE(wr_net_loss, 0)) AS profit
                    FROM            web_sales
                    LEFT OUTER JOIN web_returns
                    ON              (
                                                    ws_item_sk = wr_item_sk
                                    AND             ws_order_number = wr_order_number),
                                    date_dim,
                                    web_site,
                                    item,
                                    promotion
                    WHERE           ws_sold_date_sk = d_date_sk
                    AND             d_date BETWEEN cast('2000-08-26' AS date) AND             (
                                                    cast('2000-08-26' AS date) + INTERVAL '30' day)
                    AND             ws_web_site_sk = web_site_sk
                    AND             ws_item_sk = i_item_sk
                    AND             i_current_price > 50
                    AND             ws_promo_sk = p_promo_sk
                    AND             p_channel_tv = 'N'
                    GROUP BY        web_site_id)
    SELECT
             channel ,
             id ,
             sum(sales)   AS sales ,
             sum(returns1) AS returns1 ,
             sum(profit)  AS profit
    FROM     (
                    SELECT 'store channel' AS channel ,
                           'store'
                                  || store_id AS id ,
                           sales ,
                           returns1 ,
                           profit
                    FROM   ssr
                    UNION ALL
                    SELECT 'catalog channel' AS channel ,
                           'catalog_page'
                                  || catalog_page_id AS id ,
                           sales ,
                           returns1 ,
                           profit
                    FROM   csr
                    UNION ALL
                    SELECT 'web channel' AS channel ,
                           'web_site'
                                  || web_site_id AS id ,
                           sales ,
                           returns1 ,
                           profit
                    FROM   wsr ) x
    GROUP BY rollup (channel, id)
    ORDER BY channel ,
             id
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 80."""
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
    catalog_page = get_data(run_config.dataset_path, "catalog_page", run_config.suffix)
    web_site = get_data(run_config.dataset_path, "web_site", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    promotion = get_data(run_config.dataset_path, "promotion", run_config.suffix)
    # Filter date_dim for the 30-day period starting from 2000-08-26
    start_date = (pl.date(2000, 8, 26)).cast(pl.Datetime("us"))
    end_date = (start_date + pl.duration(days=30)).cast(pl.Datetime("us"))
    filtered_dates = date_dim.filter(
        (pl.col("d_date") >= start_date) & (pl.col("d_date") <= end_date)
    ).select("d_date_sk")
    # CTE: ssr (store sales with returns)
    ssr = (
        store_sales.join(
            store_returns,
            left_on=["ss_item_sk", "ss_ticket_number"],
            right_on=["sr_item_sk", "sr_ticket_number"],
            how="left",
        )
        .join(filtered_dates, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .join(promotion, left_on="ss_promo_sk", right_on="p_promo_sk")
        .filter((pl.col("i_current_price") > 50) & (pl.col("p_channel_tv") == "N"))
        .group_by("s_store_id")
        .agg(
            [
                pl.col("ss_ext_sales_price").count().alias("sales_count"),
                pl.col("ss_ext_sales_price").sum().alias("sales_sum"),
                pl.col("sr_return_amt").count().alias("returns1_count"),
                # pl.coalesce([pl.col("sr_return_amt"), pl.lit(0)]).sum().alias("returns1_sum"),
                pl.col("sr_return_amt").fill_null(0).sum().alias("returns1_sum"),
                pl.col("ss_net_profit").count().alias("profit_count"),
                # (pl.col("ss_net_profit") - pl.coalesce([pl.col("sr_net_loss"), pl.lit(0)])).sum().alias("profit_sum")
                (pl.col("ss_net_profit") - pl.col("sr_net_loss").fill_null(0))
                .sum()
                .alias("profit_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("sales_count") == 0)
                .then(None)
                .otherwise(pl.col("sales_sum"))
                .alias("sales"),
                pl.when(pl.col("returns1_count") == 0)
                .then(pl.lit(0))
                .otherwise(pl.col("returns1_sum"))
                .alias("returns1"),
                pl.when(pl.col("profit_count") == 0)
                .then(None)
                .otherwise(pl.col("profit_sum"))
                .alias("profit"),
                pl.col("s_store_id").alias("store_id"),
            ]
        )
        .select(["store_id", "sales", "returns1", "profit"])
    )
    # CTE: csr (catalog sales with returns)
    csr = (
        catalog_sales.join(
            catalog_returns,
            left_on=["cs_item_sk", "cs_order_number"],
            right_on=["cr_item_sk", "cr_order_number"],
            how="left",
        )
        .join(filtered_dates, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(catalog_page, left_on="cs_catalog_page_sk", right_on="cp_catalog_page_sk")
        .join(item, left_on="cs_item_sk", right_on="i_item_sk")
        .join(promotion, left_on="cs_promo_sk", right_on="p_promo_sk")
        .filter((pl.col("i_current_price") > 50) & (pl.col("p_channel_tv") == "N"))
        .group_by("cp_catalog_page_id")
        .agg(
            [
                pl.col("cs_ext_sales_price").count().alias("sales_count"),
                pl.col("cs_ext_sales_price").sum().alias("sales_sum"),
                pl.col("cr_return_amount").count().alias("returns1_count"),
                # pl.coalesce([pl.col("cr_return_amount"), pl.lit(0)]).sum().alias("returns1_sum"),
                pl.col("cr_return_amount").fill_null(0).sum().alias("returns1_sum"),
                pl.col("cs_net_profit").count().alias("profit_count"),
                # (pl.col("cs_net_profit") - pl.coalesce([pl.col("cr_net_loss"), pl.lit(0)])).sum().alias("profit_sum")
                (pl.col("cs_net_profit") - pl.col("cr_net_loss").fill_null(0))
                .sum()
                .alias("profit_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("sales_count") == 0)
                .then(None)
                .otherwise(pl.col("sales_sum"))
                .alias("sales"),
                pl.when(pl.col("returns1_count") == 0)
                .then(pl.lit(0))
                .otherwise(pl.col("returns1_sum"))
                .alias("returns1"),
                pl.when(pl.col("profit_count") == 0)
                .then(None)
                .otherwise(pl.col("profit_sum"))
                .alias("profit"),
                pl.col("cp_catalog_page_id").alias("catalog_page_id"),
            ]
        )
        .select(["catalog_page_id", "sales", "returns1", "profit"])
    )
    # CTE: wsr (web sales with returns)
    wsr = (
        web_sales.join(
            web_returns,
            left_on=["ws_item_sk", "ws_order_number"],
            right_on=["wr_item_sk", "wr_order_number"],
            how="left",
        )
        .join(filtered_dates, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(web_site, left_on="ws_web_site_sk", right_on="web_site_sk")
        .join(item, left_on="ws_item_sk", right_on="i_item_sk")
        .join(promotion, left_on="ws_promo_sk", right_on="p_promo_sk")
        .filter((pl.col("i_current_price") > 50) & (pl.col("p_channel_tv") == "N"))
        .group_by("web_site_id")
        .agg(
            [
                pl.col("ws_ext_sales_price").count().alias("sales_count"),
                pl.col("ws_ext_sales_price").sum().alias("sales_sum"),
                pl.col("wr_return_amt").count().alias("returns1_count"),
                # pl.coalesce([pl.col("wr_return_amt"), pl.lit(0)]).sum().alias("returns1_sum"),
                pl.col("wr_return_amt").fill_null(0).sum().alias("returns1_sum"),
                pl.col("ws_net_profit").count().alias("profit_count"),
                # (pl.col("ws_net_profit") - pl.coalesce([pl.col("wr_net_loss"), pl.lit(0)])).sum().alias("profit_sum")
                (pl.col("ws_net_profit") - pl.col("wr_net_loss").fill_null(0))
                .sum()
                .alias("profit_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("sales_count") == 0)
                .then(None)
                .otherwise(pl.col("sales_sum"))
                .alias("sales"),
                pl.when(pl.col("returns1_count") == 0)
                .then(pl.lit(0))
                .otherwise(pl.col("returns1_sum"))
                .alias("returns1"),
                pl.when(pl.col("profit_count") == 0)
                .then(None)
                .otherwise(pl.col("profit_sum"))
                .alias("profit"),
            ]
        )
        .select(["web_site_id", "sales", "returns1", "profit"])
    )
    # Create the three channel components
    # Store channel
    store_channel = ssr.select(
        [
            pl.lit("store channel").alias("channel"),
            (pl.lit("store") + pl.col("store_id").cast(pl.Utf8)).alias("id"),
            "sales",
            "returns1",
            "profit",
        ]
    )
    # Catalog channel
    catalog_channel = csr.select(
        [
            pl.lit("catalog channel").alias("channel"),
            (pl.lit("catalog_page") + pl.col("catalog_page_id").cast(pl.Utf8)).alias(
                "id"
            ),
            "sales",
            "returns1",
            "profit",
        ]
    )
    # Web channel
    web_channel = wsr.select(
        [
            pl.lit("web channel").alias("channel"),
            (pl.lit("web_site") + pl.col("web_site_id").cast(pl.Utf8)).alias("id"),
            "sales",
            "returns1",
            "profit",
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
    # Combine both levels and sort
    return (
        pl.concat([level1, level2], how="diagonal")
        .sort(["channel", "id"], nulls_last=True)
        .limit(100)
    )
