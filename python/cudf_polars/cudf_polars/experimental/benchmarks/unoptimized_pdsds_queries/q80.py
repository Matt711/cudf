# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q80 — naive one-for-one Polars translation of the SQL."""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data
from cudf_polars.experimental.benchmarks.unoptimized_pdsds_queries.sql_helpers import (
    sql_rollup,
    sql_sum,
)

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:  # type: ignore[return]
    raise NotImplementedError


def polars_impl(run_config: RunConfig) -> QueryResult:
    params = load_parameters(
        int(run_config.scale_factor), query_id=80, qualification=run_config.qualification
    )
    sdate = params["sdate"]
    year, month, day = map(int, sdate.split("-"))

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

    start_dt = pl.date(year, month, day)
    end_dt = start_dt + pl.duration(days=30)

    filtered_dates = (
        date_dim
        .filter(
            (pl.col("d_date").cast(pl.Date) >= start_dt)
            & (pl.col("d_date").cast(pl.Date) <= end_dt)
        )
        .select("d_date_sk")
    )

    # CTE ssr:
    # FROM store_sales LEFT OUTER JOIN store_returns ON (ss_item_sk=sr_item_sk AND ss_ticket_number=sr_ticket_number),
    #      date_dim, store, item, promotion
    # WHERE ss_sold_date_sk = d_date_sk AND d_date BETWEEN ... AND ss_store_sk = s_store_sk
    #   AND ss_item_sk = i_item_sk AND i_current_price > 50
    #   AND ss_promo_sk = p_promo_sk AND p_channel_tv = 'N'
    # GROUP BY s_store_id
    ssr = (
        store_sales
        .join(
            store_returns,
            left_on=["ss_item_sk", "ss_ticket_number"],
            right_on=["sr_item_sk", "sr_ticket_number"],
            how="left",
        )
        .join(filtered_dates, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk", how="inner")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk", how="inner")
        .join(promotion, left_on="ss_promo_sk", right_on="p_promo_sk", how="inner")
        .filter(
            (pl.col("i_current_price") > 50)
            & (pl.col("p_channel_tv") == "N")
        )
        .group_by("s_store_id")
        .agg([
            sql_sum(pl.col("ss_ext_sales_price")).alias("sales"),
            sql_sum(pl.col("sr_return_amt").fill_null(0)).alias("returns1"),
            sql_sum(
                pl.col("ss_net_profit") - pl.col("sr_net_loss").fill_null(0)
            ).alias("profit"),
        ])
    )

    # CTE csr:
    # FROM catalog_sales LEFT OUTER JOIN catalog_returns ON (cs_item_sk=cr_item_sk AND cs_order_number=cr_order_number),
    #      date_dim, catalog_page, item, promotion
    # WHERE cs_sold_date_sk = d_date_sk AND d_date BETWEEN ... AND cs_catalog_page_sk = cp_catalog_page_sk
    #   AND cs_item_sk = i_item_sk AND i_current_price > 50
    #   AND cs_promo_sk = p_promo_sk AND p_channel_tv = 'N'
    # GROUP BY cp_catalog_page_id
    csr = (
        catalog_sales
        .join(
            catalog_returns,
            left_on=["cs_item_sk", "cs_order_number"],
            right_on=["cr_item_sk", "cr_order_number"],
            how="left",
        )
        .join(filtered_dates, left_on="cs_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(
            catalog_page,
            left_on="cs_catalog_page_sk",
            right_on="cp_catalog_page_sk",
            how="inner",
        )
        .join(item, left_on="cs_item_sk", right_on="i_item_sk", how="inner")
        .join(promotion, left_on="cs_promo_sk", right_on="p_promo_sk", how="inner")
        .filter(
            (pl.col("i_current_price") > 50)
            & (pl.col("p_channel_tv") == "N")
        )
        .group_by("cp_catalog_page_id")
        .agg([
            sql_sum(pl.col("cs_ext_sales_price")).alias("sales"),
            sql_sum(pl.col("cr_return_amount").fill_null(0)).alias("returns1"),
            sql_sum(
                pl.col("cs_net_profit") - pl.col("cr_net_loss").fill_null(0)
            ).alias("profit"),
        ])
    )

    # CTE wsr:
    # FROM web_sales LEFT OUTER JOIN web_returns ON (ws_item_sk=wr_item_sk AND ws_order_number=wr_order_number),
    #      date_dim, web_site, item, promotion
    # WHERE ws_sold_date_sk = d_date_sk AND d_date BETWEEN ... AND ws_web_site_sk = web_site_sk
    #   AND ws_item_sk = i_item_sk AND i_current_price > 50
    #   AND ws_promo_sk = p_promo_sk AND p_channel_tv = 'N'
    # GROUP BY web_site_id
    wsr = (
        web_sales
        .join(
            web_returns,
            left_on=["ws_item_sk", "ws_order_number"],
            right_on=["wr_item_sk", "wr_order_number"],
            how="left",
        )
        .join(filtered_dates, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(web_site, left_on="ws_web_site_sk", right_on="web_site_sk", how="inner")
        .join(item, left_on="ws_item_sk", right_on="i_item_sk", how="inner")
        .join(promotion, left_on="ws_promo_sk", right_on="p_promo_sk", how="inner")
        .filter(
            (pl.col("i_current_price") > 50)
            & (pl.col("p_channel_tv") == "N")
        )
        .group_by("web_site_id")
        .agg([
            sql_sum(pl.col("ws_ext_sales_price")).alias("sales"),
            sql_sum(pl.col("wr_return_amt").fill_null(0)).alias("returns1"),
            sql_sum(
                pl.col("ws_net_profit") - pl.col("wr_net_loss").fill_null(0)
            ).alias("profit"),
        ])
    )

    # Build x: UNION ALL of three channel rows
    store_rows = ssr.select([
        pl.lit("store channel").alias("channel"),
        (pl.lit("store") + pl.col("s_store_id").cast(pl.Utf8)).alias("id"),
        "sales",
        "returns1",
        "profit",
    ])

    catalog_rows = csr.select([
        pl.lit("catalog channel").alias("channel"),
        (pl.lit("catalog_page") + pl.col("cp_catalog_page_id").cast(pl.Utf8)).alias("id"),
        "sales",
        "returns1",
        "profit",
    ])

    web_rows = wsr.select([
        pl.lit("web channel").alias("channel"),
        (pl.lit("web_site") + pl.col("web_site_id").cast(pl.Utf8)).alias("id"),
        "sales",
        "returns1",
        "profit",
    ])

    combined = pl.concat([store_rows, catalog_rows, web_rows], how="diagonal_relaxed")

    # GROUP BY rollup(channel, id)
    agg_exprs = [
        sql_sum(pl.col("sales")).alias("sales"),
        sql_sum(pl.col("returns1")).alias("returns1"),
        sql_sum(pl.col("profit")).alias("profit"),
    ]
    result = sql_rollup(combined, ["channel", "id"], agg_exprs)

    result = result.sort(['channel', 'id'], descending=[False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("channel", False), ("id", False)],
        limit=100,
    )
