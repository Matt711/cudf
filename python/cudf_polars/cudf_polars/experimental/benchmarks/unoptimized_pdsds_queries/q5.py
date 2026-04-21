# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q5 — naive one-for-one Polars translation of the SQL."""

from __future__ import annotations

from datetime import date, timedelta
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
        int(run_config.scale_factor),
        query_id=5,
        qualification=run_config.qualification,
    )
    sales_date_str = params["sales_date"]
    yr, mo, dy = map(int, sales_date_str.split("-"))
    start_date = date(yr, mo, dy)
    end_date = start_date + timedelta(days=14)

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(run_config.dataset_path, "store_returns", run_config.suffix)
    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    catalog_returns = get_data(run_config.dataset_path, "catalog_returns", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    web_returns = get_data(run_config.dataset_path, "web_returns", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    catalog_page = get_data(run_config.dataset_path, "catalog_page", run_config.suffix)
    web_site = get_data(run_config.dataset_path, "web_site", run_config.suffix)

    # CTE ssr: store channel sales/returns UNION ALL, then join date_dim and store
    ssr_salesreturns = pl.concat(
        [
            store_sales.select(
                pl.col("ss_store_sk").alias("store_sk"),
                pl.col("ss_sold_date_sk").alias("date_sk"),
                pl.col("ss_ext_sales_price").alias("sales_price"),
                pl.col("ss_net_profit").alias("profit"),
                pl.lit(0).cast(pl.Float64).alias("return_amt"),
                pl.lit(0).cast(pl.Float64).alias("net_loss"),
            ),
            store_returns.select(
                pl.col("sr_store_sk").alias("store_sk"),
                pl.col("sr_returned_date_sk").alias("date_sk"),
                pl.lit(0).cast(pl.Float64).alias("sales_price"),
                pl.lit(0).cast(pl.Float64).alias("profit"),
                pl.col("sr_return_amt").alias("return_amt"),
                pl.col("sr_net_loss").alias("net_loss"),
            ),
        ],
        how="diagonal_relaxed",
    )
    ssr = (
        ssr_salesreturns.join(date_dim, left_on="date_sk", right_on="d_date_sk", how="inner")
        .join(store, left_on="store_sk", right_on="s_store_sk", how="inner")
        .filter(
            pl.col("d_date").is_between(pl.lit(start_date), pl.lit(end_date), closed="both")
        )
        .group_by("s_store_id")
        .agg(
            sql_sum(pl.col("sales_price")).alias("sales"),
            sql_sum(pl.col("profit")).alias("profit"),
            sql_sum(pl.col("return_amt")).alias("returns1"),
            sql_sum(pl.col("net_loss")).alias("profit_loss"),
        )
    )

    # CTE csr: catalog channel
    csr_salesreturns = pl.concat(
        [
            catalog_sales.select(
                pl.col("cs_catalog_page_sk").alias("page_sk"),
                pl.col("cs_sold_date_sk").alias("date_sk"),
                pl.col("cs_ext_sales_price").alias("sales_price"),
                pl.col("cs_net_profit").alias("profit"),
                pl.lit(0).cast(pl.Float64).alias("return_amt"),
                pl.lit(0).cast(pl.Float64).alias("net_loss"),
            ),
            catalog_returns.select(
                pl.col("cr_catalog_page_sk").alias("page_sk"),
                pl.col("cr_returned_date_sk").alias("date_sk"),
                pl.lit(0).cast(pl.Float64).alias("sales_price"),
                pl.lit(0).cast(pl.Float64).alias("profit"),
                pl.col("cr_return_amount").alias("return_amt"),
                pl.col("cr_net_loss").alias("net_loss"),
            ),
        ],
        how="diagonal_relaxed",
    )
    csr = (
        csr_salesreturns.join(date_dim, left_on="date_sk", right_on="d_date_sk", how="inner")
        .join(catalog_page, left_on="page_sk", right_on="cp_catalog_page_sk", how="inner")
        .filter(
            pl.col("d_date").is_between(pl.lit(start_date), pl.lit(end_date), closed="both")
        )
        .group_by("cp_catalog_page_id")
        .agg(
            sql_sum(pl.col("sales_price")).alias("sales"),
            sql_sum(pl.col("profit")).alias("profit"),
            sql_sum(pl.col("return_amt")).alias("returns1"),
            sql_sum(pl.col("net_loss")).alias("profit_loss"),
        )
    )

    # CTE wsr: web channel
    # web_returns LEFT OUTER JOIN web_sales ON wr_item_sk=ws_item_sk AND wr_order_number=ws_order_number
    web_returns_with_site = web_returns.join(
        web_sales.select(["ws_item_sk", "ws_order_number", "ws_web_site_sk"]),
        left_on=["wr_item_sk", "wr_order_number"],
        right_on=["ws_item_sk", "ws_order_number"],
        how="left",
    )
    wsr_salesreturns = pl.concat(
        [
            web_sales.select(
                pl.col("ws_web_site_sk").alias("wsr_web_site_sk"),
                pl.col("ws_sold_date_sk").alias("date_sk"),
                pl.col("ws_ext_sales_price").alias("sales_price"),
                pl.col("ws_net_profit").alias("profit"),
                pl.lit(0).cast(pl.Float64).alias("return_amt"),
                pl.lit(0).cast(pl.Float64).alias("net_loss"),
            ),
            web_returns_with_site.select(
                pl.col("ws_web_site_sk").alias("wsr_web_site_sk"),
                pl.col("wr_returned_date_sk").alias("date_sk"),
                pl.lit(0).cast(pl.Float64).alias("sales_price"),
                pl.lit(0).cast(pl.Float64).alias("profit"),
                pl.col("wr_return_amt").alias("return_amt"),
                pl.col("wr_net_loss").alias("net_loss"),
            ),
        ],
        how="diagonal_relaxed",
    )
    wsr = (
        wsr_salesreturns.join(date_dim, left_on="date_sk", right_on="d_date_sk", how="inner")
        .join(web_site, left_on="wsr_web_site_sk", right_on="web_site_sk", how="inner")
        .filter(
            pl.col("d_date").is_between(pl.lit(start_date), pl.lit(end_date), closed="both")
        )
        .group_by("web_site_id")
        .agg(
            sql_sum(pl.col("sales_price")).alias("sales"),
            sql_sum(pl.col("profit")).alias("profit"),
            sql_sum(pl.col("return_amt")).alias("returns1"),
            sql_sum(pl.col("net_loss")).alias("profit_loss"),
        )
    )

    # Main query: UNION ALL of three channel frames, then GROUP BY ROLLUP(channel, id)
    store_channel = ssr.select(
        pl.lit("store channel").alias("channel"),
        (pl.lit("store") + pl.col("s_store_id")).alias("id"),
        pl.col("sales"),
        pl.col("returns1"),
        (pl.col("profit") - pl.col("profit_loss")).alias("profit"),
    )
    catalog_channel = csr.select(
        pl.lit("catalog channel").alias("channel"),
        (pl.lit("catalog_page") + pl.col("cp_catalog_page_id")).alias("id"),
        pl.col("sales"),
        pl.col("returns1"),
        (pl.col("profit") - pl.col("profit_loss")).alias("profit"),
    )
    web_channel = wsr.select(
        pl.lit("web channel").alias("channel"),
        (pl.lit("web_site") + pl.col("web_site_id")).alias("id"),
        pl.col("sales"),
        pl.col("returns1"),
        (pl.col("profit") - pl.col("profit_loss")).alias("profit"),
    )

    x = pl.concat([store_channel, catalog_channel, web_channel], how="diagonal_relaxed")

    # GROUP BY rollup(channel, id)
    result = sql_rollup(
        x,
        ["channel", "id"],
        [
            sql_sum(pl.col("sales")).alias("sales"),
            sql_sum(pl.col("returns1")).alias("returns1"),
            sql_sum(pl.col("profit")).alias("profit"),
        ],
    )

    result = result.sort(['channel', 'id'], descending=[False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("channel", False), ("id", False)],
        limit=100,
    )
