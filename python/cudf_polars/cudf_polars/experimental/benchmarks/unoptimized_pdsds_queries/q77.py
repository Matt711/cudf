# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q77 — naive one-for-one Polars translation of the SQL."""
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
        int(run_config.scale_factor), query_id=77, qualification=run_config.qualification
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
    web_page = get_data(run_config.dataset_path, "web_page", run_config.suffix)

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

    # CTE ss: FROM store_sales, date_dim, store
    # WHERE ss_sold_date_sk = d_date_sk AND d_date BETWEEN ... AND ss_store_sk = s_store_sk
    # GROUP BY s_store_sk
    ss = (
        store_sales
        .join(filtered_dates, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk", how="inner")
        .group_by("ss_store_sk")
        .agg([
            sql_sum(pl.col("ss_ext_sales_price")).alias("sales"),
            sql_sum(pl.col("ss_net_profit")).alias("profit"),
        ])
    )

    # CTE sr: FROM store_returns, date_dim, store
    # WHERE sr_returned_date_sk = d_date_sk AND ... AND sr_store_sk = s_store_sk
    # GROUP BY s_store_sk
    sr = (
        store_returns
        .join(filtered_dates, left_on="sr_returned_date_sk", right_on="d_date_sk", how="inner")
        .join(store, left_on="sr_store_sk", right_on="s_store_sk", how="inner")
        .group_by("sr_store_sk")
        .agg([
            sql_sum(pl.col("sr_return_amt")).alias("returns1"),
            sql_sum(pl.col("sr_net_loss")).alias("profit_loss"),
        ])
    )

    # CTE cs: FROM catalog_sales, date_dim
    # WHERE cs_sold_date_sk = d_date_sk AND d_date BETWEEN ...
    # GROUP BY cs_call_center_sk
    cs = (
        catalog_sales
        .join(filtered_dates, left_on="cs_sold_date_sk", right_on="d_date_sk", how="inner")
        .group_by("cs_call_center_sk")
        .agg([
            sql_sum(pl.col("cs_ext_sales_price")).alias("sales"),
            sql_sum(pl.col("cs_net_profit")).alias("profit"),
        ])
    )

    # CTE cr: FROM catalog_returns, date_dim
    # WHERE cr_returned_date_sk = d_date_sk AND d_date BETWEEN ...
    # GROUP BY cr_call_center_sk
    cr = (
        catalog_returns
        .join(filtered_dates, left_on="cr_returned_date_sk", right_on="d_date_sk", how="inner")
        .group_by("cr_call_center_sk")
        .agg([
            sql_sum(pl.col("cr_return_amount")).alias("returns1"),
            sql_sum(pl.col("cr_net_loss")).alias("profit_loss"),
        ])
    )

    # CTE ws: FROM web_sales, date_dim, web_page
    # WHERE ws_sold_date_sk = d_date_sk AND ... AND ws_web_page_sk = wp_web_page_sk
    # GROUP BY wp_web_page_sk
    ws = (
        web_sales
        .join(filtered_dates, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(web_page, left_on="ws_web_page_sk", right_on="wp_web_page_sk", how="inner")
        .group_by("ws_web_page_sk")
        .agg([
            sql_sum(pl.col("ws_ext_sales_price")).alias("sales"),
            sql_sum(pl.col("ws_net_profit")).alias("profit"),
        ])
    )

    # CTE wr: FROM web_returns, date_dim, web_page
    # WHERE wr_returned_date_sk = d_date_sk AND ... AND wr_web_page_sk = wp_web_page_sk
    # GROUP BY wp_web_page_sk
    wr = (
        web_returns
        .join(filtered_dates, left_on="wr_returned_date_sk", right_on="d_date_sk", how="inner")
        .join(web_page, left_on="wr_web_page_sk", right_on="wp_web_page_sk", how="inner")
        .group_by("wr_web_page_sk")
        .agg([
            sql_sum(pl.col("wr_return_amt")).alias("returns1"),
            sql_sum(pl.col("wr_net_loss")).alias("profit_loss"),
        ])
    )

    # Build combined x: UNION ALL of three channel subqueries
    store_channel = (
        ss
        .join(sr, left_on="ss_store_sk", right_on="sr_store_sk", how="left")
        .select([
            pl.lit("store channel").alias("channel"),
            pl.col("ss_store_sk").cast(pl.Int64).alias("id"),
            "sales",
            pl.col("returns1").fill_null(0).alias("returns1"),
            (pl.col("profit") - pl.col("profit_loss").fill_null(0)).alias("profit"),
        ])
    )

    # catalog: FROM cs, cr (comma = cross/inner join per SQL)
    catalog_channel = (
        cs
        .join(cr, how="cross")
        .select([
            pl.lit("catalog channel").alias("channel"),
            pl.col("cs_call_center_sk").cast(pl.Int64).alias("id"),
            "sales",
            "returns1",
            (pl.col("profit") - pl.col("profit_loss")).alias("profit"),
        ])
    )

    web_channel = (
        ws
        .join(wr, left_on="ws_web_page_sk", right_on="wr_web_page_sk", how="left")
        .select([
            pl.lit("web channel").alias("channel"),
            pl.col("ws_web_page_sk").cast(pl.Int64).alias("id"),
            "sales",
            pl.col("returns1").fill_null(0).alias("returns1"),
            (pl.col("profit") - pl.col("profit_loss").fill_null(0)).alias("profit"),
        ])
    )

    combined = pl.concat([store_channel, catalog_channel, web_channel], how="diagonal_relaxed")

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
