# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q94 — naive one-for-one Polars translation of the SQL."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data
from cudf_polars.experimental.benchmarks.unoptimized_pdsds_queries.sql_helpers import (
    sql_n_unique,
    sql_sum,
)

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:  # type: ignore[return]
    raise NotImplementedError


def polars_impl(run_config: RunConfig) -> QueryResult:
    """TPC-DS q94 naive Polars implementation.

    SELECT Count(DISTINCT ws_order_number) AS 'order count',
           Sum(ws_ext_ship_cost) AS 'total shipping cost',
           Sum(ws_net_profit) AS 'total net profit'
    FROM web_sales ws1, date_dim, customer_address, web_site
    WHERE d_date BETWEEN date AND date + 60 days
      AND ws1.ws_ship_date_sk = d_date_sk
      AND ws1.ws_ship_addr_sk = ca_address_sk
      AND ca_state = state
      AND ws1.ws_web_site_sk = web_site_sk
      AND web_company_name = web_company_name
      AND EXISTS (SELECT * FROM web_sales ws2
                  WHERE ws1.ws_order_number = ws2.ws_order_number
                    AND ws1.ws_warehouse_sk <> ws2.ws_warehouse_sk)
      AND NOT EXISTS (SELECT * FROM web_returns wr1
                      WHERE ws1.ws_order_number = wr1.wr_order_number)
    ORDER BY count(DISTINCT ws_order_number) LIMIT 100;

    EXISTS -> semi-join on ws_order_number (multi-warehouse orders).
    NOT EXISTS -> anti-join on wr_order_number.
    """
    params = load_parameters(
        int(run_config.scale_factor), query_id=94, qualification=run_config.qualification
    )
    date = params["date"]
    state = params["state"]
    web_company_name = params["web_company_name"]

    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer_address = get_data(run_config.dataset_path, "customer_address", run_config.suffix)
    web_site = get_data(run_config.dataset_path, "web_site", run_config.suffix)
    web_returns = get_data(run_config.dataset_path, "web_returns", run_config.suffix)

    start_date_py = datetime.strptime(date, "%Y-%m-%d").date()
    end_date_py = start_date_py + timedelta(days=60)
    start_date = pl.date(start_date_py.year, start_date_py.month, start_date_py.day)
    end_date = pl.date(end_date_py.year, end_date_py.month, end_date_py.day)

    # EXISTS (SELECT * FROM web_sales ws2
    #         WHERE ws1.ws_order_number = ws2.ws_order_number
    #           AND ws1.ws_warehouse_sk <> ws2.ws_warehouse_sk)
    mw_orders = (
        web_sales.select(["ws_order_number", "ws_warehouse_sk"])
        .join(
            web_sales.select(["ws_order_number", "ws_warehouse_sk"])
            .rename({"ws_warehouse_sk": "ws2_warehouse_sk"}),
            on="ws_order_number",
        )
        .filter(pl.col("ws_warehouse_sk") != pl.col("ws2_warehouse_sk"))
        .select("ws_order_number")
    )

    # NOT EXISTS (SELECT * FROM web_returns wr1 WHERE ws1.ws_order_number = wr1.wr_order_number)
    # Main query: web_sales ws1, date_dim, customer_address, web_site
    result = (
        web_sales
        .join(date_dim, left_on="ws_ship_date_sk", right_on="d_date_sk", how="inner")
        .join(customer_address, left_on="ws_ship_addr_sk", right_on="ca_address_sk", how="inner")
        .join(web_site, left_on="ws_web_site_sk", right_on="web_site_sk", how="inner")
        .filter(
            pl.col("d_date").is_between(start_date, end_date)
            & (pl.col("ca_state") == state)
            & (pl.col("web_company_name") == web_company_name)
        )
        .join(mw_orders, on="ws_order_number", how="semi")
        # SQL: NULL <> x = UNKNOWN, so EXISTS never holds when ws_warehouse_sk is NULL
        .filter(pl.col("ws_warehouse_sk").is_not_null())
        .join(web_returns, left_on="ws_order_number", right_on="wr_order_number", how="anti")
        .select([
            sql_n_unique(pl.col("ws_order_number")).alias("order count"),
            sql_sum(pl.col("ws_ext_ship_cost")).alias("total shipping cost"),
            sql_sum(pl.col("ws_net_profit")).alias("total net profit"),
        ])
    )

    result = result.sort('order count', descending=False, nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("order count", False)],
        limit=100,
    )
