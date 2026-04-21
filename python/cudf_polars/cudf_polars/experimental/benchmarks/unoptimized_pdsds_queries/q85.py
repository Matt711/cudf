# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q85 — naive one-for-one Polars translation of the SQL."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data
from cudf_polars.experimental.benchmarks.unoptimized_pdsds_queries.sql_helpers import (
    sql_sum,
)

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:  # type: ignore[return]
    raise NotImplementedError


def polars_impl(run_config: RunConfig) -> QueryResult:
    """TPC-DS q85 naive Polars implementation.

    SELECT Substr(r_reason_desc, 1, 20), Avg(ws_quantity),
           Avg(wr_refunded_cash), Avg(wr_fee)
    FROM web_sales, web_returns, web_page,
         customer_demographics cd1, customer_demographics cd2,
         customer_address, date_dim, reason
    WHERE ws_web_page_sk = wp_web_page_sk
      AND ws_item_sk = wr_item_sk AND ws_order_number = wr_order_number
      AND ws_sold_date_sk = d_date_sk AND d_year = year
      AND cd1.cd_demo_sk = wr_refunded_cdemo_sk
      AND cd2.cd_demo_sk = wr_returning_cdemo_sk
      AND ca_address_sk = wr_refunded_addr_sk
      AND r_reason_sk = wr_reason_sk
      AND (complex OR conditions on marital_status, education_status, price)
      AND (complex OR conditions on ca_state, ws_net_profit)
    GROUP BY r_reason_desc
    ORDER BY Substr(r_reason_desc,1,20), Avg(ws_quantity),
             Avg(wr_refunded_cash), Avg(wr_fee)
    LIMIT 100;
    """
    params = load_parameters(
        int(run_config.scale_factor), query_id=85, qualification=run_config.qualification
    )
    year = params["year"]
    ms = params["ms"]
    es = params["es"]
    price_ranges = params["price_ranges"]
    states = params["states"]
    np_min = params["np_min"]
    np_max = params["np_max"]

    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    web_returns = get_data(run_config.dataset_path, "web_returns", run_config.suffix)
    web_page = get_data(run_config.dataset_path, "web_page", run_config.suffix)
    customer_demographics = get_data(run_config.dataset_path, "customer_demographics", run_config.suffix)
    customer_address = get_data(run_config.dataset_path, "customer_address", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    reason = get_data(run_config.dataset_path, "reason", run_config.suffix)

    # cd1 and cd2 are two aliases of customer_demographics
    cd1 = customer_demographics.select([
        pl.col("cd_demo_sk").alias("cd1_demo_sk"),
        pl.col("cd_marital_status").alias("cd1_marital_status"),
        pl.col("cd_education_status").alias("cd1_education_status"),
    ])
    cd2 = customer_demographics.select([
        pl.col("cd_demo_sk").alias("cd2_demo_sk"),
        pl.col("cd_marital_status").alias("cd2_marital_status"),
        pl.col("cd_education_status").alias("cd2_education_status"),
    ])

    # Join in FROM order: web_sales, web_returns, web_page, cd1, cd2,
    # customer_address, date_dim, reason
    # ALL WHERE conditions as .filter() AFTER joins
    result = (
        web_sales
        .join(web_returns,
              left_on=["ws_item_sk", "ws_order_number"],
              right_on=["wr_item_sk", "wr_order_number"],
              how="inner")
        .join(web_page, left_on="ws_web_page_sk", right_on="wp_web_page_sk", how="inner")
        .join(cd1, left_on="wr_refunded_cdemo_sk", right_on="cd1_demo_sk", how="inner")
        .join(cd2, left_on="wr_returning_cdemo_sk", right_on="cd2_demo_sk", how="inner")
        .join(customer_address, left_on="wr_refunded_addr_sk", right_on="ca_address_sk", how="inner")
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(reason, left_on="wr_reason_sk", right_on="r_reason_sk", how="inner")
        .filter(pl.col("d_year") == year)
        .filter(
            (
                (pl.col("cd1_marital_status") == ms[0])
                & (pl.col("cd1_marital_status") == pl.col("cd2_marital_status"))
                & (pl.col("cd1_education_status") == es[0])
                & (pl.col("cd1_education_status") == pl.col("cd2_education_status"))
                & pl.col("ws_sales_price").is_between(price_ranges[0][0], price_ranges[0][1])
            )
            | (
                (pl.col("cd1_marital_status") == ms[1])
                & (pl.col("cd1_marital_status") == pl.col("cd2_marital_status"))
                & (pl.col("cd1_education_status") == es[1])
                & (pl.col("cd1_education_status") == pl.col("cd2_education_status"))
                & pl.col("ws_sales_price").is_between(price_ranges[1][0], price_ranges[1][1])
            )
            | (
                (pl.col("cd1_marital_status") == ms[2])
                & (pl.col("cd1_marital_status") == pl.col("cd2_marital_status"))
                & (pl.col("cd1_education_status") == es[2])
                & (pl.col("cd1_education_status") == pl.col("cd2_education_status"))
                & pl.col("ws_sales_price").is_between(price_ranges[2][0], price_ranges[2][1])
            )
        )
        .filter(
            (
                (pl.col("ca_country") == "United States")
                & pl.col("ca_state").is_in(states[0:3])
                & pl.col("ws_net_profit").is_between(np_min, np_max)
            )
            | (
                (pl.col("ca_country") == "United States")
                & pl.col("ca_state").is_in(states[3:6])
                & pl.col("ws_net_profit").is_between(np_min, np_max)
            )
        )
        .group_by("r_reason_desc")
        .agg([
            pl.col("ws_quantity").mean().alias("avg(ws_quantity)"),
            pl.col("wr_refunded_cash").mean().alias("avg(wr_refunded_cash)"),
            pl.col("wr_fee").mean().alias("avg(wr_fee)"),
        ])
        .select([
            pl.col("r_reason_desc").str.slice(0, 20).alias("substr(r_reason_desc, 1, 20)"),
            "avg(ws_quantity)",
            "avg(wr_refunded_cash)",
            "avg(wr_fee)",
        ])
    )

    result = result.sort(['substr(r_reason_desc, 1, 20)', 'avg(ws_quantity)', 'avg(wr_refunded_cash)', 'avg(wr_fee)'], descending=[False, False, False, False]).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[
            ("substr(r_reason_desc, 1, 20)", False),
            ("avg(ws_quantity)", False),
            ("avg(wr_refunded_cash)", False),
            ("avg(wr_fee)", False),
        ],
        limit=100,
    )
