# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q79 — naive one-for-one Polars translation of the SQL."""
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
    params = load_parameters(
        int(run_config.scale_factor), query_id=79, qualification=run_config.qualification
    )
    dep_cnt = params["dep_cnt"]
    dow = params["dow"]
    year = params["year"]
    emp_min = params["emp_min"]
    emp_max = params["emp_max"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    household_demographics = get_data(
        run_config.dataset_path, "household_demographics", run_config.suffix
    )
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)

    # Inner subquery ms:
    # FROM store_sales, date_dim, store, household_demographics
    # WHERE store_sales.ss_sold_date_sk = date_dim.d_date_sk
    #   AND store_sales.ss_store_sk = store.s_store_sk
    #   AND store_sales.ss_hdemo_sk = household_demographics.hd_demo_sk
    #   AND (hd_dep_count = dep_cnt OR hd_vehicle_count > 4)
    #   AND d_dow = dow
    #   AND d_year IN (year, year+1, year+2)
    #   AND s_number_employees BETWEEN emp_min AND emp_max
    # GROUP BY ss_ticket_number, ss_customer_sk, ss_addr_sk, store.s_city
    ms = (
        store_sales
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk", how="inner")
        .join(
            household_demographics,
            left_on="ss_hdemo_sk",
            right_on="hd_demo_sk",
            how="inner",
        )
        .filter(
            ((pl.col("hd_dep_count") == dep_cnt) | (pl.col("hd_vehicle_count") > 4))
            & (pl.col("d_dow") == dow)
            & pl.col("d_year").is_in([year, year + 1, year + 2])
            & pl.col("s_number_employees").is_between(emp_min, emp_max)
        )
        .group_by(["ss_ticket_number", "ss_customer_sk", "ss_addr_sk", "s_city"])
        .agg([
            sql_sum(pl.col("ss_coupon_amt")).alias("amt"),
            sql_sum(pl.col("ss_net_profit")).alias("profit"),
        ])
    )

    # Outer query:
    # FROM ms, customer WHERE ss_customer_sk = c_customer_sk
    # SELECT c_last_name, c_first_name, SUBSTR(s_city, 1, 30), ss_ticket_number, amt, profit
    result = (
        ms
        .join(customer, left_on="ss_customer_sk", right_on="c_customer_sk", how="inner")
        .select([
            "c_last_name",
            "c_first_name",
            pl.col("s_city").str.slice(0, 30).alias("substr(s_city, 1, 30)"),
            "ss_ticket_number",
            "amt",
            "profit",
        ])
    )

    result = result.sort(['c_last_name', 'c_first_name', 'substr(s_city, 1, 30)', 'profit'], descending=[False, False, False, False]).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[
            ("c_last_name", False),
            ("c_first_name", False),
            ("substr(s_city, 1, 30)", False),
            ("profit", False),
        ],
        limit=100,
    )
