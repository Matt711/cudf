# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 79."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import (
    QueryResult,
    get_data,
    is_duckdb_validate,
    sql_sum,
)

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 79."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=79,
        qualification=run_config.qualification,
    )
    dep_cnt = params["dep_cnt"]
    dow = params["dow"]
    year = params["year"]
    emp_min = params["emp_min"]
    emp_max = params["emp_max"]

    return f"""
    SELECT c_last_name,
                   c_first_name,
                   Substr(s_city, 1, 30),
                   ss_ticket_number,
                   amt,
                   profit
    FROM   (SELECT ss_ticket_number,
                   ss_customer_sk,
                   store.s_city,
                   Sum(ss_coupon_amt) amt,
                   Sum(ss_net_profit) profit
            FROM   store_sales,
                   date_dim,
                   store,
                   household_demographics
            WHERE  store_sales.ss_sold_date_sk = date_dim.d_date_sk
                   AND store_sales.ss_store_sk = store.s_store_sk
                   AND store_sales.ss_hdemo_sk = household_demographics.hd_demo_sk
                   AND ( household_demographics.hd_dep_count = {dep_cnt}
                          OR household_demographics.hd_vehicle_count > 4 )
                   AND date_dim.d_dow = {dow}
                   AND date_dim.d_year IN ( {year}, {year} + 1, {year} + 2 )
                   AND store.s_number_employees BETWEEN {emp_min} AND {emp_max}
            GROUP  BY ss_ticket_number,
                      ss_customer_sk,
                      ss_addr_sk,
                      store.s_city) ms,
           customer
    WHERE  ss_customer_sk = c_customer_sk
    ORDER  BY c_last_name,
              c_first_name,
              Substr(s_city, 1, 30),
              profit
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 79."""
    validate = is_duckdb_validate(run_config)
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=79,
        qualification=run_config.qualification,
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
    ms = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(household_demographics, left_on="ss_hdemo_sk", right_on="hd_demo_sk")
        .filter(
            ((pl.col("hd_dep_count") == dep_cnt) | (pl.col("hd_vehicle_count") > 4))
            & (pl.col("d_dow") == dow)
            & (pl.col("d_year").is_in([year, year + 1, year + 2]))
            & (pl.col("s_number_employees").is_between(emp_min, emp_max))
        )
        .group_by(["ss_ticket_number", "ss_customer_sk", "ss_addr_sk", "s_city"])
        .agg(
            [
                sql_sum("ss_coupon_amt", validate=validate).alias("amt"),
                sql_sum("ss_net_profit", validate=validate).alias("profit"),
            ]
        )
    )
    sort_by = {
        "c_last_name": False,
        "c_first_name": False,
        "substr(s_city, 1, 30)": False,
        "profit": False,
    }
    limit = 100
    return QueryResult(
        frame=(
            ms.join(customer, left_on="ss_customer_sk", right_on="c_customer_sk")
            .select(
                [
                    "c_last_name",
                    "c_first_name",
                    pl.col("s_city").str.slice(0, 30).alias("substr(s_city, 1, 30)"),
                    "ss_ticket_number",
                    "amt",
                    "profit",
                ]
            )
            .sort(list(sort_by.keys()), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
