# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q46 — naive one-for-one Polars translation of the SQL."""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data
from cudf_polars.experimental.benchmarks.unoptimized_pdsds_queries.sql_helpers import sql_sum

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:  # type: ignore[return]
    raise NotImplementedError


def polars_impl(run_config: RunConfig) -> QueryResult:
    params = load_parameters(
        int(run_config.scale_factor), query_id=46, qualification=run_config.qualification
    )
    year = params["year"]
    hd_dep_count = params["hd_dep_count"]
    hd_vehicle_count = params["hd_vehicle_count"]
    cities = params["cities"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    household_demographics = get_data(
        run_config.dataset_path, "household_demographics", run_config.suffix
    )
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)

    # Subquery dn:
    # FROM store_sales, date_dim, store, household_demographics, customer_address
    # WHERE ss_sold_date_sk=d_date_sk AND ss_store_sk=s_store_sk
    #   AND ss_hdemo_sk=hd_demo_sk AND ss_addr_sk=ca_address_sk
    #   AND (hd_dep_count=hd_dep_count OR hd_vehicle_count=hd_vehicle_count)
    #   AND d_dow IN (6, 0)
    #   AND d_year IN (year, year+1, year+2)
    #   AND s_city IN (cities)
    # GROUP BY ss_ticket_number, ss_customer_sk, ss_addr_sk, ca_city
    # All WHERE conditions after all joins (naive rule 2)
    dn = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(household_demographics, left_on="ss_hdemo_sk", right_on="hd_demo_sk")
        .join(customer_address, left_on="ss_addr_sk", right_on="ca_address_sk")
        .filter(
            (
                (pl.col("hd_dep_count") == hd_dep_count)
                | (pl.col("hd_vehicle_count") == hd_vehicle_count)
            )
            & (pl.col("d_dow").is_in([6, 0]))
            & (pl.col("d_year").is_in([year, year + 1, year + 2]))
            & (pl.col("s_city").is_in(cities))
        )
        .group_by(["ss_ticket_number", "ss_customer_sk", "ss_addr_sk", "ca_city"])
        .agg(
            [
                sql_sum(pl.col("ss_coupon_amt")).alias("amt"),
                sql_sum(pl.col("ss_net_profit")).alias("profit"),
            ]
        )
        .with_columns(pl.col("ca_city").alias("bought_city"))
        .select(["ss_ticket_number", "ss_customer_sk", "bought_city", "amt", "profit"])
    )

    # Outer query: FROM dn, customer, customer_address current_addr
    # WHERE ss_customer_sk = c_customer_sk
    #   AND customer.c_current_addr_sk = current_addr.ca_address_sk
    #   AND current_addr.ca_city <> bought_city
    result = (
        dn.join(customer, left_on="ss_customer_sk", right_on="c_customer_sk")
        .join(
            customer_address,
            left_on="c_current_addr_sk",
            right_on="ca_address_sk",
            suffix="_current",
        )
        .filter(pl.col("ca_city") != pl.col("bought_city"))
        .select(
            [
                "c_last_name",
                "c_first_name",
                "ca_city",
                "bought_city",
                "ss_ticket_number",
                "amt",
                "profit",
            ]
        )
    )

    result = result.sort(['c_last_name', 'c_first_name', 'ca_city', 'bought_city', 'ss_ticket_number'], descending=[False, False, False, False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[
            ("c_last_name", False),
            ("c_first_name", False),
            ("ca_city", False),
            ("bought_city", False),
            ("ss_ticket_number", False),
        ],
        limit=100,
    )
