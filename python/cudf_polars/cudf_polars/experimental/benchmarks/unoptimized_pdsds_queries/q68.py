# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q68 — naive one-for-one Polars translation of the SQL."""
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
        int(run_config.scale_factor), query_id=68, qualification=run_config.qualification
    )
    year = params["year"]
    dep_cnt = params["dep_cnt"]
    veh_cnt = params["veh_cnt"]
    city_a = params["city_a"]
    city_b = params["city_b"]

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

    # Inner subquery dn:
    # FROM store_sales, date_dim, store, household_demographics, customer_address
    # WHERE ... (all as .filter after joins, naive rule 2)
    # GROUP BY ss_ticket_number, ss_customer_sk, ss_addr_sk, ca_city
    dn = (
        store_sales
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk", how="inner")
        .join(
            household_demographics,
            left_on="ss_hdemo_sk",
            right_on="hd_demo_sk",
            how="inner",
        )
        .join(
            customer_address,
            left_on="ss_addr_sk",
            right_on="ca_address_sk",
            how="inner",
        )
        .filter(
            pl.col("d_dom").is_between(1, 2)
            & (
                (pl.col("hd_dep_count") == dep_cnt)
                | (pl.col("hd_vehicle_count") == veh_cnt)
            )
            & pl.col("d_year").is_in([year, year + 1, year + 2])
            & pl.col("s_city").is_in([city_a, city_b])
        )
        .group_by(["ss_ticket_number", "ss_customer_sk", "ss_addr_sk", "ca_city"])
        .agg([
            sql_sum(pl.col("ss_ext_sales_price")).alias("extended_price"),
            sql_sum(pl.col("ss_ext_list_price")).alias("list_price"),
            sql_sum(pl.col("ss_ext_tax")).alias("extended_tax"),
        ])
        # ca_city in the subquery is aliased bought_city in the outer SELECT
        .rename({"ca_city": "bought_city"})
    )

    # Outer query:
    # FROM dn, customer, customer_address current_addr
    # WHERE ss_customer_sk = c_customer_sk
    #   AND customer.c_current_addr_sk = current_addr.ca_address_sk
    #   AND current_addr.ca_city <> bought_city
    current_addr = customer_address.select(
        [pl.col("ca_address_sk"), pl.col("ca_city").alias("ca_city")]
    )
    result = (
        dn
        .join(customer, left_on="ss_customer_sk", right_on="c_customer_sk", how="inner")
        .join(
            current_addr,
            left_on="c_current_addr_sk",
            right_on="ca_address_sk",
            how="inner",
        )
        .filter(pl.col("ca_city") != pl.col("bought_city"))
        .select([
            "c_last_name",
            "c_first_name",
            "ca_city",
            "bought_city",
            "ss_ticket_number",
            "extended_price",
            "extended_tax",
            "list_price",
        ])
    )

    result = result.sort(['c_last_name', 'ss_ticket_number'], descending=[False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("c_last_name", False), ("ss_ticket_number", False)],
        limit=100,
    )
