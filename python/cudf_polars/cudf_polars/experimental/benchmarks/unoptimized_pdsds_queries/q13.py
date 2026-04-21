# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q13 — naive one-for-one Polars translation of the SQL."""

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
        int(run_config.scale_factor),
        query_id=13,
        qualification=run_config.qualification,
    )
    ms = params["ms"]       # 3 marital statuses
    es = params["es"]       # 3 education statuses
    state = params["state"] # 9 states

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    customer_demographics = get_data(run_config.dataset_path, "customer_demographics", run_config.suffix)
    household_demographics = get_data(run_config.dataset_path, "household_demographics", run_config.suffix)
    customer_address = get_data(run_config.dataset_path, "customer_address", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # FROM store_sales, store, customer_demographics, household_demographics, customer_address, date_dim
    # WHERE (join conditions) AND d_year=2001 AND (OR conditions for demo+address)
    # All WHERE after all joins (naive)
    result = (
        store_sales.join(store, left_on="ss_store_sk", right_on="s_store_sk", how="inner")
        .join(customer_demographics, left_on="ss_cdemo_sk", right_on="cd_demo_sk", how="inner")
        .join(household_demographics, left_on="ss_hdemo_sk", right_on="hd_demo_sk", how="inner")
        .join(customer_address, left_on="ss_addr_sk", right_on="ca_address_sk", how="inner")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(
            (pl.col("d_year") == 2001)
            & (pl.col("ca_country") == "United States")
            & (
                (
                    (pl.col("cd_marital_status") == ms[0])
                    & (pl.col("cd_education_status") == es[0])
                    & pl.col("ss_sales_price").is_between(100.00, 150.00, closed="both")
                    & (pl.col("hd_dep_count") == 3)
                )
                | (
                    (pl.col("cd_marital_status") == ms[1])
                    & (pl.col("cd_education_status") == es[1])
                    & pl.col("ss_sales_price").is_between(50.00, 100.00, closed="both")
                    & (pl.col("hd_dep_count") == 1)
                )
                | (
                    (pl.col("cd_marital_status") == ms[2])
                    & (pl.col("cd_education_status") == es[2])
                    & pl.col("ss_sales_price").is_between(150.00, 200.00, closed="both")
                    & (pl.col("hd_dep_count") == 1)
                )
            )
            & (
                (
                    pl.col("ca_state").is_in(state[0:3])
                    & pl.col("ss_net_profit").is_between(100, 200, closed="both")
                )
                | (
                    pl.col("ca_state").is_in(state[3:6])
                    & pl.col("ss_net_profit").is_between(150, 300, closed="both")
                )
                | (
                    pl.col("ca_state").is_in(state[6:9])
                    & pl.col("ss_net_profit").is_between(50, 250, closed="both")
                )
            )
        )
        .select(
            pl.col("ss_quantity").mean().alias("avg(ss_quantity)"),
            pl.col("ss_ext_sales_price").mean().alias("avg(ss_ext_sales_price)"),
            pl.col("ss_ext_wholesale_cost").mean().alias("avg(ss_ext_wholesale_cost)"),
            sql_sum(pl.col("ss_ext_wholesale_cost")).alias("sum(ss_ext_wholesale_cost)"),
        )
    )

    return QueryResult(frame=result, sort_by=[], limit=None)
