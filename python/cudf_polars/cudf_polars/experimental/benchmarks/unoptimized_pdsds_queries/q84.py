# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q84 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q84 naive Polars implementation.

    SELECT c_customer_id AS customer_id,
           c_last_name || ', ' || c_first_name AS customername
    FROM customer, customer_address, customer_demographics,
         household_demographics, income_band, store_returns
    WHERE ca_city = city
      AND c_current_addr_sk = ca_address_sk
      AND ib_lower_bound >= income AND ib_upper_bound <= income + 50000
      AND ib_income_band_sk = hd_income_band_sk
      AND cd_demo_sk = c_current_cdemo_sk
      AND hd_demo_sk = c_current_hdemo_sk
      AND sr_cdemo_sk = cd_demo_sk
    ORDER BY c_customer_id LIMIT 100;
    """
    params = load_parameters(
        int(run_config.scale_factor), query_id=84, qualification=run_config.qualification
    )
    city = params["city"]
    income = params["income"]

    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(run_config.dataset_path, "customer_address", run_config.suffix)
    customer_demographics = get_data(run_config.dataset_path, "customer_demographics", run_config.suffix)
    household_demographics = get_data(run_config.dataset_path, "household_demographics", run_config.suffix)
    income_band = get_data(run_config.dataset_path, "income_band", run_config.suffix)
    store_returns = get_data(run_config.dataset_path, "store_returns", run_config.suffix)

    # Join in FROM order: customer, customer_address, customer_demographics,
    # household_demographics, income_band, store_returns
    # ALL WHERE as .filter() after joins (naive rule 2)
    result = (
        customer
        .join(customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk", how="inner")
        .join(customer_demographics, left_on="c_current_cdemo_sk", right_on="cd_demo_sk", how="inner")
        .join(household_demographics, left_on="c_current_hdemo_sk", right_on="hd_demo_sk", how="inner")
        .join(income_band, left_on="hd_income_band_sk", right_on="ib_income_band_sk", how="inner")
        .join(store_returns, left_on="c_current_cdemo_sk", right_on="sr_cdemo_sk", how="inner")
        .filter(
            (pl.col("ca_city") == city)
            & (pl.col("ib_lower_bound") >= income)
            & (pl.col("ib_upper_bound") <= income + 50000)
        )
        .select([
            pl.col("c_customer_id").alias("customer_id"),
            (pl.col("c_last_name") + pl.lit(", ") + pl.col("c_first_name")).alias("customername"),
        ])
    )

    result = result.sort('customer_id', descending=False, nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("customer_id", False)],
        limit=100,
    )
