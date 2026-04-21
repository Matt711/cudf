# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q24 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q24 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=24, qualification=run_config.qualification
    )

    market = params["market"]
    color = params["color"][0]
    amountone = params["amountone"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(run_config.dataset_path, "store_returns", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )

    # CTE: ssales
    # FROM: store_sales, store_returns, store, item, customer, customer_address
    # WHERE conditions after all joins
    ssales = (
        store_sales
        .join(store_returns, how="inner",
              left_on=["ss_ticket_number", "ss_item_sk"],
              right_on=["sr_ticket_number", "sr_item_sk"])
        .join(store, how="inner",
              left_on="ss_store_sk", right_on="s_store_sk")
        .join(item, how="inner",
              left_on="ss_item_sk", right_on="i_item_sk")
        .join(customer, how="inner",
              left_on="ss_customer_sk", right_on="c_customer_sk")
        .join(customer_address, how="inner",
              left_on="c_current_addr_sk", right_on="ca_address_sk")
        .filter(
            (pl.col("c_birth_country") != pl.col("ca_country").str.to_uppercase())
            & (pl.col("s_zip") == pl.col("ca_zip"))
            & (pl.col("s_market_id") == market)
        )
        .group_by([
            "c_last_name", "c_first_name", "s_store_name",
            "ca_state", "s_state", "i_color",
            "i_current_price", "i_manager_id", "i_units", "i_size",
        ])
        .agg([
            sql_sum(pl.col(amountone)).alias("netpaid"),
        ])
    )

    # Compute the threshold: 0.05 * avg(netpaid) from ssales (all colors)
    threshold_lf = ssales.select(
        (pl.col("netpaid").mean() * 0.05).alias("threshold")
    )

    # Outer query: filter ssales by color, group by name/store, HAVING sum > threshold
    result = (
        ssales
        .filter(pl.col("i_color") == color)
        .group_by(["c_last_name", "c_first_name", "s_store_name"])
        .agg([
            sql_sum(pl.col("netpaid")).alias("paid"),
        ])
        .join(threshold_lf, how="cross")
        .filter(pl.col("paid") > pl.col("threshold"))
        .select(["c_last_name", "c_first_name", "s_store_name", "paid"])
    )

    sort_by = [
        ("c_last_name", False),
        ("c_first_name", False),
        ("s_store_name", False),
    ]
    limit = None

    result = result.sort(["c_last_name", "c_first_name", "s_store_name"], descending=[False, False, False], nulls_last=True)
    return QueryResult(frame=result, sort_by=sort_by, limit=limit)
