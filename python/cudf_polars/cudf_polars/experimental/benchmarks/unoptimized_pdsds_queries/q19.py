# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q19 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q19 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=19, qualification=run_config.qualification
    )

    year = params["year"]
    month = params["month"]
    manager = params["manager"]

    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    store = get_data(run_config.dataset_path, "store", run_config.suffix)

    # FROM order: date_dim, store_sales, item, customer, customer_address, store
    # All WHERE conditions after all joins
    result = (
        date_dim
        .join(store_sales, how="inner",
              left_on="d_date_sk", right_on="ss_sold_date_sk")
        .join(item, how="inner",
              left_on="ss_item_sk", right_on="i_item_sk")
        .join(customer, how="inner",
              left_on="ss_customer_sk", right_on="c_customer_sk")
        .join(customer_address, how="inner",
              left_on="c_current_addr_sk", right_on="ca_address_sk")
        .join(store, how="inner",
              left_on="ss_store_sk", right_on="s_store_sk")
        .filter(
            (pl.col("i_manager_id") == manager)
            & (pl.col("d_moy") == month)
            & (pl.col("d_year") == year)
            & (
                pl.col("ca_zip").str.slice(0, 5)
                != pl.col("s_zip").str.slice(0, 5)
            )
        )
        .group_by(["i_brand", "i_brand_id", "i_manufact_id", "i_manufact"])
        .agg([
            sql_sum(pl.col("ss_ext_sales_price")).alias("ext_price"),
        ])
        .select([
            pl.col("i_brand_id").alias("brand_id"),
            pl.col("i_brand").alias("brand"),
            pl.col("i_manufact_id"),
            pl.col("i_manufact"),
            pl.col("ext_price"),
        ])
    )

    result = result.sort(['ext_price', 'brand', 'brand_id', 'i_manufact_id', 'i_manufact'], descending=[True, False, False, False, False], nulls_last=True).limit(100)

    sort_by = [
        ("ext_price", True),
        ("brand", False),
        ("brand_id", False),
        ("i_manufact_id", False),
        ("i_manufact", False),
    ]
    limit = 100

    return QueryResult(frame=result, sort_by=sort_by, limit=limit)
