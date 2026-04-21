# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q18 — naive one-for-one Polars translation of the SQL."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data
from cudf_polars.experimental.benchmarks.unoptimized_pdsds_queries.sql_helpers import (
    sql_rollup,
    sql_sum,
)

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:  # type: ignore[return]
    raise NotImplementedError


def polars_impl(run_config: RunConfig) -> QueryResult:
    """TPC-DS q18 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=18, qualification=run_config.qualification
    )

    year = params["year"]
    month = params["month"]
    state = params["state"]
    es = params["es"]
    gen = params["gen"]

    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    customer_demographics = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    customer_demographics_2 = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # FROM order: catalog_sales, customer_demographics cd1, customer_demographics cd2,
    #             customer, customer_address, date_dim, item
    # All WHERE conditions after all joins
    base = (
        catalog_sales
        .join(customer_demographics, how="inner",
              left_on="cs_bill_cdemo_sk", right_on="cd_demo_sk")
        .join(customer, how="inner",
              left_on="cs_bill_customer_sk", right_on="c_customer_sk")
        .join(customer_demographics_2, how="inner",
              left_on="c_current_cdemo_sk", right_on="cd_demo_sk",
              suffix="_cd2")
        .join(customer_address, how="inner",
              left_on="c_current_addr_sk", right_on="ca_address_sk")
        .join(date_dim, how="inner",
              left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(item, how="inner",
              left_on="cs_item_sk", right_on="i_item_sk")
        .filter(
            (pl.col("cd_gender") == gen)
            & (pl.col("cd_education_status") == es)
            & pl.col("c_birth_month").is_in(month)
            & (pl.col("d_year") == year)
            & pl.col("ca_state").is_in(state)
            # correlated: c_current_cdemo_sk = cd2.cd_demo_sk already joined above
        )
    )

    agg_exprs = [
        pl.col("cs_quantity").mean().alias("agg1"),
        pl.col("cs_list_price").mean().alias("agg2"),
        pl.col("cs_coupon_amt").mean().alias("agg3"),
        pl.col("cs_sales_price").mean().alias("agg4"),
        pl.col("cs_net_profit").mean().alias("agg5"),
        pl.col("c_birth_year").mean().alias("agg6"),
        pl.col("cd_dep_count").mean().alias("agg7"),
    ]

    result = sql_rollup(base, ["i_item_id", "ca_country", "ca_state", "ca_county"], agg_exprs)

    result = result.sort(['ca_country', 'ca_state', 'ca_county', 'i_item_id'], descending=[False, False, False, False], nulls_last=True).limit(100)

    sort_by = [
        ("ca_country", False),
        ("ca_state", False),
        ("ca_county", False),
        ("i_item_id", False),
    ]
    limit = 100

    return QueryResult(frame=result, sort_by=sort_by, limit=limit)
