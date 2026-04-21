# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q38 — naive one-for-one Polars translation of the SQL."""
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
        int(run_config.scale_factor), query_id=38, qualification=run_config.qualification
    )
    dms = params["dms"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)

    # First INTERSECT operand: DISTINCT (c_last_name, c_first_name, d_date)
    # FROM store_sales, date_dim, customer
    # WHERE ss_sold_date_sk=d_date_sk AND ss_customer_sk=c_customer_sk
    #   AND d_month_seq BETWEEN dms AND dms+11
    store_set = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(customer, left_on="ss_customer_sk", right_on="c_customer_sk")
        .filter(pl.col("d_month_seq").is_between(dms, dms + 11))
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )

    # Second INTERSECT operand: DISTINCT (c_last_name, c_first_name, d_date)
    # FROM catalog_sales, date_dim, customer
    # WHERE cs_sold_date_sk=d_date_sk AND cs_bill_customer_sk=c_customer_sk
    catalog_set = (
        catalog_sales.join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(customer, left_on="cs_bill_customer_sk", right_on="c_customer_sk")
        .filter(pl.col("d_month_seq").is_between(dms, dms + 11))
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )

    # Third INTERSECT operand: DISTINCT (c_last_name, c_first_name, d_date)
    # FROM web_sales, date_dim, customer
    # WHERE ws_sold_date_sk=d_date_sk AND ws_bill_customer_sk=c_customer_sk
    web_set = (
        web_sales.join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(customer, left_on="ws_bill_customer_sk", right_on="c_customer_sk")
        .filter(pl.col("d_month_seq").is_between(dms, dms + 11))
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )

    # INTERSECT: chain as two sequential inner joins on (c_last_name, c_first_name, d_date)
    # then .unique() (rule 14)
    intersect_12 = store_set.join(
        catalog_set, on=["c_last_name", "c_first_name", "d_date"], how="inner"
    ).unique()
    intersect_all = intersect_12.join(
        web_set, on=["c_last_name", "c_first_name", "d_date"], how="inner"
    ).unique()

    # Outer SELECT Count(*)
    result = intersect_all.select(pl.len().cast(pl.Int64).alias("count_star()"))

    result = result.limit(100)
    return QueryResult(
        frame=result,
        sort_by=[],
        limit=100,
    )
