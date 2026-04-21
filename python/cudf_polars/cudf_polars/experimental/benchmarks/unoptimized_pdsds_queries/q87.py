# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q87 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q87 naive Polars implementation.

    SELECT count(*)
    FROM (
      (SELECT DISTINCT c_last_name, c_first_name, d_date
       FROM store_sales, date_dim, customer
       WHERE ss_sold_date_sk = d_date_sk AND ss_customer_sk = c_customer_sk
         AND d_month_seq BETWEEN d_month_seq AND d_month_seq+11)
      EXCEPT
      (SELECT DISTINCT c_last_name, c_first_name, d_date
       FROM catalog_sales, date_dim, customer
       WHERE cs_sold_date_sk = d_date_sk AND cs_bill_customer_sk = c_customer_sk
         AND d_month_seq BETWEEN d_month_seq AND d_month_seq+11)
      EXCEPT
      (SELECT DISTINCT c_last_name, c_first_name, d_date
       FROM web_sales, date_dim, customer
       WHERE ws_sold_date_sk = d_date_sk AND ws_bill_customer_sk = c_customer_sk
         AND d_month_seq BETWEEN d_month_seq AND d_month_seq+11)
    ) cool_cust;

    EXCEPT -> anti-join on all SELECT columns.
    """
    params = load_parameters(
        int(run_config.scale_factor), query_id=87, qualification=run_config.qualification
    )
    d_month_seq = params["d_month_seq"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)

    # Common filtered date_dim
    date_filtered = date_dim.filter(
        pl.col("d_month_seq").is_between(d_month_seq, d_month_seq + 11)
    )

    # First operand: DISTINCT (c_last_name, c_first_name, d_date) from store_sales
    store_set = (
        store_sales
        .join(date_filtered, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(customer, left_on="ss_customer_sk", right_on="c_customer_sk", how="inner")
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )

    # Second operand: DISTINCT from catalog_sales
    catalog_set = (
        catalog_sales
        .join(date_filtered, left_on="cs_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(customer, left_on="cs_bill_customer_sk", right_on="c_customer_sk", how="inner")
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )

    # Third operand: DISTINCT from web_sales
    web_set = (
        web_sales
        .join(date_filtered, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(customer, left_on="ws_bill_customer_sk", right_on="c_customer_sk", how="inner")
        .select(["c_last_name", "c_first_name", "d_date"])
        .unique()
    )

    # EXCEPT = anti-join on all SELECT columns.
    # NULL values in join keys cause mismatches so we use a sentinel approach
    # (rule 9: .join(other, on=[...], how="anti").unique())
    # store_set EXCEPT catalog_set EXCEPT web_set
    result = (
        store_set
        .join(catalog_set, on=["c_last_name", "c_first_name", "d_date"], how="anti")
        .unique()
        .join(web_set, on=["c_last_name", "c_first_name", "d_date"], how="anti")
        .unique()
        .select([pl.len().alias("count_star()")])
    )

    return QueryResult(
        frame=result,
        sort_by=[],
        limit=None,
    )
