# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q30 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q30 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=30, qualification=run_config.qualification
    )

    year = params["year"]
    state = params["state"]

    web_returns = get_data(run_config.dataset_path, "web_returns", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)

    # CTE: customer_total_return
    # FROM: web_returns, date_dim, customer_address
    # WHERE: wr_returned_date_sk=d_date_sk AND d_year=year AND wr_returning_addr_sk=ca_address_sk
    # GROUP BY wr_returning_customer_sk, ca_state
    customer_total_return = (
        web_returns
        .join(date_dim, how="inner",
              left_on="wr_returned_date_sk", right_on="d_date_sk")
        .join(customer_address, how="inner",
              left_on="wr_returning_addr_sk", right_on="ca_address_sk")
        .filter(pl.col("d_year") == year)
        .group_by([
            pl.col("wr_returning_customer_sk").alias("ctr_customer_sk"),
            pl.col("ca_state").alias("ctr_state"),
        ])
        .agg([
            sql_sum(pl.col("wr_return_amt")).alias("ctr_total_return"),
        ])
    )

    # Correlated subquery: avg(ctr_total_return)*1.2 per ctr_state
    # Compute as separate LazyFrame and join back on ctr_state
    state_avg = (
        customer_total_return
        .group_by("ctr_state")
        .agg([
            (pl.col("ctr_total_return").mean() * 1.2).alias("avg_threshold"),
        ])
    )

    # Main query: FROM customer_total_return ctr1, customer_address, customer
    # WHERE ctr1.ctr_total_return > avg*1.2 (correlated on ctr_state)
    #   AND ca_address_sk=c_current_addr_sk AND ca_state=state AND ctr1.ctr_customer_sk=c_customer_sk
    # Implement correlated subquery by joining ctr1 with state_avg
    ctr1 = customer_total_return.join(
        state_avg, on="ctr_state", how="inner"
    ).filter(
        pl.col("ctr_total_return") > pl.col("avg_threshold")
    )

    result = (
        ctr1
        .join(customer, how="inner",
              left_on="ctr_customer_sk", right_on="c_customer_sk")
        .join(customer_address, how="inner",
              left_on="c_current_addr_sk", right_on="ca_address_sk")
        .filter(pl.col("ca_state") == state)
        .select([
            "c_customer_id",
            "c_salutation",
            "c_first_name",
            "c_last_name",
            "c_preferred_cust_flag",
            "c_birth_day",
            "c_birth_month",
            "c_birth_year",
            "c_birth_country",
            "c_login",
            "c_email_address",
            "c_last_review_date_sk",
            "ctr_total_return",
        ])
    )

    result = result.sort(['c_customer_id', 'c_salutation', 'c_first_name', 'c_last_name', 'c_preferred_cust_flag', 'c_birth_day', 'c_birth_month', 'c_birth_year', 'c_birth_country', 'c_login', 'c_email_address', 'c_last_review_date_sk', 'ctr_total_return'], descending=[False, False, False, False, False, False, False, False, False, False, False, False, False], nulls_last=True).limit(100)

    sort_by = [
        ("c_customer_id", False),
        ("c_salutation", False),
        ("c_first_name", False),
        ("c_last_name", False),
        ("c_preferred_cust_flag", False),
        ("c_birth_day", False),
        ("c_birth_month", False),
        ("c_birth_year", False),
        ("c_birth_country", False),
        ("c_login", False),
        ("c_email_address", False),
        ("c_last_review_date_sk", False),
        ("ctr_total_return", False),
    ]
    limit = 100

    return QueryResult(frame=result, sort_by=sort_by, limit=limit)
