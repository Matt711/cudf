# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q81 — naive one-for-one Polars translation of the SQL."""
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
        int(run_config.scale_factor), query_id=81, qualification=run_config.qualification
    )
    state = params["state"]

    catalog_returns = get_data(
        run_config.dataset_path, "catalog_returns", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)

    # CTE customer_total_return:
    # FROM catalog_returns, date_dim, customer_address
    # WHERE cr_returned_date_sk = d_date_sk AND d_year = 1999
    #   AND cr_returning_addr_sk = ca_address_sk
    # GROUP BY cr_returning_customer_sk, ca_state
    customer_total_return = (
        catalog_returns
        .join(date_dim, left_on="cr_returned_date_sk", right_on="d_date_sk", how="inner")
        .join(
            customer_address,
            left_on="cr_returning_addr_sk",
            right_on="ca_address_sk",
            how="inner",
        )
        .filter(pl.col("d_year") == 1999)
        .group_by(["cr_returning_customer_sk", "ca_state"])
        .agg(
            sql_sum(pl.col("cr_return_amt_inc_tax")).alias("ctr_total_return")
        )
        .rename({
            "cr_returning_customer_sk": "ctr_customer_sk",
            "ca_state": "ctr_state",
        })
    )

    # Correlated subquery: SELECT Avg(ctr_total_return) * 1.2 FROM customer_total_return ctr2
    # WHERE ctr1.ctr_state = ctr2.ctr_state
    # => compute per-state average, then join back
    state_avg = (
        customer_total_return
        .group_by("ctr_state")
        .agg((pl.col("ctr_total_return").mean() * 1.2).alias("avg_threshold"))
    )

    # Main query:
    # FROM customer_total_return ctr1, customer_address, customer
    # WHERE ctr1.ctr_total_return > (subquery avg per state)
    #   AND ca_address_sk = c_current_addr_sk
    #   AND ca_state = state
    #   AND ctr1.ctr_customer_sk = c_customer_sk
    result = (
        customer_total_return
        .join(state_avg, on="ctr_state", how="inner")
        .filter(pl.col("ctr_total_return") > pl.col("avg_threshold"))
        .join(customer, left_on="ctr_customer_sk", right_on="c_customer_sk", how="inner")
        .join(
            customer_address,
            left_on="c_current_addr_sk",
            right_on="ca_address_sk",
            how="inner",
        )
        .filter(pl.col("ca_state") == state)
        .select([
            "c_customer_id",
            "c_salutation",
            "c_first_name",
            "c_last_name",
            "ca_street_number",
            "ca_street_name",
            "ca_street_type",
            "ca_suite_number",
            "ca_city",
            "ca_county",
            "ca_state",
            "ca_zip",
            "ca_country",
            "ca_gmt_offset",
            "ca_location_type",
            "ctr_total_return",
        ])
    )

    result = result.sort(['c_customer_id', 'c_salutation', 'c_first_name', 'c_last_name', 'ca_street_number', 'ca_street_name', 'ca_street_type', 'ca_suite_number', 'ca_city', 'ca_county', 'ca_state', 'ca_zip', 'ca_country', 'ca_gmt_offset', 'ca_location_type', 'ctr_total_return'], descending=[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[
            ("c_customer_id", False),
            ("c_salutation", False),
            ("c_first_name", False),
            ("c_last_name", False),
            ("ca_street_number", False),
            ("ca_street_name", False),
            ("ca_street_type", False),
            ("ca_suite_number", False),
            ("ca_city", False),
            ("ca_county", False),
            ("ca_state", False),
            ("ca_zip", False),
            ("ca_country", False),
            ("ca_gmt_offset", False),
            ("ca_location_type", False),
            ("ctr_total_return", False),
        ],
        limit=100,
    )
