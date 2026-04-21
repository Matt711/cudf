# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q1 — naive one-for-one Polars translation of the SQL."""

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
        query_id=1,
        qualification=run_config.qualification,
    )
    year = params["year"]
    state = params["state"]

    store_returns = get_data(run_config.dataset_path, "store_returns", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)

    # CTE: customer_total_return
    # FROM store_returns, date_dim WHERE sr_returned_date_sk = d_date_sk AND d_year = year
    # GROUP BY sr_customer_sk, sr_store_sk
    customer_total_return = (
        store_returns.join(date_dim, left_on="sr_returned_date_sk", right_on="d_date_sk", how="inner")
        .filter(pl.col("d_year") == year)
        .group_by(["sr_customer_sk", "sr_store_sk"])
        .agg(sql_sum(pl.col("sr_return_amt")).alias("ctr_total_return"))
        .rename({"sr_customer_sk": "ctr_customer_sk", "sr_store_sk": "ctr_store_sk"})
    )

    # Correlated subquery: SELECT Avg(ctr_total_return) * 1.2 FROM customer_total_return ctr2
    # WHERE ctr1.ctr_store_sk = ctr2.ctr_store_sk
    # => per-store average, joined back on ctr_store_sk
    store_avg = customer_total_return.group_by("ctr_store_sk").agg(
        (pl.col("ctr_total_return").mean() * 1.2).alias("avg_threshold")
    )

    # Main query:
    # FROM customer_total_return ctr1, store, customer
    # WHERE ctr1.ctr_total_return > (subquery)
    #   AND s_store_sk = ctr1.ctr_store_sk
    #   AND s_state = state
    #   AND ctr1.ctr_customer_sk = c_customer_sk
    # All WHERE conditions applied after all joins (naive rule 2)
    result = (
        customer_total_return.join(store_avg, on="ctr_store_sk", how="inner")
        .join(store, left_on="ctr_store_sk", right_on="s_store_sk", how="inner")
        .join(customer, left_on="ctr_customer_sk", right_on="c_customer_sk", how="inner")
        .filter(
            (pl.col("ctr_total_return") > pl.col("avg_threshold"))
            & (pl.col("s_state") == state)
        )
        .select(["c_customer_id"])
        .sort("c_customer_id", nulls_last=True)
        .limit(100)
    )

    return QueryResult(frame=result, sort_by=[("c_customer_id", False)], limit=100)
