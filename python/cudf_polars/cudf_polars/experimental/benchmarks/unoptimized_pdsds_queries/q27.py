# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q27 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q27 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=27, qualification=run_config.qualification
    )

    year = params["year"]
    gen = params["gen"]
    ms = params["ms"]
    es = params["es"]
    state = params["state"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    customer_demographics = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # FROM order: store_sales, customer_demographics, date_dim, store, item
    # All WHERE conditions after all joins
    base = (
        store_sales
        .join(customer_demographics, how="inner",
              left_on="ss_cdemo_sk", right_on="cd_demo_sk")
        .join(date_dim, how="inner",
              left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, how="inner",
              left_on="ss_store_sk", right_on="s_store_sk")
        .join(item, how="inner",
              left_on="ss_item_sk", right_on="i_item_sk")
        .filter(
            (pl.col("cd_gender") == gen)
            & (pl.col("cd_marital_status") == ms)
            & (pl.col("cd_education_status") == es)
            & (pl.col("d_year") == year)
            & pl.col("s_state").is_in(state)
        )
    )

    agg_exprs = [
        pl.col("ss_quantity").mean().alias("agg1"),
        pl.col("ss_list_price").mean().alias("agg2"),
        pl.col("ss_coupon_amt").mean().alias("agg3"),
        pl.col("ss_sales_price").mean().alias("agg4"),
    ]

    # ROLLUP(i_item_id, s_state) — two levels + grand total
    # Level 2: GROUP BY i_item_id, s_state  → g_state = 0
    # Level 1: GROUP BY i_item_id           → g_state = 1, s_state = NULL
    # Level 0: grand total                   → g_state = 1, both NULL
    # The SQL SELECT includes GROUPING(s_state) as g_state.
    # sql_rollup produces nulls for grouped-out columns; we add g_state manually.
    rollup_base = sql_rollup(base, ["i_item_id", "s_state"], agg_exprs)

    # Add g_state column: 0 when s_state is not null, 1 when s_state is null
    result = rollup_base.with_columns([
        pl.when(pl.col("s_state").is_null())
        .then(pl.lit(1, dtype=pl.Int64))
        .otherwise(pl.lit(0, dtype=pl.Int64))
        .alias("g_state")
    ]).select([
        "i_item_id", "s_state", "g_state", "agg1", "agg2", "agg3", "agg4"
    ])

    result = result.sort(['i_item_id', 's_state'], descending=[False, False], nulls_last=True).limit(100)

    sort_by = [("i_item_id", False), ("s_state", False)]
    limit = 100

    return QueryResult(frame=result, sort_by=sort_by, limit=limit)
