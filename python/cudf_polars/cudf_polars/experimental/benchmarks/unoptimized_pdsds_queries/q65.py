# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q65 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q65 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=65, qualification=run_config.qualification
    )
    dms = params["dms"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # Inline view sa (= sc base):
    # SELECT ss_store_sk, ss_item_sk, Sum(ss_sales_price) AS revenue
    # FROM store_sales, date_dim
    # WHERE ss_sold_date_sk = d_date_sk AND d_month_seq BETWEEN dms AND dms+11
    # GROUP BY ss_store_sk, ss_item_sk
    sa = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(pl.col("d_month_seq").is_between(dms, dms + 11))
        .group_by(["ss_store_sk", "ss_item_sk"])
        .agg(sql_sum(pl.col("ss_sales_price")).alias("revenue"))
    )

    # sc = sa (same query, reused)
    sc = sa

    # sb (correlated inline view):
    # SELECT ss_store_sk, Avg(revenue) AS ave FROM sa GROUP BY ss_store_sk
    sb = sa.group_by("ss_store_sk").agg(pl.col("revenue").mean().alias("ave"))

    # Main query:
    # FROM store, item, sb, sc
    # WHERE sb.ss_store_sk = sc.ss_store_sk
    #   AND sc.revenue <= 0.1 * sb.ave
    #   AND s_store_sk = sc.ss_store_sk
    #   AND i_item_sk = sc.ss_item_sk
    # SELECT s_store_name, i_item_desc, sc.revenue, i_current_price, i_wholesale_cost, i_brand
    result = (
        sc.join(sb, on="ss_store_sk", how="inner")
        .filter(pl.col("revenue") <= 0.1 * pl.col("ave"))
        .join(store, left_on="ss_store_sk", right_on="s_store_sk", how="inner")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk", how="inner")
        .select(
            "s_store_name",
            "i_item_desc",
            "revenue",
            "i_current_price",
            "i_wholesale_cost",
            "i_brand",
        )
    )

    result = result.sort(['s_store_name', 'i_item_desc', 'revenue', 'i_current_price', 'i_brand'], descending=[False, False, False, False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[
            ("s_store_name", False),
            ("i_item_desc", False),
            ("revenue", False),
            ("i_current_price", False),
            ("i_brand", False),
        ],
        limit=100,
    )
