# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q51 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q51 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=51, qualification=run_config.qualification
    )
    dms = params["dms"]

    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # web_v1 CTE:
    # SELECT ws_item_sk item_sk, d_date,
    #   sum(sum(ws_sales_price)) OVER (PARTITION BY ws_item_sk ORDER BY d_date
    #     ROWS BETWEEN unbounded preceding AND CURRENT ROW) cume_sales
    # FROM web_sales, date_dim
    # WHERE ws_sold_date_sk=d_date_sk AND d_month_seq BETWEEN dms AND dms+11
    #   AND ws_item_sk IS NOT NULL
    # GROUP BY ws_item_sk, d_date
    #
    # Step 1: join web_sales and date_dim (comma = inner join via WHERE)
    # Step 2: ALL WHERE filters after join
    # Step 3: group_by to get per-day sum
    # Step 4: cum_sum over (item_sk ordered by d_date) = running SUM(SUM(...))
    web_v1 = (
        web_sales.join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(
            pl.col("d_month_seq").is_between(dms, dms + 11)
            & pl.col("ws_item_sk").is_not_null()
        )
        .group_by(["ws_item_sk", "d_date"])
        .agg(sql_sum(pl.col("ws_sales_price")).alias("daily_sum"))
        .with_columns(
            pl.col("daily_sum")
            .cum_sum()
            .over(partition_by="ws_item_sk", order_by="d_date")
            .alias("cume_sales")
        )
        .select(
            pl.col("ws_item_sk").alias("item_sk"),
            pl.col("d_date"),
            pl.col("cume_sales").alias("web_sales"),
        )
    )

    # store_v1 CTE (same pattern, store_sales / ss_ prefix)
    store_v1 = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(
            pl.col("d_month_seq").is_between(dms, dms + 11)
            & pl.col("ss_item_sk").is_not_null()
        )
        .group_by(["ss_item_sk", "d_date"])
        .agg(sql_sum(pl.col("ss_sales_price")).alias("daily_sum"))
        .with_columns(
            pl.col("daily_sum")
            .cum_sum()
            .over(partition_by="ss_item_sk", order_by="d_date")
            .alias("cume_sales")
        )
        .select(
            pl.col("ss_item_sk").alias("item_sk"),
            pl.col("d_date"),
            pl.col("cume_sales").alias("store_sales"),
        )
    )

    # FULL OUTER JOIN web_v1 web FULL OUTER JOIN store_v1 store
    # ON (web.item_sk = store.item_sk AND web.d_date = store.d_date)
    # CASE WHEN web.item_sk IS NOT NULL THEN web.item_sk ELSE store.item_sk END item_sk
    # CASE WHEN web.d_date IS NOT NULL THEN web.d_date ELSE store.d_date END d_date
    combined = (
        web_v1.join(store_v1, on=["item_sk", "d_date"], how="full", suffix="_store")
        .with_columns(
            item_sk=pl.coalesce([pl.col("item_sk"), pl.col("item_sk_store")]),
            d_date=pl.coalesce([pl.col("d_date"), pl.col("d_date_store")]),
        )
        .select("item_sk", "d_date", "web_sales", "store_sales")
    )

    # Outer query:
    # SELECT item_sk, d_date, web_sales, store_sales,
    #   max(web_sales) OVER (PARTITION BY item_sk ORDER BY d_date ROWS UNBOUNDED PRECEDING) web_cumulative,
    #   max(store_sales) OVER (PARTITION BY item_sk ORDER BY d_date ROWS UNBOUNDED PRECEDING) store_cumulative
    # WHERE web_cumulative > store_cumulative
    # Collapse duplicate (item_sk, d_date) pairs by taking max of each side
    collapsed = (
        combined.group_by(["item_sk", "d_date"])
        .agg(
            pl.col("web_sales").max().alias("web_sales"),
            pl.col("store_sales").max().alias("store_sales"),
        )
    )

    with_cummax = collapsed.with_columns(
        pl.col("web_sales")
        .cum_max()
        .over(partition_by="item_sk", order_by="d_date")
        .alias("web_cumulative"),
        pl.col("store_sales")
        .cum_max()
        .over(partition_by="item_sk", order_by="d_date")
        .alias("store_cumulative"),
    )

    result = (
        with_cummax.filter(pl.col("web_cumulative") > pl.col("store_cumulative"))
        .select("item_sk", "d_date", "web_sales", "store_sales", "web_cumulative", "store_cumulative")
    )

    # ORDER BY item_sk NULLS FIRST, d_date NULLS FIRST => nulls_last=False
    result = result.sort(['item_sk', 'd_date'], descending=[False, False]).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("item_sk", False), ("d_date", False)],
        limit=100,
        nulls_last=False,
    )
