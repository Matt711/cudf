# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q58 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q58 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=58, qualification=run_config.qualification
    )
    sales_date = params["sales_date"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # Innermost subquery: SELECT d_week_seq FROM date_dim WHERE d_date = sales_date
    # Rule: collect the scalar, then filter date_dim for same week, get set of d_date_sk
    year, month, day = map(int, sales_date.split("-"))
    target_week_seq_df = (
        date_dim.filter(pl.col("d_date") == pl.date(year, month, day))
        .select(pl.col("d_week_seq").first())
        .collect()
    )
    target_week_seq = target_week_seq_df.item()

    # d_date IN (SELECT d_date FROM date_dim WHERE d_week_seq = that_seq)
    # => filter date_dim where d_week_seq == target_week_seq, get their d_date_sk values
    week_dates = (
        date_dim.filter(pl.col("d_week_seq") == target_week_seq)
        .select("d_date_sk")
    )

    # ss_items CTE:
    # FROM store_sales, item, date_dim
    # WHERE ss_item_sk = i_item_sk
    #   AND d_date IN (week subquery) AND ss_sold_date_sk = d_date_sk
    # GROUP BY i_item_id; SELECT i_item_id item_id, Sum(ss_ext_sales_price) ss_item_rev
    ss_items = (
        store_sales.join(item, left_on="ss_item_sk", right_on="i_item_sk", how="inner")
        .join(week_dates, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .group_by("i_item_id")
        .agg(sql_sum(pl.col("ss_ext_sales_price")).alias("ss_item_rev"))
        .select(pl.col("i_item_id").alias("item_id"), pl.col("ss_item_rev"))
    )

    # cs_items CTE (same structure with catalog_sales)
    cs_items = (
        catalog_sales.join(item, left_on="cs_item_sk", right_on="i_item_sk", how="inner")
        .join(week_dates, left_on="cs_sold_date_sk", right_on="d_date_sk", how="inner")
        .group_by("i_item_id")
        .agg(sql_sum(pl.col("cs_ext_sales_price")).alias("cs_item_rev"))
        .select(pl.col("i_item_id").alias("item_id"), pl.col("cs_item_rev"))
    )

    # ws_items CTE (same structure with web_sales)
    ws_items = (
        web_sales.join(item, left_on="ws_item_sk", right_on="i_item_sk", how="inner")
        .join(week_dates, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .group_by("i_item_id")
        .agg(sql_sum(pl.col("ws_ext_sales_price")).alias("ws_item_rev"))
        .select(pl.col("i_item_id").alias("item_id"), pl.col("ws_item_rev"))
    )

    # Final:
    # FROM ss_items, cs_items, ws_items
    # WHERE ss_items.item_id = cs_items.item_id AND ss_items.item_id = ws_items.item_id
    #   AND ss_item_rev BETWEEN 0.9*cs_item_rev AND 1.1*cs_item_rev (etc.)
    result = (
        ss_items.join(cs_items, on="item_id", how="inner")
        .join(ws_items, on="item_id", how="inner")
        .filter(
            pl.col("ss_item_rev").is_between(0.9 * pl.col("cs_item_rev"), 1.1 * pl.col("cs_item_rev"))
            & pl.col("ss_item_rev").is_between(0.9 * pl.col("ws_item_rev"), 1.1 * pl.col("ws_item_rev"))
            & pl.col("cs_item_rev").is_between(0.9 * pl.col("ss_item_rev"), 1.1 * pl.col("ss_item_rev"))
            & pl.col("cs_item_rev").is_between(0.9 * pl.col("ws_item_rev"), 1.1 * pl.col("ws_item_rev"))
            & pl.col("ws_item_rev").is_between(0.9 * pl.col("ss_item_rev"), 1.1 * pl.col("ss_item_rev"))
            & pl.col("ws_item_rev").is_between(0.9 * pl.col("cs_item_rev"), 1.1 * pl.col("cs_item_rev"))
        )
        .with_columns(
            total_rev=(
                pl.col("ss_item_rev") + pl.col("cs_item_rev") + pl.col("ws_item_rev")
            ),
        )
        .with_columns(
            ss_dev=(pl.col("ss_item_rev") / pl.col("total_rev") / 3 * 100),
            cs_dev=(pl.col("cs_item_rev") / pl.col("total_rev") / 3 * 100),
            ws_dev=(pl.col("ws_item_rev") / pl.col("total_rev") / 3 * 100),
            average=(pl.col("total_rev") / 3),
        )
        .select(["item_id", "ss_item_rev", "ss_dev", "cs_item_rev", "cs_dev",
                 "ws_item_rev", "ws_dev", "average"])
    )

    result = result.sort(['item_id', 'ss_item_rev'], descending=[False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("item_id", False), ("ss_item_rev", False)],
        limit=100,
    )
