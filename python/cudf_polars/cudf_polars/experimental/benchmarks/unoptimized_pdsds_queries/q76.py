# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q76 — naive one-for-one Polars translation of the SQL."""
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
        int(run_config.scale_factor), query_id=76, qualification=run_config.qualification
    )
    nullcol_ss = params["nullcol_ss"]
    nullcol_ws = params["nullcol_ws"]
    nullcol_cs = params["nullcol_cs"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # UNION ALL of three subqueries
    # Each: FROM sales_table, item, date_dim
    # WHERE nullcol IS NULL AND sold_date_sk = d_date_sk AND item_sk = i_item_sk
    # No .select() until final SELECT (naive rule 3) — but each subquery has explicit cols

    store_branch = (
        store_sales
        .join(item, left_on="ss_item_sk", right_on="i_item_sk", how="inner")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(pl.col(nullcol_ss).is_null())
        .select([
            pl.lit("store").alias("channel"),
            pl.lit(nullcol_ss).alias("col_name"),
            "d_year",
            "d_qoy",
            "i_category",
            pl.col("ss_ext_sales_price").alias("ext_sales_price"),
        ])
    )

    web_branch = (
        web_sales
        .join(item, left_on="ws_item_sk", right_on="i_item_sk", how="inner")
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(pl.col(nullcol_ws).is_null())
        .select([
            pl.lit("web").alias("channel"),
            pl.lit(nullcol_ws).alias("col_name"),
            "d_year",
            "d_qoy",
            "i_category",
            pl.col("ws_ext_sales_price").alias("ext_sales_price"),
        ])
    )

    catalog_branch = (
        catalog_sales
        .join(item, left_on="cs_item_sk", right_on="i_item_sk", how="inner")
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(pl.col(nullcol_cs).is_null())
        .select([
            pl.lit("catalog").alias("channel"),
            pl.lit(nullcol_cs).alias("col_name"),
            "d_year",
            "d_qoy",
            "i_category",
            pl.col("cs_ext_sales_price").alias("ext_sales_price"),
        ])
    )

    foo = pl.concat([store_branch, web_branch, catalog_branch], how="diagonal_relaxed")

    # GROUP BY channel, col_name, d_year, d_qoy, i_category
    result = (
        foo
        .group_by(["channel", "col_name", "d_year", "d_qoy", "i_category"])
        .agg([
            pl.len().alias("sales_cnt"),
            sql_sum(pl.col("ext_sales_price")).alias("sales_amt"),
        ])
        .select([
            "channel",
            "col_name",
            "d_year",
            "d_qoy",
            "i_category",
            "sales_cnt",
            "sales_amt",
        ])
    )

    result = result.sort(['channel', 'col_name', 'd_year', 'd_qoy', 'i_category'], descending=[False, False, False, False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[
            ("channel", False),
            ("col_name", False),
            ("d_year", False),
            ("d_qoy", False),
            ("i_category", False),
        ],
        limit=100,
    )
