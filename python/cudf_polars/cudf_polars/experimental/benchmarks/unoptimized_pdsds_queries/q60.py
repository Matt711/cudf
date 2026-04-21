# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q60 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q60 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=60, qualification=run_config.qualification
    )
    year = params["year"]
    month = params["month"]
    category = params["category"]
    gmt_offset = params["gmt_offset"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer_address = get_data(run_config.dataset_path, "customer_address", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # Subquery: i_item_id IN (SELECT i_item_id FROM item WHERE i_category IN (category))
    category_item_ids = (
        item.filter(pl.col("i_category") == category).select("i_item_id").unique()
    )

    # ss CTE:
    # FROM store_sales, date_dim, customer_address, item
    # WHERE i_item_id IN (category subquery) AND ss_item_sk = i_item_sk
    #   AND ss_sold_date_sk = d_date_sk AND d_year = year AND d_moy = month
    #   AND ss_addr_sk = ca_address_sk AND ca_gmt_offset = gmt_offset
    # GROUP BY i_item_id; SELECT i_item_id, Sum(ss_ext_sales_price) total_sales
    ss = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(customer_address, left_on="ss_addr_sk", right_on="ca_address_sk", how="inner")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk", how="inner")
        .join(category_item_ids, on="i_item_id", how="inner")
        .filter(
            (pl.col("d_year") == year)
            & (pl.col("d_moy") == month)
            & (pl.col("ca_gmt_offset") == gmt_offset)
        )
        .group_by("i_item_id")
        .agg(sql_sum(pl.col("ss_ext_sales_price")).alias("total_sales"))
    )

    # cs CTE (catalog_sales variant)
    cs = (
        catalog_sales.join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(customer_address, left_on="cs_bill_addr_sk", right_on="ca_address_sk", how="inner")
        .join(item, left_on="cs_item_sk", right_on="i_item_sk", how="inner")
        .join(category_item_ids, on="i_item_id", how="inner")
        .filter(
            (pl.col("d_year") == year)
            & (pl.col("d_moy") == month)
            & (pl.col("ca_gmt_offset") == gmt_offset)
        )
        .group_by("i_item_id")
        .agg(sql_sum(pl.col("cs_ext_sales_price")).alias("total_sales"))
    )

    # ws CTE (web_sales variant)
    ws = (
        web_sales.join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(customer_address, left_on="ws_bill_addr_sk", right_on="ca_address_sk", how="inner")
        .join(item, left_on="ws_item_sk", right_on="i_item_sk", how="inner")
        .join(category_item_ids, on="i_item_id", how="inner")
        .filter(
            (pl.col("d_year") == year)
            & (pl.col("d_moy") == month)
            & (pl.col("ca_gmt_offset") == gmt_offset)
        )
        .group_by("i_item_id")
        .agg(sql_sum(pl.col("ws_ext_sales_price")).alias("total_sales"))
    )

    # UNION ALL ss, cs, ws, then GROUP BY i_item_id, Sum(total_sales)
    tmp1 = pl.concat([ss, cs, ws], how="diagonal_relaxed")

    result = (
        tmp1.group_by("i_item_id")
        .agg(sql_sum(pl.col("total_sales")).alias("total_sales"))
    )

    result = result.sort(['i_item_id', 'total_sales'], descending=[False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("i_item_id", False), ("total_sales", False)],
        limit=100,
    )
