# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q33 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q33 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=33, qualification=run_config.qualification
    )

    year = params["year"]
    month = params["month"]
    gmt = params["gmt"]
    category = params["category"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )

    # Subquery: SELECT i_manufact_id FROM item WHERE i_category IN (category)
    category_manufact = (
        item
        .filter(pl.col("i_category") == category)
        .select("i_manufact_id")
    )

    # CTE: ss
    # FROM store_sales, date_dim, customer_address, item
    # WHERE i_manufact_id IN (subquery) AND ss_item_sk=i_item_sk
    #   AND ss_sold_date_sk=d_date_sk AND d_year=year AND d_moy=month
    #   AND ss_addr_sk=ca_address_sk AND ca_gmt_offset=gmt
    # GROUP BY i_manufact_id
    ss = (
        store_sales
        .join(date_dim, how="inner",
              left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(customer_address, how="inner",
              left_on="ss_addr_sk", right_on="ca_address_sk")
        .join(item, how="inner",
              left_on="ss_item_sk", right_on="i_item_sk")
        .join(category_manufact, on="i_manufact_id", how="semi")
        .filter(
            (pl.col("d_year") == year)
            & (pl.col("d_moy") == month)
            & (pl.col("ca_gmt_offset") == gmt)
        )
        .group_by("i_manufact_id")
        .agg([
            sql_sum(pl.col("ss_ext_sales_price")).alias("total_sales"),
        ])
    )

    # CTE: cs
    # FROM catalog_sales, date_dim, customer_address, item
    cs = (
        catalog_sales
        .join(date_dim, how="inner",
              left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(customer_address, how="inner",
              left_on="cs_bill_addr_sk", right_on="ca_address_sk")
        .join(item, how="inner",
              left_on="cs_item_sk", right_on="i_item_sk")
        .join(category_manufact, on="i_manufact_id", how="semi")
        .filter(
            (pl.col("d_year") == year)
            & (pl.col("d_moy") == month)
            & (pl.col("ca_gmt_offset") == gmt)
        )
        .group_by("i_manufact_id")
        .agg([
            sql_sum(pl.col("cs_ext_sales_price")).alias("total_sales"),
        ])
    )

    # CTE: ws
    # FROM web_sales, date_dim, customer_address, item
    ws = (
        web_sales
        .join(date_dim, how="inner",
              left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(customer_address, how="inner",
              left_on="ws_bill_addr_sk", right_on="ca_address_sk")
        .join(item, how="inner",
              left_on="ws_item_sk", right_on="i_item_sk")
        .join(category_manufact, on="i_manufact_id", how="semi")
        .filter(
            (pl.col("d_year") == year)
            & (pl.col("d_moy") == month)
            & (pl.col("ca_gmt_offset") == gmt)
        )
        .group_by("i_manufact_id")
        .agg([
            sql_sum(pl.col("ws_ext_sales_price")).alias("total_sales"),
        ])
    )

    # UNION ALL of ss, cs, ws, then GROUP BY i_manufact_id, ORDER BY total_sales
    result = (
        pl.concat([ss, cs, ws], how="diagonal_relaxed")
        .group_by("i_manufact_id")
        .agg([
            sql_sum(pl.col("total_sales")).alias("total_sales"),
        ])
    )

    result = result.sort('total_sales', descending=False, nulls_last=True).limit(100)

    sort_by = [("total_sales", False)]
    limit = 100

    return QueryResult(frame=result, sort_by=sort_by, limit=limit)
