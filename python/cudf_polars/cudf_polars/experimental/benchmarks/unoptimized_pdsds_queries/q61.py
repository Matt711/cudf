# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q61 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q61 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=61, qualification=run_config.qualification
    )
    year = params["year"]
    month = params["month"]
    gmt_offset = params["gmt_offset"]
    category = params["category"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    promotion = get_data(run_config.dataset_path, "promotion", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(run_config.dataset_path, "customer_address", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # Scalar subquery "promotional_sales":
    # SELECT Sum(ss_ext_sales_price) promotions
    # FROM store_sales, store, promotion, date_dim, customer, customer_address, item
    # WHERE ss_sold_date_sk=d_date_sk AND ss_store_sk=s_store_sk AND ss_promo_sk=p_promo_sk
    #   AND ss_customer_sk=c_customer_sk AND ca_address_sk=c_current_addr_sk
    #   AND ss_item_sk=i_item_sk
    #   AND ca_gmt_offset=gmt_offset AND i_category=category
    #   AND (p_channel_dmail='Y' OR p_channel_email='Y' OR p_channel_tv='Y')
    #   AND s_gmt_offset=gmt_offset AND d_year=year AND d_moy=month
    promotional_sales_lf = (
        store_sales.join(store, left_on="ss_store_sk", right_on="s_store_sk", how="inner")
        .join(promotion, left_on="ss_promo_sk", right_on="p_promo_sk", how="inner")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(customer, left_on="ss_customer_sk", right_on="c_customer_sk", how="inner")
        .join(customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk", how="inner")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk", how="inner")
        .filter(
            (pl.col("ca_gmt_offset") == gmt_offset)
            & (pl.col("i_category") == category)
            & (
                (pl.col("p_channel_dmail") == "Y")
                | (pl.col("p_channel_email") == "Y")
                | (pl.col("p_channel_tv") == "Y")
            )
            & (pl.col("s_gmt_offset") == gmt_offset)
            & (pl.col("d_year") == year)
            & (pl.col("d_moy") == month)
        )
        .select(sql_sum(pl.col("ss_ext_sales_price")).alias("promotions"))
    )

    # Scalar subquery "all_sales":
    # SELECT Sum(ss_ext_sales_price) total
    # FROM store_sales, store, date_dim, customer, customer_address, item
    # WHERE same conditions (without promotion)
    all_sales_lf = (
        store_sales.join(store, left_on="ss_store_sk", right_on="s_store_sk", how="inner")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(customer, left_on="ss_customer_sk", right_on="c_customer_sk", how="inner")
        .join(customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk", how="inner")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk", how="inner")
        .filter(
            (pl.col("ca_gmt_offset") == gmt_offset)
            & (pl.col("i_category") == category)
            & (pl.col("s_gmt_offset") == gmt_offset)
            & (pl.col("d_year") == year)
            & (pl.col("d_moy") == month)
        )
        .select(sql_sum(pl.col("ss_ext_sales_price")).alias("total"))
    )

    # Final: cross join the two scalar aggregates, compute ratio
    # SELECT promotions, total, Cast(promotions AS DECIMAL(15,4)) / Cast(total AS DECIMAL(15,4)) * 100
    result = (
        promotional_sales_lf.join(all_sales_lf, how="cross")
        .with_columns(
            pl.when(pl.col("total").is_null() | (pl.col("total") == 0))
            .then(None)
            .otherwise(
                pl.col("promotions").cast(pl.Float64) / pl.col("total").cast(pl.Float64) * 100.0
            )
            .alias(
                "((CAST(promotions AS DECIMAL(15, 4)) / CAST(total AS DECIMAL(15, 4))) * 100)"
            )
        )
    )

    result = result.sort(['promotions', 'total'], descending=[False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("promotions", False), ("total", False)],
        limit=100,
    )
