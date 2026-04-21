# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q54 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q54 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=54, qualification=run_config.qualification
    )
    category = params["category"]
    class_name = params["class"]
    month = params["month"]
    year = params["year"]

    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(run_config.dataset_path, "customer_address", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)

    # UNION ALL of catalog_sales and web_sales with renamed cols
    cs_or_ws = pl.concat(
        [
            catalog_sales.select(
                pl.col("cs_sold_date_sk").alias("sold_date_sk"),
                pl.col("cs_bill_customer_sk").alias("customer_sk"),
                pl.col("cs_item_sk").alias("item_sk"),
            ),
            web_sales.select(
                pl.col("ws_sold_date_sk").alias("sold_date_sk"),
                pl.col("ws_bill_customer_sk").alias("customer_sk"),
                pl.col("ws_item_sk").alias("item_sk"),
            ),
        ],
        how="diagonal_relaxed",
    )

    # my_customers CTE:
    # FROM cs_or_ws_sales, item, date_dim, customer
    # WHERE sold_date_sk = d_date_sk AND item_sk = i_item_sk
    #   AND i_category = category AND i_class = class_name
    #   AND c_customer_sk = customer_sk
    #   AND d_moy = month AND d_year = year
    # SELECT DISTINCT c_customer_sk, c_current_addr_sk
    my_customers = (
        cs_or_ws.join(item, left_on="item_sk", right_on="i_item_sk", how="inner")
        .join(date_dim, left_on="sold_date_sk", right_on="d_date_sk", how="inner")
        .join(customer, left_on="customer_sk", right_on="c_customer_sk", how="inner")
        .filter(
            (pl.col("i_category") == category)
            & (pl.col("i_class") == class_name)
            & (pl.col("d_moy") == month)
            & (pl.col("d_year") == year)
        )
        .select(
            pl.col("customer_sk").alias("c_customer_sk"),
            pl.col("c_current_addr_sk"),
        )
        .unique()
    )

    # Subquery: SELECT DISTINCT d_month_seq + 1 FROM date_dim WHERE d_year=year AND d_moy=month
    # and SELECT DISTINCT d_month_seq + 3 FROM date_dim WHERE d_year=year AND d_moy=month
    # Since both reference the same base scalar, compute once and derive the range
    target_seq_df = (
        date_dim.filter((pl.col("d_year") == year) & (pl.col("d_moy") == month))
        .select(pl.col("d_month_seq").first())
        .collect()
    )
    target_seq = target_seq_df.item()
    seq_start = target_seq + 1
    seq_end = target_seq + 3

    # my_revenue CTE:
    # FROM my_customers, store_sales, customer_address, store, date_dim
    # WHERE c_current_addr_sk = ca_address_sk AND ca_county = s_county
    #   AND ca_state = s_state AND ss_sold_date_sk = d_date_sk
    #   AND c_customer_sk = ss_customer_sk
    #   AND d_month_seq BETWEEN seq_start AND seq_end
    # GROUP BY c_customer_sk
    # SELECT c_customer_sk, Sum(ss_ext_sales_price) AS revenue
    my_revenue = (
        my_customers.join(customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk", how="inner")
        .join(store, left_on=["ca_county", "ca_state"], right_on=["s_county", "s_state"], how="inner")
        .join(store_sales, left_on="c_customer_sk", right_on="ss_customer_sk", how="inner")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(pl.col("d_month_seq").is_between(seq_start, seq_end))
        .group_by("c_customer_sk")
        .agg(sql_sum(pl.col("ss_ext_sales_price")).alias("revenue"))
    )

    # segments CTE:
    # SELECT Cast((revenue / 50) AS INT) AS segment FROM my_revenue
    segments = my_revenue.select(
        (pl.col("revenue") / 50).cast(pl.Int32).alias("segment")
    )

    # Final:
    # SELECT segment, Count(*) AS num_customers, segment * 50 AS segment_base
    # GROUP BY segment ORDER BY segment, num_customers LIMIT 100
    result = (
        segments.group_by("segment")
        .agg(pl.len().alias("num_customers"))
        .with_columns((pl.col("segment") * 50).alias("segment_base"))
        .select("segment", "num_customers", "segment_base")
    )

    result = result.sort(['segment', 'num_customers'], descending=[False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("segment", False), ("num_customers", False)],
        limit=100,
    )
