# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q17 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q17 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=17, qualification=run_config.qualification
    )

    year = params["year"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(run_config.dataset_path, "store_returns", run_config.suffix)
    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # Alias date_dim copies for d1, d2, d3
    d1 = date_dim.select([
        pl.col("d_date_sk").alias("d1_date_sk"),
        pl.col("d_quarter_name").alias("d1_quarter_name"),
    ])
    d2 = date_dim.select([
        pl.col("d_date_sk").alias("d2_date_sk"),
        pl.col("d_quarter_name").alias("d2_quarter_name"),
    ])
    d3 = date_dim.select([
        pl.col("d_date_sk").alias("d3_date_sk"),
        pl.col("d_quarter_name").alias("d3_quarter_name"),
    ])

    # Join in FROM order: store_sales, store_returns, catalog_sales, d1, d2, d3, store, item
    # Then filter ALL WHERE conditions after all joins
    result = (
        store_sales
        .join(store_returns, how="inner",
              left_on=["ss_customer_sk", "ss_item_sk", "ss_ticket_number"],
              right_on=["sr_customer_sk", "sr_item_sk", "sr_ticket_number"])
        .join(catalog_sales, how="inner",
              left_on=["ss_customer_sk", "ss_item_sk"],
              right_on=["cs_bill_customer_sk", "cs_item_sk"])
        .join(d1, how="inner", left_on="ss_sold_date_sk", right_on="d1_date_sk")
        .join(d2, how="inner", left_on="sr_returned_date_sk", right_on="d2_date_sk")
        .join(d3, how="inner", left_on="cs_sold_date_sk", right_on="d3_date_sk")
        .join(store, how="inner", left_on="ss_store_sk", right_on="s_store_sk")
        .join(item, how="inner", left_on="ss_item_sk", right_on="i_item_sk")
        .filter(
            (pl.col("d1_quarter_name") == f"{year}Q1")
            & pl.col("d2_quarter_name").is_in([f"{year}Q1", f"{year}Q2", f"{year}Q3"])
            & pl.col("d3_quarter_name").is_in([f"{year}Q1", f"{year}Q2", f"{year}Q3"])
        )
        .group_by(["i_item_id", "i_item_desc", "s_state"])
        .agg([
            pl.col("ss_quantity").count().alias("store_sales_quantitycount"),
            pl.col("ss_quantity").mean().alias("store_sales_quantityave"),
            pl.col("ss_quantity").std(ddof=1).alias("store_sales_quantitystdev"),
            (pl.col("ss_quantity").std(ddof=1) / pl.col("ss_quantity").mean()).alias(
                "store_sales_quantitycov"
            ),
            pl.col("sr_return_quantity").count().alias("store_returns_quantitycount"),
            pl.col("sr_return_quantity").mean().alias("store_returns_quantityave"),
            pl.col("sr_return_quantity").std(ddof=1).alias("store_returns_quantitystdev"),
            (
                pl.col("sr_return_quantity").std(ddof=1)
                / pl.col("sr_return_quantity").mean()
            ).alias("store_returns_quantitycov"),
            pl.col("cs_quantity").count().alias("catalog_sales_quantitycount"),
            pl.col("cs_quantity").mean().alias("catalog_sales_quantityave"),
            (pl.col("cs_quantity").std(ddof=1) / pl.col("cs_quantity").mean()).alias(
                "catalog_sales_quantitystdev"
            ),
            (pl.col("cs_quantity").std(ddof=1) / pl.col("cs_quantity").mean()).alias(
                "catalog_sales_quantitycov"
            ),
        ])
    )

    result = result.sort(['i_item_id', 'i_item_desc', 's_state'], descending=[False, False, False], nulls_last=True).limit(100)

    sort_by = [("i_item_id", False), ("i_item_desc", False), ("s_state", False)]
    limit = 100

    return QueryResult(frame=result, sort_by=sort_by, limit=limit)
