# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q29 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q29 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=29, qualification=run_config.qualification
    )

    year = params["year"]
    month = params["month"]
    agg = params["agg"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(run_config.dataset_path, "store_returns", run_config.suffix)
    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # Three aliased copies of date_dim
    d1 = date_dim.select([
        pl.col("d_date_sk").alias("d1_date_sk"),
        pl.col("d_moy").alias("d1_moy"),
        pl.col("d_year").alias("d1_year"),
    ])
    d2 = date_dim.select([
        pl.col("d_date_sk").alias("d2_date_sk"),
        pl.col("d_moy").alias("d2_moy"),
        pl.col("d_year").alias("d2_year"),
    ])
    d3 = date_dim.select([
        pl.col("d_date_sk").alias("d3_date_sk"),
        pl.col("d_year").alias("d3_year"),
    ])

    # Map SQL agg to Polars
    if agg == "avg":
        polars_agg = "mean"
    elif agg == "stddev_samp":
        polars_agg = "std"
    else:
        polars_agg = agg

    # FROM order: store_sales, store_returns, catalog_sales, d1, d2, d3, store, item
    # All WHERE conditions after all joins
    result = (
        store_sales
        .join(store_returns, how="inner",
              left_on=["ss_customer_sk", "ss_item_sk", "ss_ticket_number"],
              right_on=["sr_customer_sk", "sr_item_sk", "sr_ticket_number"])
        .join(catalog_sales, how="inner",
              left_on=["ss_customer_sk", "ss_item_sk"],
              right_on=["cs_bill_customer_sk", "cs_item_sk"])
        .join(d1, how="inner",
              left_on="ss_sold_date_sk", right_on="d1_date_sk")
        .join(d2, how="inner",
              left_on="sr_returned_date_sk", right_on="d2_date_sk")
        .join(d3, how="inner",
              left_on="cs_sold_date_sk", right_on="d3_date_sk")
        .join(store, how="inner",
              left_on="ss_store_sk", right_on="s_store_sk")
        .join(item, how="inner",
              left_on="ss_item_sk", right_on="i_item_sk")
        .filter(
            (pl.col("d1_moy") == month)
            & (pl.col("d1_year") == year)
            & pl.col("d2_moy").is_between(month, month + 3)
            & (pl.col("d2_year") == year)
            & pl.col("d3_year").is_in([year, year + 1, year + 2])
        )
        .group_by(["i_item_id", "i_item_desc", "s_store_id", "s_store_name"])
        .agg([
            getattr(pl.col("ss_quantity"), polars_agg)().alias("store_sales_quantity"),
            getattr(pl.col("sr_return_quantity"), polars_agg)().alias(
                "store_returns_quantity"
            ),
            getattr(pl.col("cs_quantity"), polars_agg)().alias("catalog_sales_quantity"),
        ])
    )

    result = result.sort(['i_item_id', 'i_item_desc', 's_store_id', 's_store_name'], descending=[False, False, False, False], nulls_last=True).limit(100)

    sort_by = [
        ("i_item_id", False),
        ("i_item_desc", False),
        ("s_store_id", False),
        ("s_store_name", False),
    ]
    limit = 100

    return QueryResult(frame=result, sort_by=sort_by, limit=limit)
