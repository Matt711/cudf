# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q26 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q26 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=26, qualification=run_config.qualification
    )

    year = params["year"]
    gen = params["gen"]
    ms = params["ms"]
    es = params["es"]

    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    customer_demographics = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    promotion = get_data(run_config.dataset_path, "promotion", run_config.suffix)

    # FROM order: catalog_sales, customer_demographics, date_dim, item, promotion
    # All WHERE conditions after all joins
    result = (
        catalog_sales
        .join(customer_demographics, how="inner",
              left_on="cs_bill_cdemo_sk", right_on="cd_demo_sk")
        .join(date_dim, how="inner",
              left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(item, how="inner",
              left_on="cs_item_sk", right_on="i_item_sk")
        .join(promotion, how="inner",
              left_on="cs_promo_sk", right_on="p_promo_sk")
        .filter(
            (pl.col("cd_gender") == gen)
            & (pl.col("cd_marital_status") == ms)
            & (pl.col("cd_education_status") == es)
            & (
                (pl.col("p_channel_email") == "N")
                | (pl.col("p_channel_event") == "N")
            )
            & (pl.col("d_year") == year)
        )
        .group_by("i_item_id")
        .agg([
            pl.col("cs_quantity").mean().alias("agg1"),
            pl.col("cs_list_price").mean().alias("agg2"),
            pl.col("cs_coupon_amt").mean().alias("agg3"),
            pl.col("cs_sales_price").mean().alias("agg4"),
        ])
    )

    result = result.sort('i_item_id', descending=False, nulls_last=True).limit(100)

    sort_by = [("i_item_id", False)]
    limit = 100

    return QueryResult(frame=result, sort_by=sort_by, limit=limit)
