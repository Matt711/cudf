# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q7 — naive one-for-one Polars translation of the SQL."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:  # type: ignore[return]
    raise NotImplementedError


def polars_impl(run_config: RunConfig) -> QueryResult:
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=7,
        qualification=run_config.qualification,
    )
    year = params["year"]
    gender = params["gender"]
    marital_status = params["marital_status"]
    education_status = params["education_status"]
    promo_channel = params["promo_channel"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    customer_demographics = get_data(run_config.dataset_path, "customer_demographics", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    promotion = get_data(run_config.dataset_path, "promotion", run_config.suffix)

    # FROM store_sales, customer_demographics, date_dim, item, promotion
    # All WHERE conditions applied after all joins (naive rule 2)
    result = (
        store_sales.join(customer_demographics, left_on="ss_cdemo_sk", right_on="cd_demo_sk", how="inner")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk", how="inner")
        .join(promotion, left_on="ss_promo_sk", right_on="p_promo_sk", how="inner")
        .filter(
            (pl.col("cd_gender") == gender)
            & (pl.col("cd_marital_status") == marital_status)
            & (pl.col("cd_education_status") == education_status)
            & (
                (pl.col("p_channel_email") == promo_channel)
                | (pl.col("p_channel_event") == promo_channel)
            )
            & (pl.col("d_year") == year)
        )
        .group_by("i_item_id")
        .agg(
            pl.col("ss_quantity").mean().alias("agg1"),
            pl.col("ss_list_price").mean().alias("agg2"),
            pl.col("ss_coupon_amt").mean().alias("agg3"),
            pl.col("ss_sales_price").mean().alias("agg4"),
        )
    )

    result = result.sort('i_item_id', descending=False, nulls_last=True).limit(100)
    return QueryResult(frame=result, sort_by=[("i_item_id", False)], limit=100)
