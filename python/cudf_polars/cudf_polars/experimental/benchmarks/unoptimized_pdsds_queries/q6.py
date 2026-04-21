# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q6 — naive one-for-one Polars translation of the SQL."""

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
        query_id=6,
        qualification=run_config.qualification,
    )
    year = params["year"]
    month = params["month"]

    customer_address = get_data(run_config.dataset_path, "customer_address", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # Subquery 1: DISTINCT d_month_seq for given year and month
    # d.d_month_seq = (SELECT DISTINCT (d_month_seq) FROM date_dim WHERE d_year=year AND d_moy=month)
    target_month_seqs = (
        date_dim.filter((pl.col("d_year") == year) & (pl.col("d_moy") == month))
        .select("d_month_seq")
        .unique()
    )

    # Subquery 2: i.i_current_price > 1.2 * (SELECT Avg(j.i_current_price) FROM item j WHERE j.i_category = i.i_category)
    # => per-category average, joined back on i_category
    avg_price_per_category = item.group_by("i_category").agg(
        pl.col("i_current_price").mean().alias("avg_category_price")
    )

    # Main query: FROM customer_address a, customer c, store_sales s, date_dim d, item i
    # All WHERE applied after all joins (naive)
    result = (
        customer_address.join(customer, left_on="ca_address_sk", right_on="c_current_addr_sk", how="inner")
        .join(store_sales, left_on="c_customer_sk", right_on="ss_customer_sk", how="inner")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk", how="inner")
        .join(target_month_seqs, on="d_month_seq", how="semi")
        .join(avg_price_per_category, on="i_category", how="inner")
        .filter(pl.col("i_current_price") > 1.2 * pl.col("avg_category_price"))
        .group_by("ca_state")
        .agg(pl.len().alias("cnt"))
        .filter(pl.col("cnt") >= 10)
        .select(
            pl.col("ca_state").alias("state"),
            pl.col("cnt"),
        )
    )

    result = result.sort(['cnt', 'state'], descending=[False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("cnt", False), ("state", False)],
        limit=100,
    )
