# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q3 — naive one-for-one Polars translation of the SQL."""

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
        int(run_config.scale_factor),
        query_id=3,
        qualification=run_config.qualification,
    )
    aggc = params["aggc"]
    month = params["month"]
    manufact = params["manufact"]

    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # FROM date_dim dt, store_sales, item
    # Join in left-to-right FROM order; all WHERE filters after all joins
    result = (
        date_dim.join(store_sales, left_on="d_date_sk", right_on="ss_sold_date_sk", how="inner")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk", how="inner")
        .filter(
            (pl.col("i_manufact_id") == manufact)
            & (pl.col("d_moy") == month)
        )
        .group_by(["d_year", "i_brand", "i_brand_id"])
        .agg(sql_sum(pl.col(aggc)).alias("sum_agg"))
        .select(
            pl.col("d_year"),
            pl.col("i_brand_id").alias("brand_id"),
            pl.col("i_brand").alias("brand"),
            pl.col("sum_agg"),
        )
    )

    result = result.sort(['d_year', 'sum_agg', 'brand_id'], descending=[False, True, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("d_year", False), ("sum_agg", True), ("brand_id", False)],
        limit=100,
    )
