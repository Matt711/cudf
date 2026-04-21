# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q12 — naive one-for-one Polars translation of the SQL."""

from __future__ import annotations

from datetime import date, timedelta
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
        query_id=12,
        qualification=run_config.qualification,
    )
    sdate = params["sdate"]
    categories = params["category"]

    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    start_date = date.fromisoformat(sdate)
    end_date = start_date + timedelta(days=30)

    # FROM web_sales, item, date_dim
    # WHERE ws_item_sk=i_item_sk AND i_category IN categories
    #   AND ws_sold_date_sk=d_date_sk
    #   AND d_date BETWEEN sdate AND sdate+30days
    # All WHERE after joins (naive)
    result = (
        web_sales.join(item, left_on="ws_item_sk", right_on="i_item_sk", how="inner")
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(
            pl.col("i_category").is_in(categories)
            & pl.col("d_date").is_between(pl.lit(start_date), pl.lit(end_date), closed="both")
        )
        .group_by(["i_item_id", "i_item_desc", "i_category", "i_class", "i_current_price"])
        .agg(
            sql_sum(pl.col("ws_ext_sales_price")).alias("itemrevenue")
        )
        .with_columns(
            (
                pl.col("itemrevenue") * 100
                / sql_sum(pl.col("itemrevenue")).over("i_class")
            ).alias("revenueratio")
        )
    )

    result = result.sort(['i_category', 'i_class', 'i_item_id', 'i_item_desc', 'revenueratio'], descending=[False, False, False, False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[
            ("i_category", False),
            ("i_class", False),
            ("i_item_id", False),
            ("i_item_desc", False),
            ("revenueratio", False),
        ],
        limit=100,
    )
