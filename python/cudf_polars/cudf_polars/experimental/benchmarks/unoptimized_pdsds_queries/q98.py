# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q98 — naive one-for-one Polars translation of the SQL."""

from __future__ import annotations

from datetime import datetime, timedelta
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
    """TPC-DS q98 naive Polars implementation.

    SELECT i_item_id, i_item_desc, i_category, i_class, i_current_price,
           Sum(ss_ext_sales_price) AS itemrevenue,
           Sum(ss_ext_sales_price) * 100 / Sum(Sum(ss_ext_sales_price))
             OVER (PARTITION BY i_class) AS revenueratio
    FROM store_sales, item, date_dim
    WHERE ss_item_sk = i_item_sk
      AND i_category IN (categories)
      AND ss_sold_date_sk = d_date_sk
      AND d_date BETWEEN date AND date + 30 days
    GROUP BY i_item_id, i_item_desc, i_category, i_class, i_current_price
    ORDER BY i_category, i_class, i_item_id, i_item_desc, revenueratio;
    """
    params = load_parameters(
        int(run_config.scale_factor), query_id=98, qualification=run_config.qualification
    )
    date = params["date"]
    categories = params["categories"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    start_date_py = datetime.strptime(date, "%Y-%m-%d").date()
    end_date_py = start_date_py + timedelta(days=30)
    start_date = pl.date(start_date_py.year, start_date_py.month, start_date_py.day)
    end_date = pl.date(end_date_py.year, end_date_py.month, end_date_py.day)

    # Join in FROM order: store_sales, item, date_dim
    # ALL WHERE conditions after joins (naive rule 2)
    result = (
        store_sales
        .join(item, left_on="ss_item_sk", right_on="i_item_sk", how="inner")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(
            pl.col("i_category").is_in(categories)
            & pl.col("d_date").is_between(start_date, end_date)
        )
        .group_by(["i_item_id", "i_item_desc", "i_category", "i_class", "i_current_price"])
        .agg([sql_sum(pl.col("ss_ext_sales_price")).alias("itemrevenue")])
        # AVG(SUM(col)) OVER (PARTITION BY i_class) - rule 13:
        # group_by sum done above; .sum().over("i_class") gives per-class total
        .with_columns(
            (
                pl.col("itemrevenue") * 100.0
                / pl.col("itemrevenue").sum().over("i_class")
            ).alias("revenueratio")
        )
        .select([
            "i_item_id", "i_item_desc", "i_category", "i_class",
            "i_current_price", "itemrevenue", "revenueratio",
        ])
    )

    return QueryResult(
        frame=result,
        sort_by=[
            ("i_category", False),
            ("i_class", False),
            ("i_item_id", False),
            ("i_item_desc", False),
            ("revenueratio", False),
        ],
        limit=None,
    )
