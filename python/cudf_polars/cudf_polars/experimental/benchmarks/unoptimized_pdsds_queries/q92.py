# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q92 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q92 naive Polars implementation.

    SELECT Sum(ws_ext_discount_amt) AS 'Excess Discount Amount'
    FROM web_sales, item, date_dim
    WHERE i_manufact_id = manufact_id
      AND i_item_sk = ws_item_sk
      AND d_date BETWEEN date AND date + 90 days
      AND d_date_sk = ws_sold_date_sk
      AND ws_ext_discount_amt > (
            SELECT 1.3 * avg(ws_ext_discount_amt)
            FROM web_sales, date_dim
            WHERE ws_item_sk = i_item_sk
              AND d_date BETWEEN date AND date + 90 days
              AND d_date_sk = ws_sold_date_sk)
    ORDER BY sum(ws_ext_discount_amt) LIMIT 100;

    Correlated subquery: compute avg_discount per item_sk, join back (rule: q92 note).
    """
    params = load_parameters(
        int(run_config.scale_factor), query_id=92, qualification=run_config.qualification
    )
    manufact_id = params["manufact_id"]
    date = params["date"]

    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    start_date_py = datetime.strptime(date, "%Y-%m-%d").date()
    end_date_py = start_date_py + timedelta(days=90)
    start_date = pl.date(start_date_py.year, start_date_py.month, start_date_py.day)
    end_date = pl.date(end_date_py.year, end_date_py.month, end_date_py.day)

    # Correlated subquery: 1.3 * avg(ws_ext_discount_amt) per item_sk
    # FROM web_sales, date_dim WHERE d_date BETWEEN ... AND d_date_sk = ws_sold_date_sk
    # (no item filter here — the correlation is ws_item_sk = i_item_sk from outer)
    avg_discount = (
        web_sales
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(pl.col("d_date").is_between(start_date, end_date))
        .group_by("ws_item_sk")
        .agg((pl.col("ws_ext_discount_amt").mean() * 1.3).alias("threshold_discount"))
    )

    # Main query: web_sales, item, date_dim (comma-sep = inner join via WHERE)
    # ALL WHERE conditions after joins
    result = (
        web_sales
        .join(item, left_on="ws_item_sk", right_on="i_item_sk", how="inner")
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(avg_discount, left_on="ws_item_sk", right_on="ws_item_sk", how="inner")
        .filter(
            (pl.col("i_manufact_id") == manufact_id)
            & pl.col("d_date").is_between(start_date, end_date)
            & (pl.col("ws_ext_discount_amt") > pl.col("threshold_discount"))
        )
        .select([sql_sum(pl.col("ws_ext_discount_amt")).alias("Excess Discount Amount")])
    )

    result = result.sort('Excess Discount Amount', descending=False, nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("Excess Discount Amount", False)],
        limit=100,
    )
