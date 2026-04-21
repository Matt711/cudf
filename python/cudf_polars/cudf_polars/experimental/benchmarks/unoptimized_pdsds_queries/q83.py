# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q83 — naive one-for-one Polars translation of the SQL."""

from __future__ import annotations

import datetime
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
    """TPC-DS q83 naive Polars implementation.

    WITH sr_items AS (
        SELECT i_item_id item_id, Sum(sr_return_quantity) sr_item_qty
        FROM store_returns, item, date_dim
        WHERE sr_item_sk = i_item_sk
          AND d_date IN (SELECT d_date FROM date_dim WHERE d_week_seq IN
                         (SELECT d_week_seq FROM date_dim WHERE d_date IN (dates)))
          AND sr_returned_date_sk = d_date_sk
        GROUP BY i_item_id),
    ... (similar for cr_items, wr_items)
    SELECT ... FROM sr_items, cr_items, wr_items
    WHERE sr_items.item_id = cr_items.item_id AND sr_items.item_id = wr_items.item_id
    ORDER BY sr_items.item_id, sr_item_qty LIMIT 100;
    """
    params = load_parameters(
        int(run_config.scale_factor), query_id=83, qualification=run_config.qualification
    )
    dates = params["dates"]

    store_returns = get_data(run_config.dataset_path, "store_returns", run_config.suffix)
    catalog_returns = get_data(run_config.dataset_path, "catalog_returns", run_config.suffix)
    web_returns = get_data(run_config.dataset_path, "web_returns", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # Resolve the nested subquery:
    # d_date IN (SELECT d_date FROM date_dim WHERE d_week_seq IN
    #             (SELECT d_week_seq FROM date_dim WHERE d_date IN (dates_list)))
    # Step 1: find week_seqs for the literal dates
    dates_as_date = [datetime.date.fromisoformat(d) for d in dates]
    week_seqs = (
        date_dim.filter(pl.col("d_date").is_in(dates_as_date))
        .select("d_week_seq")
        .collect()["d_week_seq"]
        .to_list()
    )
    # Step 2: find all d_dates that belong to those week_seqs
    valid_dates = (
        date_dim.filter(pl.col("d_week_seq").is_in(week_seqs))
        .select("d_date")
        .collect()["d_date"]
        .to_list()
    )

    # date_dim filtered to rows whose d_date is in valid_dates
    date_dim_filtered = date_dim.filter(pl.col("d_date").is_in(valid_dates))

    # CTE sr_items: store_returns, item, date_dim (comma-sep = inner join via WHERE)
    # WHERE sr_item_sk = i_item_sk AND sr_returned_date_sk = d_date_sk
    sr_items = (
        store_returns
        .join(item, left_on="sr_item_sk", right_on="i_item_sk", how="inner")
        .join(date_dim_filtered, left_on="sr_returned_date_sk", right_on="d_date_sk", how="inner")
        .group_by("i_item_id")
        .agg(sql_sum(pl.col("sr_return_quantity")).alias("sr_item_qty"))
        .select([pl.col("i_item_id").alias("item_id"), "sr_item_qty"])
    )

    # CTE cr_items: catalog_returns, item, date_dim
    cr_items = (
        catalog_returns
        .join(item, left_on="cr_item_sk", right_on="i_item_sk", how="inner")
        .join(date_dim_filtered, left_on="cr_returned_date_sk", right_on="d_date_sk", how="inner")
        .group_by("i_item_id")
        .agg(sql_sum(pl.col("cr_return_quantity")).alias("cr_item_qty"))
        .select([pl.col("i_item_id").alias("item_id"), "cr_item_qty"])
    )

    # CTE wr_items: web_returns, item, date_dim
    wr_items = (
        web_returns
        .join(item, left_on="wr_item_sk", right_on="i_item_sk", how="inner")
        .join(date_dim_filtered, left_on="wr_returned_date_sk", right_on="d_date_sk", how="inner")
        .group_by("i_item_id")
        .agg(sql_sum(pl.col("wr_return_quantity")).alias("wr_item_qty"))
        .select([pl.col("i_item_id").alias("item_id"), "wr_item_qty"])
    )

    # Main SELECT: sr_items, cr_items, wr_items (inner join on item_id)
    result = (
        sr_items
        .join(cr_items, on="item_id", how="inner")
        .join(wr_items, on="item_id", how="inner")
        .select([
            "item_id",
            "sr_item_qty",
            (pl.col("sr_item_qty") / (pl.col("sr_item_qty") + pl.col("cr_item_qty") + pl.col("wr_item_qty")) / 3.0 * 100).alias("sr_dev"),
            "cr_item_qty",
            (pl.col("cr_item_qty") / (pl.col("sr_item_qty") + pl.col("cr_item_qty") + pl.col("wr_item_qty")) / 3.0 * 100).alias("cr_dev"),
            "wr_item_qty",
            (pl.col("wr_item_qty") / (pl.col("sr_item_qty") + pl.col("cr_item_qty") + pl.col("wr_item_qty")) / 3.0 * 100).alias("wr_dev"),
            ((pl.col("sr_item_qty") + pl.col("cr_item_qty") + pl.col("wr_item_qty")) / 3.0).alias("average"),
        ])
    )

    result = result.sort(['item_id', 'sr_item_qty'], descending=[False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("item_id", False), ("sr_item_qty", False)],
        limit=100,
    )
