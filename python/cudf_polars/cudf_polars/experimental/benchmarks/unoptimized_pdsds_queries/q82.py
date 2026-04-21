# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q82 — naive one-for-one Polars translation of the SQL."""
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
        int(run_config.scale_factor), query_id=82, qualification=run_config.qualification
    )
    price = params["price"]
    sdate = params["sdate"]
    manufact = params["manufact"]
    inv_min = params["inv_min"]
    inv_max = params["inv_max"]

    year, month, day = map(int, sdate.split("-"))
    start_dt = pl.date(year, month, day)
    end_dt = start_dt + pl.duration(days=60)

    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    inventory = get_data(run_config.dataset_path, "inventory", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)

    # FROM item, inventory, date_dim, store_sales (comma = inner join)
    # WHERE i_current_price BETWEEN price AND price+30
    #   AND inv_item_sk = i_item_sk
    #   AND d_date_sk = inv_date_sk
    #   AND d_date BETWEEN ... AND ...
    #   AND i_manufact_id IN (manufact)
    #   AND inv_quantity_on_hand BETWEEN inv_min AND inv_max
    #   AND ss_item_sk = i_item_sk
    # GROUP BY i_item_id, i_item_desc, i_current_price
    result = (
        item
        .join(inventory, left_on="i_item_sk", right_on="inv_item_sk", how="inner")
        .join(date_dim, left_on="inv_date_sk", right_on="d_date_sk", how="inner")
        .join(store_sales, left_on="i_item_sk", right_on="ss_item_sk", how="inner")
        .filter(
            pl.col("i_current_price").is_between(price, price + 30)
            & (pl.col("d_date").cast(pl.Date) >= start_dt)
            & (pl.col("d_date").cast(pl.Date) <= end_dt)
            & pl.col("i_manufact_id").is_in(manufact)
            & pl.col("inv_quantity_on_hand").is_between(inv_min, inv_max)
        )
        .group_by(["i_item_id", "i_item_desc", "i_current_price"])
        .agg([])
        .select(["i_item_id", "i_item_desc", "i_current_price"])
    )

    result = result.sort('i_item_id', descending=False, nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("i_item_id", False)],
        limit=100,
    )
