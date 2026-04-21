# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q37 — naive one-for-one Polars translation of the SQL."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data
from cudf_polars.experimental.benchmarks.unoptimized_pdsds_queries.sql_helpers import sql_sum

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:  # type: ignore[return]
    raise NotImplementedError


def polars_impl(run_config: RunConfig) -> QueryResult:
    params = load_parameters(
        int(run_config.scale_factor), query_id=37, qualification=run_config.qualification
    )
    price = params["price"]
    manufact = params["manufact"]
    invdate = params["invdate"]

    start_date_obj = datetime.strptime(invdate, "%Y-%m-%d")
    end_date_obj = start_date_obj + timedelta(days=60)
    start_date = pl.date(start_date_obj.year, start_date_obj.month, start_date_obj.day)
    end_date = pl.date(end_date_obj.year, end_date_obj.month, end_date_obj.day)

    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    inventory = get_data(run_config.dataset_path, "inventory", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)

    # FROM item, inventory, date_dim, catalog_sales
    # WHERE i_current_price BETWEEN price AND price+30
    #   AND inv_item_sk = i_item_sk AND d_date_sk=inv_date_sk
    #   AND d_date BETWEEN invdate AND invdate+60days
    #   AND i_manufact_id IN (manufact)
    #   AND inv_quantity_on_hand BETWEEN 100 AND 500
    #   AND cs_item_sk = i_item_sk
    # All WHERE conditions after all joins (naive rule 2)
    result = (
        item.join(inventory, left_on="i_item_sk", right_on="inv_item_sk")
        .join(date_dim, left_on="inv_date_sk", right_on="d_date_sk")
        .join(catalog_sales, left_on="i_item_sk", right_on="cs_item_sk")
        .filter(
            (pl.col("i_current_price").is_between(price, price + 30))
            & (pl.col("d_date").is_between(start_date, end_date))
            & (pl.col("i_manufact_id").is_in(manufact))
            & (pl.col("inv_quantity_on_hand").is_between(100, 500))
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
