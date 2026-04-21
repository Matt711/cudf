# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q22 — naive one-for-one Polars translation of the SQL."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data
from cudf_polars.experimental.benchmarks.unoptimized_pdsds_queries.sql_helpers import (
    sql_rollup,
    sql_sum,
)

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:  # type: ignore[return]
    raise NotImplementedError


def polars_impl(run_config: RunConfig) -> QueryResult:
    """TPC-DS q22 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=22, qualification=run_config.qualification
    )

    dms = params["dms"]

    inventory = get_data(run_config.dataset_path, "inventory", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    warehouse = get_data(run_config.dataset_path, "warehouse", run_config.suffix)

    # FROM order: inventory, date_dim, item, warehouse
    # All WHERE conditions after all joins
    base = (
        inventory
        .join(date_dim, how="inner",
              left_on="inv_date_sk", right_on="d_date_sk")
        .join(item, how="inner",
              left_on="inv_item_sk", right_on="i_item_sk")
        .join(warehouse, how="inner",
              left_on="inv_warehouse_sk", right_on="w_warehouse_sk")
        .filter(
            pl.col("d_month_seq").is_between(dms, dms + 11)
        )
    )

    agg_exprs = [
        pl.col("inv_quantity_on_hand").mean().alias("qoh"),
    ]

    result = sql_rollup(
        base,
        ["i_product_name", "i_brand", "i_class", "i_category"],
        agg_exprs,
    )

    result = result.sort(['qoh', 'i_product_name', 'i_brand', 'i_class', 'i_category'], descending=[False, False, False, False, False], nulls_last=True).limit(100)

    sort_by = [
        ("qoh", False),
        ("i_product_name", False),
        ("i_brand", False),
        ("i_class", False),
        ("i_category", False),
    ]
    limit = 100

    return QueryResult(frame=result, sort_by=sort_by, limit=limit)
