# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q67 — naive one-for-one Polars translation of the SQL."""
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
    params = load_parameters(
        int(run_config.scale_factor), query_id=67, qualification=run_config.qualification
    )
    dms = params["dms"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # FROM store_sales, date_dim, store, item — comma = inner join
    # WHERE conditions applied after all joins (naive rule 2)
    base = (
        store_sales
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk", how="inner")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk", how="inner")
        .filter(pl.col("d_month_seq").is_between(dms, dms + 11))
    )

    # GROUP BY rollup(i_category, i_class, i_brand, i_product_name,
    #                 d_year, d_qoy, d_moy, s_store_id)
    # sum(coalesce(ss_sales_price*ss_quantity, 0)) sumsales
    rollup_cols = [
        "i_category", "i_class", "i_brand", "i_product_name",
        "d_year", "d_qoy", "d_moy", "s_store_id",
    ]
    agg_exprs = [
        sql_sum(
            pl.coalesce([
                pl.col("ss_sales_price") * pl.col("ss_quantity"),
                pl.lit(0.0),
            ])
        ).alias("sumsales")
    ]

    dw1 = sql_rollup(base, rollup_cols, agg_exprs)

    # RANK() OVER (PARTITION BY i_category ORDER BY sumsales DESC) rk
    dw2 = dw1.with_columns(
        pl.col("sumsales")
        .rank(method="min", descending=True)
        .over("i_category")
        .alias("rk")
    )

    # WHERE rk <= 100
    result = dw2.filter(pl.col("rk") <= 100)

    result = result.sort(['i_category', 'i_class', 'i_brand', 'i_product_name', 'd_year', 'd_qoy', 'd_moy', 's_store_id', 'sumsales', 'rk'], descending=[False, False, False, False, False, False, False, False, False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[
            ("i_category", False),
            ("i_class", False),
            ("i_brand", False),
            ("i_product_name", False),
            ("d_year", False),
            ("d_qoy", False),
            ("d_moy", False),
            ("s_store_id", False),
            ("sumsales", False),
            ("rk", False),
        ],
        limit=100,
    )
