# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q36 — naive one-for-one Polars translation of the SQL."""
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
        int(run_config.scale_factor), query_id=36, qualification=run_config.qualification
    )
    year = params["year"]
    state = params["state"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)

    # FROM store_sales, date_dim d1, item, store
    # WHERE d1.d_year = year AND d1.d_date_sk = ss_sold_date_sk
    #   AND i_item_sk = ss_item_sk AND s_store_sk = ss_store_sk
    #   AND s_state IN (state)
    # All WHERE conditions after all joins (naive rule 2)
    base_lf = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .filter((pl.col("d_year") == year) & (pl.col("s_state").is_in(state)))
    )

    # GROUP BY rollup(i_category, i_class)
    # Aggregates: Sum(ss_net_profit), Sum(ss_ext_sales_price)
    rolled = sql_rollup(
        base_lf,
        ["i_category", "i_class"],
        [
            sql_sum(pl.col("ss_net_profit")).alias("sum_net_profit"),
            sql_sum(pl.col("ss_ext_sales_price")).alias("sum_ext_sales_price"),
        ],
    )

    # lochierarchy = Grouping(i_category) + Grouping(i_class) = _rollup_lochierarchy
    # gross_margin = sum_net_profit / sum_ext_sales_price
    rolled = rolled.with_columns(
        pl.col("_rollup_lochierarchy").alias("lochierarchy"),
        (
            pl.col("sum_net_profit").cast(pl.Float64)
            / pl.col("sum_ext_sales_price").cast(pl.Float64)
        ).alias("gross_margin"),
    )

    # RANK() OVER (PARTITION BY Grouping(i_category)+Grouping(i_class),
    #              CASE WHEN Grouping(i_class)=0 THEN i_category END
    #              ORDER BY gross_margin ASC)
    # Grouping(i_class)=0 iff _rollup_lochierarchy==0 (finest level)
    rank_partition_key = (
        pl.when(pl.col("_rollup_lochierarchy") == 0)
        .then(pl.col("i_category"))
        .otherwise(pl.lit(None))
    )

    result = (
        rolled.with_columns(rank_partition_key.alias("_rank_key"))
        .with_columns(
            pl.col("gross_margin")
            .rank(method="min")
            .over(["lochierarchy", "_rank_key"])
            .alias("rank_within_parent")
        )
        .select(
            [
                "gross_margin",
                "i_category",
                "i_class",
                "lochierarchy",
                "rank_within_parent",
            ]
        )
    )

    # ORDER BY lochierarchy DESC, CASE WHEN lochierarchy=0 THEN i_category END, rank_within_parent
    result = (
        result
        .sort(
            [
                pl.col("lochierarchy"),
                pl.when(pl.col("lochierarchy") == 0)
                .then(pl.col("i_category"))
                .otherwise(pl.lit(None)),
                pl.col("rank_within_parent"),
            ],
            descending=[True, False, False],
            nulls_last=True,
        )
        .limit(100)
    )
    return QueryResult(
        frame=result,
        sort_by=[
            ("lochierarchy", True),
            ("rank_within_parent", False),
        ],
        limit=100,
        sort_keys=[
            (pl.col("lochierarchy"), True),
            (
                pl.when(pl.col("lochierarchy") == 0)
                .then(pl.col("i_category"))
                .otherwise(pl.lit(None)),
                False,
            ),
            (pl.col("rank_within_parent"), False),
        ],
    )
