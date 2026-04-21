# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q70 — naive one-for-one Polars translation of the SQL."""
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
        int(run_config.scale_factor), query_id=70, qualification=run_config.qualification
    )
    dms = params["dms"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)

    # Inner subquery: rank states by total net_profit, keep ranking <= 5
    # FROM store_sales, store, date_dim WHERE d_month_seq BETWEEN dms AND dms+11
    #   AND d_date_sk = ss_sold_date_sk AND s_store_sk = ss_store_sk
    # GROUP BY s_state
    # RANK() OVER (PARTITION BY s_state ORDER BY Sum(ss_net_profit) DESC) AS ranking
    inner_base = (
        store_sales
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk", how="inner")
        .filter(pl.col("d_month_seq").is_between(dms, dms + 11))
        .group_by("s_state")
        .agg(sql_sum(pl.col("ss_net_profit")).alias("state_profit"))
    )
    # rank within each state (partition BY s_state, only one row per state → rank=1 always)
    # The subquery filters ranking <= 5 → keep top-5 states by total net_profit
    top_states = (
        inner_base
        .with_columns(
            pl.col("state_profit")
            .rank(method="min", descending=True)
            .over(pl.lit("all"))
            .alias("ranking")
        )
        .filter(pl.col("ranking") <= 5)
        .select("s_state")
    )

    # Main query:
    # FROM store_sales, date_dim d1, store
    # WHERE d1.d_month_seq BETWEEN dms AND dms+11
    #   AND d1.d_date_sk = ss_sold_date_sk
    #   AND s_store_sk = ss_store_sk
    #   AND s_state IN (top_states subquery)
    # GROUP BY rollup(s_state, s_county)
    main_base = (
        store_sales
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk", how="inner")
        .filter(pl.col("d_month_seq").is_between(dms, dms + 11))
        .join(top_states, on="s_state", how="semi")
    )

    agg_exprs = [sql_sum(pl.col("ss_net_profit")).alias("total_sum")]
    rollup_result = sql_rollup(main_base, ["s_state", "s_county"], agg_exprs)

    # Grouping(s_state) + Grouping(s_county) AS lochierarchy = _rollup_lochierarchy
    # RANK() OVER (PARTITION BY lochierarchy,
    #              CASE WHEN Grouping(s_county)=0 THEN s_state END
    #              ORDER BY Sum(ss_net_profit) DESC)
    # Grouping(s_county)=0 iff _rollup_lochierarchy==0 (finest level)
    result = (
        rollup_result
        .with_columns([
            pl.col("_rollup_lochierarchy").cast(pl.Int32).alias("lochierarchy"),
        ])
        .with_columns([
            pl.when(pl.col("_rollup_lochierarchy") == 0)
            .then(pl.col("s_state"))
            .otherwise(pl.lit(None))
            .alias("_rank_partition_key"),
        ])
        .with_columns([
            pl.col("total_sum")
            .rank(method="min", descending=True)
            .over(["lochierarchy", "_rank_partition_key"])
            .alias("rank_within_parent"),
        ])
        .select([
            "total_sum",
            "s_state",
            "s_county",
            "lochierarchy",
            "rank_within_parent",
        ])
    )

    result = result.sort(['lochierarchy', 'rank_within_parent'], descending=[True, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("lochierarchy", True), ("rank_within_parent", False)],
        limit=100,
        sort_keys=[
            (pl.col("lochierarchy"), True),
            (
                pl.when(pl.col("lochierarchy") == 0)
                .then(pl.col("s_state"))
                .otherwise(pl.lit(None)),
                False,
            ),
            (pl.col("rank_within_parent"), False),
        ],
    )
