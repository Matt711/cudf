# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q86 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q86 naive Polars implementation.

    SELECT Sum(ws_net_paid) AS total_sum, i_category, i_class,
           Grouping(i_category) + Grouping(i_class) AS lochierarchy,
           Rank() OVER (PARTITION BY lochierarchy,
                        CASE WHEN Grouping(i_class)=0 THEN i_category END
                        ORDER BY Sum(ws_net_paid) DESC) AS rank_within_parent
    FROM web_sales, date_dim d1, item
    WHERE d1.d_month_seq BETWEEN d_month_seq AND d_month_seq+11
      AND d1.d_date_sk = ws_sold_date_sk AND i_item_sk = ws_item_sk
    GROUP BY rollup(i_category, i_class)
    ORDER BY lochierarchy DESC,
             CASE WHEN lochierarchy=0 THEN i_category END,
             rank_within_parent
    LIMIT 100;
    """
    params = load_parameters(
        int(run_config.scale_factor), query_id=86, qualification=run_config.qualification
    )
    d_month_seq = params["d_month_seq"]

    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # Join in FROM order: web_sales, date_dim d1, item
    # ALL WHERE conditions after joins (naive rule 2)
    base = (
        web_sales
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(item, left_on="ws_item_sk", right_on="i_item_sk", how="inner")
        .filter(pl.col("d_month_seq").is_between(d_month_seq, d_month_seq + 11))
    )

    # GROUP BY rollup(i_category, i_class) -> sql_rollup
    # The rollup produces lochierarchy implicitly via nulled columns.
    # lochierarchy = Grouping(i_category) + Grouping(i_class)
    #   level 2 (all nulled): Grouping(cat)=1, Grouping(class)=1 -> lochierarchy=2
    #   level 1 (class nulled): Grouping(cat)=0, Grouping(class)=1 -> lochierarchy=1
    #   level 0 (none nulled): Grouping(cat)=0, Grouping(class)=0 -> lochierarchy=0
    rolled = sql_rollup(
        base,
        ["i_category", "i_class"],
        [sql_sum(pl.col("ws_net_paid")).alias("total_sum")],
    )

    # lochierarchy = Grouping(i_category) + Grouping(i_class) = _rollup_lochierarchy
    rolled = rolled.with_columns(
        pl.col("_rollup_lochierarchy").alias("lochierarchy")
    )

    # RANK() OVER (PARTITION BY Grouping(i_category)+Grouping(i_class),
    #              CASE WHEN Grouping(i_class)=0 THEN i_category END
    #              ORDER BY Sum(ws_net_paid) DESC)
    # Grouping(i_class)=0 iff _rollup_lochierarchy==0 (finest level)
    result = (
        rolled
        .with_columns(
            pl.when(pl.col("_rollup_lochierarchy") == 0)
            .then(pl.col("i_category"))
            .otherwise(None)
            .alias("_partition_category")
        )
        .with_columns(
            pl.col("total_sum")
            .rank(method="min", descending=True)
            .over(["lochierarchy", "_partition_category"])
            .cast(pl.Int64)
            .alias("rank_within_parent")
        )
        .select([
            "total_sum",
            "i_category",
            "i_class",
            "lochierarchy",
            "rank_within_parent",
        ])
        .sort(
            [
                "lochierarchy",
                pl.when(pl.col("lochierarchy") == 0).then(pl.col("i_category")).otherwise(None),
                "rank_within_parent",
            ],
            descending=[True, False, False],
            nulls_last=True,
        )
        .limit(100)
    )

    return QueryResult(
        frame=result,
        sort_by=[("lochierarchy", True), ("rank_within_parent", False)],
        limit=100,
        sort_keys=[
            (pl.col("lochierarchy"), True),
            (
                pl.when(pl.col("lochierarchy") == 0)
                .then(pl.col("i_category"))
                .otherwise(None),
                False,
            ),
            (pl.col("rank_within_parent"), False),
        ],
    )
