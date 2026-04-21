# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q44 — naive one-for-one Polars translation of the SQL."""
from __future__ import annotations

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
        int(run_config.scale_factor), query_id=44, qualification=run_config.qualification
    )
    store_sk = params["store_sk"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # Correlated subquery: SELECT Avg(ss_net_profit) FROM store_sales
    # WHERE ss_store_sk = store_sk AND ss_cdemo_sk IS NULL GROUP BY ss_store_sk
    # => scalar threshold per store
    benchmark = (
        store_sales.filter(
            (pl.col("ss_store_sk") == store_sk) & (pl.col("ss_cdemo_sk").is_null())
        )
        .group_by("ss_store_sk")
        .agg(pl.col("ss_net_profit").mean().alias("benchmark_profit"))
        .select("benchmark_profit")
    )

    # Inner subquery V1/V2: SELECT ss_item_sk item_sk, Avg(ss_net_profit) rank_col
    # FROM store_sales ss1 WHERE ss_store_sk = store_sk GROUP BY ss_item_sk
    # HAVING Avg(ss_net_profit) > 0.9 * benchmark
    item_profits = (
        store_sales.filter(pl.col("ss_store_sk") == store_sk)
        .group_by("ss_item_sk")
        .agg(pl.col("ss_net_profit").mean().alias("rank_col"))
        .join(benchmark, how="cross")
        .filter(pl.col("rank_col") > 0.9 * pl.col("benchmark_profit"))
        .select(["ss_item_sk", "rank_col"])
    )

    # ascending: RANK() OVER (ORDER BY rank_col ASC)
    # rule 12/note 44: use rank(method="min")
    ascending = (
        item_profits.with_columns(
            pl.col("rank_col").rank(method="min").alias("rnk")
        )
        .filter(pl.col("rnk") < 11)
        .select([pl.col("ss_item_sk").alias("asc_item_sk"), "rnk"])
    )

    # descending: RANK() OVER (ORDER BY rank_col DESC)
    descending = (
        item_profits.with_columns(
            pl.col("rank_col").rank(method="min", descending=True).alias("rnk")
        )
        .filter(pl.col("rnk") < 11)
        .select([pl.col("ss_item_sk").alias("desc_item_sk"), "rnk"])
    )

    # FROM ascending, descending, item i1, item i2
    # WHERE ascending.rnk = descending.rnk
    #   AND i1.i_item_sk = ascending.item_sk
    #   AND i2.i_item_sk = descending.item_sk
    result = (
        ascending.join(descending, on="rnk", how="inner")
        .join(item, left_on="asc_item_sk", right_on="i_item_sk", how="inner")
        .join(
            item,
            left_on="desc_item_sk",
            right_on="i_item_sk",
            how="inner",
            suffix="_worst",
        )
        .select(
            [
                pl.col("rnk"),
                pl.col("i_product_name").alias("best_performing"),
                pl.col("i_product_name_worst").alias("worst_performing"),
            ]
        )
    )

    result = result.sort('rnk', descending=False, nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("rnk", False)],
        limit=100,
    )
