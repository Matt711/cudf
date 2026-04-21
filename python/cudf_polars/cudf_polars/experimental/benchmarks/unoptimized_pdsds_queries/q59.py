# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q59 — naive one-for-one Polars translation of the SQL."""

from __future__ import annotations

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
    """TPC-DS q59 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=59, qualification=run_config.qualification
    )
    dms = params["dms"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)

    # wss CTE:
    # FROM store_sales, date_dim WHERE d_date_sk = ss_sold_date_sk
    # GROUP BY d_week_seq, ss_store_sk
    # SELECT d_week_seq, ss_store_sk,
    #   Sum(CASE WHEN d_day_name='Sunday' THEN ss_sales_price ELSE NULL END) sun_sales, ...
    wss = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .group_by(["d_week_seq", "ss_store_sk"])
        .agg(
            pl.when(pl.col("d_day_name") == "Sunday")
            .then(pl.col("ss_sales_price"))
            .otherwise(None)
            .sum()
            .alias("sun_sales"),
            pl.when(pl.col("d_day_name") == "Monday")
            .then(pl.col("ss_sales_price"))
            .otherwise(None)
            .sum()
            .alias("mon_sales"),
            pl.when(pl.col("d_day_name") == "Tuesday")
            .then(pl.col("ss_sales_price"))
            .otherwise(None)
            .sum()
            .alias("tue_sales"),
            pl.when(pl.col("d_day_name") == "Wednesday")
            .then(pl.col("ss_sales_price"))
            .otherwise(None)
            .sum()
            .alias("wed_sales"),
            pl.when(pl.col("d_day_name") == "Thursday")
            .then(pl.col("ss_sales_price"))
            .otherwise(None)
            .sum()
            .alias("thu_sales"),
            pl.when(pl.col("d_day_name") == "Friday")
            .then(pl.col("ss_sales_price"))
            .otherwise(None)
            .sum()
            .alias("fri_sales"),
            pl.when(pl.col("d_day_name") == "Saturday")
            .then(pl.col("ss_sales_price"))
            .otherwise(None)
            .sum()
            .alias("sat_sales"),
        )
    )

    # Inline view y:
    # FROM wss, store, date_dim d
    # WHERE d.d_week_seq = wss.d_week_seq AND ss_store_sk = s_store_sk
    #   AND d_month_seq BETWEEN dms AND dms+11
    # SELECT ... aliases with suffix 1
    wss_with_store = wss.join(
        store.select(["s_store_sk", "s_store_id", "s_store_name"]),
        left_on="ss_store_sk",
        right_on="s_store_sk",
        how="inner",
    ).join(
        date_dim.select(["d_week_seq", "d_month_seq"]),
        on="d_week_seq",
        how="inner",
    )

    y = (
        wss_with_store.filter(pl.col("d_month_seq").is_between(dms, dms + 11))
        .select(
            pl.col("s_store_name").alias("s_store_name1"),
            pl.col("s_store_id").alias("s_store_id1"),
            pl.col("d_week_seq").alias("d_week_seq1"),
            pl.col("sun_sales").alias("sun_sales1"),
            pl.col("mon_sales").alias("mon_sales1"),
            pl.col("tue_sales").alias("tue_sales1"),
            pl.col("wed_sales").alias("wed_sales1"),
            pl.col("thu_sales").alias("thu_sales1"),
            pl.col("fri_sales").alias("fri_sales1"),
            pl.col("sat_sales").alias("sat_sales1"),
        )
    )

    # Inline view x:
    # FROM wss, store, date_dim d
    # WHERE d.d_week_seq = wss.d_week_seq AND ss_store_sk = s_store_sk
    #   AND d_month_seq BETWEEN dms+12 AND dms+23
    # SELECT ... aliases with suffix 2
    x = (
        wss_with_store.filter(pl.col("d_month_seq").is_between(dms + 12, dms + 23))
        .select(
            pl.col("s_store_id").alias("s_store_id2"),
            # d_week_seq2 - 52 for the join: d_week_seq1 = d_week_seq2 - 52
            (pl.col("d_week_seq") - 52).alias("d_week_seq_join"),
            pl.col("sun_sales").alias("sun_sales2"),
            pl.col("mon_sales").alias("mon_sales2"),
            pl.col("tue_sales").alias("tue_sales2"),
            pl.col("wed_sales").alias("wed_sales2"),
            pl.col("thu_sales").alias("thu_sales2"),
            pl.col("fri_sales").alias("fri_sales2"),
            pl.col("sat_sales").alias("sat_sales2"),
        )
    )

    # Final:
    # FROM y, x WHERE s_store_id1 = s_store_id2 AND d_week_seq1 = d_week_seq2 - 52
    # SELECT s_store_name1, s_store_id1, d_week_seq1, ratio columns
    result = (
        y.join(
            x,
            left_on=["s_store_id1", "d_week_seq1"],
            right_on=["s_store_id2", "d_week_seq_join"],
            how="inner",
        )
        .select(
            "s_store_name1",
            "s_store_id1",
            "d_week_seq1",
            (pl.col("sun_sales1") / pl.col("sun_sales2")).alias("(sun_sales1 / sun_sales2)"),
            (pl.col("mon_sales1") / pl.col("mon_sales2")).alias("(mon_sales1 / mon_sales2)"),
            (pl.col("tue_sales1") / pl.col("tue_sales2")).alias("(tue_sales1 / tue_sales2)"),
            (pl.col("wed_sales1") / pl.col("wed_sales2")).alias("(wed_sales1 / wed_sales2)"),
            (pl.col("thu_sales1") / pl.col("thu_sales2")).alias("(thu_sales1 / thu_sales2)"),
            (pl.col("fri_sales1") / pl.col("fri_sales2")).alias("(fri_sales1 / fri_sales2)"),
            (pl.col("sat_sales1") / pl.col("sat_sales2")).alias("(sat_sales1 / sat_sales2)"),
        )
    )

    result = result.sort(['s_store_name1', 's_store_id1', 'd_week_seq1'], descending=[False, False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("s_store_name1", False), ("s_store_id1", False), ("d_week_seq1", False)],
        limit=100,
    )
