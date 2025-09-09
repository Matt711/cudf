# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 59."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 59."""
    return """
    WITH wss
         AS (SELECT d_week_seq,
                    ss_store_sk,
                    Sum(CASE
                          WHEN ( d_day_name = 'Sunday' ) THEN ss_sales_price
                          ELSE NULL
                        END) sun_sales,
                    Sum(CASE
                          WHEN ( d_day_name = 'Monday' ) THEN ss_sales_price
                          ELSE NULL
                        END) mon_sales,
                    Sum(CASE
                          WHEN ( d_day_name = 'Tuesday' ) THEN ss_sales_price
                          ELSE NULL
                        END) tue_sales,
                    Sum(CASE
                          WHEN ( d_day_name = 'Wednesday' ) THEN ss_sales_price
                          ELSE NULL
                        END) wed_sales,
                    Sum(CASE
                          WHEN ( d_day_name = 'Thursday' ) THEN ss_sales_price
                          ELSE NULL
                        END) thu_sales,
                    Sum(CASE
                          WHEN ( d_day_name = 'Friday' ) THEN ss_sales_price
                          ELSE NULL
                        END) fri_sales,
                    Sum(CASE
                          WHEN ( d_day_name = 'Saturday' ) THEN ss_sales_price
                          ELSE NULL
                        END) sat_sales
             FROM   store_sales,
                    date_dim
             WHERE  d_date_sk = ss_sold_date_sk
             GROUP  BY d_week_seq,
                       ss_store_sk)
    SELECT s_store_name1,
                   s_store_id1,
                   d_week_seq1,
                   sun_sales1 / sun_sales2,
                   mon_sales1 / mon_sales2,
                   tue_sales1 / tue_sales2,
                   wed_sales1 / wed_sales2,
                   thu_sales1 / thu_sales2,
                   fri_sales1 / fri_sales2,
                   sat_sales1 / sat_sales2
    FROM   (SELECT s_store_name   s_store_name1,
                   wss.d_week_seq d_week_seq1,
                   s_store_id     s_store_id1,
                   sun_sales      sun_sales1,
                   mon_sales      mon_sales1,
                   tue_sales      tue_sales1,
                   wed_sales      wed_sales1,
                   thu_sales      thu_sales1,
                   fri_sales      fri_sales1,
                   sat_sales      sat_sales1
            FROM   wss,
                   store,
                   date_dim d
            WHERE  d.d_week_seq = wss.d_week_seq
                   AND ss_store_sk = s_store_sk
                   AND d_month_seq BETWEEN 1196 AND 1196 + 11) y,
           (SELECT s_store_name   s_store_name2,
                   wss.d_week_seq d_week_seq2,
                   s_store_id     s_store_id2,
                   sun_sales      sun_sales2,
                   mon_sales      mon_sales2,
                   tue_sales      tue_sales2,
                   wed_sales      wed_sales2,
                   thu_sales      thu_sales2,
                   fri_sales      fri_sales2,
                   sat_sales      sat_sales2
            FROM   wss,
                   store,
                   date_dim d
            WHERE  d.d_week_seq = wss.d_week_seq
                   AND ss_store_sk = s_store_sk
                   AND d_month_seq BETWEEN 1196 + 12 AND 1196 + 23) x
    WHERE  s_store_id1 = s_store_id2
           AND d_week_seq1 = d_week_seq2 - 52
    ORDER  BY s_store_name1,
              s_store_id1,
              d_week_seq1
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 59."""
    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    # CTE: wss - Weekly store sales pivoted by day of week
    # Create base data with week and store
    base_data = store_sales.join(
        date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk"
    ).select(["d_week_seq", "ss_store_sk", "d_day_name", "ss_sales_price"])
    # Create separate queries for each day
    days = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ]
    day_queries = {}
    for day in days:
        day_queries[day] = (
            base_data
            # TODO: There's some bug in cudf_polars with FILTER on Column with NULLs
            # .filter(pl.col("d_day_name") == day)
            .filter(pl.col("d_day_name").is_not_null() & (pl.col("d_day_name") == day))
            .group_by(["d_week_seq", "ss_store_sk"])
            .agg(
                [
                    pl.col("ss_sales_price").count().alias(f"{day.lower()[:3]}_count"),
                    pl.col("ss_sales_price").sum().alias(f"{day.lower()[:3]}_sum"),
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col(f"{day.lower()[:3]}_count") == 0)
                    .then(None)
                    .otherwise(pl.col(f"{day.lower()[:3]}_sum"))
                    .alias(f"{day.lower()[:3]}_sales")
                ]
            )
            .select(["d_week_seq", "ss_store_sk", f"{day.lower()[:3]}_sales"])
        )
    # Get all unique week/store combinations
    wss_base = base_data.select(["d_week_seq", "ss_store_sk"]).unique()
    # Join all days together
    wss = wss_base
    for day in days:
        wss = wss.join(day_queries[day], on=["d_week_seq", "ss_store_sk"], how="left")
    wss = wss.select(
        [
            "d_week_seq",
            "ss_store_sk",
            "sun_sales",
            "mon_sales",
            "tue_sales",
            "wed_sales",
            "thu_sales",
            "fri_sales",
            "sat_sales",
        ]
    )
    # Subquery y: Year 1 data (month_seq 1196-1207)
    y = (
        wss.join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(date_dim, left_on="d_week_seq", right_on="d_week_seq")
        .filter(pl.col("d_month_seq").is_between(1196, 1196 + 11))
        .select(
            [
                pl.col("s_store_name").alias("s_store_name1"),
                pl.col("d_week_seq").alias("d_week_seq1"),
                pl.col("s_store_id").alias("s_store_id1"),
                pl.col("sun_sales").alias("sun_sales1"),
                pl.col("mon_sales").alias("mon_sales1"),
                pl.col("tue_sales").alias("tue_sales1"),
                pl.col("wed_sales").alias("wed_sales1"),
                pl.col("thu_sales").alias("thu_sales1"),
                pl.col("fri_sales").alias("fri_sales1"),
                pl.col("sat_sales").alias("sat_sales1"),
            ]
        )
    )
    # Subquery x: Year 2 data (month_seq 1208-1219)
    x = (
        wss.join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(date_dim, left_on="d_week_seq", right_on="d_week_seq")
        .filter(pl.col("d_month_seq").is_between(1196 + 12, 1196 + 23))
        .select(
            [
                pl.col("s_store_name").alias("s_store_name2"),
                pl.col("d_week_seq").alias("d_week_seq2"),
                pl.col("s_store_id").alias("s_store_id2"),
                pl.col("sun_sales").alias("sun_sales2"),
                pl.col("mon_sales").alias("mon_sales2"),
                pl.col("tue_sales").alias("tue_sales2"),
                pl.col("wed_sales").alias("wed_sales2"),
                pl.col("thu_sales").alias("thu_sales2"),
                pl.col("fri_sales").alias("fri_sales2"),
                pl.col("sat_sales").alias("sat_sales2"),
            ]
        )
    )
    # Cross join with year-over-year matching and ratio calculations
    return (
        y.join(x, left_on=["s_store_id1"], right_on=["s_store_id2"], how="inner")
        .filter(
            # Same store and corresponding week (52 weeks apart)
            pl.col("d_week_seq1") == pl.col("d_week_seq2") - 52
        )
        .sort(
            ["s_store_name1", "s_store_id1", "d_week_seq1"],
            nulls_last=True,
            descending=[False, False, False],
        )
        .limit(100)
        .select(
            [
                "s_store_name1",
                "s_store_id1",
                "d_week_seq1",
                (pl.col("sun_sales1") / pl.col("sun_sales2")).alias(
                    "(sun_sales1 / sun_sales2)"
                ),
                (pl.col("mon_sales1") / pl.col("mon_sales2")).alias(
                    "(mon_sales1 / mon_sales2)"
                ),
                (pl.col("tue_sales1") / pl.col("tue_sales2")).alias(
                    "(tue_sales1 / tue_sales2)"
                ),
                (pl.col("wed_sales1") / pl.col("wed_sales2")).alias(
                    "(wed_sales1 / wed_sales2)"
                ),
                (pl.col("thu_sales1") / pl.col("thu_sales2")).alias(
                    "(thu_sales1 / thu_sales2)"
                ),
                (pl.col("fri_sales1") / pl.col("fri_sales2")).alias(
                    "(fri_sales1 / fri_sales2)"
                ),
                (pl.col("sat_sales1") / pl.col("sat_sales2")).alias(
                    "(sat_sales1 / sat_sales2)"
                ),
            ]
        )
    )
