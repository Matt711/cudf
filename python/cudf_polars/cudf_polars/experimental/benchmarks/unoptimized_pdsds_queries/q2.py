# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q2 — naive one-for-one Polars translation of the SQL."""

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
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=2,
        qualification=run_config.qualification,
    )
    year = params["year"]

    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # CTE wscs: UNION ALL of web_sales and catalog_sales
    wscs = pl.concat(
        [
            web_sales.select(
                pl.col("ws_sold_date_sk").alias("sold_date_sk"),
                pl.col("ws_ext_sales_price").alias("sales_price"),
            ),
            catalog_sales.select(
                pl.col("cs_sold_date_sk").alias("sold_date_sk"),
                pl.col("cs_ext_sales_price").alias("sales_price"),
            ),
        ],
        how="diagonal_relaxed",
    )

    # CTE wswscs: FROM wscs, date_dim WHERE d_date_sk = sold_date_sk GROUP BY d_week_seq
    # SUM(CASE WHEN d_day_name = 'X' THEN sales_price ELSE NULL END)
    wswscs = (
        wscs.join(date_dim, left_on="sold_date_sk", right_on="d_date_sk", how="inner")
        .group_by("d_week_seq")
        .agg(
            sql_sum(pl.when(pl.col("d_day_name") == "Sunday").then(pl.col("sales_price")).otherwise(None)).alias("sun_sales"),
            sql_sum(pl.when(pl.col("d_day_name") == "Monday").then(pl.col("sales_price")).otherwise(None)).alias("mon_sales"),
            sql_sum(pl.when(pl.col("d_day_name") == "Tuesday").then(pl.col("sales_price")).otherwise(None)).alias("tue_sales"),
            sql_sum(pl.when(pl.col("d_day_name") == "Wednesday").then(pl.col("sales_price")).otherwise(None)).alias("wed_sales"),
            sql_sum(pl.when(pl.col("d_day_name") == "Thursday").then(pl.col("sales_price")).otherwise(None)).alias("thu_sales"),
            sql_sum(pl.when(pl.col("d_day_name") == "Friday").then(pl.col("sales_price")).otherwise(None)).alias("fri_sales"),
            sql_sum(pl.when(pl.col("d_day_name") == "Saturday").then(pl.col("sales_price")).otherwise(None)).alias("sat_sales"),
        )
    )

    # Subquery y: FROM wswscs, date_dim WHERE date_dim.d_week_seq = wswscs.d_week_seq AND d_year = year
    y = (
        wswscs.join(date_dim, on="d_week_seq", how="inner")
        .filter(pl.col("d_year") == year)
        .select(
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

    # Subquery z: FROM wswscs, date_dim WHERE date_dim.d_week_seq = wswscs.d_week_seq AND d_year = year+1
    z = (
        wswscs.join(date_dim, on="d_week_seq", how="inner")
        .filter(pl.col("d_year") == year + 1)
        .select(
            pl.col("d_week_seq").alias("d_week_seq2"),
            pl.col("sun_sales").alias("sun_sales2"),
            pl.col("mon_sales").alias("mon_sales2"),
            pl.col("tue_sales").alias("tue_sales2"),
            pl.col("wed_sales").alias("wed_sales2"),
            pl.col("thu_sales").alias("thu_sales2"),
            pl.col("fri_sales").alias("fri_sales2"),
            pl.col("sat_sales").alias("sat_sales2"),
        )
    )

    # Main SELECT: FROM y, z WHERE d_week_seq1 = d_week_seq2 - 53
    result = (
        y.join(z, how="inner", left_on="d_week_seq1", right_on=pl.col("d_week_seq2") - 53)
        .filter(pl.col("d_week_seq1") == pl.col("d_week_seq2") - 53)
        .select(
            pl.col("d_week_seq1"),
            (pl.col("sun_sales1") / pl.col("sun_sales2")).round(2).alias("round((sun_sales1 / sun_sales2), 2)"),
            (pl.col("mon_sales1") / pl.col("mon_sales2")).round(2).alias("round((mon_sales1 / mon_sales2), 2)"),
            (pl.col("tue_sales1") / pl.col("tue_sales2")).round(2).alias("round((tue_sales1 / tue_sales2), 2)"),
            (pl.col("wed_sales1") / pl.col("wed_sales2")).round(2).alias("round((wed_sales1 / wed_sales2), 2)"),
            (pl.col("thu_sales1") / pl.col("thu_sales2")).round(2).alias("round((thu_sales1 / thu_sales2), 2)"),
            (pl.col("fri_sales1") / pl.col("fri_sales2")).round(2).alias("round((fri_sales1 / fri_sales2), 2)"),
            (pl.col("sat_sales1") / pl.col("sat_sales2")).round(2).alias("round((sat_sales1 / sat_sales2), 2)"),
        )
    )

    result = result.sort("d_week_seq1", nulls_last=True)
    return QueryResult(frame=result, sort_by=[("d_week_seq1", False)], limit=None)
