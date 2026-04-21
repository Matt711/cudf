# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q43 — naive one-for-one Polars translation of the SQL."""
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
        int(run_config.scale_factor), query_id=43, qualification=run_config.qualification
    )
    year = params["year"]
    gmt = params["gmt"]

    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)

    # FROM date_dim, store_sales, store
    # WHERE d_date_sk = ss_sold_date_sk AND s_store_sk = ss_store_sk
    #   AND s_gmt_offset = gmt AND d_year = year
    # All WHERE conditions after all joins (naive rule 2)
    result = (
        date_dim.join(store_sales, left_on="d_date_sk", right_on="ss_sold_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .filter((pl.col("s_gmt_offset") == gmt) & (pl.col("d_year") == year))
        .group_by(["s_store_name", "s_store_id"])
        .agg(
            [
                sql_sum(
                    pl.when(pl.col("d_day_name") == "Sunday")
                    .then(pl.col("ss_sales_price"))
                    .otherwise(pl.lit(None))
                ).alias("sun_sales"),
                sql_sum(
                    pl.when(pl.col("d_day_name") == "Monday")
                    .then(pl.col("ss_sales_price"))
                    .otherwise(pl.lit(None))
                ).alias("mon_sales"),
                sql_sum(
                    pl.when(pl.col("d_day_name") == "Tuesday")
                    .then(pl.col("ss_sales_price"))
                    .otherwise(pl.lit(None))
                ).alias("tue_sales"),
                sql_sum(
                    pl.when(pl.col("d_day_name") == "Wednesday")
                    .then(pl.col("ss_sales_price"))
                    .otherwise(pl.lit(None))
                ).alias("wed_sales"),
                sql_sum(
                    pl.when(pl.col("d_day_name") == "Thursday")
                    .then(pl.col("ss_sales_price"))
                    .otherwise(pl.lit(None))
                ).alias("thu_sales"),
                sql_sum(
                    pl.when(pl.col("d_day_name") == "Friday")
                    .then(pl.col("ss_sales_price"))
                    .otherwise(pl.lit(None))
                ).alias("fri_sales"),
                sql_sum(
                    pl.when(pl.col("d_day_name") == "Saturday")
                    .then(pl.col("ss_sales_price"))
                    .otherwise(pl.lit(None))
                ).alias("sat_sales"),
            ]
        )
        .select(
            [
                "s_store_name",
                "s_store_id",
                "sun_sales",
                "mon_sales",
                "tue_sales",
                "wed_sales",
                "thu_sales",
                "fri_sales",
                "sat_sales",
            ]
        )
    )

    result = result.sort(['s_store_name', 's_store_id', 'sun_sales', 'mon_sales', 'tue_sales', 'wed_sales', 'thu_sales', 'fri_sales', 'sat_sales'], descending=[False, False, False, False, False, False, False, False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[
            ("s_store_name", False),
            ("s_store_id", False),
            ("sun_sales", False),
            ("mon_sales", False),
            ("tue_sales", False),
            ("wed_sales", False),
            ("thu_sales", False),
            ("fri_sales", False),
            ("sat_sales", False),
        ],
        limit=100,
    )
