# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q50 — naive one-for-one Polars translation of the SQL."""
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
        int(run_config.scale_factor), query_id=50, qualification=run_config.qualification
    )
    year = params["year"]
    month = params["month"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(run_config.dataset_path, "store_returns", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # FROM store_sales, store_returns, store, date_dim d1, date_dim d2
    # WHERE d2.d_year=year AND d2.d_moy=month
    #   AND ss_ticket_number=sr_ticket_number AND ss_item_sk=sr_item_sk
    #   AND ss_sold_date_sk=d1.d_date_sk AND sr_returned_date_sk=d2.d_date_sk
    #   AND ss_customer_sk=sr_customer_sk AND ss_store_sk=s_store_sk
    # All WHERE conditions after all joins (naive rule 2)
    # Two aliases of date_dim: d1 (for sold) and d2 (for returned, filtered by year+month)
    d1 = date_dim.select(["d_date_sk"])
    d2 = date_dim.filter((pl.col("d_year") == year) & (pl.col("d_moy") == month)).select(
        pl.col("d_date_sk").alias("d2_date_sk")
    )

    group_cols = [
        "s_store_name",
        "s_company_id",
        "s_street_number",
        "s_street_name",
        "s_street_type",
        "s_suite_number",
        "s_city",
        "s_county",
        "s_state",
        "s_zip",
    ]

    diff = pl.col("sr_returned_date_sk") - pl.col("ss_sold_date_sk")

    result = (
        store_sales.join(store_returns, left_on=["ss_ticket_number", "ss_item_sk", "ss_customer_sk"],
                         right_on=["sr_ticket_number", "sr_item_sk", "sr_customer_sk"])
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(d1, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(d2, left_on="sr_returned_date_sk", right_on="d2_date_sk")
        .group_by(group_cols)
        .agg(
            [
                sql_sum(
                    pl.when(diff <= 30).then(pl.lit(1)).otherwise(pl.lit(0))
                ).alias("30 days"),
                sql_sum(
                    pl.when((diff > 30) & (diff <= 60)).then(pl.lit(1)).otherwise(pl.lit(0))
                ).alias("31-60 days"),
                sql_sum(
                    pl.when((diff > 60) & (diff <= 90)).then(pl.lit(1)).otherwise(pl.lit(0))
                ).alias("61-90 days"),
                sql_sum(
                    pl.when((diff > 90) & (diff <= 120)).then(pl.lit(1)).otherwise(pl.lit(0))
                ).alias("91-120 days"),
                sql_sum(
                    pl.when(diff > 120).then(pl.lit(1)).otherwise(pl.lit(0))
                ).alias(">120 days"),
            ]
        )
        .select(
            group_cols
            + ["30 days", "31-60 days", "61-90 days", "91-120 days", ">120 days"]
        )
    )

    result = result.sort(group_cols, descending=False, nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[(col, False) for col in group_cols],
        limit=100,
    )
