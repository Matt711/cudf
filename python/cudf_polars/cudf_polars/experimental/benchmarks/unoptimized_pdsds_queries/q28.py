# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q28 — naive one-for-one Polars translation of the SQL."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data
from cudf_polars.experimental.benchmarks.unoptimized_pdsds_queries.sql_helpers import (
    sql_n_unique,
    sql_sum,
)

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:  # type: ignore[return]
    raise NotImplementedError


def polars_impl(run_config: RunConfig) -> QueryResult:
    """TPC-DS q28 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=28, qualification=run_config.qualification
    )

    lp = params["listprice"]
    ca = params["couponamt"]
    wc = params["wholesalecost"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)

    def make_block(
        q_lo: int,
        q_hi: int,
        lp_min: int,
        ca_min: int,
        wc_min: int,
        prefix: str,
    ) -> pl.LazyFrame:
        return (
            store_sales
            .filter(
                pl.col("ss_quantity").is_between(q_lo, q_hi)
                & (
                    pl.col("ss_list_price").is_between(lp_min, lp_min + 10)
                    | pl.col("ss_coupon_amt").is_between(ca_min, ca_min + 1000)
                    | pl.col("ss_wholesale_cost").is_between(wc_min, wc_min + 20)
                )
            )
            .select([
                pl.col("ss_list_price").mean().alias(f"{prefix}_LP"),
                pl.col("ss_list_price").count().alias(f"{prefix}_CNT"),
                sql_n_unique(pl.col("ss_list_price")).cast(pl.Int64).alias(f"{prefix}_CNTD"),
            ])
        )

    b1 = make_block(0, 5, lp[0], ca[0], wc[0], "B1")
    b2 = make_block(6, 10, lp[1], ca[1], wc[1], "B2")
    b3 = make_block(11, 15, lp[2], ca[2], wc[2], "B3")
    b4 = make_block(16, 20, lp[3], ca[3], wc[3], "B4")
    b5 = make_block(21, 25, lp[4], ca[4], wc[4], "B5")
    b6 = make_block(26, 30, lp[5], ca[5], wc[5], "B6")

    result = (
        b1.join(b2, how="cross")
        .join(b3, how="cross")
        .join(b4, how="cross")
        .join(b5, how="cross")
        .join(b6, how="cross")
    )

    sort_by: list = []
    limit = 100

    result = result.limit(limit)
    return QueryResult(frame=result, sort_by=sort_by, limit=limit)
