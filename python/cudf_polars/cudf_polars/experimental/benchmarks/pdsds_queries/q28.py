# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 28."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 28."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=28,
        qualification=run_config.qualification,
    )

    lp = params["listprice"]
    ca = params["couponamt"]
    wc = params["wholesalecost"]

    return f"""
    SELECT *
    FROM   (SELECT Avg(ss_list_price)            B1_LP,
                   Count(ss_list_price)          B1_CNT,
                   Count(DISTINCT ss_list_price) B1_CNTD
            FROM   store_sales
            WHERE  ss_quantity BETWEEN 0 AND 5
                   AND ( ss_list_price BETWEEN {lp[0]} AND {lp[0]} + 10
                          OR ss_coupon_amt BETWEEN {ca[0]} AND {ca[0]} + 1000
                          OR ss_wholesale_cost BETWEEN {wc[0]} AND {wc[0]} + 20 )) B1,
           (SELECT Avg(ss_list_price)            B2_LP,
                   Count(ss_list_price)          B2_CNT,
                   Count(DISTINCT ss_list_price) B2_CNTD
            FROM   store_sales
            WHERE  ss_quantity BETWEEN 6 AND 10
                   AND ( ss_list_price BETWEEN {lp[1]} AND {lp[1]} + 10
                          OR ss_coupon_amt BETWEEN {ca[1]} AND {ca[1]} + 1000
                          OR ss_wholesale_cost BETWEEN {wc[1]} AND {wc[1]} + 20 )) B2,
           (SELECT Avg(ss_list_price)            B3_LP,
                   Count(ss_list_price)          B3_CNT,
                   Count(DISTINCT ss_list_price) B3_CNTD
            FROM   store_sales
            WHERE  ss_quantity BETWEEN 11 AND 15
                   AND ( ss_list_price BETWEEN {lp[2]} AND {lp[2]} + 10
                          OR ss_coupon_amt BETWEEN {ca[2]} AND {ca[2]} + 1000
                          OR ss_wholesale_cost BETWEEN {wc[2]} AND {wc[2]} + 20 )) B3,
           (SELECT Avg(ss_list_price)            B4_LP,
                   Count(ss_list_price)          B4_CNT,
                   Count(DISTINCT ss_list_price) B4_CNTD
            FROM   store_sales
            WHERE  ss_quantity BETWEEN 16 AND 20
                   AND ( ss_list_price BETWEEN {lp[3]} AND {lp[3]} + 10
                          OR ss_coupon_amt BETWEEN {ca[3]} AND {ca[3]} + 1000
                          OR ss_wholesale_cost BETWEEN {wc[3]} AND {wc[3]} + 20 )) B4,
           (SELECT Avg(ss_list_price)            B5_LP,
                   Count(ss_list_price)          B5_CNT,
                   Count(DISTINCT ss_list_price) B5_CNTD
            FROM   store_sales
            WHERE  ss_quantity BETWEEN 21 AND 25
                   AND ( ss_list_price BETWEEN {lp[4]} AND {lp[4]} + 10
                          OR ss_coupon_amt BETWEEN {ca[4]} AND {ca[4]} + 1000
                          OR ss_wholesale_cost BETWEEN {wc[4]} AND {wc[4]} + 20 )) B5,
           (SELECT Avg(ss_list_price)            B6_LP,
                   Count(ss_list_price)          B6_CNT,
                   Count(DISTINCT ss_list_price) B6_CNTD
            FROM   store_sales
            WHERE  ss_quantity BETWEEN 26 AND 30
                   AND ( ss_list_price BETWEEN {lp[5]} AND {lp[5]} + 10
                          OR ss_coupon_amt BETWEEN {ca[5]} AND {ca[5]} + 1000
                          OR ss_wholesale_cost BETWEEN {wc[5]} AND {wc[5]} + 20 )) B6
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 28."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=28,
        qualification=run_config.qualification,
    )

    lp = params["listprice"]
    ca = params["couponamt"]
    wc = params["wholesalecost"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)

    buckets = [
        (0, 5, 0, "B1"),
        (6, 10, 1, "B2"),
        (11, 15, 2, "B3"),
        (16, 20, 3, "B4"),
        (21, 25, 4, "B5"),
        (26, 30, 5, "B6"),
    ]

    exprs = []
    for q_lo, q_hi, i, prefix in buckets:
        c = pl.col("ss_quantity").is_between(q_lo, q_hi) & (
            pl.col("ss_list_price").is_between(lp[i], lp[i] + 10)
            | pl.col("ss_coupon_amt").is_between(ca[i], ca[i] + 1000)
            | pl.col("ss_wholesale_cost").is_between(wc[i], wc[i] + 20)
        )
        lp_masked = pl.col("ss_list_price").filter(c)
        exprs.extend(
            [
                lp_masked.mean().alias(f"{prefix}_LP"),
                lp_masked.count().alias(f"{prefix}_CNT"),
                lp_masked.drop_nulls().n_unique().alias(f"{prefix}_CNTD"),
            ]
        )

    limit = 100
    return QueryResult(
        frame=store_sales.select(exprs).limit(limit),
        sort_by=[],
        limit=limit,
    )
