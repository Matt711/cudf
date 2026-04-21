# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q9 — naive one-for-one Polars translation of the SQL."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:  # type: ignore[return]
    raise NotImplementedError


def polars_impl(run_config: RunConfig) -> QueryResult:
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=9,
        qualification=run_config.qualification,
    )
    aggcthen = params["aggcthen"]
    aggcelse = params["aggcelse"]
    rc = params["rc"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    reason = get_data(run_config.dataset_path, "reason", run_config.suffix)

    # q9 has scalar correlated subqueries inside CASE WHEN.
    # Each bucket: (min, max, count_threshold)
    buckets = [
        (1, 20, rc[0]),
        (21, 40, rc[1]),
        (41, 60, rc[2]),
        (61, 80, rc[3]),
        (81, 100, rc[4]),
    ]

    # Compute all bucket counts and avgs eagerly (scalar subqueries)
    bucket_vals: list[float | None] = []
    for min_qty, max_qty, threshold in buckets:
        bucket_ss = (
            store_sales.filter(
                pl.col("ss_quantity").is_between(min_qty, max_qty, closed="both")
            )
            .select(
                pl.len().alias("cnt"),
                pl.col(aggcthen).mean().alias("avg_then"),
                pl.col(aggcelse).mean().alias("avg_else"),
            )
            .collect()
        )
        cnt_val = bucket_ss["cnt"][0]
        avg_then_val = bucket_ss["avg_then"][0]
        avg_else_val = bucket_ss["avg_else"][0]
        bucket_vals.append(avg_then_val if cnt_val > threshold else avg_else_val)

    # Build a single-row result anchored to reason WHERE r_reason_sk=1
    result = reason.filter(pl.col("r_reason_sk") == 1).select(
        pl.lit(bucket_vals[0]).alias("bucket1"),
        pl.lit(bucket_vals[1]).alias("bucket2"),
        pl.lit(bucket_vals[2]).alias("bucket3"),
        pl.lit(bucket_vals[3]).alias("bucket4"),
        pl.lit(bucket_vals[4]).alias("bucket5"),
    )

    return QueryResult(frame=result, sort_by=[], limit=None)
