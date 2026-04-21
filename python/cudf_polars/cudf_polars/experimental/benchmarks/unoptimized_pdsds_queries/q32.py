# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q32 — naive one-for-one Polars translation of the SQL."""

from __future__ import annotations

from datetime import date, timedelta
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
    """TPC-DS q32 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=32, qualification=run_config.qualification
    )

    imid = params["imid"]
    csdate = params["csdate"]

    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    start_date_obj = date.fromisoformat(csdate)
    end_date_obj = start_date_obj + timedelta(days=90)
    start_date_lit = pl.lit(start_date_obj)
    end_date_lit = pl.lit(end_date_obj)

    # Correlated subquery on i_item_sk:
    # SELECT 1.3 * avg(cs_ext_discount_amt)
    # FROM catalog_sales, date_dim
    # WHERE cs_item_sk = i_item_sk AND d_date BETWEEN ... AND d_date_sk=cs_sold_date_sk
    # Compute per-item avg, join back
    item_avg = (
        catalog_sales
        .join(date_dim, how="inner",
              left_on="cs_sold_date_sk", right_on="d_date_sk")
        .filter(
            pl.col("d_date").is_between(start_date_lit, end_date_lit)
        )
        .group_by("cs_item_sk")
        .agg([
            (pl.col("cs_ext_discount_amt").mean() * 1.3).alias("threshold"),
        ])
    )

    # Main query:
    # FROM catalog_sales, item, date_dim
    # WHERE i_manufact_id=imid AND i_item_sk=cs_item_sk
    #   AND d_date BETWEEN ... AND d_date_sk=cs_sold_date_sk
    #   AND cs_ext_discount_amt > correlated_threshold
    result = (
        catalog_sales
        .join(item, how="inner",
              left_on="cs_item_sk", right_on="i_item_sk")
        .join(date_dim, how="inner",
              left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(item_avg, how="inner",
              left_on="cs_item_sk", right_on="cs_item_sk")
        .filter(
            (pl.col("i_manufact_id") == imid)
            & pl.col("d_date").is_between(start_date_lit, end_date_lit)
            & (pl.col("cs_ext_discount_amt") > pl.col("threshold"))
        )
        .select([
            sql_sum(pl.col("cs_ext_discount_amt")).alias("excess discount amount"),
        ])
    )

    sort_by: list = []
    limit = 100

    result = result.limit(limit)
    return QueryResult(frame=result, sort_by=sort_by, limit=limit)
