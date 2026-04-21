# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q97 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q97 naive Polars implementation.

    WITH ssci AS (
      SELECT ss_customer_sk customer_sk, ss_item_sk item_sk
      FROM store_sales, date_dim
      WHERE ss_sold_date_sk = d_date_sk
        AND d_month_seq BETWEEN d_month_seq AND d_month_seq+11
      GROUP BY ss_customer_sk, ss_item_sk),
    csci AS (
      SELECT cs_bill_customer_sk customer_sk, cs_item_sk item_sk
      FROM catalog_sales, date_dim
      WHERE cs_sold_date_sk = d_date_sk
        AND d_month_seq BETWEEN d_month_seq AND d_month_seq+11
      GROUP BY cs_bill_customer_sk, cs_item_sk)
    SELECT Sum(CASE WHEN ssci.customer_sk IS NOT NULL AND csci.customer_sk IS NULL THEN 1 ELSE 0 END) store_only,
           Sum(CASE WHEN ssci.customer_sk IS NULL AND csci.customer_sk IS NOT NULL THEN 1 ELSE 0 END) catalog_only,
           Sum(CASE WHEN ssci.customer_sk IS NOT NULL AND csci.customer_sk IS NOT NULL THEN 1 ELSE 0 END) store_and_catalog
    FROM ssci FULL OUTER JOIN csci ON (ssci.customer_sk = csci.customer_sk AND ssci.item_sk = csci.item_sk)
    LIMIT 100;
    """
    params = load_parameters(
        int(run_config.scale_factor), query_id=97, qualification=run_config.qualification
    )
    d_month_seq = params["d_month_seq"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # CTE ssci: store_sales, date_dim (comma-sep = inner join via WHERE)
    # GROUP BY ss_customer_sk, ss_item_sk (DISTINCT customer+item pairs)
    ssci = (
        store_sales
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(pl.col("d_month_seq").is_between(d_month_seq, d_month_seq + 11))
        .group_by(["ss_customer_sk", "ss_item_sk"])
        .agg([])
        .select([
            pl.col("ss_customer_sk").alias("customer_sk"),
            pl.col("ss_item_sk").alias("item_sk"),
        ])
    )

    # CTE csci: catalog_sales, date_dim
    csci = (
        catalog_sales
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(pl.col("d_month_seq").is_between(d_month_seq, d_month_seq + 11))
        .group_by(["cs_bill_customer_sk", "cs_item_sk"])
        .agg([])
        .select([
            pl.col("cs_bill_customer_sk").alias("customer_sk"),
            pl.col("cs_item_sk").alias("item_sk"),
        ])
    )

    # FULL OUTER JOIN ssci, csci ON (customer_sk = customer_sk AND item_sk = item_sk)
    # Polars full join with suffix for right-side columns (rule 11)
    result = (
        ssci
        .join(csci, on=["customer_sk", "item_sk"], how="full", suffix="_catalog")
        .select([
            pl.when(
                pl.col("customer_sk").is_not_null() & pl.col("customer_sk_catalog").is_null()
            ).then(1).otherwise(0).sum().alias("store_only"),
            pl.when(
                pl.col("customer_sk").is_null() & pl.col("customer_sk_catalog").is_not_null()
            ).then(1).otherwise(0).sum().alias("catalog_only"),
            pl.when(
                pl.col("customer_sk").is_not_null() & pl.col("customer_sk_catalog").is_not_null()
            ).then(1).otherwise(0).sum().alias("store_and_catalog"),
        ])
        .limit(100)
    )

    return QueryResult(
        frame=result,
        sort_by=[],
        limit=100,
    )
