# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q93 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q93 naive Polars implementation.

    SELECT ss_customer_sk, Sum(act_sales) sumsales
    FROM (
      SELECT ss_item_sk, ss_ticket_number, ss_customer_sk,
             CASE WHEN sr_return_quantity IS NOT NULL
                  THEN (ss_quantity - sr_return_quantity) * ss_sales_price
                  ELSE (ss_quantity * ss_sales_price) END act_sales
      FROM store_sales
           LEFT OUTER JOIN store_returns ON (sr_item_sk = ss_item_sk
                                             AND sr_ticket_number = ss_ticket_number),
           reason
      WHERE sr_reason_sk = r_reason_sk AND r_reason_desc = reason_desc
    ) t
    GROUP BY ss_customer_sk
    ORDER BY sumsales, ss_customer_sk LIMIT 100;
    """
    params = load_parameters(
        int(run_config.scale_factor), query_id=93, qualification=run_config.qualification
    )
    reason_desc = params["reason_desc"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(run_config.dataset_path, "store_returns", run_config.suffix)
    reason = get_data(run_config.dataset_path, "reason", run_config.suffix)

    # Subquery t:
    # store_sales LEFT OUTER JOIN store_returns ON (sr_item_sk = ss_item_sk AND sr_ticket_number = ss_ticket_number),
    # reason  (comma-sep = inner join via WHERE: sr_reason_sk = r_reason_sk)
    # WHERE r_reason_desc = reason_desc
    # ALL WHERE conditions after joins (naive rule 2)
    t = (
        store_sales
        .join(store_returns,
              left_on=["ss_item_sk", "ss_ticket_number"],
              right_on=["sr_item_sk", "sr_ticket_number"],
              how="left")
        .join(reason, how="inner", left_on="sr_reason_sk", right_on="r_reason_sk")
        .filter(pl.col("r_reason_desc") == reason_desc)
        .with_columns(
            pl.when(pl.col("sr_return_quantity").is_not_null())
            .then((pl.col("ss_quantity") - pl.col("sr_return_quantity")) * pl.col("ss_sales_price"))
            .otherwise(pl.col("ss_quantity") * pl.col("ss_sales_price"))
            .alias("act_sales")
        )
    )

    # Main query: GROUP BY ss_customer_sk, SUM(act_sales) sumsales
    result = (
        t
        .group_by("ss_customer_sk")
        .agg([sql_sum(pl.col("act_sales")).alias("sumsales")])
        .select(["ss_customer_sk", "sumsales"])
    )

    result = result.sort(['sumsales', 'ss_customer_sk'], descending=[False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("sumsales", False), ("ss_customer_sk", False)],
        limit=100,
    )
