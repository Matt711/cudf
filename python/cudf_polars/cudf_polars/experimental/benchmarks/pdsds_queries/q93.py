# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 93."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 93."""
    return """
    SELECT ss_customer_sk,
                   Sum(act_sales) sumsales
    FROM   (SELECT ss_item_sk,
                   ss_ticket_number,
                   ss_customer_sk,
                   CASE
                     WHEN sr_return_quantity IS NOT NULL THEN
                     ( ss_quantity - sr_return_quantity ) * ss_sales_price
                     ELSE ( ss_quantity * ss_sales_price )
                   END act_sales
            FROM   store_sales
                   LEFT OUTER JOIN store_returns
                                ON ( sr_item_sk = ss_item_sk
                                     AND sr_ticket_number = ss_ticket_number ),
                   reason
            WHERE  sr_reason_sk = r_reason_sk
                   AND r_reason_desc = 'reason 38') t
    GROUP  BY ss_customer_sk
    ORDER  BY sumsales,
              ss_customer_sk
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 93."""
    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(
        run_config.dataset_path, "store_returns", run_config.suffix
    )
    reason = get_data(run_config.dataset_path, "reason", run_config.suffix)
    # The SQL: FROM store_sales LEFT OUTER JOIN store_returns ..., reason
    # WHERE sr_reason_sk = r_reason_sk AND r_reason_desc = 'reason 38'
    # The comma creates a cross join with reason, then WHERE filters
    # This means we need sales that have returns AND those returns have reason 38
    return (
        store_sales.join(
            store_returns,
            left_on=["ss_item_sk", "ss_ticket_number"],
            right_on=["sr_item_sk", "sr_ticket_number"],
            how="left",
        )
        # Cross join with reason table (but we'll filter immediately)
        .join(reason, how="cross")
        .filter(
            # WHERE sr_reason_sk = r_reason_sk AND r_reason_desc = 'reason 38'
            (pl.col("sr_reason_sk") == pl.col("r_reason_sk"))
            & (pl.col("r_reason_desc") == "reason 38")
        )
        .with_columns(
            [
                # Calculate actual sales using CASE logic with proper NULL handling
                pl.when(pl.col("sr_return_quantity").is_not_null())
                .then(
                    pl.when(
                        pl.col("ss_quantity").is_not_null()
                        & pl.col("sr_return_quantity").is_not_null()
                        & pl.col("ss_sales_price").is_not_null()
                    )
                    .then(
                        (pl.col("ss_quantity") - pl.col("sr_return_quantity"))
                        * pl.col("ss_sales_price")
                    )
                    .otherwise(None)
                )
                .otherwise(
                    pl.when(
                        pl.col("ss_quantity").is_not_null()
                        & pl.col("ss_sales_price").is_not_null()
                    )
                    .then(pl.col("ss_quantity") * pl.col("ss_sales_price"))
                    .otherwise(None)
                )
                .alias("act_sales")
            ]
        )
        .group_by("ss_customer_sk")
        .agg(
            [
                pl.col("act_sales").count().alias("sumsales_count"),
                pl.col("act_sales").sum().alias("sumsales_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("sumsales_count") == 0)
                .then(None)
                .otherwise(pl.col("sumsales_sum"))
                .alias("sumsales")
            ]
        )
        .select(["ss_customer_sk", "sumsales"])
        .sort(["sumsales", "ss_customer_sk"], nulls_last=True)
        .limit(100)
    )
