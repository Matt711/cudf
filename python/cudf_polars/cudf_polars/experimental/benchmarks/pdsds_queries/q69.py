# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 69."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 69."""
    return """
    SELECT cd_gender,
                   cd_marital_status,
                   cd_education_status,
                   Count(*) cnt1,
                   cd_purchase_estimate,
                   Count(*) cnt2,
                   cd_credit_rating,
                   Count(*) cnt3
    FROM   customer c,
           customer_address ca,
           customer_demographics
    WHERE  c.c_current_addr_sk = ca.ca_address_sk
           AND ca_state IN ( 'KS', 'AZ', 'NE' )
           AND cd_demo_sk = c.c_current_cdemo_sk
           AND EXISTS (SELECT *
                       FROM   store_sales,
                              date_dim
                       WHERE  c.c_customer_sk = ss_customer_sk
                              AND ss_sold_date_sk = d_date_sk
                              AND d_year = 2004
                              AND d_moy BETWEEN 3 AND 3 + 2)
           AND ( NOT EXISTS (SELECT *
                             FROM   web_sales,
                                    date_dim
                             WHERE  c.c_customer_sk = ws_bill_customer_sk
                                    AND ws_sold_date_sk = d_date_sk
                                    AND d_year = 2004
                                    AND d_moy BETWEEN 3 AND 3 + 2)
                 AND NOT EXISTS (SELECT *
                                 FROM   catalog_sales,
                                        date_dim
                                 WHERE  c.c_customer_sk = cs_ship_customer_sk
                                        AND cs_sold_date_sk = d_date_sk
                                        AND d_year = 2004
                                        AND d_moy BETWEEN 3 AND 3 + 2) )
    GROUP  BY cd_gender,
              cd_marital_status,
              cd_education_status,
              cd_purchase_estimate,
              cd_credit_rating
    ORDER  BY cd_gender,
              cd_marital_status,
              cd_education_status,
              cd_purchase_estimate,
              cd_credit_rating
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 69."""
    # Load required tables using scan_parquet for lazy evaluation
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    customer_demographics = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    # Filter date_dim for the target period (2004, months 3-5)
    target_dates = date_dim.filter(
        (pl.col("d_year") == 2004) & (pl.col("d_moy").is_between(3, 5))
    ).select("d_date_sk")
    # Create EXISTS condition: customers who made store purchases in target period
    store_customers = (
        store_sales.join(target_dates, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .select("ss_customer_sk")
        .unique()
    )
    # Create NOT EXISTS conditions: customers who made web/catalog purchases in target period
    web_customers = (
        web_sales.join(target_dates, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .select(pl.col("ws_bill_customer_sk").alias("customer_sk"))
        .unique()
    )
    catalog_customers = (
        catalog_sales.join(
            target_dates, left_on="cs_sold_date_sk", right_on="d_date_sk"
        )
        .select(pl.col("cs_ship_customer_sk").alias("customer_sk"))
        .unique()
    )
    # Combine web and catalog customers to exclude
    exclude_customers = pl.concat([web_customers, catalog_customers]).unique()
    # Main query: start with customer demographics and address filtering
    return (
        customer
        # Join with customer_address and filter by state
        .join(customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk")
        .filter(pl.col("ca_state").is_in(["KS", "AZ", "NE"]))
        # Join with customer_demographics
        .join(
            customer_demographics, left_on="c_current_cdemo_sk", right_on="cd_demo_sk"
        )
        # Apply EXISTS condition: must have store purchases
        .join(store_customers, left_on="c_customer_sk", right_on="ss_customer_sk")
        # Apply NOT EXISTS condition: exclude web/catalog customers
        .join(
            exclude_customers,
            left_on="c_customer_sk",
            right_on="customer_sk",
            how="anti",
        )
        # Group by demographics and count
        .group_by(
            [
                "cd_gender",
                "cd_marital_status",
                "cd_education_status",
                "cd_purchase_estimate",
                "cd_credit_rating",
            ]
        )
        .agg(
            [
                # Cast -> Int64 to match DuckDB
                pl.len().cast(pl.Int64).alias("cnt1"),
                pl.len().cast(pl.Int64).alias("cnt2"),
                pl.len().cast(pl.Int64).alias("cnt3"),
            ]
        )
        # Select and order columns to match SQL output
        .select(
            [
                "cd_gender",
                "cd_marital_status",
                "cd_education_status",
                "cnt1",
                "cd_purchase_estimate",
                "cnt2",
                "cd_credit_rating",
                "cnt3",
            ]
        )
        .sort(
            [
                "cd_gender",
                "cd_marital_status",
                "cd_education_status",
                "cd_purchase_estimate",
                "cd_credit_rating",
            ],
            nulls_last=True,
        )
        .limit(100)
    )
