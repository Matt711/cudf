# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q10 — naive one-for-one Polars translation of the SQL."""

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
        query_id=10,
        qualification=run_config.qualification,
    )
    counties = params["county"]

    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(run_config.dataset_path, "customer_address", run_config.suffix)
    customer_demographics = get_data(run_config.dataset_path, "customer_demographics", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # EXISTS (SELECT * FROM store_sales, date_dim WHERE c.c_customer_sk=ss_customer_sk
    #          AND ss_sold_date_sk=d_date_sk AND d_year=2002 AND d_moy BETWEEN 4 AND 4+3)
    # => semi-join: unique customer_sks satisfying store_sales EXISTS condition
    store_date_customers = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(
            (pl.col("d_year") == 2002)
            & (pl.col("d_moy").is_between(4, 7, closed="both"))
        )
        .select(pl.col("ss_customer_sk").alias("customer_sk"))
        .unique()
    )

    # EXISTS (SELECT * FROM web_sales, date_dim WHERE c.c_customer_sk=ws_bill_customer_sk ...)
    web_date_customers = (
        web_sales.join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(
            (pl.col("d_year") == 2002)
            & (pl.col("d_moy").is_between(4, 7, closed="both"))
        )
        .select(pl.col("ws_bill_customer_sk").alias("customer_sk"))
        .unique()
    )

    # EXISTS (SELECT * FROM catalog_sales, date_dim WHERE c.c_customer_sk=cs_ship_customer_sk ...)
    catalog_date_customers = (
        catalog_sales.join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(
            (pl.col("d_year") == 2002)
            & (pl.col("d_moy").is_between(4, 7, closed="both"))
        )
        .select(pl.col("cs_ship_customer_sk").alias("customer_sk"))
        .unique()
    )

    # OR EXISTS web OR EXISTS catalog => UNION ALL + unique
    web_or_catalog = pl.concat([web_date_customers, catalog_date_customers], how="diagonal_relaxed").unique()

    # Main query:
    # FROM customer c, customer_address ca, customer_demographics
    # WHERE c.c_current_addr_sk=ca.ca_address_sk
    #   AND ca_county IN counties
    #   AND cd_demo_sk=c.c_current_cdemo_sk
    #   AND EXISTS (store_sales)
    #   AND (EXISTS (web_sales) OR EXISTS (catalog_sales))
    # Apply all WHERE after all joins (naive)
    result = (
        customer.join(customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk", how="inner")
        .join(customer_demographics, left_on="c_current_cdemo_sk", right_on="cd_demo_sk", how="inner")
        .filter(pl.col("ca_county").is_in(counties))
        .join(store_date_customers, left_on="c_customer_sk", right_on="customer_sk", how="semi")
        .join(web_or_catalog, left_on="c_customer_sk", right_on="customer_sk", how="semi")
        .group_by([
            "cd_gender", "cd_marital_status", "cd_education_status",
            "cd_purchase_estimate", "cd_credit_rating",
            "cd_dep_count", "cd_dep_employed_count", "cd_dep_college_count",
        ])
        .agg(
            pl.len().alias("cnt1"),
            pl.len().alias("cnt2"),
            pl.len().alias("cnt3"),
            pl.len().alias("cnt4"),
            pl.len().alias("cnt5"),
            pl.len().alias("cnt6"),
        )
        .select([
            "cd_gender", "cd_marital_status", "cd_education_status",
            "cnt1", "cd_purchase_estimate", "cnt2",
            "cd_credit_rating", "cnt3", "cd_dep_count", "cnt4",
            "cd_dep_employed_count", "cnt5", "cd_dep_college_count", "cnt6",
        ])
    )

    result = result.sort(['cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_purchase_estimate', 'cd_credit_rating', 'cd_dep_count', 'cd_dep_employed_count', 'cd_dep_college_count'], descending=[False, False, False, False, False, False, False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[
            ("cd_gender", False),
            ("cd_marital_status", False),
            ("cd_education_status", False),
            ("cd_purchase_estimate", False),
            ("cd_credit_rating", False),
            ("cd_dep_count", False),
            ("cd_dep_employed_count", False),
            ("cd_dep_college_count", False),
        ],
        limit=100,
    )
