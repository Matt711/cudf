# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q69 — naive one-for-one Polars translation of the SQL."""
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
        int(run_config.scale_factor), query_id=69, qualification=run_config.qualification
    )
    year = params["year"]
    month = params["month"]
    states = params["states"]

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

    # EXISTS subquery: store_sales joined with date_dim
    # WHERE c.c_customer_sk = ss_customer_sk AND ss_sold_date_sk = d_date_sk
    #   AND d_year = year AND d_moy BETWEEN month AND month+2
    store_customer_keys = (
        store_sales
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(
            (pl.col("d_year") == year)
            & pl.col("d_moy").is_between(month, month + 2)
        )
        .select(pl.col("ss_customer_sk").alias("customer_sk"))
        .unique()
    )

    # NOT EXISTS: web_sales joined date_dim
    web_customer_keys = (
        web_sales
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(
            (pl.col("d_year") == year)
            & pl.col("d_moy").is_between(month, month + 2)
        )
        .select(pl.col("ws_bill_customer_sk").alias("customer_sk"))
        .unique()
    )

    # NOT EXISTS: catalog_sales joined date_dim
    catalog_customer_keys = (
        catalog_sales
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(
            (pl.col("d_year") == year)
            & pl.col("d_moy").is_between(month, month + 2)
        )
        .select(pl.col("cs_ship_customer_sk").alias("customer_sk"))
        .unique()
    )

    # Combine NOT EXISTS exclusion sets
    exclude_keys = pl.concat([web_customer_keys, catalog_customer_keys]).unique()

    # Main query:
    # FROM customer c, customer_address ca, customer_demographics
    # WHERE c.c_current_addr_sk = ca.ca_address_sk
    #   AND ca_state IN (states)
    #   AND cd_demo_sk = c.c_current_cdemo_sk
    #   AND EXISTS (store subquery)
    #   AND NOT EXISTS (web subquery) AND NOT EXISTS (catalog subquery)
    # No .select() until final SELECT (naive rule 3)
    result = (
        customer
        .join(
            customer_address,
            left_on="c_current_addr_sk",
            right_on="ca_address_sk",
            how="inner",
        )
        .join(
            customer_demographics,
            left_on="c_current_cdemo_sk",
            right_on="cd_demo_sk",
            how="inner",
        )
        .filter(pl.col("ca_state").is_in(states))
        # EXISTS → semi join
        .join(store_customer_keys, left_on="c_customer_sk", right_on="customer_sk", how="semi")
        # NOT EXISTS → anti join
        .join(exclude_keys, left_on="c_customer_sk", right_on="customer_sk", how="anti")
        .group_by([
            "cd_gender",
            "cd_marital_status",
            "cd_education_status",
            "cd_purchase_estimate",
            "cd_credit_rating",
        ])
        .agg([
            pl.len().alias("cnt1"),
            pl.len().alias("cnt2"),
            pl.len().alias("cnt3"),
        ])
        .select([
            "cd_gender",
            "cd_marital_status",
            "cd_education_status",
            "cnt1",
            "cd_purchase_estimate",
            "cnt2",
            "cd_credit_rating",
            "cnt3",
        ])
    )

    result = result.sort(['cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_purchase_estimate', 'cd_credit_rating'], descending=[False, False, False, False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[
            ("cd_gender", False),
            ("cd_marital_status", False),
            ("cd_education_status", False),
            ("cd_purchase_estimate", False),
            ("cd_credit_rating", False),
        ],
        limit=100,
    )
