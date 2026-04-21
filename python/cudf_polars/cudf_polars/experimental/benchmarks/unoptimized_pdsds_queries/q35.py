# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q35 — naive one-for-one Polars translation of the SQL."""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data
from cudf_polars.experimental.benchmarks.unoptimized_pdsds_queries.sql_helpers import sql_sum

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:  # type: ignore[return]
    raise NotImplementedError


def _agg_expr(col_name: str, agg_func: str, alias: str) -> pl.Expr:
    col = pl.col(col_name)
    if agg_func == "sum":
        return sql_sum(col).alias(alias)
    elif agg_func == "min":
        return col.min().alias(alias)
    elif agg_func == "max":
        return col.max().alias(alias)
    elif agg_func == "avg":
        return col.mean().alias(alias)
    elif agg_func == "stddev_samp":
        return col.std(ddof=1).alias(alias)
    else:
        msg = f"Unknown aggregation function: {agg_func}"
        raise ValueError(msg)


def polars_impl(run_config: RunConfig) -> QueryResult:
    params = load_parameters(
        int(run_config.scale_factor), query_id=35, qualification=run_config.qualification
    )
    year = params["year"]
    aggone = params["aggone"]
    aggtwo = params["aggtwo"]
    aggthree = params["aggthree"]

    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    customer_demographics = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)

    # EXISTS subquery: store_sales JOIN date_dim WHERE c.c_customer_sk = ss_customer_sk
    #   AND d_year = year AND d_qoy < 4
    store_exists = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .filter((pl.col("d_year") == year) & (pl.col("d_qoy") < 4))
        .select("ss_customer_sk")
        .unique()
    )

    # EXISTS subquery: web_sales JOIN date_dim WHERE c.c_customer_sk = ws_bill_customer_sk
    web_exists = (
        web_sales.join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .filter((pl.col("d_year") == year) & (pl.col("d_qoy") < 4))
        .select(pl.col("ws_bill_customer_sk").alias("customer_sk"))
        .unique()
    )

    # EXISTS subquery: catalog_sales JOIN date_dim WHERE c.c_customer_sk = cs_ship_customer_sk
    catalog_exists = (
        catalog_sales.join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .filter((pl.col("d_year") == year) & (pl.col("d_qoy") < 4))
        .select(pl.col("cs_ship_customer_sk").alias("customer_sk"))
        .unique()
    )

    # OR EXISTS (web) OR EXISTS (catalog) => union then deduplicate
    web_or_catalog = pl.concat([web_exists, catalog_exists]).unique()

    group_cols = [
        "ca_state",
        "cd_gender",
        "cd_marital_status",
        "cd_dep_count",
        "cd_dep_employed_count",
        "cd_dep_college_count",
    ]

    # FROM customer c, customer_address ca, customer_demographics
    # WHERE c.c_current_addr_sk = ca.ca_address_sk
    #   AND cd_demo_sk = c.c_current_cdemo_sk
    #   AND EXISTS (store subquery)
    #   AND (EXISTS (web subquery) OR EXISTS (catalog subquery))
    result = (
        customer.join(
            customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk"
        )
        .join(
            customer_demographics, left_on="c_current_cdemo_sk", right_on="cd_demo_sk"
        )
        .join(store_exists, left_on="c_customer_sk", right_on="ss_customer_sk", how="semi")
        .join(web_or_catalog, left_on="c_customer_sk", right_on="customer_sk", how="semi")
        .group_by(group_cols)
        .agg(
            [
                pl.len().alias("cnt1"),
                _agg_expr("cd_dep_count", aggone, f"{aggone}(cd_dep_count)"),
                _agg_expr("cd_dep_count", aggtwo, f"{aggtwo}(cd_dep_count)"),
                _agg_expr("cd_dep_count", aggthree, f"{aggthree}(cd_dep_count)_1"),
                pl.len().alias("cnt2"),
                _agg_expr(
                    "cd_dep_employed_count", aggone, f"{aggone}(cd_dep_employed_count)"
                ),
                _agg_expr(
                    "cd_dep_employed_count", aggtwo, f"{aggtwo}(cd_dep_employed_count)"
                ),
                _agg_expr(
                    "cd_dep_employed_count",
                    aggthree,
                    f"{aggthree}(cd_dep_employed_count)_1",
                ),
                pl.len().alias("cnt3"),
                _agg_expr(
                    "cd_dep_college_count", aggone, f"{aggone}(cd_dep_college_count)"
                ),
                _agg_expr(
                    "cd_dep_college_count", aggtwo, f"{aggtwo}(cd_dep_college_count)"
                ),
                _agg_expr(
                    "cd_dep_college_count",
                    aggthree,
                    f"{aggthree}(cd_dep_college_count)_1",
                ),
            ]
        )
        .select(
            [
                "ca_state",
                "cd_gender",
                "cd_marital_status",
                "cd_dep_count",
                "cnt1",
                f"{aggone}(cd_dep_count)",
                f"{aggtwo}(cd_dep_count)",
                f"{aggthree}(cd_dep_count)_1",
                "cd_dep_employed_count",
                "cnt2",
                f"{aggone}(cd_dep_employed_count)",
                f"{aggtwo}(cd_dep_employed_count)",
                f"{aggthree}(cd_dep_employed_count)_1",
                "cd_dep_college_count",
                "cnt3",
                f"{aggone}(cd_dep_college_count)",
                f"{aggtwo}(cd_dep_college_count)",
                f"{aggthree}(cd_dep_college_count)_1",
            ]
        )
    )

    result = result.sort(['ca_state', 'cd_gender', 'cd_marital_status', 'cd_dep_count', 'cd_dep_employed_count', 'cd_dep_college_count'], descending=[False, False, False, False, False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[
            ("ca_state", False),
            ("cd_gender", False),
            ("cd_marital_status", False),
            ("cd_dep_count", False),
            ("cd_dep_employed_count", False),
            ("cd_dep_college_count", False),
        ],
        limit=100,
    )
