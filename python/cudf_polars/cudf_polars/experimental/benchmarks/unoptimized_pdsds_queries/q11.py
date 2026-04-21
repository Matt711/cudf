# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q11 — naive one-for-one Polars translation of the SQL."""

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
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=11,
        qualification=run_config.qualification,
    )
    year_first = params["year"]
    year_second = year_first + 1

    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    _cust_cols = [
        "c_customer_id", "c_first_name", "c_last_name", "c_preferred_cust_flag",
        "c_birth_country", "c_login", "c_email_address",
    ]
    _rename_map = {
        "c_customer_id": "customer_id",
        "c_first_name": "customer_first_name",
        "c_last_name": "customer_last_name",
        "c_preferred_cust_flag": "customer_preferred_cust_flag",
        "c_birth_country": "customer_birth_country",
        "c_login": "customer_login",
        "c_email_address": "customer_email_address",
        "d_year": "dyear",
    }

    # CTE year_total — store sales leg ('s')
    # FROM customer, store_sales, date_dim WHERE c_customer_sk=ss_customer_sk AND ss_sold_date_sk=d_date_sk
    yt_store = (
        customer.join(store_sales, left_on="c_customer_sk", right_on="ss_customer_sk", how="inner")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .group_by([*_cust_cols, "d_year"])
        .agg(
            sql_sum(pl.col("ss_ext_list_price") - pl.col("ss_ext_discount_amt")).alias("year_total"),
            pl.lit("s").alias("sale_type"),
        )
        .rename(_rename_map)
    )

    # CTE year_total — web sales leg ('w')
    # FROM customer, web_sales, date_dim WHERE c_customer_sk=ws_bill_customer_sk AND ws_sold_date_sk=d_date_sk
    yt_web = (
        customer.join(web_sales, left_on="c_customer_sk", right_on="ws_bill_customer_sk", how="inner")
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .group_by([*_cust_cols, "d_year"])
        .agg(
            sql_sum(pl.col("ws_ext_list_price") - pl.col("ws_ext_discount_amt")).alias("year_total"),
            pl.lit("w").alias("sale_type"),
        )
        .rename(_rename_map)
    )

    year_total = pl.concat([yt_store, yt_web], how="diagonal_relaxed")

    # Main query aliases (per SQL WHERE conditions):
    # t_s_firstyear: sale_type='s' AND dyear=year_first AND year_total>0
    # t_s_secyear:   sale_type='s' AND dyear=year_second
    # t_w_firstyear: sale_type='w' AND dyear=year_first AND year_total>0
    # t_w_secyear:   sale_type='w' AND dyear=year_second

    t_s_firstyear = (
        year_total
        .filter((pl.col("sale_type") == "s") & (pl.col("dyear") == year_first) & (pl.col("year_total") > 0))
        .select("customer_id", pl.col("year_total").alias("s_fy_total"))
    )
    t_s_secyear = (
        year_total
        .filter((pl.col("sale_type") == "s") & (pl.col("dyear") == year_second))
        .rename({"year_total": "s_sy_total"})
    )
    t_w_firstyear = (
        year_total
        .filter((pl.col("sale_type") == "w") & (pl.col("dyear") == year_first) & (pl.col("year_total") > 0))
        .select("customer_id", pl.col("year_total").alias("w_fy_total"))
    )
    t_w_secyear = (
        year_total
        .filter((pl.col("sale_type") == "w") & (pl.col("dyear") == year_second))
        .select("customer_id", pl.col("year_total").alias("w_sy_total"))
    )

    # Main SELECT: FROM t_s_firstyear, t_s_secyear, t_w_firstyear, t_w_secyear
    # WHERE conditions include ratio comparison; all applied after all joins
    result = (
        t_s_secyear.join(t_s_firstyear, on="customer_id", how="inner")
        .join(t_w_firstyear, on="customer_id", how="inner")
        .join(t_w_secyear, on="customer_id", how="inner")
        .filter(
            (
                pl.when(pl.col("w_fy_total") > 0)
                .then(pl.col("w_sy_total") / pl.col("w_fy_total"))
                .otherwise(pl.lit(0.0))
            )
            > (
                pl.when(pl.col("s_fy_total") > 0)
                .then(pl.col("s_sy_total") / pl.col("s_fy_total"))
                .otherwise(pl.lit(0.0))
            )
        )
        .select([
            pl.col("customer_id"),
            pl.col("customer_first_name"),
            pl.col("customer_last_name"),
            pl.col("customer_birth_country"),
        ])
    )

    result = result.sort(['customer_id', 'customer_first_name', 'customer_last_name', 'customer_birth_country'], descending=[False, False, False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[
            ("customer_id", False),
            ("customer_first_name", False),
            ("customer_last_name", False),
            ("customer_birth_country", False),
        ],
        limit=100,
    )
