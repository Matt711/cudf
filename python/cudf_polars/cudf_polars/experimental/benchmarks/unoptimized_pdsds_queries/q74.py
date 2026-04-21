# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q74 — naive one-for-one Polars translation of the SQL."""
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
        int(run_config.scale_factor), query_id=74, qualification=run_config.qualification
    )
    year = params["year"]

    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # CTE year_total:
    # UNION ALL of store_sales and web_sales,
    # each joined with customer and date_dim (comma = inner join)
    # WHERE c_customer_sk = ss/ws_customer_sk
    #   AND ss/ws_sold_date_sk = d_date_sk
    #   AND d_year IN (year, year+1)
    # GROUP BY c_customer_id, c_first_name, c_last_name, d_year

    # Store sales half
    ss_half = (
        customer
        .join(store_sales, left_on="c_customer_sk", right_on="ss_customer_sk", how="inner")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(pl.col("d_year").is_in([year, year + 1]))
        .group_by(["c_customer_id", "c_first_name", "c_last_name", "d_year"])
        .agg(pl.col("ss_net_paid").std(ddof=1).alias("year_total"))
        .with_columns(pl.lit("s").alias("sale_type"))
        .rename({"d_year": "year1"})
    )

    # Web sales half
    ws_half = (
        customer
        .join(web_sales, left_on="c_customer_sk", right_on="ws_bill_customer_sk", how="inner")
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(pl.col("d_year").is_in([year, year + 1]))
        .group_by(["c_customer_id", "c_first_name", "c_last_name", "d_year"])
        .agg(pl.col("ws_net_paid").std(ddof=1).alias("year_total"))
        .with_columns(pl.lit("w").alias("sale_type"))
        .rename({"d_year": "year1"})
    )

    year_total = pl.concat([ss_half, ws_half], how="diagonal_relaxed")

    # The outer query uses four references to year_total:
    # t_s_firstyear, t_s_secyear, t_w_firstyear, t_w_secyear
    t_s_firstyear = (
        year_total
        .filter(
            (pl.col("sale_type") == "s")
            & (pl.col("year1") == year)
            & (pl.col("year_total") > 0)
        )
        .select([
            pl.col("c_customer_id").alias("s_first_customer_id"),
            pl.col("c_first_name").alias("s_first_first_name"),
            pl.col("c_last_name").alias("s_first_last_name"),
            pl.col("year_total").alias("s_first_year_total"),
        ])
    )

    t_s_secyear = (
        year_total
        .filter(
            (pl.col("sale_type") == "s")
            & (pl.col("year1") == year + 1)
        )
        .select([
            pl.col("c_customer_id").alias("s_sec_customer_id"),
            pl.col("c_first_name").alias("s_sec_first_name"),
            pl.col("c_last_name").alias("s_sec_last_name"),
            pl.col("year_total").alias("s_sec_year_total"),
        ])
    )

    t_w_firstyear = (
        year_total
        .filter(
            (pl.col("sale_type") == "w")
            & (pl.col("year1") == year)
            & (pl.col("year_total") > 0)
        )
        .select([
            pl.col("c_customer_id").alias("w_first_customer_id"),
            pl.col("year_total").alias("w_first_year_total"),
        ])
    )

    t_w_secyear = (
        year_total
        .filter(
            (pl.col("sale_type") == "w")
            & (pl.col("year1") == year + 1)
        )
        .select([
            pl.col("c_customer_id").alias("w_sec_customer_id"),
            pl.col("year_total").alias("w_sec_year_total"),
        ])
    )

    # FROM year_total t_s_firstyear, year_total t_s_secyear,
    #      year_total t_w_firstyear, year_total t_w_secyear
    # WHERE t_s_secyear.customer_id = t_s_firstyear.customer_id
    #   AND t_s_firstyear.customer_id = t_w_secyear.customer_id
    #   AND t_s_firstyear.customer_id = t_w_firstyear.customer_id
    #   AND ... (filter conditions already applied above via filtering)
    #   AND CASE WHEN t_w_firstyear.year_total > 0 THEN ... > CASE WHEN ...
    result = (
        t_s_firstyear
        .join(t_s_secyear, left_on="s_first_customer_id", right_on="s_sec_customer_id", how="inner")
        .join(t_w_firstyear, left_on="s_first_customer_id", right_on="w_first_customer_id", how="inner")
        .join(t_w_secyear, left_on="s_first_customer_id", right_on="w_sec_customer_id", how="inner")
        .filter(
            pl.col("w_sec_year_total") / pl.col("w_first_year_total")
            > pl.col("s_sec_year_total") / pl.col("s_first_year_total")
        )
        .select([
            pl.col("s_first_customer_id").alias("customer_id"),
            pl.col("s_sec_first_name").alias("customer_first_name"),
            pl.col("s_sec_last_name").alias("customer_last_name"),
        ])
    )

    result = result.sort(['customer_id', 'customer_first_name', 'customer_last_name'], descending=[False, False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[
            ("customer_id", False),
            ("customer_first_name", False),
            ("customer_last_name", False),
        ],
        limit=100,
    )
