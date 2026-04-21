# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q4 — naive one-for-one Polars translation of the SQL."""

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
        query_id=4,
        qualification=run_config.qualification,
    )
    year = params["year"]

    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    _group_cols = [
        "c_customer_id", "c_first_name", "c_last_name", "c_preferred_cust_flag",
        "c_birth_country", "c_login", "c_email_address", "d_year",
    ]

    # CTE year_total — store sales leg ('s')
    # FROM customer, store_sales, date_dim WHERE c_customer_sk=ss_customer_sk AND ss_sold_date_sk=d_date_sk
    yt_store = (
        customer.join(store_sales, left_on="c_customer_sk", right_on="ss_customer_sk", how="inner")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .group_by(_group_cols)
        .agg(
            sql_sum(
                (
                    (pl.col("ss_ext_list_price") - pl.col("ss_ext_wholesale_cost") - pl.col("ss_ext_discount_amt"))
                    + pl.col("ss_ext_sales_price")
                ) / 2
            ).alias("year_total"),
            pl.lit("s").alias("sale_type"),
        )
        .rename({
            "c_customer_id": "customer_id",
            "c_first_name": "customer_first_name",
            "c_last_name": "customer_last_name",
            "c_preferred_cust_flag": "customer_preferred_cust_flag",
            "c_birth_country": "customer_birth_country",
            "c_login": "customer_login",
            "c_email_address": "customer_email_address",
            "d_year": "dyear",
        })
    )

    # CTE year_total — catalog sales leg ('c')
    # FROM customer, catalog_sales, date_dim WHERE c_customer_sk=cs_bill_customer_sk AND cs_sold_date_sk=d_date_sk
    yt_catalog = (
        customer.join(catalog_sales, left_on="c_customer_sk", right_on="cs_bill_customer_sk", how="inner")
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk", how="inner")
        .group_by(_group_cols)
        .agg(
            sql_sum(
                (
                    (pl.col("cs_ext_list_price") - pl.col("cs_ext_wholesale_cost") - pl.col("cs_ext_discount_amt"))
                    + pl.col("cs_ext_sales_price")
                ) / 2
            ).alias("year_total"),
            pl.lit("c").alias("sale_type"),
        )
        .rename({
            "c_customer_id": "customer_id",
            "c_first_name": "customer_first_name",
            "c_last_name": "customer_last_name",
            "c_preferred_cust_flag": "customer_preferred_cust_flag",
            "c_birth_country": "customer_birth_country",
            "c_login": "customer_login",
            "c_email_address": "customer_email_address",
            "d_year": "dyear",
        })
    )

    # CTE year_total — web sales leg ('w')
    # FROM customer, web_sales, date_dim WHERE c_customer_sk=ws_bill_customer_sk AND ws_sold_date_sk=d_date_sk
    yt_web = (
        customer.join(web_sales, left_on="c_customer_sk", right_on="ws_bill_customer_sk", how="inner")
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .group_by(_group_cols)
        .agg(
            sql_sum(
                (
                    (pl.col("ws_ext_list_price") - pl.col("ws_ext_wholesale_cost") - pl.col("ws_ext_discount_amt"))
                    + pl.col("ws_ext_sales_price")
                ) / 2
            ).alias("year_total"),
            pl.lit("w").alias("sale_type"),
        )
        .rename({
            "c_customer_id": "customer_id",
            "c_first_name": "customer_first_name",
            "c_last_name": "customer_last_name",
            "c_preferred_cust_flag": "customer_preferred_cust_flag",
            "c_birth_country": "customer_birth_country",
            "c_login": "customer_login",
            "c_email_address": "customer_email_address",
            "d_year": "dyear",
        })
    )

    # CTE is the UNION ALL of the three legs
    year_total = pl.concat([yt_store, yt_catalog, yt_web], how="diagonal_relaxed")

    # Main query:
    # FROM year_total t_s_firstyear, year_total t_s_secyear,
    #      year_total t_c_firstyear, year_total t_c_secyear,
    #      year_total t_w_firstyear, year_total t_w_secyear
    # WHERE ... (all filter conditions applied after all joins per naive rule 2)

    # Filter each alias before joining to reduce cardinality (but no WHERE push-down — only sale_type/dyear
    # predicates are safe to evaluate without any other table's data, so we apply them here as part
    # of producing the named alias, which in SQL is defined purely by the CTE + WHERE equality)
    t_s_firstyear = (
        year_total
        .filter((pl.col("sale_type") == "s") & (pl.col("dyear") == year) & (pl.col("year_total") > 0))
        .rename({"year_total": "s_fy_total", "customer_id": "cid"})
        .select(["cid", "s_fy_total"])
    )
    t_s_secyear = (
        year_total
        .filter((pl.col("sale_type") == "s") & (pl.col("dyear") == year + 1))
        .rename({"year_total": "s_sy_total"})
    )
    t_c_firstyear = (
        year_total
        .filter((pl.col("sale_type") == "c") & (pl.col("dyear") == year) & (pl.col("year_total") > 0))
        .rename({"year_total": "c_fy_total", "customer_id": "cid"})
        .select(["cid", "c_fy_total"])
    )
    t_c_secyear = (
        year_total
        .filter((pl.col("sale_type") == "c") & (pl.col("dyear") == year + 1))
        .rename({"year_total": "c_sy_total", "customer_id": "cid"})
        .select(["cid", "c_sy_total"])
    )
    t_w_firstyear = (
        year_total
        .filter((pl.col("sale_type") == "w") & (pl.col("dyear") == year) & (pl.col("year_total") > 0))
        .rename({"year_total": "w_fy_total", "customer_id": "cid"})
        .select(["cid", "w_fy_total"])
    )
    t_w_secyear = (
        year_total
        .filter((pl.col("sale_type") == "w") & (pl.col("dyear") == year + 1))
        .rename({"year_total": "w_sy_total", "customer_id": "cid"})
        .select(["cid", "w_sy_total"])
    )

    result = (
        t_s_secyear.join(t_s_firstyear, left_on="customer_id", right_on="cid", how="inner")
        .join(t_c_firstyear, left_on="customer_id", right_on="cid", how="inner")
        .join(t_c_secyear, left_on="customer_id", right_on="cid", how="inner")
        .join(t_w_firstyear, left_on="customer_id", right_on="cid", how="inner")
        .join(t_w_secyear, left_on="customer_id", right_on="cid", how="inner")
        .filter(
            (
                pl.when(pl.col("c_fy_total") > 0)
                .then(pl.col("c_sy_total") / pl.col("c_fy_total"))
                .otherwise(None)
                > pl.when(pl.col("s_fy_total") > 0)
                .then(pl.col("s_sy_total") / pl.col("s_fy_total"))
                .otherwise(None)
            )
            & (
                pl.when(pl.col("c_fy_total") > 0)
                .then(pl.col("c_sy_total") / pl.col("c_fy_total"))
                .otherwise(None)
                > pl.when(pl.col("w_fy_total") > 0)
                .then(pl.col("w_sy_total") / pl.col("w_fy_total"))
                .otherwise(None)
            )
        )
        .select([
            pl.col("customer_id"),
            pl.col("customer_first_name"),
            pl.col("customer_last_name"),
            pl.col("customer_preferred_cust_flag"),
        ])
    )

    result = result.sort(['customer_id', 'customer_first_name', 'customer_last_name', 'customer_preferred_cust_flag'], descending=[False, False, False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[
            ("customer_id", False),
            ("customer_first_name", False),
            ("customer_last_name", False),
            ("customer_preferred_cust_flag", False),
        ],
        limit=100,
    )
