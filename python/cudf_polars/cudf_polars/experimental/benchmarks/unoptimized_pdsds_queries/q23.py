# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q23 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q23 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=23, qualification=run_config.qualification
    )

    year = params["year"]
    month = params["month"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)

    # CTE: frequent_ss_items
    # FROM: store_sales, date_dim, (SELECT SUBSTRING(i_item_desc,1,30) itemdesc, * FROM item) sq1
    # WHERE ss_sold_date_sk=d_date_sk AND ss_item_sk=i_item_sk AND d_year IN (...)
    # GROUP BY itemdesc, i_item_sk, d_date HAVING count(*) > 4
    frequent_ss_items = (
        store_sales
        .join(date_dim, how="inner",
              left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(
            item.with_columns(
                pl.col("i_item_desc").str.slice(0, 30).alias("itemdesc")
            ),
            how="inner",
            left_on="ss_item_sk", right_on="i_item_sk",
        )
        .filter(
            pl.col("d_year").is_in([year, year + 1, year + 2, year + 3])
        )
        .group_by(["itemdesc", "ss_item_sk", "d_date"])
        .agg([pl.len().alias("cnt")])
        .filter(pl.col("cnt") > 4)
        .select([pl.col("ss_item_sk").alias("item_sk")])
    )

    # CTE: max_store_sales
    # Inner: SELECT c_customer_sk, sum(ss_quantity*ss_sales_price) csales
    #        FROM store_sales, customer, date_dim
    #        WHERE ... GROUP BY c_customer_sk
    # Outer: SELECT max(csales) tpcds_cmax
    per_customer_sales = (
        store_sales
        .join(customer, how="inner",
              left_on="ss_customer_sk", right_on="c_customer_sk")
        .join(date_dim, how="inner",
              left_on="ss_sold_date_sk", right_on="d_date_sk")
        .filter(
            pl.col("d_year").is_in([year, year + 1, year + 2, year + 3])
        )
        .group_by("ss_customer_sk")
        .agg([
            sql_sum(pl.col("ss_quantity") * pl.col("ss_sales_price")).alias("csales")
        ])
    )

    max_store_sales = per_customer_sales.select(
        pl.col("csales").max().alias("tpcds_cmax")
    )

    # CTE: best_ss_customer
    # FROM store_sales, customer, max_store_sales
    # WHERE ss_customer_sk = c_customer_sk
    # GROUP BY c_customer_sk
    # HAVING sum(ss_quantity*ss_sales_price) > (95/100.0) * max(tpcds_cmax)
    best_ss_customer = (
        store_sales
        .join(customer, how="inner",
              left_on="ss_customer_sk", right_on="c_customer_sk")
        .join(max_store_sales, how="cross")
        .group_by("ss_customer_sk")
        .agg([
            sql_sum(pl.col("ss_quantity") * pl.col("ss_sales_price")).alias("ssales"),
            pl.col("tpcds_cmax").first().alias("tpcds_cmax"),
        ])
        .filter(pl.col("ssales") > (95 / 100.0) * pl.col("tpcds_cmax"))
        .select("ss_customer_sk")
    )

    # date_dim filtered for target year/month
    date_target = date_dim.filter(
        (pl.col("d_year") == year) & (pl.col("d_moy") == month)
    )

    # First UNION ALL branch: catalog_sales, customer, date_dim, frequent_ss_items, best_ss_customer
    catalog_part = (
        catalog_sales
        .join(customer, how="inner",
              left_on="cs_bill_customer_sk", right_on="c_customer_sk")
        .join(date_target, how="inner",
              left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(frequent_ss_items, how="inner",
              left_on="cs_item_sk", right_on="item_sk")
        .join(best_ss_customer, how="inner",
              left_on="cs_bill_customer_sk", right_on="ss_customer_sk")
        .group_by(["c_last_name", "c_first_name"])
        .agg([
            sql_sum(pl.col("cs_quantity") * pl.col("cs_list_price")).alias("sales")
        ])
    )

    # Second UNION ALL branch: web_sales, customer, date_dim, frequent_ss_items, best_ss_customer
    web_part = (
        web_sales
        .join(customer, how="inner",
              left_on="ws_bill_customer_sk", right_on="c_customer_sk")
        .join(date_target, how="inner",
              left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(frequent_ss_items, how="inner",
              left_on="ws_item_sk", right_on="item_sk")
        .join(best_ss_customer, how="inner",
              left_on="ws_bill_customer_sk", right_on="ss_customer_sk")
        .group_by(["c_last_name", "c_first_name"])
        .agg([
            sql_sum(pl.col("ws_quantity") * pl.col("ws_list_price")).alias("sales")
        ])
    )

    result = pl.concat([catalog_part, web_part], how="diagonal_relaxed")

    sort_by = [("c_last_name", False), ("c_first_name", False), ("sales", False)]
    limit = 100

    result = result.sort(["c_last_name", "c_first_name", "sales"], descending=[False, False, False]).limit(limit)
    return QueryResult(frame=result, sort_by=sort_by, limit=limit, nulls_last=False)
