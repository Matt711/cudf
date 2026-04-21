# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q45 — naive one-for-one Polars translation of the SQL."""
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


def polars_impl(run_config: RunConfig) -> QueryResult:
    params = load_parameters(
        int(run_config.scale_factor), query_id=45, qualification=run_config.qualification
    )
    year = params["year"]
    qoy = params["qoy"]
    zip_codes = params["zip_codes"]
    item_sks = params["item_sks"]

    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # Subquery: SELECT i_item_id FROM item WHERE i_item_sk IN (item_sks)
    subquery_item_ids = (
        item.filter(pl.col("i_item_sk").is_in(item_sks)).select("i_item_id").unique()
    )

    # FROM web_sales, customer, customer_address, date_dim, item
    # WHERE ws_bill_customer_sk = c_customer_sk
    #   AND c_current_addr_sk = ca_address_sk
    #   AND ws_item_sk = i_item_sk
    #   AND ws_sold_date_sk = d_date_sk
    #   AND d_qoy = qoy AND d_year = year
    # All WHERE conditions after all joins (naive rule 2)
    base = (
        web_sales.join(customer, left_on="ws_bill_customer_sk", right_on="c_customer_sk")
        .join(customer_address, left_on="c_current_addr_sk", right_on="ca_address_sk")
        .join(item, left_on="ws_item_sk", right_on="i_item_sk")
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .filter((pl.col("d_qoy") == qoy) & (pl.col("d_year") == year))
        .with_columns(pl.col("ca_zip").str.slice(0, 5).alias("zip_prefix"))
    )

    # OR condition:
    # Substr(ca_zip,1,5) IN (zip_codes)  OR  i_item_id IN (subquery)
    zip_match = base.filter(pl.col("zip_prefix").is_in(zip_codes))
    item_match = base.join(subquery_item_ids, on="i_item_id", how="semi")

    result = (
        pl.concat([zip_match, item_match], how="diagonal_relaxed")
        .group_by(["ca_zip", "ca_state"])
        .agg(sql_sum(pl.col("ws_sales_price")).alias("sum(ws_sales_price)"))
        .select(["ca_zip", "ca_state", "sum(ws_sales_price)"])
    )

    result = result.sort(['ca_zip', 'ca_state'], descending=[False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("ca_zip", False), ("ca_state", False)],
        limit=100,
    )
