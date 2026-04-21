# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q8 — naive one-for-one Polars translation of the SQL."""

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
        query_id=8,
        qualification=run_config.qualification,
    )
    year = params["year"]
    qoy = params["qoy"]
    zip_codes = params["zip_codes"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    customer_address = get_data(run_config.dataset_path, "customer_address", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)

    # Inline subquery V1: INTERSECT of two zip code sets
    # First set: Substr(ca_zip,1,5) IN zip_codes (FROM customer_address)
    zip_set_a = (
        customer_address.select(pl.col("ca_zip").str.slice(0, 5).alias("ca_zip"))
        .filter(pl.col("ca_zip").is_in(zip_codes))
        .unique()
    )

    # Second set (A1): FROM customer_address, customer WHERE ca_address_sk=c_current_addr_sk
    #   AND c_preferred_cust_flag='Y' GROUP BY ca_zip HAVING Count(*) > 10
    zip_set_b = (
        customer_address.join(customer, left_on="ca_address_sk", right_on="c_current_addr_sk", how="inner")
        .filter(pl.col("c_preferred_cust_flag") == "Y")
        .group_by(pl.col("ca_zip").str.slice(0, 5).alias("ca_zip"))
        .agg(pl.len().alias("cnt"))
        .filter(pl.col("cnt") > 10)
        .select("ca_zip")
    )

    # INTERSECT (A2): inner join on ca_zip, deduplicate
    v1 = zip_set_a.join(zip_set_b, on="ca_zip", how="inner").select("ca_zip").unique()

    # Main query:
    # FROM store_sales, date_dim, store, V1
    # WHERE ss_store_sk=s_store_sk AND ss_sold_date_sk=d_date_sk
    #   AND d_qoy=qoy AND d_year=year AND Substr(s_zip,1,2)=Substr(V1.ca_zip,1,2)
    # All WHERE after all joins (naive)
    result = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk", how="inner")
        .join(v1, left_on=pl.col("s_zip").str.slice(0, 2), right_on=pl.col("ca_zip").str.slice(0, 2), how="inner")
        .filter(
            (pl.col("d_qoy") == qoy)
            & (pl.col("d_year") == year)
        )
        .group_by("s_store_name")
        .agg(sql_sum(pl.col("ss_net_profit")).alias("sum(ss_net_profit)"))
    )

    result = result.sort('s_store_name', descending=False, nulls_last=True).limit(100)
    return QueryResult(frame=result, sort_by=[("s_store_name", False)], limit=100)
