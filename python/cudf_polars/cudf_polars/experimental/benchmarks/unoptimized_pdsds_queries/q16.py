# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q16 — naive one-for-one Polars translation of the SQL."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data
from cudf_polars.experimental.benchmarks.unoptimized_pdsds_queries.sql_helpers import (
    sql_n_unique,
    sql_sum,
)

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:  # type: ignore[return]
    raise NotImplementedError


def polars_impl(run_config: RunConfig) -> QueryResult:
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=16,
        qualification=run_config.qualification,
    )
    year = params["year"]
    month = params["month"]
    state = params["state"]
    county = params["county"]

    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    catalog_returns = get_data(run_config.dataset_path, "catalog_returns", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer_address = get_data(run_config.dataset_path, "customer_address", run_config.suffix)
    call_center = get_data(run_config.dataset_path, "call_center", run_config.suffix)

    start_date_obj = date(year, month, 1)
    end_date_obj = start_date_obj + timedelta(days=60)

    # Main FROM: catalog_sales cs1, date_dim, customer_address, call_center
    # WHERE d_date BETWEEN start AND end
    #   AND cs1.cs_ship_date_sk=d_date_sk
    #   AND cs1.cs_ship_addr_sk=ca_address_sk
    #   AND ca_state=state
    #   AND cs1.cs_call_center_sk=cc_call_center_sk
    #   AND cc_county IN county
    # All WHERE after all joins (naive)
    cs1 = (
        catalog_sales.join(date_dim, left_on="cs_ship_date_sk", right_on="d_date_sk", how="inner")
        .join(customer_address, left_on="cs_ship_addr_sk", right_on="ca_address_sk", how="inner")
        .join(call_center, left_on="cs_call_center_sk", right_on="cc_call_center_sk", how="inner")
        .filter(
            pl.col("d_date").is_between(pl.lit(start_date_obj), pl.lit(end_date_obj), closed="both")
            & (pl.col("ca_state") == state)
            & pl.col("cc_county").is_in(county)
        )
    )

    # EXISTS (SELECT * FROM catalog_sales cs2
    #         WHERE cs1.cs_order_number=cs2.cs_order_number AND cs1.cs_warehouse_sk<>cs2.cs_warehouse_sk)
    mw_orders = (
        catalog_sales.select(["cs_order_number", "cs_warehouse_sk"])
        .join(
            catalog_sales.select(["cs_order_number", "cs_warehouse_sk"])
            .rename({"cs_warehouse_sk": "cs2_warehouse_sk"}),
            on="cs_order_number",
        )
        .filter(pl.col("cs_warehouse_sk") != pl.col("cs2_warehouse_sk"))
        .select("cs_order_number")
    )

    result = (
        cs1.join(mw_orders, on="cs_order_number", how="semi")
        # SQL: NULL <> x = UNKNOWN, so EXISTS never holds when cs_warehouse_sk is NULL
        .filter(pl.col("cs_warehouse_sk").is_not_null())
        # NOT EXISTS (SELECT * FROM catalog_returns cr1 WHERE cs1.cs_order_number=cr1.cr_order_number)
        .join(catalog_returns, left_on="cs_order_number", right_on="cr_order_number", how="anti")
        .select([
            sql_n_unique(pl.col("cs_order_number")).alias("order count"),
            sql_sum(pl.col("cs_ext_ship_cost")).alias("total shipping cost"),
            sql_sum(pl.col("cs_net_profit")).alias("total net profit"),
        ])
    )

    result = result.sort('order count', descending=False, nulls_last=True).limit(100)
    return QueryResult(frame=result, sort_by=[("order count", False)], limit=100)
