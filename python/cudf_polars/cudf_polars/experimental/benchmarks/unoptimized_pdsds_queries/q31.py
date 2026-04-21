# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q31 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q31 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=31, qualification=run_config.qualification
    )

    year = params["year"]
    agg = params["agg"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )

    # CTE: ss
    # FROM store_sales, date_dim, customer_address
    # WHERE ss_sold_date_sk=d_date_sk AND ss_addr_sk=ca_address_sk
    # GROUP BY ca_county, d_qoy, d_year
    ss = (
        store_sales
        .join(date_dim, how="inner",
              left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(customer_address, how="inner",
              left_on="ss_addr_sk", right_on="ca_address_sk")
        .group_by(["ca_county", "d_qoy", "d_year"])
        .agg([
            sql_sum(pl.col("ss_ext_sales_price")).alias("store_sales"),
        ])
    )

    # CTE: ws
    # FROM web_sales, date_dim, customer_address
    # WHERE ws_sold_date_sk=d_date_sk AND ws_bill_addr_sk=ca_address_sk
    # GROUP BY ca_county, d_qoy, d_year
    ws = (
        web_sales
        .join(date_dim, how="inner",
              left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(customer_address, how="inner",
              left_on="ws_bill_addr_sk", right_on="ca_address_sk")
        .group_by(["ca_county", "d_qoy", "d_year"])
        .agg([
            sql_sum(pl.col("ws_ext_sales_price")).alias("web_sales"),
        ])
    )

    # Filter each alias to specific quarter/year
    ss1 = ss.filter((pl.col("d_qoy") == 1) & (pl.col("d_year") == year)).select(
        [pl.col("ca_county"), pl.col("store_sales").alias("ss1_store_sales")]
    )
    ss2 = ss.filter((pl.col("d_qoy") == 2) & (pl.col("d_year") == year)).select(
        [pl.col("ca_county"), pl.col("store_sales").alias("ss2_store_sales")]
    )
    ss3 = ss.filter((pl.col("d_qoy") == 3) & (pl.col("d_year") == year)).select(
        [pl.col("ca_county"), pl.col("store_sales").alias("ss3_store_sales")]
    )
    ws1 = ws.filter((pl.col("d_qoy") == 1) & (pl.col("d_year") == year)).select(
        [pl.col("ca_county"), pl.col("web_sales").alias("ws1_web_sales")]
    )
    ws2 = ws.filter((pl.col("d_qoy") == 2) & (pl.col("d_year") == year)).select(
        [pl.col("ca_county"), pl.col("web_sales").alias("ws2_web_sales")]
    )
    ws3 = ws.filter((pl.col("d_qoy") == 3) & (pl.col("d_year") == year)).select(
        [pl.col("ca_county"), pl.col("web_sales").alias("ws3_web_sales")]
    )

    # Join all 6 aliases: ss1, ss2, ss3, ws1, ws2, ws3
    # WHERE ss1.ca_county = ss2.ca_county = ss3.ca_county = ws1.ca_county = ws2.ca_county = ws3.ca_county
    # Plus the ratio-comparison WHERE conditions
    joined = (
        ss1
        .join(ss2, on="ca_county", how="inner")
        .join(ss3, on="ca_county", how="inner")
        .join(ws1, on="ca_county", how="inner")
        .join(ws2, on="ca_county", how="inner")
        .join(ws3, on="ca_county", how="inner")
        .filter(
            # CASE WHEN ws1.web_sales>0 THEN ws2/ws1 ELSE NULL END
            # > CASE WHEN ss1.store_sales>0 THEN ss2/ss1 ELSE NULL END
            (
                pl.when(pl.col("ws1_web_sales") > 0)
                .then(pl.col("ws2_web_sales") / pl.col("ws1_web_sales"))
                .otherwise(None)
                >
                pl.when(pl.col("ss1_store_sales") > 0)
                .then(pl.col("ss2_store_sales") / pl.col("ss1_store_sales"))
                .otherwise(None)
            )
            &
            # CASE WHEN ws2.web_sales>0 THEN ws3/ws2 ELSE NULL END
            # > CASE WHEN ss2.store_sales>0 THEN ss3/ss2 ELSE NULL END
            (
                pl.when(pl.col("ws2_web_sales") > 0)
                .then(pl.col("ws3_web_sales") / pl.col("ws2_web_sales"))
                .otherwise(None)
                >
                pl.when(pl.col("ss2_store_sales") > 0)
                .then(pl.col("ss3_store_sales") / pl.col("ss2_store_sales"))
                .otherwise(None)
            )
        )
        .select([
            pl.col("ca_county"),
            pl.lit(year, dtype=pl.Int64).alias("d_year"),
            (pl.col("ws2_web_sales") / pl.col("ws1_web_sales")).alias("web_q1_q2_increase"),
            (pl.col("ss2_store_sales") / pl.col("ss1_store_sales")).alias(
                "store_q1_q2_increase"
            ),
            (pl.col("ws3_web_sales") / pl.col("ws2_web_sales")).alias("web_q2_q3_increase"),
            (pl.col("ss3_store_sales") / pl.col("ss2_store_sales")).alias(
                "store_q2_q3_increase"
            ),
        ])
    )

    # ORDER BY {agg} — agg may be "ss1.ca_county" etc., strip table prefix
    polars_agg = agg.replace("ss1.", "").replace("ss2.", "").replace("ws1.", "")
    sort_by = [(polars_agg, False)]

    joined = joined.sort(polars_agg, descending=False, nulls_last=True)
    return QueryResult(frame=joined, sort_by=sort_by, limit=None)
