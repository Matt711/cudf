# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q78 — naive one-for-one Polars translation of the SQL."""
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
        int(run_config.scale_factor), query_id=78, qualification=run_config.qualification
    )
    year = params["year"]

    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    web_returns = get_data(run_config.dataset_path, "web_returns", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    catalog_returns = get_data(
        run_config.dataset_path, "catalog_returns", run_config.suffix
    )
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(
        run_config.dataset_path, "store_returns", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # CTE ws:
    # FROM web_sales LEFT JOIN web_returns ON wr_order_number=ws_order_number AND ws_item_sk=wr_item_sk
    # JOIN date_dim ON ws_sold_date_sk = d_date_sk
    # WHERE wr_order_number IS NULL
    # GROUP BY d_year, ws_item_sk, ws_bill_customer_sk
    ws = (
        web_sales
        .join(
            web_returns,
            left_on=["ws_order_number", "ws_item_sk"],
            right_on=["wr_order_number", "wr_item_sk"],
            how="left",
        )
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(pl.col("wr_return_amt").is_null())
        .group_by(["d_year", "ws_item_sk", "ws_bill_customer_sk"])
        .agg([
            sql_sum(pl.col("ws_quantity")).alias("ws_qty"),
            sql_sum(pl.col("ws_wholesale_cost")).alias("ws_wc"),
            sql_sum(pl.col("ws_sales_price")).alias("ws_sp"),
        ])
        .rename({"ws_bill_customer_sk": "ws_customer_sk", "d_year": "ws_sold_year"})
    )

    # CTE cs:
    # FROM catalog_sales LEFT JOIN catalog_returns ON cr_order_number=cs_order_number AND cs_item_sk=cr_item_sk
    # JOIN date_dim ON cs_sold_date_sk = d_date_sk
    # WHERE cr_order_number IS NULL
    # GROUP BY d_year, cs_item_sk, cs_bill_customer_sk
    cs = (
        catalog_sales
        .join(
            catalog_returns,
            left_on=["cs_order_number", "cs_item_sk"],
            right_on=["cr_order_number", "cr_item_sk"],
            how="left",
        )
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(pl.col("cr_return_amt").is_null())
        .group_by(["d_year", "cs_item_sk", "cs_bill_customer_sk"])
        .agg([
            sql_sum(pl.col("cs_quantity")).alias("cs_qty"),
            sql_sum(pl.col("cs_wholesale_cost")).alias("cs_wc"),
            sql_sum(pl.col("cs_sales_price")).alias("cs_sp"),
        ])
        .rename({"cs_bill_customer_sk": "cs_customer_sk", "d_year": "cs_sold_year"})
    )

    # CTE ss:
    # FROM store_sales LEFT JOIN store_returns ON sr_ticket_number=ss_ticket_number AND ss_item_sk=sr_item_sk
    # JOIN date_dim ON ss_sold_date_sk = d_date_sk
    # WHERE sr_ticket_number IS NULL
    # GROUP BY d_year, ss_item_sk, ss_customer_sk
    ss = (
        store_sales
        .join(
            store_returns,
            left_on=["ss_ticket_number", "ss_item_sk"],
            right_on=["sr_ticket_number", "sr_item_sk"],
            how="left",
        )
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(pl.col("sr_return_amt").is_null())
        .group_by(["d_year", "ss_item_sk", "ss_customer_sk"])
        .agg([
            sql_sum(pl.col("ss_quantity")).alias("ss_qty"),
            sql_sum(pl.col("ss_wholesale_cost")).alias("ss_wc"),
            sql_sum(pl.col("ss_sales_price")).alias("ss_sp"),
        ])
        .rename({"d_year": "ss_sold_year"})
    )

    # Main query:
    # FROM ss LEFT JOIN ws ON (ws_sold_year=ss_sold_year AND ws_item_sk=ss_item_sk AND ws_customer_sk=ss_customer_sk)
    #         LEFT JOIN cs ON (cs_sold_year=ss_sold_year AND cs_item_sk=ss_item_sk AND cs_customer_sk=ss_customer_sk)
    # WHERE (coalesce(ws_qty,0)>0 OR coalesce(cs_qty,0)>0)
    #   AND ss_sold_year=year
    result = (
        ss
        .join(
            ws,
            left_on=["ss_sold_year", "ss_item_sk", "ss_customer_sk"],
            right_on=["ws_sold_year", "ws_item_sk", "ws_customer_sk"],
            how="left",
        )
        .join(
            cs,
            left_on=["ss_sold_year", "ss_item_sk", "ss_customer_sk"],
            right_on=["cs_sold_year", "cs_item_sk", "cs_customer_sk"],
            how="left",
        )
        .filter(
            ((pl.col("ws_qty").fill_null(0) > 0) | (pl.col("cs_qty").fill_null(0) > 0))
            & (pl.col("ss_sold_year") == year)
        )
        .select([
            "ss_sold_year",
            "ss_item_sk",
            "ss_customer_sk",
            (
                pl.col("ss_qty").cast(pl.Float64)
                / (pl.col("ws_qty").fill_null(0) + pl.col("cs_qty").fill_null(0))
            ).round(2).alias("ratio"),
            pl.col("ss_qty").alias("store_qty"),
            pl.col("ss_wc").alias("store_wholesale_cost"),
            pl.col("ss_sp").alias("store_sales_price"),
            (pl.col("ws_qty").fill_null(0) + pl.col("cs_qty").fill_null(0)).alias(
                "other_chan_qty"
            ),
            (pl.col("ws_wc").fill_null(0) + pl.col("cs_wc").fill_null(0)).alias(
                "other_chan_wholesale_cost"
            ),
            (pl.col("ws_sp").fill_null(0) + pl.col("cs_sp").fill_null(0)).alias(
                "other_chan_sales_price"
            ),
        ])
    )

    result = result.sort(['ss_sold_year', 'ss_item_sk', 'ss_customer_sk', 'store_qty', 'store_wholesale_cost', 'store_sales_price', 'other_chan_qty', 'other_chan_wholesale_cost', 'other_chan_sales_price', 'ratio'], descending=[False, False, False, True, True, True, False, False, False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[
            ("ss_sold_year", False),
            ("ss_item_sk", False),
            ("ss_customer_sk", False),
            ("store_qty", True),
            ("store_wholesale_cost", True),
            ("store_sales_price", True),
            ("other_chan_qty", False),
            ("other_chan_wholesale_cost", False),
            ("other_chan_sales_price", False),
            ("ratio", False),
        ],
        limit=100,
    )
