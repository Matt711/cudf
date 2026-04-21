# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q72 — naive one-for-one Polars translation of the SQL."""
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
        int(run_config.scale_factor), query_id=72, qualification=run_config.qualification
    )
    year = params["year"]
    bp = params["bp"]
    ms = params["ms"]

    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    inventory = get_data(run_config.dataset_path, "inventory", run_config.suffix)
    warehouse = get_data(run_config.dataset_path, "warehouse", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    customer_demographics = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    household_demographics = get_data(
        run_config.dataset_path, "household_demographics", run_config.suffix
    )
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    promotion = get_data(run_config.dataset_path, "promotion", run_config.suffix)
    catalog_returns = get_data(
        run_config.dataset_path, "catalog_returns", run_config.suffix
    )

    # Three aliased copies of date_dim: d1, d2, d3
    d1 = date_dim.select([
        pl.col("d_date_sk").alias("d1_date_sk"),
        pl.col("d_week_seq").alias("d1_week_seq"),
        pl.col("d_date").alias("d1_date"),
        pl.col("d_year").alias("d1_year"),
    ])
    d2 = date_dim.select([
        pl.col("d_date_sk").alias("d2_date_sk"),
        pl.col("d_week_seq").alias("d2_week_seq"),
    ])
    d3 = date_dim.select([
        pl.col("d_date_sk").alias("d3_date_sk"),
        pl.col("d_date").alias("d3_date"),
    ])

    # SQL join order (explicit JOIN...ON syntax):
    # catalog_sales
    # JOIN inventory ON cs_item_sk = inv_item_sk
    # JOIN warehouse ON w_warehouse_sk = inv_warehouse_sk
    # JOIN item ON i_item_sk = cs_item_sk
    # JOIN customer_demographics ON cs_bill_cdemo_sk = cd_demo_sk
    # JOIN household_demographics ON cs_bill_hdemo_sk = hd_demo_sk
    # JOIN date_dim d1 ON cs_sold_date_sk = d1.d_date_sk
    # JOIN date_dim d2 ON inv_date_sk = d2.d_date_sk
    # JOIN date_dim d3 ON cs_ship_date_sk = d3.d_date_sk
    # LEFT OUTER JOIN promotion ON cs_promo_sk = p_promo_sk
    # LEFT OUTER JOIN catalog_returns ON cr_item_sk = cs_item_sk AND cr_order_number = cs_order_number
    result = (
        catalog_sales
        .join(inventory, left_on="cs_item_sk", right_on="inv_item_sk", how="inner")
        .join(warehouse, left_on="inv_warehouse_sk", right_on="w_warehouse_sk", how="inner")
        .join(item, left_on="cs_item_sk", right_on="i_item_sk", how="inner")
        .join(
            customer_demographics,
            left_on="cs_bill_cdemo_sk",
            right_on="cd_demo_sk",
            how="inner",
        )
        .join(
            household_demographics,
            left_on="cs_bill_hdemo_sk",
            right_on="hd_demo_sk",
            how="inner",
        )
        .join(d1, left_on="cs_sold_date_sk", right_on="d1_date_sk", how="inner")
        .join(d2, left_on="inv_date_sk", right_on="d2_date_sk", how="inner")
        .join(d3, left_on="cs_ship_date_sk", right_on="d3_date_sk", how="inner")
        .join(promotion, left_on="cs_promo_sk", right_on="p_promo_sk", how="left")
        .join(
            catalog_returns,
            left_on=["cs_item_sk", "cs_order_number"],
            right_on=["cr_item_sk", "cr_order_number"],
            how="left",
        )
        # WHERE conditions after all joins (naive rule 2)
        .filter(
            (pl.col("d1_week_seq") == pl.col("d2_week_seq"))
            & (pl.col("inv_quantity_on_hand") < pl.col("cs_quantity"))
            & (
                pl.col("d3_date").cast(pl.Date)
                > pl.col("d1_date").cast(pl.Date) + pl.duration(days=5)
            )
            & (pl.col("hd_buy_potential") == bp)
            & (pl.col("d1_year") == year)
            & (pl.col("cd_marital_status") == ms)
        )
        .with_columns([
            pl.when(pl.col("p_promo_name").is_null())
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("_no_promo_flag"),
            pl.when(pl.col("p_promo_name").is_not_null())
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("_promo_flag"),
        ])
        .group_by(["i_item_desc", "w_warehouse_name", "d1_week_seq"])
        .agg([
            sql_sum(pl.col("_no_promo_flag")).alias("no_promo"),
            sql_sum(pl.col("_promo_flag")).alias("promo"),
            pl.len().alias("total_cnt"),
        ])
        .select([
            "i_item_desc",
            "w_warehouse_name",
            pl.col("d1_week_seq").alias("d_week_seq"),
            "no_promo",
            "promo",
            "total_cnt",
        ])
    )

    result = result.sort(['total_cnt', 'i_item_desc', 'w_warehouse_name', 'd_week_seq'], descending=[True, False, False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[
            ("total_cnt", True),
            ("i_item_desc", False),
            ("w_warehouse_name", False),
            ("d_week_seq", False),
        ],
        limit=100,
    )
