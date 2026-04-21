# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q57 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q57 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=57, qualification=run_config.qualification
    )
    year = params["year"]

    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    call_center = get_data(run_config.dataset_path, "call_center", run_config.suffix)

    # v1 CTE:
    # FROM item, catalog_sales, date_dim, call_center
    # WHERE cs_item_sk = i_item_sk AND cs_sold_date_sk = d_date_sk
    #   AND cc_call_center_sk = cs_call_center_sk
    #   AND (d_year = year OR (d_year=year-1 AND d_moy=12) OR (d_year=year+1 AND d_moy=1))
    # GROUP BY i_category, i_brand, cc_name, d_year, d_moy
    # sum_sales = Sum(cs_sales_price)
    # avg_monthly_sales = Avg(Sum(...)) OVER (PARTITION BY i_category, i_brand, cc_name, d_year)
    # rn = Rank() OVER (PARTITION BY i_category, i_brand, cc_name ORDER BY d_year, d_moy)
    #
    # Rule 13: group_by (i_category, i_brand, cc_name, d_year, d_moy) first to get sum_sales,
    #   then .mean().over([i_category, i_brand, cc_name, d_year]) for avg_monthly_sales
    # Rule 12: .rank(method="min").over([i_category, i_brand, cc_name], order_by=[d_year, d_moy])
    v1 = (
        item.join(catalog_sales, left_on="i_item_sk", right_on="cs_item_sk", how="inner")
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk", how="inner")
        .join(call_center, left_on="cs_call_center_sk", right_on="cc_call_center_sk", how="inner")
        .filter(
            (pl.col("d_year") == year)
            | ((pl.col("d_year") == year - 1) & (pl.col("d_moy") == 12))
            | ((pl.col("d_year") == year + 1) & (pl.col("d_moy") == 1))
        )
        .group_by(["i_category", "i_brand", "cc_name", "d_year", "d_moy"])
        .agg(sql_sum(pl.col("cs_sales_price")).alias("sum_sales"))
        .with_columns(
            pl.col("sum_sales")
            .mean()
            .over(["i_category", "i_brand", "cc_name", "d_year"])
            .alias("avg_monthly_sales"),
            pl.col("d_year")
            .rank(method="min")
            .over(["i_category", "i_brand", "cc_name"], order_by=["d_year", "d_moy"])
            .alias("rn"),
        )
    )

    # v2 CTE:
    # FROM v1, v1 v1_lag, v1 v1_lead
    # WHERE v1.i_category = v1_lag.i_category AND v1.i_category = v1_lead.i_category
    #   AND v1.i_brand = v1_lag.i_brand AND v1.i_brand = v1_lead.i_brand
    #   AND v1.cc_name = v1_lag.cc_name AND v1.cc_name = v1_lead.cc_name
    #   AND v1.rn = v1_lag.rn + 1
    #   AND v1.rn = v1_lead.rn - 1
    # SELECT v1.i_brand, v1.d_year, v1.avg_monthly_sales, v1.sum_sales,
    #   v1_lag.sum_sales psum, v1_lead.sum_sales nsum
    v1_lag = v1.select(
        pl.col("i_category").alias("lag_i_category"),
        pl.col("i_brand").alias("lag_i_brand"),
        pl.col("cc_name").alias("lag_cc_name"),
        pl.col("rn").alias("lag_rn"),
        pl.col("sum_sales").alias("psum"),
    )
    v1_lead = v1.select(
        pl.col("i_category").alias("lead_i_category"),
        pl.col("i_brand").alias("lead_i_brand"),
        pl.col("cc_name").alias("lead_cc_name"),
        pl.col("rn").alias("lead_rn"),
        pl.col("sum_sales").alias("nsum"),
    )

    # Join v1 with v1_lag on matching category/brand/cc_name, then filter rn == lag_rn + 1
    v2 = (
        v1.join(
            v1_lag,
            left_on=["i_category", "i_brand", "cc_name"],
            right_on=["lag_i_category", "lag_i_brand", "lag_cc_name"],
            how="inner",
        )
        .filter(pl.col("rn") == pl.col("lag_rn") + 1)
        .join(
            v1_lead,
            left_on=["i_category", "i_brand", "cc_name"],
            right_on=["lead_i_category", "lead_i_brand", "lead_cc_name"],
            how="inner",
        )
        .filter(pl.col("rn") == pl.col("lead_rn") - 1)
        .select(["i_brand", "d_year", "avg_monthly_sales", "sum_sales", "psum", "nsum"])
    )

    # Final:
    # WHERE d_year = year AND avg_monthly_sales > 0
    #   AND CASE WHEN avg_monthly_sales > 0
    #     THEN Abs(sum_sales - avg_monthly_sales) / avg_monthly_sales ELSE NULL END > 0.1
    # ORDER BY sum_sales - avg_monthly_sales, 3 (avg_monthly_sales)
    result = v2.filter(
        (pl.col("d_year") == year)
        & (pl.col("avg_monthly_sales") > 0)
        & (
            pl.when(pl.col("avg_monthly_sales") > 0)
            .then(
                (pl.col("sum_sales") - pl.col("avg_monthly_sales")).abs()
                / pl.col("avg_monthly_sales")
            )
            .otherwise(None)
            > 0.1
        )
    )

    result = result.sort('avg_monthly_sales', descending=False, nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("avg_monthly_sales", False)],
        limit=100,
        sort_keys=[
            (pl.col("sum_sales") - pl.col("avg_monthly_sales"), False),
            (pl.col("avg_monthly_sales"), False),
        ],
    )
