# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q47 — naive one-for-one Polars translation of the SQL."""
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
        int(run_config.scale_factor), query_id=47, qualification=run_config.qualification
    )
    year = params["year"]

    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)

    # CTE v1: FROM item, store_sales, date_dim, store
    # WHERE ss_item_sk=i_item_sk AND ss_sold_date_sk=d_date_sk AND ss_store_sk=s_store_sk
    #   AND (d_year=year OR (d_year=year-1 AND d_moy=12) OR (d_year=year+1 AND d_moy=1))
    # GROUP BY i_category, i_brand, s_store_name, s_company_name, d_year, d_moy
    # Window: AVG(SUM(ss_sales_price)) OVER (PARTITION BY i_category,i_brand,s_store_name,s_company_name,d_year)
    #       = avg_monthly_sales
    # Window: RANK() OVER (PARTITION BY i_category,i_brand,s_store_name,s_company_name ORDER BY d_year,d_moy)
    #       = rn
    # All WHERE conditions after all joins (naive rule 2)
    v1 = (
        item.join(store_sales, left_on="i_item_sk", right_on="ss_item_sk")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .filter(
            (pl.col("d_year") == year)
            | ((pl.col("d_year") == year - 1) & (pl.col("d_moy") == 12))
            | ((pl.col("d_year") == year + 1) & (pl.col("d_moy") == 1))
        )
        .group_by(
            ["i_category", "i_brand", "s_store_name", "s_company_name", "d_year", "d_moy"]
        )
        .agg(sql_sum(pl.col("ss_sales_price")).alias("sum_sales"))
        # AVG(SUM(...)) OVER (PARTITION BY ..., d_year) — compute mean over the per-year group
        .with_columns(
            pl.col("sum_sales")
            .mean()
            .over(["i_category", "i_brand", "s_store_name", "s_company_name", "d_year"])
            .alias("avg_monthly_sales")
        )
        # RANK() OVER (PARTITION BY ... ORDER BY d_year, d_moy)
        .with_columns(
            pl.col("d_year")
            .rank(method="min")
            .over(
                ["i_category", "i_brand", "s_store_name", "s_company_name"],
                order_by=["d_year", "d_moy"],
            )
            .alias("rn")
        )
    )

    # CTE v2: FROM v1, v1 v1_lag, v1 v1_lead
    # WHERE v1.i_category=v1_lag.i_category AND ... AND v1.rn=v1_lag.rn+1 AND v1.rn=v1_lead.rn-1
    # Implement as two self-joins (rule for q47: join v1 with itself twice on keys + rn offset)
    partition_keys = ["i_category", "i_brand", "s_store_name", "s_company_name"]

    v1_lag = v1.select(
        [
            pl.col(k).alias(f"{k}_lag") for k in partition_keys
        ]
        + [pl.col("rn").alias("rn_lag"), pl.col("sum_sales").alias("psum")]
    )

    v1_lead = v1.select(
        [
            pl.col(k).alias(f"{k}_lead") for k in partition_keys
        ]
        + [pl.col("rn").alias("rn_lead"), pl.col("sum_sales").alias("nsum")]
    )

    v2 = (
        v1
        # join with lag: v1.rn = v1_lag.rn + 1 on partition keys
        .join(
            v1_lag,
            left_on=partition_keys,
            right_on=[f"{k}_lag" for k in partition_keys],
            how="inner",
        )
        .filter(pl.col("rn") == pl.col("rn_lag") + 1)
        # join with lead: v1.rn = v1_lead.rn - 1 on partition keys
        .join(
            v1_lead,
            left_on=partition_keys,
            right_on=[f"{k}_lead" for k in partition_keys],
            how="inner",
        )
        .filter(pl.col("rn") == pl.col("rn_lead") - 1)
        .select(
            [
                "i_category",
                "d_year",
                "d_moy",
                "avg_monthly_sales",
                "sum_sales",
                "psum",
                "nsum",
            ]
        )
    )

    # Final: WHERE d_year=year AND avg_monthly_sales > 0
    #   AND ABS(sum_sales-avg_monthly_sales)/avg_monthly_sales > 0.1
    result = v2.filter(
        (pl.col("d_year") == year)
        & (pl.col("avg_monthly_sales") > 0)
        & (
            pl.when(pl.col("avg_monthly_sales") > 0)
            .then(
                (pl.col("sum_sales") - pl.col("avg_monthly_sales")).abs()
                / pl.col("avg_monthly_sales")
            )
            .otherwise(pl.lit(None))
            > 0.1
        )
    )

    result = result.sort('d_moy', descending=False, nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[("d_moy", False)],
        limit=100,
        sort_keys=[
            (pl.col("sum_sales") - pl.col("avg_monthly_sales"), False),
            (pl.col("d_moy"), False),
        ],
    )
