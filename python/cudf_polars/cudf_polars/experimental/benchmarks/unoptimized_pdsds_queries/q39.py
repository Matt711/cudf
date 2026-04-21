# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q39 — naive one-for-one Polars translation of the SQL."""
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
        int(run_config.scale_factor), query_id=39, qualification=run_config.qualification
    )
    year = params["year"]
    month = params["month"]

    inventory = get_data(run_config.dataset_path, "inventory", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    warehouse = get_data(run_config.dataset_path, "warehouse", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # CTE inv (inner foo subquery):
    # FROM inventory, item, warehouse, date_dim
    # WHERE inv_item_sk = i_item_sk AND inv_warehouse_sk = w_warehouse_sk
    #   AND inv_date_sk = d_date_sk AND d_year = year
    # GROUP BY w_warehouse_name, w_warehouse_sk, i_item_sk, d_moy
    # HAVING cov > 1
    # All WHERE conditions after all joins (naive rule 2)
    base_agg = (
        inventory.join(item, left_on="inv_item_sk", right_on="i_item_sk")
        .join(warehouse, left_on="inv_warehouse_sk", right_on="w_warehouse_sk")
        .join(date_dim, left_on="inv_date_sk", right_on="d_date_sk")
        .filter(pl.col("d_year") == year)
        .group_by(["w_warehouse_name", "inv_warehouse_sk", "inv_item_sk", "d_moy"])
        .agg(
            [
                (pl.col("inv_quantity_on_hand").std(ddof=1) * 1.0).alias("stdev"),
                pl.col("inv_quantity_on_hand").mean().alias("mean"),
            ]
        )
    )

    # cov = CASE mean WHEN 0 THEN NULL ELSE stdev/mean END
    # Filter: CASE mean WHEN 0 THEN 0 ELSE stdev/mean END > 1
    inv_cte = (
        base_agg.with_columns(
            pl.when(pl.col("mean") == 0)
            .then(pl.lit(None))
            .otherwise(pl.col("stdev") / pl.col("mean"))
            .alias("cov")
        )
        .filter(
            pl.when(pl.col("mean") == 0)
            .then(pl.lit(False))
            .otherwise(pl.col("stdev") / pl.col("mean") > 1.0)
        )
    )

    # Self-join inv1 (d_moy=month) with inv2 (d_moy=month+1)
    # WHERE inv1.i_item_sk = inv2.i_item_sk AND inv1.w_warehouse_sk = inv2.w_warehouse_sk
    inv1 = inv_cte.filter(pl.col("d_moy") == month).select(
        [
            pl.col("inv_warehouse_sk").alias("wsk1"),
            pl.col("inv_item_sk").alias("isk1"),
            pl.col("d_moy").alias("dmoy1"),
            pl.col("mean").alias("mean1"),
            pl.col("cov").alias("cov1"),
            "inv_warehouse_sk",
            "inv_item_sk",
        ]
    )

    inv2 = inv_cte.filter(pl.col("d_moy") == month + 1).select(
        [
            "inv_warehouse_sk",
            "inv_item_sk",
            pl.col("d_moy"),
            pl.col("mean"),
            pl.col("cov"),
        ]
    )

    result = (
        inv1.join(
            inv2,
            left_on=["inv_warehouse_sk", "inv_item_sk"],
            right_on=["inv_warehouse_sk", "inv_item_sk"],
            how="inner",
        )
        .select(
            [
                "wsk1",
                "isk1",
                "dmoy1",
                "mean1",
                "cov1",
                pl.col("inv_warehouse_sk"),
                pl.col("inv_item_sk"),
                pl.col("d_moy"),
                pl.col("mean"),
                pl.col("cov"),
            ]
        )
    )

    return QueryResult(
        frame=result,
        sort_by=[
            ("wsk1", False),
            ("isk1", False),
            ("dmoy1", False),
            ("mean1", False),
            ("cov1", False),
            ("d_moy", False),
            ("mean", False),
            ("cov", False),
        ],
        limit=None,
        nulls_last=False,
    )
