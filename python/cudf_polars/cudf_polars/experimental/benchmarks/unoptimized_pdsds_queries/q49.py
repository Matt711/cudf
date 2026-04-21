# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q49 — naive one-for-one Polars translation of the SQL."""
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
        int(run_config.scale_factor), query_id=49, qualification=run_config.qualification
    )
    year = params["year"]
    month = params["month"]

    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    web_returns = get_data(run_config.dataset_path, "web_returns", run_config.suffix)
    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    catalog_returns = get_data(
        run_config.dataset_path, "catalog_returns", run_config.suffix
    )
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(run_config.dataset_path, "store_returns", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # Web channel:
    # FROM web_sales ws LEFT OUTER JOIN web_returns wr
    #   ON (ws_order_number=wr_order_number AND ws_item_sk=wr_item_sk), date_dim
    # WHERE wr_return_amt > 10000 AND ws_net_profit > 1 AND ws_net_paid > 0
    #   AND ws_quantity > 0 AND ws_sold_date_sk=d_date_sk AND d_year=year AND d_moy=month
    # GROUP BY ws_item_sk
    # return_ratio = SUM(COALESCE(wr_return_quantity,0)) / SUM(COALESCE(ws_quantity,0))
    # currency_ratio = SUM(COALESCE(wr_return_amt,0)) / SUM(COALESCE(ws_net_paid,0))
    # All WHERE conditions after all joins (naive rule 2)
    web_inner = (
        web_sales.join(
            web_returns,
            left_on=["ws_order_number", "ws_item_sk"],
            right_on=["wr_order_number", "wr_item_sk"],
            how="left",
        )
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .filter(
            (pl.col("wr_return_amt").fill_null(0) > 10000)
            & (pl.col("ws_net_profit") > 1)
            & (pl.col("ws_net_paid") > 0)
            & (pl.col("ws_quantity") > 0)
            & (pl.col("d_year") == year)
            & (pl.col("d_moy") == month)
        )
        .group_by("ws_item_sk")
        .agg(
            [
                (
                    pl.col("wr_return_quantity").fill_null(0).sum().cast(pl.Float64)
                    / pl.col("ws_quantity").fill_null(0).sum().cast(pl.Float64)
                ).alias("return_ratio"),
                (
                    pl.col("wr_return_amt").fill_null(0).sum().cast(pl.Float64)
                    / pl.col("ws_net_paid").fill_null(0).sum().cast(pl.Float64)
                ).alias("currency_ratio"),
            ]
        )
        .rename({"ws_item_sk": "item"})
    )

    web_ranked = (
        web_inner.with_columns(
            [
                pl.col("return_ratio").rank(method="min").alias("return_rank"),
                pl.col("currency_ratio").rank(method="min").alias("currency_rank"),
            ]
        )
        .filter(
            (pl.col("return_rank") <= 10) | (pl.col("currency_rank") <= 10)
        )
        .with_columns(pl.lit("web").alias("channel"))
        .select(["channel", "item", "return_ratio", "return_rank", "currency_rank"])
    )

    # Catalog channel:
    # FROM catalog_sales cs LEFT OUTER JOIN catalog_returns cr
    #   ON (cs_order_number=cr_order_number AND cs_item_sk=cr_item_sk), date_dim
    # WHERE cr_return_amount > 10000 AND cs_net_profit > 1 AND cs_net_paid > 0
    #   AND cs_quantity > 0 AND cs_sold_date_sk=d_date_sk AND d_year=year AND d_moy=month
    catalog_inner = (
        catalog_sales.join(
            catalog_returns,
            left_on=["cs_order_number", "cs_item_sk"],
            right_on=["cr_order_number", "cr_item_sk"],
            how="left",
        )
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .filter(
            (pl.col("cr_return_amount").fill_null(0) > 10000)
            & (pl.col("cs_net_profit") > 1)
            & (pl.col("cs_net_paid") > 0)
            & (pl.col("cs_quantity") > 0)
            & (pl.col("d_year") == year)
            & (pl.col("d_moy") == month)
        )
        .group_by("cs_item_sk")
        .agg(
            [
                (
                    pl.col("cr_return_quantity").fill_null(0).sum().cast(pl.Float64)
                    / pl.col("cs_quantity").fill_null(0).sum().cast(pl.Float64)
                ).alias("return_ratio"),
                (
                    pl.col("cr_return_amount").fill_null(0).sum().cast(pl.Float64)
                    / pl.col("cs_net_paid").fill_null(0).sum().cast(pl.Float64)
                ).alias("currency_ratio"),
            ]
        )
        .rename({"cs_item_sk": "item"})
    )

    catalog_ranked = (
        catalog_inner.with_columns(
            [
                pl.col("return_ratio").rank(method="min").alias("return_rank"),
                pl.col("currency_ratio").rank(method="min").alias("currency_rank"),
            ]
        )
        .filter(
            (pl.col("return_rank") <= 10) | (pl.col("currency_rank") <= 10)
        )
        .with_columns(pl.lit("catalog").alias("channel"))
        .select(["channel", "item", "return_ratio", "return_rank", "currency_rank"])
    )

    # Store channel:
    # FROM store_sales sts LEFT OUTER JOIN store_returns sr
    #   ON (sts.ss_ticket_number=sr.sr_ticket_number AND sts.ss_item_sk=sr.sr_item_sk), date_dim
    # WHERE sr_return_amt > 10000 AND sts.ss_net_profit > 1 AND sts.ss_net_paid > 0
    #   AND sts.ss_quantity > 0 AND ss_sold_date_sk=d_date_sk AND d_year=year AND d_moy=month
    store_inner = (
        store_sales.join(
            store_returns,
            left_on=["ss_ticket_number", "ss_item_sk"],
            right_on=["sr_ticket_number", "sr_item_sk"],
            how="left",
        )
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .filter(
            (pl.col("sr_return_amt").fill_null(0) > 10000)
            & (pl.col("ss_net_profit") > 1)
            & (pl.col("ss_net_paid") > 0)
            & (pl.col("ss_quantity") > 0)
            & (pl.col("d_year") == year)
            & (pl.col("d_moy") == month)
        )
        .group_by("ss_item_sk")
        .agg(
            [
                (
                    pl.col("sr_return_quantity").fill_null(0).sum().cast(pl.Float64)
                    / pl.col("ss_quantity").fill_null(0).sum().cast(pl.Float64)
                ).alias("return_ratio"),
                (
                    pl.col("sr_return_amt").fill_null(0).sum().cast(pl.Float64)
                    / pl.col("ss_net_paid").fill_null(0).sum().cast(pl.Float64)
                ).alias("currency_ratio"),
            ]
        )
        .rename({"ss_item_sk": "item"})
    )

    store_ranked = (
        store_inner.with_columns(
            [
                pl.col("return_ratio").rank(method="min").alias("return_rank"),
                pl.col("currency_ratio").rank(method="min").alias("currency_rank"),
            ]
        )
        .filter(
            (pl.col("return_rank") <= 10) | (pl.col("currency_rank") <= 10)
        )
        .with_columns(pl.lit("store").alias("channel"))
        .select(["channel", "item", "return_ratio", "return_rank", "currency_rank"])
    )

    # UNION (deduplication) of three channels
    result = pl.concat(
        [web_ranked, catalog_ranked, store_ranked], how="diagonal_relaxed"
    ).unique()

    result = result.sort(['channel', 'return_rank', 'currency_rank'], descending=[False, False, False], nulls_last=True).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[
            ("channel", False),
            ("return_rank", False),
            ("currency_rank", False),
        ],
        limit=100,
    )
