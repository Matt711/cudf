# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS q14 — naive one-for-one Polars translation of the SQL."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data
from cudf_polars.experimental.benchmarks.unoptimized_pdsds_queries.sql_helpers import (
    sql_rollup,
    sql_sum,
)

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:  # type: ignore[return]
    raise NotImplementedError


def polars_impl(run_config: RunConfig) -> QueryResult:
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=14,
        qualification=run_config.qualification,
    )
    year = params["year"]
    day = params["day"]

    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # CTE cross_items:
    # Three INTERSECTs of (i_brand_id, i_class_id, i_category_id) from the three channels.
    # Each channel: FROM <sales>, item <iss/ics/iws>, date_dim <d>
    #   WHERE <sales_item_sk>=<item_sk> AND <sales_date_sk>=d.d_date_sk
    #     AND d.d_year BETWEEN year AND year+2
    # INTERSECT => inner-join approach (rule 11)

    store_brands = (
        store_sales.join(item.select(["i_item_sk", "i_brand_id", "i_class_id", "i_category_id"]),
                         left_on="ss_item_sk", right_on="i_item_sk", how="inner")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(pl.col("d_year").is_between(year, year + 2, closed="both"))
        .select(["i_brand_id", "i_class_id", "i_category_id"])
        .unique()
    )

    catalog_brands = (
        catalog_sales.join(item.select(["i_item_sk", "i_brand_id", "i_class_id", "i_category_id"]),
                           left_on="cs_item_sk", right_on="i_item_sk", how="inner")
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(pl.col("d_year").is_between(year, year + 2, closed="both"))
        .select(["i_brand_id", "i_class_id", "i_category_id"])
        .unique()
    )

    web_brands = (
        web_sales.join(item.select(["i_item_sk", "i_brand_id", "i_class_id", "i_category_id"]),
                       left_on="ws_item_sk", right_on="i_item_sk", how="inner")
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(pl.col("d_year").is_between(year, year + 2, closed="both"))
        .select(["i_brand_id", "i_class_id", "i_category_id"])
        .unique()
    )

    # INTERSECT three sets: inner join on all three projected columns, then unique
    intersect_brands = (
        store_brands.join(catalog_brands, on=["i_brand_id", "i_class_id", "i_category_id"], how="inner")
        .join(web_brands, on=["i_brand_id", "i_class_id", "i_category_id"], how="inner")
        .unique()
    )

    # cross_items: FROM item WHERE i_brand_id=brand_id AND i_class_id=class_id AND i_category_id=category_id
    cross_items = (
        item.join(intersect_brands, on=["i_brand_id", "i_class_id", "i_category_id"], how="inner")
        .select(pl.col("i_item_sk").alias("ss_item_sk"))
    )

    # CTE avg_sales:
    # UNION ALL of store/catalog/web (quantity, list_price) joined with date_dim WHERE d_year BETWEEN year AND year+2
    # Then AVG(quantity*list_price)
    store_qty = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(pl.col("d_year").is_between(year, year + 2, closed="both"))
        .select(
            pl.col("ss_quantity").alias("quantity"),
            pl.col("ss_list_price").alias("list_price"),
        )
    )
    catalog_qty = (
        catalog_sales.join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(pl.col("d_year").is_between(year, year + 2, closed="both"))
        .select(
            pl.col("cs_quantity").alias("quantity"),
            pl.col("cs_list_price").alias("list_price"),
        )
    )
    web_qty = (
        web_sales.join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .filter(pl.col("d_year").is_between(year, year + 2, closed="both"))
        .select(
            pl.col("ws_quantity").alias("quantity"),
            pl.col("ws_list_price").alias("list_price"),
        )
    )

    sq2 = pl.concat([store_qty, catalog_qty, web_qty], how="diagonal_relaxed")
    avg_sales = sq2.select(
        (pl.col("quantity") * pl.col("list_price")).mean().alias("average_sales")
    )

    # Target d_week_seq: FROM date_dim WHERE d_year=year+1 AND d_moy=12 AND d_dom=day
    target_week_seq = (
        date_dim.filter(
            (pl.col("d_year") == year + 1)
            & (pl.col("d_moy") == 12)
            & (pl.col("d_dom") == day)
        )
        .select("d_week_seq")
        .unique()
    )
    # All date_sks in that week
    week_date_sks = (
        date_dim.join(target_week_seq, on="d_week_seq", how="inner")
        .select("d_date_sk")
    )

    # Store channel:
    # FROM store_sales, item, date_dim
    # WHERE ss_item_sk IN cross_items AND ss_item_sk=i_item_sk
    #   AND ss_sold_date_sk=d_date_sk AND d_week_seq=(scalar)
    # GROUP BY i_brand_id, i_class_id, i_category_id
    # HAVING sum(ss_quantity*ss_list_price) > average_sales
    store_ch = (
        store_sales.join(cross_items, left_on="ss_item_sk", right_on="ss_item_sk", how="inner")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk", how="inner")
        .join(week_date_sks, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner")
        .group_by(["i_brand_id", "i_class_id", "i_category_id"])
        .agg(
            sql_sum(pl.col("ss_quantity") * pl.col("ss_list_price")).alias("sales"),
            pl.len().alias("number_sales"),
        )
        .join(avg_sales, how="cross")
        .filter(pl.col("sales") > pl.col("average_sales"))
        .select(
            pl.lit("store").alias("channel"),
            "i_brand_id", "i_class_id", "i_category_id", "sales", "number_sales",
        )
    )

    # Catalog channel
    catalog_ch = (
        catalog_sales.join(cross_items.rename({"ss_item_sk": "item_sk"}), left_on="cs_item_sk", right_on="item_sk", how="inner")
        .join(item, left_on="cs_item_sk", right_on="i_item_sk", how="inner")
        .join(week_date_sks, left_on="cs_sold_date_sk", right_on="d_date_sk", how="inner")
        .group_by(["i_brand_id", "i_class_id", "i_category_id"])
        .agg(
            sql_sum(pl.col("cs_quantity") * pl.col("cs_list_price")).alias("sales"),
            pl.len().alias("number_sales"),
        )
        .join(avg_sales, how="cross")
        .filter(pl.col("sales") > pl.col("average_sales"))
        .select(
            pl.lit("catalog").alias("channel"),
            "i_brand_id", "i_class_id", "i_category_id", "sales", "number_sales",
        )
    )

    # Web channel
    web_ch = (
        web_sales.join(cross_items.rename({"ss_item_sk": "item_sk"}), left_on="ws_item_sk", right_on="item_sk", how="inner")
        .join(item, left_on="ws_item_sk", right_on="i_item_sk", how="inner")
        .join(week_date_sks, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner")
        .group_by(["i_brand_id", "i_class_id", "i_category_id"])
        .agg(
            sql_sum(pl.col("ws_quantity") * pl.col("ws_list_price")).alias("sales"),
            pl.len().alias("number_sales"),
        )
        .join(avg_sales, how="cross")
        .filter(pl.col("sales") > pl.col("average_sales"))
        .select(
            pl.lit("web").alias("channel"),
            "i_brand_id", "i_class_id", "i_category_id", "sales", "number_sales",
        )
    )

    # UNION ALL of three channels => y
    y = pl.concat([store_ch, catalog_ch, web_ch], how="diagonal_relaxed")

    # GROUP BY ROLLUP(channel, i_brand_id, i_class_id, i_category_id)
    result = sql_rollup(
        y,
        ["channel", "i_brand_id", "i_class_id", "i_category_id"],
        [
            sql_sum(pl.col("sales")).alias("sum_sales"),
            sql_sum(pl.col("number_sales")).alias("sum_number_sales"),
        ],
    )

    result = result.sort(['channel', 'i_brand_id', 'i_class_id', 'i_category_id'], descending=[False, False, False, False]).limit(100)
    return QueryResult(
        frame=result,
        sort_by=[
            ("channel", False),
            ("i_brand_id", False),
            ("i_class_id", False),
            ("i_category_id", False),
        ],
        limit=100,
        nulls_last=False,
    )
