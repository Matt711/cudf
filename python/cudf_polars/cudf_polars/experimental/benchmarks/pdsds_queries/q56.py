# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 56."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 56."""
    return """
    WITH ss
         AS (SELECT i_item_id,
                    Sum(ss_ext_sales_price) total_sales
             FROM   store_sales,
                    date_dim,
                    customer_address,
                    item
             WHERE  i_item_id IN (SELECT i_item_id
                                  FROM   item
                                  WHERE  i_color IN ( 'firebrick', 'rosy', 'white' )
                                 )
                    AND ss_item_sk = i_item_sk
                    AND ss_sold_date_sk = d_date_sk
                    AND d_year = 1998
                    AND d_moy = 3
                    AND ss_addr_sk = ca_address_sk
                    AND ca_gmt_offset = -6
             GROUP  BY i_item_id),
         cs
         AS (SELECT i_item_id,
                    Sum(cs_ext_sales_price) total_sales
             FROM   catalog_sales,
                    date_dim,
                    customer_address,
                    item
             WHERE  i_item_id IN (SELECT i_item_id
                                  FROM   item
                                  WHERE  i_color IN ( 'firebrick', 'rosy', 'white' )
                                 )
                    AND cs_item_sk = i_item_sk
                    AND cs_sold_date_sk = d_date_sk
                    AND d_year = 1998
                    AND d_moy = 3
                    AND cs_bill_addr_sk = ca_address_sk
                    AND ca_gmt_offset = -6
             GROUP  BY i_item_id),
         ws
         AS (SELECT i_item_id,
                    Sum(ws_ext_sales_price) total_sales
             FROM   web_sales,
                    date_dim,
                    customer_address,
                    item
             WHERE  i_item_id IN (SELECT i_item_id
                                  FROM   item
                                  WHERE  i_color IN ( 'firebrick', 'rosy', 'white' )
                                 )
                    AND ws_item_sk = i_item_sk
                    AND ws_sold_date_sk = d_date_sk
                    AND d_year = 1998
                    AND d_moy = 3
                    AND ws_bill_addr_sk = ca_address_sk
                    AND ca_gmt_offset = -6
             GROUP  BY i_item_id)
    SELECT i_item_id,
                   Sum(total_sales) total_sales
    FROM   (SELECT *
            FROM   ss
            UNION ALL
            SELECT *
            FROM   cs
            UNION ALL
            SELECT *
            FROM   ws) tmp1
    GROUP  BY i_item_id
    ORDER  BY total_sales
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 56."""
    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # Item IDs with specified colors
    color_item_ids_lf = (
        item.filter(pl.col("i_color").is_in(["firebrick", "rosy", "white"]))
        .select(["i_item_id"])
        .unique()
    )

    # CTE 1: ss - Store sales for colored items
    ss = (
        store_sales.join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .join(color_item_ids_lf, on="i_item_id")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(customer_address, left_on="ss_addr_sk", right_on="ca_address_sk")
        # TODO: There's some bug in cudf_polars with FILTER on Column with NULLs
        .filter(
            pl.col("i_item_id").is_not_null()
            & pl.col("d_year").is_not_null()
            & (pl.col("d_year") == 1998)
            & pl.col("d_moy").is_not_null()
            & (pl.col("d_moy") == 3)
            & pl.col("ca_gmt_offset").is_not_null()
            & (pl.col("ca_gmt_offset") == -6)
        )
        .group_by("i_item_id")
        .agg(
            [
                pl.col("ss_ext_sales_price").count().alias("count_sales"),
                pl.col("ss_ext_sales_price").sum().alias("sum_sales"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("count_sales") == 0)
                .then(None)
                .otherwise(pl.col("sum_sales"))
                .alias("total_sales")
            ]
        )
        .select(["i_item_id", "total_sales"])
    )

    # CTE 2: cs - Catalog sales for colored items
    cs = (
        catalog_sales.join(item, left_on="cs_item_sk", right_on="i_item_sk")
        .join(color_item_ids_lf, on="i_item_id")
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(customer_address, left_on="cs_bill_addr_sk", right_on="ca_address_sk")
        # TODO: There's some bug in cudf_polars with FILTER on Column with NULLs
        .filter(
            pl.col("i_item_id").is_not_null()
            & pl.col("d_year").is_not_null()
            & (pl.col("d_year") == 1998)
            & pl.col("d_moy").is_not_null()
            & (pl.col("d_moy") == 3)
            & pl.col("ca_gmt_offset").is_not_null()
            & (pl.col("ca_gmt_offset") == -6)
        )
        .group_by("i_item_id")
        .agg(
            [
                pl.col("cs_ext_sales_price").count().alias("count_sales"),
                pl.col("cs_ext_sales_price").sum().alias("sum_sales"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("count_sales") == 0)
                .then(None)
                .otherwise(pl.col("sum_sales"))
                .alias("total_sales")
            ]
        )
        .select(["i_item_id", "total_sales"])
    )

    # CTE 3: ws - Web sales for colored items
    ws = (
        web_sales.join(item, left_on="ws_item_sk", right_on="i_item_sk")
        .join(color_item_ids_lf, on="i_item_id")
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(customer_address, left_on="ws_bill_addr_sk", right_on="ca_address_sk")
        # TODO: There's some bug in cudf_polars with FILTER on Column with NULLs
        .filter(
            pl.col("i_item_id").is_not_null()
            & pl.col("d_year").is_not_null()
            & (pl.col("d_year") == 1998)
            & pl.col("d_moy").is_not_null()
            & (pl.col("d_moy") == 3)
            & pl.col("ca_gmt_offset").is_not_null()
            & (pl.col("ca_gmt_offset") == -6)
        )
        .group_by("i_item_id")
        .agg(
            [
                pl.col("ws_ext_sales_price").count().alias("count_sales"),
                pl.col("ws_ext_sales_price").sum().alias("sum_sales"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("count_sales") == 0)
                .then(None)
                .otherwise(pl.col("sum_sales"))
                .alias("total_sales")
            ]
        )
        .select(["i_item_id", "total_sales"])
    )

    # UNION ALL and final aggregation with postprocessing
    return (
        pl.concat([ss, cs, ws])  # UNION ALL equivalent
        .group_by("i_item_id")
        .agg(
            [
                pl.col("total_sales").count().alias("count_total"),
                pl.col("total_sales").sum().alias("sum_total"),
            ]
        )
        # Postprocessing: replace sum with null when count is 0
        .with_columns(
            [
                pl.when(pl.col("count_total") == 0)
                .then(None)
                .otherwise(pl.col("sum_total"))
                .alias("total_sales")
            ]
        )
        .select(["i_item_id", "total_sales"])
        .sort(
            ["total_sales"],
            nulls_last=True,
            descending=[False],  # ORDER BY total_sales ASC
        )
        .limit(100)
    )
