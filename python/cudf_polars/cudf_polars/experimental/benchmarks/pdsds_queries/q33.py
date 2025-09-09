# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 33."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 33."""
    return """
    WITH ss
         AS (SELECT i_manufact_id,
                    Sum(ss_ext_sales_price) total_sales
             FROM   store_sales,
                    date_dim,
                    customer_address,
                    item
             WHERE  i_manufact_id IN (SELECT i_manufact_id
                                      FROM   item
                                      WHERE  i_category IN ( 'Books' ))
                    AND ss_item_sk = i_item_sk
                    AND ss_sold_date_sk = d_date_sk
                    AND d_year = 1999
                    AND d_moy = 3
                    AND ss_addr_sk = ca_address_sk
                    AND ca_gmt_offset = -5
             GROUP  BY i_manufact_id),
         cs
         AS (SELECT i_manufact_id,
                    Sum(cs_ext_sales_price) total_sales
             FROM   catalog_sales,
                    date_dim,
                    customer_address,
                    item
             WHERE  i_manufact_id IN (SELECT i_manufact_id
                                      FROM   item
                                      WHERE  i_category IN ( 'Books' ))
                    AND cs_item_sk = i_item_sk
                    AND cs_sold_date_sk = d_date_sk
                    AND d_year = 1999
                    AND d_moy = 3
                    AND cs_bill_addr_sk = ca_address_sk
                    AND ca_gmt_offset = -5
             GROUP  BY i_manufact_id),
         ws
         AS (SELECT i_manufact_id,
                    Sum(ws_ext_sales_price) total_sales
             FROM   web_sales,
                    date_dim,
                    customer_address,
                    item
             WHERE  i_manufact_id IN (SELECT i_manufact_id
                                      FROM   item
                                      WHERE  i_category IN ( 'Books' ))
                    AND ws_item_sk = i_item_sk
                    AND ws_sold_date_sk = d_date_sk
                    AND d_year = 1999
                    AND d_moy = 3
                    AND ws_bill_addr_sk = ca_address_sk
                    AND ca_gmt_offset = -5
             GROUP  BY i_manufact_id)
    SELECT i_manufact_id,
                   Sum(total_sales) total_sales
    FROM   (SELECT *
            FROM   ss
            UNION ALL
            SELECT *
            FROM   cs
            UNION ALL
            SELECT *
            FROM   ws) tmp1
    GROUP  BY i_manufact_id
    ORDER  BY total_sales
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 33."""
    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    _ = get_data(run_config.dataset_path, "store", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    # Get manufacturers that have items in 'Books' category
    books_manufacturers = (
        item.filter(pl.col("i_category") == "Books").select("i_manufact_id").unique()
    )
    # CTE: ss (store sales)
    ss = (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(customer_address, left_on="ss_addr_sk", right_on="ca_address_sk")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .join(books_manufacturers, on="i_manufact_id")
        # TODO: Bug in cudf_polars with FILTER on Column with NULLs
        # .filter(
        #     (pl.col("d_year") == 1999) &
        #     (pl.col("d_moy") == 3) &
        #     (pl.col("ca_gmt_offset") == -5)
        # )
        .filter(
            pl.col("d_year").is_not_null()
            & (pl.col("d_year") == 1999)
            & pl.col("d_moy").is_not_null()
            & (pl.col("d_moy") == 3)
            & pl.col("ca_gmt_offset").is_not_null()
            & (pl.col("ca_gmt_offset") == -5)
        )
        .group_by("i_manufact_id")
        .agg([pl.col("ss_ext_sales_price").sum().alias("total_sales")])
    )
    # CTE: cs (catalog sales)
    cs = (
        catalog_sales.join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(customer_address, left_on="cs_bill_addr_sk", right_on="ca_address_sk")
        .join(item, left_on="cs_item_sk", right_on="i_item_sk")
        .join(books_manufacturers, on="i_manufact_id")
        # TODO: Bug in cudf_polars with FILTER on Column with NULLs
        # .filter(
        #     (pl.col("d_year") == 1999) &
        #     (pl.col("d_moy") == 3) &
        #     (pl.col("ca_gmt_offset") == -5)
        # )
        .filter(
            pl.col("d_year").is_not_null()
            & (pl.col("d_year") == 1999)
            & pl.col("d_moy").is_not_null()
            & (pl.col("d_moy") == 3)
            & pl.col("ca_gmt_offset").is_not_null()
            & (pl.col("ca_gmt_offset") == -5)
        )
        .group_by("i_manufact_id")
        .agg([pl.col("cs_ext_sales_price").sum().alias("total_sales")])
    )
    # CTE: ws (web sales)
    ws = (
        web_sales.join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(customer_address, left_on="ws_bill_addr_sk", right_on="ca_address_sk")
        .join(item, left_on="ws_item_sk", right_on="i_item_sk")
        .join(books_manufacturers, on="i_manufact_id")
        # TODO: Bug in cudf_polars with FILTER on Column with NULLs
        # .filter(
        #     (pl.col("d_year") == 1999) &
        #     (pl.col("d_moy") == 3) &
        #     (pl.col("ca_gmt_offset") == -5)
        # )
        .filter(
            pl.col("d_year").is_not_null()
            & (pl.col("d_year") == 1999)
            & pl.col("d_moy").is_not_null()
            & (pl.col("d_moy") == 3)
            & pl.col("ca_gmt_offset").is_not_null()
            & (pl.col("ca_gmt_offset") == -5)
        )
        .group_by("i_manufact_id")
        .agg([pl.col("ws_ext_sales_price").sum().alias("total_sales")])
    )
    # Union all channels and sum total sales by manufacturer
    return (
        pl.concat([ss, cs, ws])
        .group_by("i_manufact_id")
        .agg([pl.col("total_sales").sum().alias("total_sales")])
        .select(["i_manufact_id", "total_sales"])
        .sort("total_sales")
        .limit(100)
    )
