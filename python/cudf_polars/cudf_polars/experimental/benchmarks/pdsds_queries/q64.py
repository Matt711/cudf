# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 64."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 64."""
    return """
    WITH cs_ui
         AS (SELECT cs_item_sk,
                    Sum(cs_ext_list_price) AS sale,
                    Sum(cr_refunded_cash + cr_reversed_charge
                        + cr_store_credit) AS refund
             FROM   catalog_sales,
                    catalog_returns
             WHERE  cs_item_sk = cr_item_sk
                    AND cs_order_number = cr_order_number
             GROUP  BY cs_item_sk
             HAVING Sum(cs_ext_list_price) > 2 * Sum(
                    cr_refunded_cash + cr_reversed_charge
                    + cr_store_credit)),
         cross_sales
         AS (SELECT i_product_name         product_name,
                    i_item_sk              item_sk,
                    s_store_name           store_name,
                    s_zip                  store_zip,
                    ad1.ca_street_number   b_street_number,
                    ad1.ca_street_name     b_streen_name,
                    ad1.ca_city            b_city,
                    ad1.ca_zip             b_zip,
                    ad2.ca_street_number   c_street_number,
                    ad2.ca_street_name     c_street_name,
                    ad2.ca_city            c_city,
                    ad2.ca_zip             c_zip,
                    d1.d_year              AS syear,
                    d2.d_year              AS fsyear,
                    d3.d_year              s2year,
                    Count(*)               cnt,
                    Sum(ss_wholesale_cost) s1,
                    Sum(ss_list_price)     s2,
                    Sum(ss_coupon_amt)     s3
             FROM   store_sales,
                    store_returns,
                    cs_ui,
                    date_dim d1,
                    date_dim d2,
                    date_dim d3,
                    store,
                    customer,
                    customer_demographics cd1,
                    customer_demographics cd2,
                    promotion,
                    household_demographics hd1,
                    household_demographics hd2,
                    customer_address ad1,
                    customer_address ad2,
                    income_band ib1,
                    income_band ib2,
                    item
             WHERE  ss_store_sk = s_store_sk
                    AND ss_sold_date_sk = d1.d_date_sk
                    AND ss_customer_sk = c_customer_sk
                    AND ss_cdemo_sk = cd1.cd_demo_sk
                    AND ss_hdemo_sk = hd1.hd_demo_sk
                    AND ss_addr_sk = ad1.ca_address_sk
                    AND ss_item_sk = i_item_sk
                    AND ss_item_sk = sr_item_sk
                    AND ss_ticket_number = sr_ticket_number
                    AND ss_item_sk = cs_ui.cs_item_sk
                    AND c_current_cdemo_sk = cd2.cd_demo_sk
                    AND c_current_hdemo_sk = hd2.hd_demo_sk
                    AND c_current_addr_sk = ad2.ca_address_sk
                    AND c_first_sales_date_sk = d2.d_date_sk
                    AND c_first_shipto_date_sk = d3.d_date_sk
                    AND ss_promo_sk = p_promo_sk
                    AND hd1.hd_income_band_sk = ib1.ib_income_band_sk
                    AND hd2.hd_income_band_sk = ib2.ib_income_band_sk
                    AND cd1.cd_marital_status <> cd2.cd_marital_status
                    AND i_color IN ( 'cyan', 'peach', 'blush', 'frosted',
                                     'powder', 'orange' )
                    AND i_current_price BETWEEN 58 AND 58 + 10
                    AND i_current_price BETWEEN 58 + 1 AND 58 + 15
             GROUP  BY i_product_name,
                       i_item_sk,
                       s_store_name,
                       s_zip,
                       ad1.ca_street_number,
                       ad1.ca_street_name,
                       ad1.ca_city,
                       ad1.ca_zip,
                       ad2.ca_street_number,
                       ad2.ca_street_name,
                       ad2.ca_city,
                       ad2.ca_zip,
                       d1.d_year,
                       d2.d_year,
                       d3.d_year)
    SELECT cs1.product_name,
           cs1.store_name,
           cs1.store_zip,
           cs1.b_street_number,
           cs1.b_streen_name,
           cs1.b_city,
           cs1.b_zip,
           cs1.c_street_number,
           cs1.c_street_name,
           cs1.c_city,
           cs1.c_zip,
           cs1.syear,
           cs1.cnt,
           cs1.s1,
           cs1.s2,
           cs1.s3,
           cs2.s1,
           cs2.s2,
           cs2.s3,
           cs2.syear,
           cs2.cnt
    FROM   cross_sales cs1,
           cross_sales cs2
    WHERE  cs1.item_sk = cs2.item_sk
           AND cs1.syear = 2001
           AND cs2.syear = 2001 + 1
           AND cs2.cnt <= cs1.cnt
           AND cs1.store_name = cs2.store_name
           AND cs1.store_zip = cs2.store_zip
    ORDER  BY cs1.product_name,
              cs1.store_name,
              cs2.cnt,
              cs1.s1, --added for deterministic ordering
              cs2.s1; --added for deterministic ordering
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 64."""
    # Load all required tables
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
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_demographics = get_data(
        run_config.dataset_path, "customer_demographics", run_config.suffix
    )
    promotion = get_data(run_config.dataset_path, "promotion", run_config.suffix)
    household_demographics = get_data(
        run_config.dataset_path, "household_demographics", run_config.suffix
    )
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    income_band = get_data(run_config.dataset_path, "income_band", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    # CTE 1: cs_ui - catalog sales items with low return rates
    cs_ui = (
        catalog_sales.join(
            catalog_returns,
            left_on=["cs_item_sk", "cs_order_number"],
            right_on=["cr_item_sk", "cr_order_number"],
        )
        .group_by("cs_item_sk")
        .agg(
            [
                pl.col("cs_ext_list_price").sum().alias("sale"),
                (
                    pl.col("cr_refunded_cash")
                    + pl.col("cr_reversed_charge")
                    + pl.col("cr_store_credit")
                )
                .sum()
                .alias("refund"),
            ]
        )
        .filter(pl.col("sale") > 2 * pl.col("refund"))
        .select("cs_item_sk")
    )
    # Filter items early to reduce data size
    filtered_items = item.filter(
        pl.col("i_color").is_in(
            ["cyan", "peach", "blush", "frosted", "powder", "orange"]
        )
        & pl.col("i_current_price").is_between(58, 68)
        & pl.col("i_current_price").is_between(59, 73)
    ).select(["i_item_sk", "i_product_name", "i_color", "i_current_price"])
    # Start with a much smaller base by filtering early
    store_sales_filtered = (
        store_sales.join(filtered_items, left_on="ss_item_sk", right_on="i_item_sk")
        .join(cs_ui, left_on="ss_item_sk", right_on="cs_item_sk")
        .join(
            store_returns,
            left_on=["ss_item_sk", "ss_ticket_number"],
            right_on=["sr_item_sk", "sr_ticket_number"],
        )
    )
    # Add only essential joins with immediate filtering
    cross_sales = (
        store_sales_filtered.join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .filter(
            # Filter for specific years early
            pl.col("d_year").is_in([2001, 2002])
        )
        .join(customer, left_on="ss_customer_sk", right_on="c_customer_sk")
        .join(
            customer_demographics,
            left_on="ss_cdemo_sk",
            right_on="cd_demo_sk",
            suffix="_cd1",
        )
        .join(
            customer_demographics,
            left_on="c_current_cdemo_sk",
            right_on="cd_demo_sk",
            suffix="_cd2",
        )
        .filter(
            # Apply marital status filter early
            pl.col("cd_marital_status") != pl.col("cd_marital_status_cd2")
        )
        .join(
            household_demographics,
            left_on="ss_hdemo_sk",
            right_on="hd_demo_sk",
            suffix="_hd1",
        )
        .join(
            household_demographics,
            left_on="c_current_hdemo_sk",
            right_on="hd_demo_sk",
            suffix="_hd2",
        )
        .join(
            customer_address,
            left_on="ss_addr_sk",
            right_on="ca_address_sk",
            suffix="_ad1",
        )
        .join(
            customer_address,
            left_on="c_current_addr_sk",
            right_on="ca_address_sk",
            suffix="_ad2",
        )
        .join(
            date_dim,
            left_on="c_first_sales_date_sk",
            right_on="d_date_sk",
            suffix="_d2",
        )
        .join(
            date_dim,
            left_on="c_first_shipto_date_sk",
            right_on="d_date_sk",
            suffix="_d3",
        )
        .join(promotion, left_on="ss_promo_sk", right_on="p_promo_sk")
        .join(
            income_band,
            left_on="hd_income_band_sk",
            right_on="ib_income_band_sk",
            suffix="_ib1",
        )
        .join(
            income_band,
            left_on="hd_income_band_sk_hd2",
            right_on="ib_income_band_sk",
            suffix="_ib2",
        )
        .group_by(
            [
                "i_product_name",
                "ss_item_sk",
                "s_store_name",
                "s_zip",
                "ca_street_number",
                "ca_street_name",
                "ca_city",
                "ca_zip",
                "ca_street_number_ad2",
                "ca_street_name_ad2",
                "ca_city_ad2",
                "ca_zip_ad2",
                "d_year",
                "d_year_d2",
                "d_year_d3",
            ]
        )
        .agg(
            [
                # Cast -> Int64 to match DuckDB
                pl.len().cast(pl.Int64).alias("cnt"),
                # Count and sum aggregations for postprocessing
                pl.col("ss_wholesale_cost").count().alias("s1_count"),
                pl.col("ss_wholesale_cost").sum().alias("s1_sum"),
                pl.col("ss_list_price").count().alias("s2_count"),
                pl.col("ss_list_price").sum().alias("s2_sum"),
                pl.col("ss_coupon_amt").count().alias("s3_count"),
                pl.col("ss_coupon_amt").sum().alias("s3_sum"),
            ]
        )
        .with_columns(
            [
                # Postprocessing: replace sum with null when count is 0
                pl.when(pl.col("s1_count") == 0)
                .then(None)
                .otherwise(pl.col("s1_sum"))
                .alias("s1"),
                pl.when(pl.col("s2_count") == 0)
                .then(None)
                .otherwise(pl.col("s2_sum"))
                .alias("s2"),
                pl.when(pl.col("s3_count") == 0)
                .then(None)
                .otherwise(pl.col("s3_sum"))
                .alias("s3"),
            ]
        )
        .with_columns(
            [
                pl.col("i_product_name").alias("product_name"),
                pl.col("ss_item_sk").alias("item_sk"),
                pl.col("s_store_name").alias("store_name"),
                pl.col("s_zip").alias("store_zip"),
                pl.col("ca_street_number").alias("b_street_number"),
                pl.col("ca_street_name").alias("b_streen_name"),
                pl.col("ca_city").alias("b_city"),
                pl.col("ca_zip").alias("b_zip"),
                pl.col("ca_street_number_ad2").alias("c_street_number"),
                pl.col("ca_street_name_ad2").alias("c_street_name"),
                pl.col("ca_city_ad2").alias("c_city"),
                pl.col("ca_zip_ad2").alias("c_zip"),
                pl.col("d_year").alias("syear"),
                pl.col("d_year_d2").alias("fsyear"),
                pl.col("d_year_d3").alias("s2year"),
            ]
        )
    )
    # Final query: Self-join cross_sales for consecutive years
    return (
        cross_sales.join(
            cross_sales,
            left_on=["item_sk", "store_name", "store_zip"],
            right_on=["item_sk", "store_name", "store_zip"],
            suffix="_cs2",
        )
        .filter(
            (pl.col("syear") == 2001)
            & (pl.col("syear_cs2") == 2002)
            & (pl.col("cnt_cs2") <= pl.col("cnt"))
        )
        .sort(
            ["product_name", "store_name", "cnt_cs2", "s1", "s1_cs2"],
            nulls_last=True,
            descending=[False, False, False, False, False],
        )
        .select(
            [
                "product_name",
                "store_name",
                "store_zip",
                "b_street_number",
                "b_streen_name",
                "b_city",
                "b_zip",
                "c_street_number",
                "c_street_name",
                "c_city",
                "c_zip",
                "syear",
                "cnt",
                "s1",
                "s2",
                "s3",
                pl.col("s1_cs2").alias("s1_1"),
                pl.col("s2_cs2").alias("s2_1"),
                pl.col("s3_cs2").alias("s3_1"),
                pl.col("syear_cs2").alias("syear_1"),
                # Cast -> Int64 to match DuckDB
                pl.col("cnt_cs2").cast(pl.Int64).alias("cnt_1"),
            ]
        )
    )
