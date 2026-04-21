# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""TPC-DS q64 — naive one-for-one Polars translation of the SQL."""

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
    """TPC-DS q64 naive Polars implementation."""
    params = load_parameters(
        int(run_config.scale_factor), query_id=64, qualification=run_config.qualification
    )
    year = params["year"]
    price = params["price"]
    colors = params["colors"]

    catalog_sales = get_data(run_config.dataset_path, "catalog_sales", run_config.suffix)
    catalog_returns = get_data(run_config.dataset_path, "catalog_returns", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(run_config.dataset_path, "store_returns", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_demographics = get_data(run_config.dataset_path, "customer_demographics", run_config.suffix)
    customer_address = get_data(run_config.dataset_path, "customer_address", run_config.suffix)
    household_demographics = get_data(run_config.dataset_path, "household_demographics", run_config.suffix)
    income_band = get_data(run_config.dataset_path, "income_band", run_config.suffix)
    promotion = get_data(run_config.dataset_path, "promotion", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # cs_ui CTE:
    # FROM catalog_sales, catalog_returns
    # WHERE cs_item_sk = cr_item_sk AND cs_order_number = cr_order_number
    # GROUP BY cs_item_sk
    # HAVING Sum(cs_ext_list_price) > 2 * Sum(cr_refunded_cash + cr_reversed_charge + cr_store_credit)
    cs_ui = (
        catalog_sales.join(
            catalog_returns,
            left_on=["cs_item_sk", "cs_order_number"],
            right_on=["cr_item_sk", "cr_order_number"],
            how="inner",
        )
        .group_by("cs_item_sk")
        .agg(
            sql_sum(pl.col("cs_ext_list_price")).alias("sale"),
            sql_sum(
                pl.col("cr_refunded_cash") + pl.col("cr_reversed_charge") + pl.col("cr_store_credit")
            ).alias("refund"),
        )
        .filter(pl.col("sale") > 2 * pl.col("refund"))
        .select("cs_item_sk")
    )

    # Prepare aliased copies of tables used multiple times
    # date_dim: d1 (sold date), d2 (first sales date), d3 (first shipto date)
    d1 = date_dim.select(
        pl.col("d_date_sk").alias("d1_date_sk"),
        pl.col("d_year").alias("d1_year"),
    )
    d2 = date_dim.select(
        pl.col("d_date_sk").alias("d2_date_sk"),
        pl.col("d_year").alias("d2_year"),
    )
    d3 = date_dim.select(
        pl.col("d_date_sk").alias("d3_date_sk"),
        pl.col("d_year").alias("d3_year"),
    )

    # customer_demographics: cd1 (sale demo), cd2 (current demo)
    cd1 = customer_demographics.select(
        pl.col("cd_demo_sk").alias("cd1_demo_sk"),
        pl.col("cd_marital_status").alias("cd1_marital_status"),
    )
    cd2 = customer_demographics.select(
        pl.col("cd_demo_sk").alias("cd2_demo_sk"),
        pl.col("cd_marital_status").alias("cd2_marital_status"),
    )

    # household_demographics: hd1 (sale hdemo), hd2 (current hdemo)
    hd1 = household_demographics.select(
        pl.col("hd_demo_sk").alias("hd1_demo_sk"),
        pl.col("hd_income_band_sk").alias("hd1_income_band_sk"),
    )
    hd2 = household_demographics.select(
        pl.col("hd_demo_sk").alias("hd2_demo_sk"),
        pl.col("hd_income_band_sk").alias("hd2_income_band_sk"),
    )

    # income_band: ib1, ib2
    ib1 = income_band.select(pl.col("ib_income_band_sk").alias("ib1_income_band_sk"))
    ib2 = income_band.select(pl.col("ib_income_band_sk").alias("ib2_income_band_sk"))

    # customer_address: ad1 (sale address), ad2 (current address)
    ad1 = customer_address.select(
        pl.col("ca_address_sk").alias("ad1_address_sk"),
        pl.col("ca_street_number").alias("b_street_number"),
        pl.col("ca_street_name").alias("b_streen_name"),
        pl.col("ca_city").alias("b_city"),
        pl.col("ca_zip").alias("b_zip"),
    )
    ad2 = customer_address.select(
        pl.col("ca_address_sk").alias("ad2_address_sk"),
        pl.col("ca_street_number").alias("c_street_number"),
        pl.col("ca_street_name").alias("c_street_name"),
        pl.col("ca_city").alias("c_city"),
        pl.col("ca_zip").alias("c_zip"),
    )

    # cross_sales CTE:
    # FROM store_sales, store_returns, cs_ui, d1, d2, d3,
    #      store, customer, cd1, cd2, promotion, hd1, hd2, ad1, ad2, ib1, ib2, item
    # WHERE (all the equality join conditions in the WHERE clause)
    #   AND cd1.cd_marital_status <> cd2.cd_marital_status
    #   AND i_color IN (colors)
    #   AND i_current_price BETWEEN price AND price+10
    #   AND i_current_price BETWEEN price+1 AND price+15
    # GROUP BY (all non-agg selected columns)
    cross_sales = (
        store_sales
        .join(store_returns,
              left_on=["ss_item_sk", "ss_ticket_number"],
              right_on=["sr_item_sk", "sr_ticket_number"],
              how="inner")
        .join(cs_ui, left_on="ss_item_sk", right_on="cs_item_sk", how="inner")
        .join(d1, left_on="ss_sold_date_sk", right_on="d1_date_sk", how="inner")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk", how="inner")
        .join(customer, left_on="ss_customer_sk", right_on="c_customer_sk", how="inner")
        .join(cd1, left_on="ss_cdemo_sk", right_on="cd1_demo_sk", how="inner")
        .join(cd2, left_on="c_current_cdemo_sk", right_on="cd2_demo_sk", how="inner")
        .join(hd1, left_on="ss_hdemo_sk", right_on="hd1_demo_sk", how="inner")
        .join(hd2, left_on="c_current_hdemo_sk", right_on="hd2_demo_sk", how="inner")
        .join(ad1, left_on="ss_addr_sk", right_on="ad1_address_sk", how="inner")
        .join(ad2, left_on="c_current_addr_sk", right_on="ad2_address_sk", how="inner")
        .join(d2, left_on="c_first_sales_date_sk", right_on="d2_date_sk", how="inner")
        .join(d3, left_on="c_first_shipto_date_sk", right_on="d3_date_sk", how="inner")
        .join(promotion, left_on="ss_promo_sk", right_on="p_promo_sk", how="inner")
        .join(ib1, left_on="hd1_income_band_sk", right_on="ib1_income_band_sk", how="inner")
        .join(ib2, left_on="hd2_income_band_sk", right_on="ib2_income_band_sk", how="inner")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk", how="inner")
        .filter(
            (pl.col("cd1_marital_status") != pl.col("cd2_marital_status"))
            & pl.col("i_color").is_in(colors)
            & pl.col("i_current_price").is_between(price, price + 10)
            & pl.col("i_current_price").is_between(price + 1, price + 15)
        )
        .group_by([
            "i_product_name",
            "ss_item_sk",
            "s_store_name",
            "s_zip",
            "b_street_number",
            "b_streen_name",
            "b_city",
            "b_zip",
            "c_street_number",
            "c_street_name",
            "c_city",
            "c_zip",
            "d1_year",
            "d2_year",
            "d3_year",
        ])
        .agg(
            pl.len().alias("cnt"),
            sql_sum(pl.col("ss_wholesale_cost")).alias("s1"),
            sql_sum(pl.col("ss_list_price")).alias("s2"),
            sql_sum(pl.col("ss_coupon_amt")).alias("s3"),
        )
        .select(
            pl.col("i_product_name").alias("product_name"),
            pl.col("ss_item_sk").alias("item_sk"),
            pl.col("s_store_name").alias("store_name"),
            pl.col("s_zip").alias("store_zip"),
            "b_street_number",
            "b_streen_name",
            "b_city",
            "b_zip",
            "c_street_number",
            "c_street_name",
            "c_city",
            "c_zip",
            pl.col("d1_year").alias("syear"),
            pl.col("d2_year").alias("fsyear"),
            pl.col("d3_year").alias("s2year"),
            "cnt",
            "s1",
            "s2",
            "s3",
        )
    )

    # Final:
    # FROM cross_sales cs1, cross_sales cs2
    # WHERE cs1.item_sk = cs2.item_sk AND cs1.syear = year AND cs2.syear = year+1
    #   AND cs2.cnt <= cs1.cnt AND cs1.store_name = cs2.store_name AND cs1.store_zip = cs2.store_zip
    # SELECT cs1.product_name, cs1.store_name, cs1.store_zip, ... cs2.s1, cs2.s2, cs2.s3, cs2.syear, cs2.cnt
    cs1 = cross_sales.filter(pl.col("syear") == year)
    cs2 = cross_sales.filter(pl.col("syear") == year + 1)

    result = (
        cs1.join(
            cs2,
            left_on=["item_sk", "store_name", "store_zip"],
            right_on=["item_sk", "store_name", "store_zip"],
            how="inner",
            suffix="_cs2",
        )
        .filter(pl.col("cnt_cs2") <= pl.col("cnt"))
        .select(
            pl.col("product_name"),
            pl.col("store_name"),
            pl.col("store_zip"),
            pl.col("b_street_number"),
            pl.col("b_streen_name"),
            pl.col("b_city"),
            pl.col("b_zip"),
            pl.col("c_street_number"),
            pl.col("c_street_name"),
            pl.col("c_city"),
            pl.col("c_zip"),
            pl.col("syear"),
            pl.col("cnt"),
            pl.col("s1"),
            pl.col("s2"),
            pl.col("s3"),
            pl.col("s1_cs2"),
            pl.col("s2_cs2"),
            pl.col("s3_cs2"),
            pl.col("syear_cs2"),
            pl.col("cnt_cs2"),
        )
    )

    return QueryResult(
        frame=result,
        sort_by=[
            ("product_name", False),
            ("store_name", False),
            ("cnt_cs2", False),
            ("s1", False),
            ("s1_cs2", False),
        ],
        limit=None,
    )
