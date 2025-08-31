# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 24."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig

S_MARKET_ID = 2
I_COLOR1 = "turquoise"
I_COLOR2 = "smoke"


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 24."""
    return f"""
    WITH ssales
        AS (SELECT c_last_name,
                    c_first_name,
                    s_store_name,
                    ca_state,
                    s_state,
                    i_color,
                    i_current_price,
                    i_manager_id,
                    i_units,
                    i_size,
                    Sum(ss_net_profit) netpaid
            FROM   store_sales,
                    store_returns,
                    store,
                    item,
                    customer,
                    customer_address
            WHERE  ss_ticket_number = sr_ticket_number
                    AND ss_item_sk = sr_item_sk
                    AND ss_customer_sk = c_customer_sk
                    AND ss_item_sk = i_item_sk
                    AND ss_store_sk = s_store_sk
                    AND c_birth_country = Upper(ca_country)
                    AND s_zip = ca_zip
                    AND s_market_id = {S_MARKET_ID}
            GROUP  BY c_last_name,
                    c_first_name,
                    s_store_name,
                    ca_state,
                    s_state,
                    i_color,
                    i_current_price,
                    i_manager_id,
                    i_units,
                    i_size)
    SELECT c_last_name,
        c_first_name,
        s_store_name,
        Sum(netpaid) paid
    FROM   ssales
    WHERE  i_color = '{I_COLOR1}'
    GROUP  BY c_last_name,
            c_first_name,
            s_store_name
    HAVING Sum(netpaid) > (SELECT 0.05 * Avg(netpaid)
                        FROM   ssales);

    WITH ssales
        AS (SELECT c_last_name,
                    c_first_name,
                    s_store_name,
                    ca_state,
                    s_state,
                    i_color,
                    i_current_price,
                    i_manager_id,
                    i_units,
                    i_size,
                    Sum(ss_net_profit) netpaid
            FROM   store_sales,
                    store_returns,
                    store,
                    item,
                    customer,
                    customer_address
            WHERE  ss_ticket_number = sr_ticket_number
                    AND ss_item_sk = sr_item_sk
                    AND ss_customer_sk = c_customer_sk
                    AND ss_item_sk = i_item_sk
                    AND ss_store_sk = s_store_sk
                    AND c_birth_country = Upper(ca_country)
                    AND s_zip = ca_zip
                    AND s_market_id = {S_MARKET_ID}
            GROUP  BY c_last_name,
                    c_first_name,
                    s_store_name,
                    ca_state,
                    s_state,
                    i_color,
                    i_current_price,
                    i_manager_id,
                    i_units,
                    i_size)
    SELECT c_last_name,
        c_first_name,
        s_store_name,
        Sum(netpaid) paid
    FROM   ssales
    WHERE  i_color = '{I_COLOR2}'
    GROUP  BY c_last_name,
            c_first_name,
            s_store_name
    HAVING Sum(netpaid) > (SELECT 0.05 * Avg(netpaid)
                        FROM   ssales)

    ORDER BY c_last_name, c_first_name, s_store_name; -- added for consistent ordering

    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 24."""
    # Load tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    store_returns = get_data(
        run_config.dataset_path, "store_returns", run_config.suffix
    )
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    item = get_data(run_config.dataset_path, "item", run_config.suffix)
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )

    # CTE: ssales
    ssales = (
        store_sales.join(
            store_returns,
            left_on=["ss_ticket_number", "ss_item_sk"],
            right_on=["sr_ticket_number", "sr_item_sk"],
        )
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(item, left_on="ss_item_sk", right_on="i_item_sk")
        .join(customer, left_on="ss_customer_sk", right_on="c_customer_sk")
        .join(
            customer_address.with_columns(
                pl.col("ca_country").str.to_uppercase().alias("ca_country_upper")
            ),
            left_on=["c_birth_country", "s_zip"],
            right_on=["ca_country_upper", "ca_zip"],
            how="inner",
        )
        .filter(pl.col("s_market_id") == S_MARKET_ID)
        .group_by(
            [
                "c_last_name",
                "c_first_name",
                "s_store_name",
                "ca_state",
                "s_state",
                "i_color",
                "i_current_price",
                "i_manager_id",
                "i_units",
                "i_size",
            ]
        )
        .agg(
            [
                pl.col("ss_net_profit").count().alias("ss_net_profit_count"),
                pl.col("ss_net_profit").sum().alias("ss_net_profit_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("ss_net_profit_count") == 0)
                .then(None)
                .otherwise(pl.col("ss_net_profit_sum"))
                .alias("netpaid")
            ]
        )
        .drop(["ss_net_profit_count", "ss_net_profit_sum"])
    )

    # Calculate average netpaid for threshold
    threshold_table = (
        ssales.select(pl.col("netpaid").mean().alias("avg_netpaid"))
        .with_columns((pl.col("avg_netpaid") * 0.05).alias("threshold"))
        .select("threshold")
    )

    # Query 1: first color (discarded?)
    _ = (
        ssales.filter(pl.col("i_color") == I_COLOR1)
        .group_by(["c_last_name", "c_first_name", "s_store_name"])
        .agg(
            [
                pl.col("netpaid").count().alias("netpaid_count"),
                pl.col("netpaid").sum().alias("netpaid_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("netpaid_count") == 0)
                .then(None)
                .otherwise(pl.col("netpaid_sum"))
                .alias("paid")
            ]
        )
        .join(threshold_table, how="cross")
        .filter(pl.col("paid") > pl.col("threshold"))
        .select(["c_last_name", "c_first_name", "s_store_name", "paid"])
    )

    # Query 2: second color
    color2_result = (
        ssales.filter(pl.col("i_color") == I_COLOR2)
        .group_by(["c_last_name", "c_first_name", "s_store_name"])
        .agg(
            [
                pl.col("netpaid").count().alias("netpaid_count"),
                pl.col("netpaid").sum().alias("netpaid_sum"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("netpaid_count") == 0)
                .then(None)
                .otherwise(pl.col("netpaid_sum"))
                .alias("paid")
            ]
        )
        .join(threshold_table, how="cross")
        .filter(pl.col("paid") > pl.col("threshold"))
        .select(["c_last_name", "c_first_name", "s_store_name", "paid"])
    )

    # SQL has two separate queries
    # TODO: Should we concatenate both results? Why does does DuckDB discard the result from the first subquery?
    return color2_result.sort(
        ["c_last_name", "c_first_name", "s_store_name"], nulls_last=True
    )
