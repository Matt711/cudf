# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 74."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 74."""
    return """
    WITH year_total
         AS (SELECT c_customer_id    customer_id,
                    c_first_name     customer_first_name,
                    c_last_name      customer_last_name,
                    d_year           AS year1,
                    Sum(ss_net_paid) year_total,
                    's'              sale_type
             FROM   customer,
                    store_sales,
                    date_dim
             WHERE  c_customer_sk = ss_customer_sk
                    AND ss_sold_date_sk = d_date_sk
                    AND d_year IN ( 1999, 1999 + 1 )
             GROUP  BY c_customer_id,
                       c_first_name,
                       c_last_name,
                       d_year
             UNION ALL
             SELECT c_customer_id    customer_id,
                    c_first_name     customer_first_name,
                    c_last_name      customer_last_name,
                    d_year           AS year1,
                    Sum(ws_net_paid) year_total,
                    'w'              sale_type
             FROM   customer,
                    web_sales,
                    date_dim
             WHERE  c_customer_sk = ws_bill_customer_sk
                    AND ws_sold_date_sk = d_date_sk
                    AND d_year IN ( 1999, 1999 + 1 )
             GROUP  BY c_customer_id,
                       c_first_name,
                       c_last_name,
                       d_year)
    SELECT t_s_secyear.customer_id,
                   t_s_secyear.customer_first_name,
                   t_s_secyear.customer_last_name
    FROM   year_total t_s_firstyear,
           year_total t_s_secyear,
           year_total t_w_firstyear,
           year_total t_w_secyear
    WHERE  t_s_secyear.customer_id = t_s_firstyear.customer_id
           AND t_s_firstyear.customer_id = t_w_secyear.customer_id
           AND t_s_firstyear.customer_id = t_w_firstyear.customer_id
           AND t_s_firstyear.sale_type = 's'
           AND t_w_firstyear.sale_type = 'w'
           AND t_s_secyear.sale_type = 's'
           AND t_w_secyear.sale_type = 'w'
           AND t_s_firstyear.year1 = 1999
           AND t_s_secyear.year1 = 1999 + 1
           AND t_w_firstyear.year1 = 1999
           AND t_w_secyear.year1 = 1999 + 1
           AND t_s_firstyear.year_total > 0
           AND t_w_firstyear.year_total > 0
           AND CASE
                 WHEN t_w_firstyear.year_total > 0 THEN t_w_secyear.year_total /
                                                        t_w_firstyear.year_total
                 ELSE NULL
               END > CASE
                       WHEN t_s_firstyear.year_total > 0 THEN
                       t_s_secyear.year_total /
                       t_s_firstyear.year_total
                       ELSE NULL
                     END
    ORDER  BY 1,
              2,
              3
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 74."""
    # Load required tables using scan_parquet for lazy evaluation
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    # Filter date_dim for 1999 and 2000
    filtered_dates = date_dim.filter(pl.col("d_year").is_in([1999, 2000])).select(
        ["d_date_sk", "d_year"]
    )
    # Create year_total CTE equivalent - Store sales component
    # Start with filtered sales data to reduce size early
    store_component = (
        store_sales.join(
            filtered_dates, left_on="ss_sold_date_sk", right_on="d_date_sk"
        )
        .join(
            customer.select(
                ["c_customer_sk", "c_customer_id", "c_first_name", "c_last_name"]
            ),
            left_on="ss_customer_sk",
            right_on="c_customer_sk",
        )
        .group_by(["c_customer_id", "c_first_name", "c_last_name", "d_year"])
        .agg(
            [
                pl.col("ss_net_paid").count().alias("count_paid"),
                pl.col("ss_net_paid").sum().alias("sum_paid"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("count_paid") == 0)
                .then(None)
                .otherwise(pl.col("sum_paid"))
                .alias("year_total"),
                pl.col("d_year").alias("year1"),
                pl.lit("s").alias("sale_type"),
            ]
        )
        .select(
            [
                pl.col("c_customer_id").alias("customer_id"),
                pl.col("c_first_name").alias("customer_first_name"),
                pl.col("c_last_name").alias("customer_last_name"),
                "year1",
                "year_total",
                "sale_type",
            ]
        )
    )
    # Create year_total CTE equivalent - Web sales component
    # Start with filtered sales data to reduce size early
    web_component = (
        web_sales.join(filtered_dates, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(
            customer.select(
                ["c_customer_sk", "c_customer_id", "c_first_name", "c_last_name"]
            ),
            left_on="ws_bill_customer_sk",
            right_on="c_customer_sk",
        )
        .group_by(["c_customer_id", "c_first_name", "c_last_name", "d_year"])
        .agg(
            [
                pl.col("ws_net_paid").count().alias("count_paid"),
                pl.col("ws_net_paid").sum().alias("sum_paid"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("count_paid") == 0)
                .then(None)
                .otherwise(pl.col("sum_paid"))
                .alias("year_total"),
                pl.col("d_year").alias("year1"),
                pl.lit("w").alias("sale_type"),
            ]
        )
        .select(
            [
                pl.col("c_customer_id").alias("customer_id"),
                pl.col("c_first_name").alias("customer_first_name"),
                pl.col("c_last_name").alias("customer_last_name"),
                "year1",
                "year_total",
                "sale_type",
            ]
        )
    )
    # Combine store and web components (UNION ALL)
    year_total = pl.concat([store_component, web_component])
    # Create the four filtered views for self-joins
    t_s_firstyear = year_total.filter(
        (pl.col("sale_type") == "s") & (pl.col("year1") == 1999)
    ).select(
        [
            pl.col("customer_id").alias("s_first_customer_id"),
            pl.col("year_total").alias("s_first_total"),
        ]
    )
    t_s_secyear = year_total.filter(
        (pl.col("sale_type") == "s") & (pl.col("year1") == 2000)
    ).select(
        [
            pl.col("customer_id").alias("s_sec_customer_id"),
            pl.col("customer_first_name").alias("customer_first_name"),
            pl.col("customer_last_name").alias("customer_last_name"),
            pl.col("year_total").alias("s_sec_total"),
        ]
    )
    t_w_firstyear = year_total.filter(
        (pl.col("sale_type") == "w") & (pl.col("year1") == 1999)
    ).select(
        [
            pl.col("customer_id").alias("w_first_customer_id"),
            pl.col("year_total").alias("w_first_total"),
        ]
    )
    t_w_secyear = year_total.filter(
        (pl.col("sale_type") == "w") & (pl.col("year1") == 2000)
    ).select(
        [
            pl.col("customer_id").alias("w_sec_customer_id"),
            pl.col("year_total").alias("w_sec_total"),
        ]
    )
    # Main query: join all four views and apply conditions
    return (
        t_s_secyear.join(
            t_s_firstyear, left_on="s_sec_customer_id", right_on="s_first_customer_id"
        )
        .join(
            t_w_firstyear, left_on="s_sec_customer_id", right_on="w_first_customer_id"
        )
        .join(t_w_secyear, left_on="s_sec_customer_id", right_on="w_sec_customer_id")
        # Apply filtering conditions
        .filter(
            (pl.col("s_first_total") > 0)
            & (pl.col("w_first_total") > 0)
            &
            # Web growth ratio > Store growth ratio
            (
                pl.when(pl.col("w_first_total") > 0)
                .then(pl.col("w_sec_total") / pl.col("w_first_total"))
                .otherwise(None)
                > pl.when(pl.col("s_first_total") > 0)
                .then(pl.col("s_sec_total") / pl.col("s_first_total"))
                .otherwise(None)
            )
        )
        # Select final columns
        .select(
            [
                pl.col("s_sec_customer_id").alias("customer_id"),
                "customer_first_name",
                "customer_last_name",
            ]
        )
        # Sort by customer_id, first_name, last_name
        .sort(
            ["customer_id", "customer_first_name", "customer_last_name"],
            nulls_last=True,
        )
        .limit(100)
    )
