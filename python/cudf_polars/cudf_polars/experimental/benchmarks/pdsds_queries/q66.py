# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 66."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 66."""
    return """
    SELECT w_warehouse_name,
                   w_warehouse_sq_ft,
                   w_city,
                   w_county,
                   w_state,
                   w_country,
                   ship_carriers,
                   year1,
                   Sum(jan_sales)                     AS jan_sales,
                   Sum(feb_sales)                     AS feb_sales,
                   Sum(mar_sales)                     AS mar_sales,
                   Sum(apr_sales)                     AS apr_sales,
                   Sum(may_sales)                     AS may_sales,
                   Sum(jun_sales)                     AS jun_sales,
                   Sum(jul_sales)                     AS jul_sales,
                   Sum(aug_sales)                     AS aug_sales,
                   Sum(sep_sales)                     AS sep_sales,
                   Sum(oct_sales)                     AS oct_sales,
                   Sum(nov_sales)                     AS nov_sales,
                   Sum(dec_sales)                     AS dec_sales,
                   Sum(jan_sales / w_warehouse_sq_ft) AS jan_sales_per_sq_foot,
                   Sum(feb_sales / w_warehouse_sq_ft) AS feb_sales_per_sq_foot,
                   Sum(mar_sales / w_warehouse_sq_ft) AS mar_sales_per_sq_foot,
                   Sum(apr_sales / w_warehouse_sq_ft) AS apr_sales_per_sq_foot,
                   Sum(may_sales / w_warehouse_sq_ft) AS may_sales_per_sq_foot,
                   Sum(jun_sales / w_warehouse_sq_ft) AS jun_sales_per_sq_foot,
                   Sum(jul_sales / w_warehouse_sq_ft) AS jul_sales_per_sq_foot,
                   Sum(aug_sales / w_warehouse_sq_ft) AS aug_sales_per_sq_foot,
                   Sum(sep_sales / w_warehouse_sq_ft) AS sep_sales_per_sq_foot,
                   Sum(oct_sales / w_warehouse_sq_ft) AS oct_sales_per_sq_foot,
                   Sum(nov_sales / w_warehouse_sq_ft) AS nov_sales_per_sq_foot,
                   Sum(dec_sales / w_warehouse_sq_ft) AS dec_sales_per_sq_foot,
                   Sum(jan_net)                       AS jan_net,
                   Sum(feb_net)                       AS feb_net,
                   Sum(mar_net)                       AS mar_net,
                   Sum(apr_net)                       AS apr_net,
                   Sum(may_net)                       AS may_net,
                   Sum(jun_net)                       AS jun_net,
                   Sum(jul_net)                       AS jul_net,
                   Sum(aug_net)                       AS aug_net,
                   Sum(sep_net)                       AS sep_net,
                   Sum(oct_net)                       AS oct_net,
                   Sum(nov_net)                       AS nov_net,
                   Sum(dec_net)                       AS dec_net
    FROM   (SELECT w_warehouse_name,
                   w_warehouse_sq_ft,
                   w_city,
                   w_county,
                   w_state,
                   w_country,
                   'ZOUROS'
                   || ','
                   || 'ZHOU' AS ship_carriers,
                   d_year    AS year1,
                   Sum(CASE
                         WHEN d_moy = 1 THEN ws_ext_sales_price * ws_quantity
                         ELSE 0
                       END)  AS jan_sales,
                   Sum(CASE
                         WHEN d_moy = 2 THEN ws_ext_sales_price * ws_quantity
                         ELSE 0
                       END)  AS feb_sales,
                   Sum(CASE
                         WHEN d_moy = 3 THEN ws_ext_sales_price * ws_quantity
                         ELSE 0
                       END)  AS mar_sales,
                   Sum(CASE
                         WHEN d_moy = 4 THEN ws_ext_sales_price * ws_quantity
                         ELSE 0
                       END)  AS apr_sales,
                   Sum(CASE
                         WHEN d_moy = 5 THEN ws_ext_sales_price * ws_quantity
                         ELSE 0
                       END)  AS may_sales,
                   Sum(CASE
                         WHEN d_moy = 6 THEN ws_ext_sales_price * ws_quantity
                         ELSE 0
                       END)  AS jun_sales,
                   Sum(CASE
                         WHEN d_moy = 7 THEN ws_ext_sales_price * ws_quantity
                         ELSE 0
                       END)  AS jul_sales,
                   Sum(CASE
                         WHEN d_moy = 8 THEN ws_ext_sales_price * ws_quantity
                         ELSE 0
                       END)  AS aug_sales,
                   Sum(CASE
                         WHEN d_moy = 9 THEN ws_ext_sales_price * ws_quantity
                         ELSE 0
                       END)  AS sep_sales,
                   Sum(CASE
                         WHEN d_moy = 10 THEN ws_ext_sales_price * ws_quantity
                         ELSE 0
                       END)  AS oct_sales,
                   Sum(CASE
                         WHEN d_moy = 11 THEN ws_ext_sales_price * ws_quantity
                         ELSE 0
                       END)  AS nov_sales,
                   Sum(CASE
                         WHEN d_moy = 12 THEN ws_ext_sales_price * ws_quantity
                         ELSE 0
                       END)  AS dec_sales,
                   Sum(CASE
                         WHEN d_moy = 1 THEN ws_net_paid_inc_ship * ws_quantity
                         ELSE 0
                       END)  AS jan_net,
                   Sum(CASE
                         WHEN d_moy = 2 THEN ws_net_paid_inc_ship * ws_quantity
                         ELSE 0
                       END)  AS feb_net,
                   Sum(CASE
                         WHEN d_moy = 3 THEN ws_net_paid_inc_ship * ws_quantity
                         ELSE 0
                       END)  AS mar_net,
                   Sum(CASE
                         WHEN d_moy = 4 THEN ws_net_paid_inc_ship * ws_quantity
                         ELSE 0
                       END)  AS apr_net,
                   Sum(CASE
                         WHEN d_moy = 5 THEN ws_net_paid_inc_ship * ws_quantity
                         ELSE 0
                       END)  AS may_net,
                   Sum(CASE
                         WHEN d_moy = 6 THEN ws_net_paid_inc_ship * ws_quantity
                         ELSE 0
                       END)  AS jun_net,
                   Sum(CASE
                         WHEN d_moy = 7 THEN ws_net_paid_inc_ship * ws_quantity
                         ELSE 0
                       END)  AS jul_net,
                   Sum(CASE
                         WHEN d_moy = 8 THEN ws_net_paid_inc_ship * ws_quantity
                         ELSE 0
                       END)  AS aug_net,
                   Sum(CASE
                         WHEN d_moy = 9 THEN ws_net_paid_inc_ship * ws_quantity
                         ELSE 0
                       END)  AS sep_net,
                   Sum(CASE
                         WHEN d_moy = 10 THEN ws_net_paid_inc_ship * ws_quantity
                         ELSE 0
                       END)  AS oct_net,
                   Sum(CASE
                         WHEN d_moy = 11 THEN ws_net_paid_inc_ship * ws_quantity
                         ELSE 0
                       END)  AS nov_net,
                   Sum(CASE
                         WHEN d_moy = 12 THEN ws_net_paid_inc_ship * ws_quantity
                         ELSE 0
                       END)  AS dec_net
            FROM   web_sales,
                   warehouse,
                   date_dim,
                   time_dim,
                   ship_mode
            WHERE  ws_warehouse_sk = w_warehouse_sk
                   AND ws_sold_date_sk = d_date_sk
                   AND ws_sold_time_sk = t_time_sk
                   AND ws_ship_mode_sk = sm_ship_mode_sk
                   AND d_year = 1998
                   AND t_time BETWEEN 7249 AND 7249 + 28800
                   AND sm_carrier IN ( 'ZOUROS', 'ZHOU' )
            GROUP  BY w_warehouse_name,
                      w_warehouse_sq_ft,
                      w_city,
                      w_county,
                      w_state,
                      w_country,
                      d_year
            UNION ALL
            SELECT w_warehouse_name,
                   w_warehouse_sq_ft,
                   w_city,
                   w_county,
                   w_state,
                   w_country,
                   'ZOUROS'
                   || ','
                   || 'ZHOU' AS ship_carriers,
                   d_year    AS year1,
                   Sum(CASE
                         WHEN d_moy = 1 THEN cs_ext_sales_price * cs_quantity
                         ELSE 0
                       END)  AS jan_sales,
                   Sum(CASE
                         WHEN d_moy = 2 THEN cs_ext_sales_price * cs_quantity
                         ELSE 0
                       END)  AS feb_sales,
                   Sum(CASE
                         WHEN d_moy = 3 THEN cs_ext_sales_price * cs_quantity
                         ELSE 0
                       END)  AS mar_sales,
                   Sum(CASE
                         WHEN d_moy = 4 THEN cs_ext_sales_price * cs_quantity
                         ELSE 0
                       END)  AS apr_sales,
                   Sum(CASE
                         WHEN d_moy = 5 THEN cs_ext_sales_price * cs_quantity
                         ELSE 0
                       END)  AS may_sales,
                   Sum(CASE
                         WHEN d_moy = 6 THEN cs_ext_sales_price * cs_quantity
                         ELSE 0
                       END)  AS jun_sales,
                   Sum(CASE
                         WHEN d_moy = 7 THEN cs_ext_sales_price * cs_quantity
                         ELSE 0
                       END)  AS jul_sales,
                   Sum(CASE
                         WHEN d_moy = 8 THEN cs_ext_sales_price * cs_quantity
                         ELSE 0
                       END)  AS aug_sales,
                   Sum(CASE
                         WHEN d_moy = 9 THEN cs_ext_sales_price * cs_quantity
                         ELSE 0
                       END)  AS sep_sales,
                   Sum(CASE
                         WHEN d_moy = 10 THEN cs_ext_sales_price * cs_quantity
                         ELSE 0
                       END)  AS oct_sales,
                   Sum(CASE
                         WHEN d_moy = 11 THEN cs_ext_sales_price * cs_quantity
                         ELSE 0
                       END)  AS nov_sales,
                   Sum(CASE
                         WHEN d_moy = 12 THEN cs_ext_sales_price * cs_quantity
                         ELSE 0
                       END)  AS dec_sales,
                   Sum(CASE
                         WHEN d_moy = 1 THEN cs_net_paid * cs_quantity
                         ELSE 0
                       END)  AS jan_net,
                   Sum(CASE
                         WHEN d_moy = 2 THEN cs_net_paid * cs_quantity
                         ELSE 0
                       END)  AS feb_net,
                   Sum(CASE
                         WHEN d_moy = 3 THEN cs_net_paid * cs_quantity
                         ELSE 0
                       END)  AS mar_net,
                   Sum(CASE
                         WHEN d_moy = 4 THEN cs_net_paid * cs_quantity
                         ELSE 0
                       END)  AS apr_net,
                   Sum(CASE
                         WHEN d_moy = 5 THEN cs_net_paid * cs_quantity
                         ELSE 0
                       END)  AS may_net,
                   Sum(CASE
                         WHEN d_moy = 6 THEN cs_net_paid * cs_quantity
                         ELSE 0
                       END)  AS jun_net,
                   Sum(CASE
                         WHEN d_moy = 7 THEN cs_net_paid * cs_quantity
                         ELSE 0
                       END)  AS jul_net,
                   Sum(CASE
                         WHEN d_moy = 8 THEN cs_net_paid * cs_quantity
                         ELSE 0
                       END)  AS aug_net,
                   Sum(CASE
                         WHEN d_moy = 9 THEN cs_net_paid * cs_quantity
                         ELSE 0
                       END)  AS sep_net,
                   Sum(CASE
                         WHEN d_moy = 10 THEN cs_net_paid * cs_quantity
                         ELSE 0
                       END)  AS oct_net,
                   Sum(CASE
                         WHEN d_moy = 11 THEN cs_net_paid * cs_quantity
                         ELSE 0
                       END)  AS nov_net,
                   Sum(CASE
                         WHEN d_moy = 12 THEN cs_net_paid * cs_quantity
                         ELSE 0
                       END)  AS dec_net
            FROM   catalog_sales,
                   warehouse,
                   date_dim,
                   time_dim,
                   ship_mode
            WHERE  cs_warehouse_sk = w_warehouse_sk
                   AND cs_sold_date_sk = d_date_sk
                   AND cs_sold_time_sk = t_time_sk
                   AND cs_ship_mode_sk = sm_ship_mode_sk
                   AND d_year = 1998
                   AND t_time BETWEEN 7249 AND 7249 + 28800
                   AND sm_carrier IN ( 'ZOUROS', 'ZHOU' )
            GROUP  BY w_warehouse_name,
                      w_warehouse_sq_ft,
                      w_city,
                      w_county,
                      w_state,
                      w_country,
                      d_year) x
    GROUP  BY w_warehouse_name,
              w_warehouse_sq_ft,
              w_city,
              w_county,
              w_state,
              w_country,
              ship_carriers,
              year1
    ORDER  BY w_warehouse_name
    LIMIT 100;
    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 66."""
    # Load tables
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    catalog_sales = get_data(
        run_config.dataset_path, "catalog_sales", run_config.suffix
    )
    warehouse = get_data(run_config.dataset_path, "warehouse", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    time_dim = get_data(run_config.dataset_path, "time_dim", run_config.suffix)
    ship_mode = get_data(run_config.dataset_path, "ship_mode", run_config.suffix)
    # Web sales part of UNION ALL
    web_sales_monthly = (
        web_sales.join(warehouse, left_on="ws_warehouse_sk", right_on="w_warehouse_sk")
        .join(date_dim, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .join(time_dim, left_on="ws_sold_time_sk", right_on="t_time_sk")
        .join(ship_mode, left_on="ws_ship_mode_sk", right_on="sm_ship_mode_sk")
        # TODO: There's some bug in cudf_polars with FILTER on Column with NULLs
        # .filter(
        #     (pl.col("d_year") == 1998) &
        #     pl.col("t_time").is_between(7249, 7249 + 28800) &
        #     pl.col("sm_carrier").is_in(["ZOUROS", "ZHOU"])
        # )
        .filter(
            (pl.col("d_year").is_not_null() & (pl.col("d_year") == 1998))
            & (
                pl.col("t_time").is_not_null()
                & pl.col("t_time").is_between(7249, 7249 + 28800)
            )
            & (
                pl.col("sm_carrier").is_not_null()
                & pl.col("sm_carrier").is_in(["ZOUROS", "ZHOU"])
            )
        )
        .with_columns(
            [
                # Preprocess: Create monthly sales columns (price * quantity for the month, 0 otherwise)
                pl.when(pl.col("d_moy") == 1)
                .then(pl.col("ws_ext_sales_price") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("jan_sales_val"),
                pl.when(pl.col("d_moy") == 2)
                .then(pl.col("ws_ext_sales_price") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("feb_sales_val"),
                pl.when(pl.col("d_moy") == 3)
                .then(pl.col("ws_ext_sales_price") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("mar_sales_val"),
                pl.when(pl.col("d_moy") == 4)
                .then(pl.col("ws_ext_sales_price") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("apr_sales_val"),
                pl.when(pl.col("d_moy") == 5)
                .then(pl.col("ws_ext_sales_price") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("may_sales_val"),
                pl.when(pl.col("d_moy") == 6)
                .then(pl.col("ws_ext_sales_price") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("jun_sales_val"),
                pl.when(pl.col("d_moy") == 7)
                .then(pl.col("ws_ext_sales_price") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("jul_sales_val"),
                pl.when(pl.col("d_moy") == 8)
                .then(pl.col("ws_ext_sales_price") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("aug_sales_val"),
                pl.when(pl.col("d_moy") == 9)
                .then(pl.col("ws_ext_sales_price") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("sep_sales_val"),
                pl.when(pl.col("d_moy") == 10)
                .then(pl.col("ws_ext_sales_price") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("oct_sales_val"),
                pl.when(pl.col("d_moy") == 11)
                .then(pl.col("ws_ext_sales_price") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("nov_sales_val"),
                pl.when(pl.col("d_moy") == 12)
                .then(pl.col("ws_ext_sales_price") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("dec_sales_val"),
                # Preprocess: Create monthly net columns (net * quantity for the month, 0 otherwise)
                pl.when(pl.col("d_moy") == 1)
                .then(pl.col("ws_net_paid_inc_ship") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("jan_net_val"),
                pl.when(pl.col("d_moy") == 2)
                .then(pl.col("ws_net_paid_inc_ship") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("feb_net_val"),
                pl.when(pl.col("d_moy") == 3)
                .then(pl.col("ws_net_paid_inc_ship") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("mar_net_val"),
                pl.when(pl.col("d_moy") == 4)
                .then(pl.col("ws_net_paid_inc_ship") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("apr_net_val"),
                pl.when(pl.col("d_moy") == 5)
                .then(pl.col("ws_net_paid_inc_ship") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("may_net_val"),
                pl.when(pl.col("d_moy") == 6)
                .then(pl.col("ws_net_paid_inc_ship") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("jun_net_val"),
                pl.when(pl.col("d_moy") == 7)
                .then(pl.col("ws_net_paid_inc_ship") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("jul_net_val"),
                pl.when(pl.col("d_moy") == 8)
                .then(pl.col("ws_net_paid_inc_ship") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("aug_net_val"),
                pl.when(pl.col("d_moy") == 9)
                .then(pl.col("ws_net_paid_inc_ship") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("sep_net_val"),
                pl.when(pl.col("d_moy") == 10)
                .then(pl.col("ws_net_paid_inc_ship") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("oct_net_val"),
                pl.when(pl.col("d_moy") == 11)
                .then(pl.col("ws_net_paid_inc_ship") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("nov_net_val"),
                pl.when(pl.col("d_moy") == 12)
                .then(pl.col("ws_net_paid_inc_ship") * pl.col("ws_quantity"))
                .otherwise(0)
                .alias("dec_net_val"),
            ]
        )
        .group_by(
            [
                "w_warehouse_name",
                "w_warehouse_sq_ft",
                "w_city",
                "w_county",
                "w_state",
                "w_country",
                "d_year",
            ]
        )
        .agg(
            [
                # Simple sum aggregations of preprocessed monthly columns
                pl.col("jan_sales_val").sum().alias("jan_sales"),
                pl.col("feb_sales_val").sum().alias("feb_sales"),
                pl.col("mar_sales_val").sum().alias("mar_sales"),
                pl.col("apr_sales_val").sum().alias("apr_sales"),
                pl.col("may_sales_val").sum().alias("may_sales"),
                pl.col("jun_sales_val").sum().alias("jun_sales"),
                pl.col("jul_sales_val").sum().alias("jul_sales"),
                pl.col("aug_sales_val").sum().alias("aug_sales"),
                pl.col("sep_sales_val").sum().alias("sep_sales"),
                pl.col("oct_sales_val").sum().alias("oct_sales"),
                pl.col("nov_sales_val").sum().alias("nov_sales"),
                pl.col("dec_sales_val").sum().alias("dec_sales"),
                pl.col("jan_net_val").sum().alias("jan_net"),
                pl.col("feb_net_val").sum().alias("feb_net"),
                pl.col("mar_net_val").sum().alias("mar_net"),
                pl.col("apr_net_val").sum().alias("apr_net"),
                pl.col("may_net_val").sum().alias("may_net"),
                pl.col("jun_net_val").sum().alias("jun_net"),
                pl.col("jul_net_val").sum().alias("jul_net"),
                pl.col("aug_net_val").sum().alias("aug_net"),
                pl.col("sep_net_val").sum().alias("sep_net"),
                pl.col("oct_net_val").sum().alias("oct_net"),
                pl.col("nov_net_val").sum().alias("nov_net"),
                pl.col("dec_net_val").sum().alias("dec_net"),
            ]
        )
        .with_columns(
            [
                pl.lit("ZOUROS,ZHOU").alias("ship_carriers"),
                pl.col("d_year").alias("year1"),
            ]
        )
    )
    # Catalog sales part of UNION ALL
    catalog_sales_monthly = (
        catalog_sales.join(
            warehouse, left_on="cs_warehouse_sk", right_on="w_warehouse_sk"
        )
        .join(date_dim, left_on="cs_sold_date_sk", right_on="d_date_sk")
        .join(time_dim, left_on="cs_sold_time_sk", right_on="t_time_sk")
        .join(ship_mode, left_on="cs_ship_mode_sk", right_on="sm_ship_mode_sk")
        # TODO: There's some bug in cudf_polars with FILTER on Column with NULLs
        # .filter(
        #     (pl.col("d_year") == 1998) &
        #     pl.col("t_time").is_between(7249, 7249 + 28800) &
        #     pl.col("sm_carrier").is_in(["ZOUROS", "ZHOU"])
        # )
        .filter(
            (pl.col("d_year").is_not_null() & (pl.col("d_year") == 1998))
            & (
                pl.col("t_time").is_not_null()
                & pl.col("t_time").is_between(7249, 7249 + 28800)
            )
            & (
                pl.col("sm_carrier").is_not_null()
                & pl.col("sm_carrier").is_in(["ZOUROS", "ZHOU"])
            )
        )
        .with_columns(
            [
                # Preprocess: Create monthly sales columns (price * quantity for the month, 0 otherwise)
                pl.when(pl.col("d_moy") == 1)
                .then(pl.col("cs_ext_sales_price") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("jan_sales_val"),
                pl.when(pl.col("d_moy") == 2)
                .then(pl.col("cs_ext_sales_price") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("feb_sales_val"),
                pl.when(pl.col("d_moy") == 3)
                .then(pl.col("cs_ext_sales_price") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("mar_sales_val"),
                pl.when(pl.col("d_moy") == 4)
                .then(pl.col("cs_ext_sales_price") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("apr_sales_val"),
                pl.when(pl.col("d_moy") == 5)
                .then(pl.col("cs_ext_sales_price") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("may_sales_val"),
                pl.when(pl.col("d_moy") == 6)
                .then(pl.col("cs_ext_sales_price") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("jun_sales_val"),
                pl.when(pl.col("d_moy") == 7)
                .then(pl.col("cs_ext_sales_price") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("jul_sales_val"),
                pl.when(pl.col("d_moy") == 8)
                .then(pl.col("cs_ext_sales_price") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("aug_sales_val"),
                pl.when(pl.col("d_moy") == 9)
                .then(pl.col("cs_ext_sales_price") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("sep_sales_val"),
                pl.when(pl.col("d_moy") == 10)
                .then(pl.col("cs_ext_sales_price") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("oct_sales_val"),
                pl.when(pl.col("d_moy") == 11)
                .then(pl.col("cs_ext_sales_price") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("nov_sales_val"),
                pl.when(pl.col("d_moy") == 12)
                .then(pl.col("cs_ext_sales_price") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("dec_sales_val"),
                # Preprocess: Create monthly net columns (net * quantity for the month, 0 otherwise)
                pl.when(pl.col("d_moy") == 1)
                .then(pl.col("cs_net_paid") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("jan_net_val"),
                pl.when(pl.col("d_moy") == 2)
                .then(pl.col("cs_net_paid") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("feb_net_val"),
                pl.when(pl.col("d_moy") == 3)
                .then(pl.col("cs_net_paid") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("mar_net_val"),
                pl.when(pl.col("d_moy") == 4)
                .then(pl.col("cs_net_paid") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("apr_net_val"),
                pl.when(pl.col("d_moy") == 5)
                .then(pl.col("cs_net_paid") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("may_net_val"),
                pl.when(pl.col("d_moy") == 6)
                .then(pl.col("cs_net_paid") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("jun_net_val"),
                pl.when(pl.col("d_moy") == 7)
                .then(pl.col("cs_net_paid") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("jul_net_val"),
                pl.when(pl.col("d_moy") == 8)
                .then(pl.col("cs_net_paid") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("aug_net_val"),
                pl.when(pl.col("d_moy") == 9)
                .then(pl.col("cs_net_paid") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("sep_net_val"),
                pl.when(pl.col("d_moy") == 10)
                .then(pl.col("cs_net_paid") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("oct_net_val"),
                pl.when(pl.col("d_moy") == 11)
                .then(pl.col("cs_net_paid") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("nov_net_val"),
                pl.when(pl.col("d_moy") == 12)
                .then(pl.col("cs_net_paid") * pl.col("cs_quantity"))
                .otherwise(0)
                .alias("dec_net_val"),
            ]
        )
        .group_by(
            [
                "w_warehouse_name",
                "w_warehouse_sq_ft",
                "w_city",
                "w_county",
                "w_state",
                "w_country",
                "d_year",
            ]
        )
        .agg(
            [
                # Simple sum aggregations of preprocessed monthly columns
                pl.col("jan_sales_val").sum().alias("jan_sales"),
                pl.col("feb_sales_val").sum().alias("feb_sales"),
                pl.col("mar_sales_val").sum().alias("mar_sales"),
                pl.col("apr_sales_val").sum().alias("apr_sales"),
                pl.col("may_sales_val").sum().alias("may_sales"),
                pl.col("jun_sales_val").sum().alias("jun_sales"),
                pl.col("jul_sales_val").sum().alias("jul_sales"),
                pl.col("aug_sales_val").sum().alias("aug_sales"),
                pl.col("sep_sales_val").sum().alias("sep_sales"),
                pl.col("oct_sales_val").sum().alias("oct_sales"),
                pl.col("nov_sales_val").sum().alias("nov_sales"),
                pl.col("dec_sales_val").sum().alias("dec_sales"),
                pl.col("jan_net_val").sum().alias("jan_net"),
                pl.col("feb_net_val").sum().alias("feb_net"),
                pl.col("mar_net_val").sum().alias("mar_net"),
                pl.col("apr_net_val").sum().alias("apr_net"),
                pl.col("may_net_val").sum().alias("may_net"),
                pl.col("jun_net_val").sum().alias("jun_net"),
                pl.col("jul_net_val").sum().alias("jul_net"),
                pl.col("aug_net_val").sum().alias("aug_net"),
                pl.col("sep_net_val").sum().alias("sep_net"),
                pl.col("oct_net_val").sum().alias("oct_net"),
                pl.col("nov_net_val").sum().alias("nov_net"),
                pl.col("dec_net_val").sum().alias("dec_net"),
            ]
        )
        .with_columns(
            [
                pl.lit("ZOUROS,ZHOU").alias("ship_carriers"),
                pl.col("d_year").alias("year1"),
            ]
        )
    )
    # UNION ALL and final aggregation
    return (
        pl.concat([web_sales_monthly, catalog_sales_monthly])
        .group_by(
            [
                "w_warehouse_name",
                "w_warehouse_sq_ft",
                "w_city",
                "w_county",
                "w_state",
                "w_country",
                "ship_carriers",
                "year1",
            ]
        )
        .agg(
            [
                # Sum monthly sales
                pl.col("jan_sales").sum().alias("jan_sales"),
                pl.col("feb_sales").sum().alias("feb_sales"),
                pl.col("mar_sales").sum().alias("mar_sales"),
                pl.col("apr_sales").sum().alias("apr_sales"),
                pl.col("may_sales").sum().alias("may_sales"),
                pl.col("jun_sales").sum().alias("jun_sales"),
                pl.col("jul_sales").sum().alias("jul_sales"),
                pl.col("aug_sales").sum().alias("aug_sales"),
                pl.col("sep_sales").sum().alias("sep_sales"),
                pl.col("oct_sales").sum().alias("oct_sales"),
                pl.col("nov_sales").sum().alias("nov_sales"),
                pl.col("dec_sales").sum().alias("dec_sales"),
                # Sum monthly net
                pl.col("jan_net").sum().alias("jan_net"),
                pl.col("feb_net").sum().alias("feb_net"),
                pl.col("mar_net").sum().alias("mar_net"),
                pl.col("apr_net").sum().alias("apr_net"),
                pl.col("may_net").sum().alias("may_net"),
                pl.col("jun_net").sum().alias("jun_net"),
                pl.col("jul_net").sum().alias("jul_net"),
                pl.col("aug_net").sum().alias("aug_net"),
                pl.col("sep_net").sum().alias("sep_net"),
                pl.col("oct_net").sum().alias("oct_net"),
                pl.col("nov_net").sum().alias("nov_net"),
                pl.col("dec_net").sum().alias("dec_net"),
            ]
        )
        .with_columns(
            [
                # Calculate per-square-foot metrics
                (pl.col("jan_sales") / pl.col("w_warehouse_sq_ft")).alias(
                    "jan_sales_per_sq_foot"
                ),
                (pl.col("feb_sales") / pl.col("w_warehouse_sq_ft")).alias(
                    "feb_sales_per_sq_foot"
                ),
                (pl.col("mar_sales") / pl.col("w_warehouse_sq_ft")).alias(
                    "mar_sales_per_sq_foot"
                ),
                (pl.col("apr_sales") / pl.col("w_warehouse_sq_ft")).alias(
                    "apr_sales_per_sq_foot"
                ),
                (pl.col("may_sales") / pl.col("w_warehouse_sq_ft")).alias(
                    "may_sales_per_sq_foot"
                ),
                (pl.col("jun_sales") / pl.col("w_warehouse_sq_ft")).alias(
                    "jun_sales_per_sq_foot"
                ),
                (pl.col("jul_sales") / pl.col("w_warehouse_sq_ft")).alias(
                    "jul_sales_per_sq_foot"
                ),
                (pl.col("aug_sales") / pl.col("w_warehouse_sq_ft")).alias(
                    "aug_sales_per_sq_foot"
                ),
                (pl.col("sep_sales") / pl.col("w_warehouse_sq_ft")).alias(
                    "sep_sales_per_sq_foot"
                ),
                (pl.col("oct_sales") / pl.col("w_warehouse_sq_ft")).alias(
                    "oct_sales_per_sq_foot"
                ),
                (pl.col("nov_sales") / pl.col("w_warehouse_sq_ft")).alias(
                    "nov_sales_per_sq_foot"
                ),
                (pl.col("dec_sales") / pl.col("w_warehouse_sq_ft")).alias(
                    "dec_sales_per_sq_foot"
                ),
            ]
        )
        .sort(["w_warehouse_name"], nulls_last=True, descending=[False])
        .limit(100)
    )
