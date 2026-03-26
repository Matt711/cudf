# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 74."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 74."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=74,
        qualification=run_config.qualification,
    )
    year = params["year"]

    return f"""
    WITH year_total
         AS (SELECT c_customer_id    customer_id,
                    c_first_name     customer_first_name,
                    c_last_name      customer_last_name,
                    d_year           AS year1,
                    STDDEV_SAMP(ss_net_paid) year_total,
                    's'              sale_type
             FROM   customer,
                    store_sales,
                    date_dim
             WHERE  c_customer_sk = ss_customer_sk
                    AND ss_sold_date_sk = d_date_sk
                    AND d_year IN ( {year}, {year} + 1 )
             GROUP  BY c_customer_id,
                       c_first_name,
                       c_last_name,
                       d_year
             UNION ALL
             SELECT c_customer_id    customer_id,
                    c_first_name     customer_first_name,
                    c_last_name      customer_last_name,
                    d_year           AS year1,
                    STDDEV_SAMP(ws_net_paid) year_total,
                    'w'              sale_type
             FROM   customer,
                    web_sales,
                    date_dim
             WHERE  c_customer_sk = ws_bill_customer_sk
                    AND ws_sold_date_sk = d_date_sk
                    AND d_year IN ( {year}, {year} + 1 )
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
           AND t_s_firstyear.year1 = {year}
           AND t_s_secyear.year1 = {year} + 1
           AND t_w_firstyear.year1 = {year}
           AND t_w_secyear.year1 = {year} + 1
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


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 74."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=74,
        qualification=run_config.qualification,
    )

    year = params["year"]

    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    web_sales = get_data(run_config.dataset_path, "web_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)

    # Filter date_dim early
    date_filtered = date_dim.filter(pl.col("d_year").is_in([year, year + 1]))

    # Store sales aggregate (group by integer sk + year - smaller keys than string customer_id)
    store_agg = (
        store_sales.join(date_filtered, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .group_by(["ss_customer_sk", "d_year"])
        .agg(pl.col("ss_net_paid").std().alias("total"))
    )

    # Web sales aggregate
    web_agg = (
        web_sales.join(date_filtered, left_on="ws_sold_date_sk", right_on="d_date_sk")
        .group_by(["ws_bill_customer_sk", "d_year"])
        .agg(pl.col("ws_net_paid").std().alias("total"))
    )

    # Separate by year
    s_first = store_agg.filter(pl.col("d_year") == year).select(
        pl.col("ss_customer_sk"), pl.col("total").alias("s_first")
    )
    s_second = store_agg.filter(pl.col("d_year") == year + 1).select(
        pl.col("ss_customer_sk"), pl.col("total").alias("s_second")
    )
    w_first = web_agg.filter(pl.col("d_year") == year).select(
        pl.col("ws_bill_customer_sk"), pl.col("total").alias("w_first")
    )
    w_second = web_agg.filter(pl.col("d_year") == year + 1).select(
        pl.col("ws_bill_customer_sk"), pl.col("total").alias("w_second")
    )

    # Join year slices, then join customer last
    res = (
        s_first.join(s_second, on="ss_customer_sk")
        .join(w_first, left_on="ss_customer_sk", right_on="ws_bill_customer_sk")
        .join(w_second, left_on="ss_customer_sk", right_on="ws_bill_customer_sk")
        .filter(
            (pl.col("s_first") > 0)
            & (pl.col("w_first") > 0)
            & ((pl.col("w_second") / pl.col("w_first")) > (pl.col("s_second") / pl.col("s_first")))
        )
        .join(customer, left_on="ss_customer_sk", right_on="c_customer_sk")
    )

    sort_by = {
        "customer_id": False,
        "customer_first_name": False,
        "customer_last_name": False,
    }
    limit = 100
    return QueryResult(
        frame=(
            res.select(
                pl.col("c_customer_id").alias("customer_id"),
                pl.col("c_first_name").alias("customer_first_name"),
                pl.col("c_last_name").alias("customer_last_name"),
            )
            .sort(sort_by.keys(), nulls_last=True)
            .limit(limit)
        ),
        sort_by=list(sort_by.items()),
        limit=limit,
    )
