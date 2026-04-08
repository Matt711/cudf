# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 67."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import QueryResult, get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 67."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=67,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    return f"""
        select *
        from (select i_category
                    ,i_class
                    ,i_brand
                    ,i_product_name
                    ,d_year
                    ,d_qoy
                    ,d_moy
                    ,s_store_id
                    ,sumsales
                    ,rank() over (partition by i_category order by sumsales desc) rk
            from (select i_category
                        ,i_class
                        ,i_brand
                        ,i_product_name
                        ,d_year
                        ,d_qoy
                        ,d_moy
                        ,s_store_id
                        ,sum(coalesce(ss_sales_price*ss_quantity,0)) sumsales
                    from store_sales
                        ,date_dim
                        ,store
                        ,item
            where  ss_sold_date_sk=d_date_sk
                and ss_item_sk=i_item_sk
                and ss_store_sk = s_store_sk
                and d_month_seq between {dms} and {dms}+11
            group by  rollup(i_category, i_class, i_brand, i_product_name, d_year, d_qoy, d_moy,s_store_id))dw1) dw2
        where rk <= 100
        order by i_category
                ,i_class
                ,i_brand
                ,i_product_name
                ,d_year
                ,d_qoy
                ,d_moy
                ,s_store_id
                ,sumsales
                ,rk
        limit 100;
    """


def polars_impl(run_config: RunConfig) -> QueryResult:
    """Query 67.

    SQL structure
    -------------
    SELECT *, rank() OVER (PARTITION BY i_category ORDER BY sumsales DESC) rk
    FROM (
        SELECT i_category, i_class, i_brand, i_product_name,
               d_year, d_qoy, d_moy, s_store_id,
               SUM(COALESCE(ss_sales_price * ss_quantity, 0)) sumsales
        FROM store_sales, date_dim, store, item
        WHERE ss_sold_date_sk = d_date_sk
          AND ss_item_sk = i_item_sk
          AND ss_store_sk = s_store_sk
          AND d_month_seq BETWEEN {dms} AND {dms}+11
        GROUP BY ROLLUP(i_category, i_class, i_brand, i_product_name,
                        d_year, d_qoy, d_moy, s_store_id)
    )
    WHERE rk <= 100
    ORDER BY i_category, i_class, i_brand, i_product_name,
             d_year, d_qoy, d_moy, s_store_id, sumsales, rk
    LIMIT 100
    """
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=67,
        qualification=run_config.qualification,
    )

    dms = params["dms"]

    # Pre-filter / project dimension tables once.
    # These are small (~365, ~1K, ~360K rows at SF1000) so their 9-consumer
    # fanout buffers in the streaming executor are negligible.
    date_filtered = (
        get_data(run_config.dataset_path, "date_dim", run_config.suffix)
        .filter(pl.col("d_month_seq").is_between(dms, dms + 11))
        .select(["d_date_sk", "d_year", "d_qoy", "d_moy"])
    )
    item_sel = get_data(run_config.dataset_path, "item", run_config.suffix).select(
        ["i_item_sk", "i_category", "i_class", "i_brand", "i_product_name"]
    )
    store_sel = get_data(run_config.dataset_path, "store", run_config.suffix).select(
        ["s_store_sk", "s_store_id"]
    )

    # Each make_base() call issues a fresh scan_parquet() so the 9 rollup
    # pipelines are independent in the streaming actor graph — no fanout and
    # no back-pressure on the large store_sales table.
    def make_base() -> pl.LazyFrame:
        return (
            get_data(run_config.dataset_path, "store_sales", run_config.suffix)
            .join(date_filtered, left_on="ss_sold_date_sk", right_on="d_date_sk")
            .join(item_sel, left_on="ss_item_sk", right_on="i_item_sk")
            .join(store_sel, left_on="ss_store_sk", right_on="s_store_sk")
        )

    sales_expr = (
        (pl.col("ss_sales_price") * pl.col("ss_quantity"))
        .fill_null(0)
        .sum()
        .alias("sumsales")
    )

    # Output column order matches the SQL SELECT list.
    out_cols = [
        "i_category",
        "i_class",
        "i_brand",
        "i_product_name",
        "d_year",
        "d_qoy",
        "d_moy",
        "s_store_id",
        "sumsales",
    ]

    # ROLLUP(i_category, i_class, i_brand, i_product_name, d_year, d_qoy, d_moy, s_store_id)
    # produces 9 levels: the full grouping plus each successive right-side drop.
    level1 = (
        make_base()
        .group_by(["i_category", "i_class", "i_brand", "i_product_name",
                   "d_year", "d_qoy", "d_moy", "s_store_id"])
        .agg(sales_expr)
        .select(out_cols)
    )
    level2 = (
        make_base()
        .group_by(["i_category", "i_class", "i_brand", "i_product_name",
                   "d_year", "d_qoy", "d_moy"])
        .agg(sales_expr)
        .with_columns(pl.lit(None, dtype=pl.String).alias("s_store_id"))
        .select(out_cols)
    )
    level3 = (
        make_base()
        .group_by(["i_category", "i_class", "i_brand", "i_product_name",
                   "d_year", "d_qoy"])
        .agg(sales_expr)
        .with_columns(
            pl.lit(None, dtype=pl.Int64).alias("d_moy"),
            pl.lit(None, dtype=pl.String).alias("s_store_id"),
        )
        .select(out_cols)
    )
    level4 = (
        make_base()
        .group_by(["i_category", "i_class", "i_brand", "i_product_name", "d_year"])
        .agg(sales_expr)
        .with_columns(
            pl.lit(None, dtype=pl.Int64).alias("d_qoy"),
            pl.lit(None, dtype=pl.Int64).alias("d_moy"),
            pl.lit(None, dtype=pl.String).alias("s_store_id"),
        )
        .select(out_cols)
    )
    level5 = (
        make_base()
        .group_by(["i_category", "i_class", "i_brand", "i_product_name"])
        .agg(sales_expr)
        .with_columns(
            pl.lit(None, dtype=pl.Int64).alias("d_year"),
            pl.lit(None, dtype=pl.Int64).alias("d_qoy"),
            pl.lit(None, dtype=pl.Int64).alias("d_moy"),
            pl.lit(None, dtype=pl.String).alias("s_store_id"),
        )
        .select(out_cols)
    )
    level6 = (
        make_base()
        .group_by(["i_category", "i_class", "i_brand"])
        .agg(sales_expr)
        .with_columns(
            pl.lit(None, dtype=pl.String).alias("i_product_name"),
            pl.lit(None, dtype=pl.Int64).alias("d_year"),
            pl.lit(None, dtype=pl.Int64).alias("d_qoy"),
            pl.lit(None, dtype=pl.Int64).alias("d_moy"),
            pl.lit(None, dtype=pl.String).alias("s_store_id"),
        )
        .select(out_cols)
    )
    level7 = (
        make_base()
        .group_by(["i_category", "i_class"])
        .agg(sales_expr)
        .with_columns(
            pl.lit(None, dtype=pl.String).alias("i_brand"),
            pl.lit(None, dtype=pl.String).alias("i_product_name"),
            pl.lit(None, dtype=pl.Int64).alias("d_year"),
            pl.lit(None, dtype=pl.Int64).alias("d_qoy"),
            pl.lit(None, dtype=pl.Int64).alias("d_moy"),
            pl.lit(None, dtype=pl.String).alias("s_store_id"),
        )
        .select(out_cols)
    )
    level8 = (
        make_base()
        .group_by(["i_category"])
        .agg(sales_expr)
        .with_columns(
            pl.lit(None, dtype=pl.String).alias("i_class"),
            pl.lit(None, dtype=pl.String).alias("i_brand"),
            pl.lit(None, dtype=pl.String).alias("i_product_name"),
            pl.lit(None, dtype=pl.Int64).alias("d_year"),
            pl.lit(None, dtype=pl.Int64).alias("d_qoy"),
            pl.lit(None, dtype=pl.Int64).alias("d_moy"),
            pl.lit(None, dtype=pl.String).alias("s_store_id"),
        )
        .select(out_cols)
    )
    # Grand total — no group-by keys; all dimension columns become NULL.
    level9 = (
        make_base()
        .select(sales_expr)
        .with_columns(
            pl.lit(None, dtype=pl.String).alias("i_category"),
            pl.lit(None, dtype=pl.String).alias("i_class"),
            pl.lit(None, dtype=pl.String).alias("i_brand"),
            pl.lit(None, dtype=pl.String).alias("i_product_name"),
            pl.lit(None, dtype=pl.Int64).alias("d_year"),
            pl.lit(None, dtype=pl.Int64).alias("d_qoy"),
            pl.lit(None, dtype=pl.Int64).alias("d_moy"),
            pl.lit(None, dtype=pl.String).alias("s_store_id"),
        )
        .select(out_cols)
    )

    all_levels = pl.concat(
        [level1, level2, level3, level4, level5, level6, level7, level8, level9]
    )

    # rank() OVER (PARTITION BY i_category ORDER BY sumsales DESC)
    # The HStack handler shuffles by i_category so this runs partition-locally.
    sort_cols = [
        "i_category", "i_class", "i_brand", "i_product_name",
        "d_year", "d_qoy", "d_moy", "s_store_id", "sumsales", "rk",
    ]
    return QueryResult(
        frame=(
            all_levels
            .with_columns(
                pl.col("sumsales")
                .rank(method="min", descending=True)
                .over("i_category")
                .alias("rk")
            )
            .filter(pl.col("rk") <= 100)
            .sort(sort_cols, nulls_last=True)
            .limit(100)
        ),
        sort_by=[
            ("i_category", False),
            ("i_class", False),
            ("i_brand", False),
            ("i_product_name", False),
            ("d_year", False),
            ("d_qoy", False),
            ("d_moy", False),
            ("s_store_id", False),
            ("sumsales", False),
            ("rk", False),
        ],
        limit=100,
    )
