# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for equi joins that Polars plans with a fused predicate."""

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.versions import POLARS_VERSION_LT_21

pytestmark = pytest.mark.skipif(
    POLARS_VERSION_LT_21,
    reason="Polars 2.0 does not expose fused join predicates to the visitor",
)


@pytest.fixture
def orders() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "key": [1, 2, 3, 3, None],
            "supp_nation": ["FRANCE", "GERMANY", "FRANCE", "GERMANY", "FRANCE"],
            "v": [1, 2, 3, 4, 5],
        }
    )


@pytest.fixture
def customers() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "key": [1, 2, 3, None, 3],
            "cust_nation": ["GERMANY", "FRANCE", "FRANCE", "GERMANY", "GERMANY"],
            "w": [4, 3, 2, 1, 9],
        }
    )


def test_sql_equi_join_with_not_equal(engine: pl.GPUEngine, orders, customers):
    q = pl.SQLContext(orders=orders, customers=customers).execute(
        "SELECT * FROM orders, customers "
        "WHERE orders.key = customers.key AND supp_nation <> cust_nation"
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("nulls_equal", [False, True])
def test_inner_join_then_filter(engine: pl.GPUEngine, orders, customers, nulls_equal):
    q = orders.join(customers, on="key", how="inner", nulls_equal=nulls_equal).filter(
        pl.col("supp_nation") != pl.col("cust_nation")
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_join_where_equality_and_inequality(engine: pl.GPUEngine, orders, customers):
    q = orders.join_where(
        customers,
        pl.col("key") == pl.col("key_right"),
        pl.col("v") > pl.col("w"),
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_predicate_on_suffixed_columns(engine: pl.GPUEngine):
    left = pl.LazyFrame({"k": [1, 2, 3], "x": [1, 2, 3]})
    right = pl.LazyFrame({"k": [1, 2, 3], "x": [3, 2, 1]})
    q = left.join(right, on="k", how="inner", suffix="_r").filter(
        pl.col("x") != pl.col("x_r")
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("negate", [False, True], ids=["exists", "not-exists"])
def test_correlated_exists(engine: pl.GPUEngine, orders, customers, negate):
    # The shape used by TPC-H queries 21 and 22 and similar.
    not_ = "NOT " if negate else ""
    q = pl.SQLContext(orders=orders, customers=customers).execute(
        f"SELECT * FROM orders WHERE {not_}EXISTS "
        "(SELECT 1 FROM customers WHERE customers.key = orders.key "
        "AND customers.cust_nation <> orders.supp_nation)"
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_empty_result(engine: pl.GPUEngine, orders, customers):
    q = orders.join(customers, on="key", how="inner").filter(
        pl.col("supp_nation") != pl.col("supp_nation")
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)
