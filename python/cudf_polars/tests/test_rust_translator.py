# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Rust translator path."""

from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl
from polars import polars as plrs

from cudf_polars.testing.asserts import assert_gpu_result_equal

if not hasattr(plrs, "cudf"):
    pytest.skip(
        "polars Rust runtime built without cudf translator",
        allow_module_level=True,
    )


@pytest.fixture(autouse=True)
def _enable_rust_translator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDF_POLARS_USE_RUST_TRANSLATOR", "1")


@pytest.fixture
def left_parquet(tmp_path: Path) -> str:
    path = tmp_path / "left.parquet"
    pl.DataFrame(
        {
            "key": [1, 2, 3, 4, 5, 6, 7],
            "x": [10, 20, 30, 40, 50, 60, 70],
        }
    ).write_parquet(path)
    return str(path)


@pytest.fixture
def right_parquet(tmp_path: Path) -> str:
    path = tmp_path / "right.parquet"
    pl.DataFrame(
        {
            "key": [1, 2, 3, 4, 5],
            "y": [1.5, 2.5, 3.5, 4.5, 5.5],
            "name": ["a", "b", "c", "d", "e"],
        }
    ).write_parquet(path)
    return str(path)


def test_scan_filter_select_cast_binop_literal(
    engine: pl.GPUEngine, left_parquet: str
) -> None:
    """Exercises Scan, Filter, Select, Cast, BinExpr, Column, and Literal."""
    query = (
        pl.scan_parquet(left_parquet)
        .filter(pl.col("x") > 25)
        .select((pl.col("x").cast(pl.Float64) + pl.lit(1.0)).alias("z"))
    )
    assert_gpu_result_equal(query, engine=engine)


def test_join_with_ternary(
    engine: pl.GPUEngine, left_parquet: str, right_parquet: str
) -> None:
    """Exercises Join plus a ``when/then/otherwise`` ternary."""
    left = pl.scan_parquet(left_parquet).filter(pl.col("x") > 15)
    right = pl.scan_parquet(right_parquet).select(
        pl.col("key"),
        (pl.col("y").cast(pl.Float64) + pl.lit(1.0)).alias("z"),
        pl.col("name"),
    )
    query = left.join(right, on="key").select(
        pl.when(pl.col("z") > pl.lit(2.0))
        .then(pl.col("x"))
        .otherwise(pl.lit(0))
        .alias("result")
    )
    assert_gpu_result_equal(query, engine=engine)


def test_group_by_raises_not_implemented(
    engine: pl.GPUEngine, left_parquet: str
) -> None:
    """``IR::GroupBy`` is not translated and must raise."""
    query = pl.scan_parquet(left_parquet).group_by("key").agg(pl.col("x").sum())
    with pytest.raises(
        pl.exceptions.ComputeError, match="IR::GroupBy: not yet implemented"
    ):
        query.collect(engine=engine)


def test_string_function_raises_not_implemented(
    engine: pl.GPUEngine, right_parquet: str
) -> None:
    """``AExpr::Function`` (a string function) is not translated."""
    query = pl.scan_parquet(right_parquet).select(
        pl.col("name").str.to_uppercase().alias("upper")
    )
    with pytest.raises(
        pl.exceptions.ComputeError, match="AExpr::Function: not yet implemented"
    ):
        query.collect(engine=engine)
