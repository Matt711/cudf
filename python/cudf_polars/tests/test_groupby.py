# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.fixture
def df():
    return pl.LazyFrame(
        {
            "key1": [1, 1, 1, 2, 3, 1, 4, 6, 7],
            "key2": [2, 2, 2, 2, 6, 1, 4, 6, 8],
            "int": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "float": [7.0, 1, 2, 3, 4, 5, 6, 7, 8],
        }
    )


@pytest.fixture(
    params=[
        ["key1"],
        ["key2"],
        [pl.col("key1") * pl.col("key2")],
        ["key1", "key2"],
        [pl.col("key1") == pl.col("key2")],
        ["key2", pl.col("key1") == pl.lit(1, dtype=pl.Int64)],
    ],
    ids=lambda keys: "-".join(map(str, keys)),
)
def keys(request):
    return request.param


@pytest.fixture(
    params=[
        ["int"],
        ["float", "int"],
        [pl.col("float") + pl.col("int")],
        [pl.col("float").max() - pl.col("int").min()],
        [pl.col("float").mean(), pl.col("int").std()],
        [(pl.col("float") - pl.lit(2)).max()],
        [pl.col("float").sum().round(decimals=1)],
        [pl.col("float").round(decimals=1).sum()],
    ],
    ids=lambda aggs: "-".join(map(str, aggs)),
)
def exprs(request):
    return request.param


@pytest.fixture(
    params=[
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                reason="Maintaining order in groupby not implemented"
            ),
        ),
    ],
    ids=["no_maintain_order", "maintain_order"],
)
def maintain_order(request):
    return request.param


def test_groupby(df: pl.LazyFrame, maintain_order, keys, exprs):
    q = df.group_by(*keys, maintain_order=maintain_order).agg(*exprs)

    if not maintain_order:
        sort_keys = list(q.collect_schema().keys())[: len(keys)]
        q = q.sort(*sort_keys)

    assert_gpu_result_equal(q, check_exact=False)


def test_groupby_len(df, keys):
    q = df.group_by(*keys).agg(pl.len())

    assert_gpu_result_equal(q, check_row_order=False)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("float").is_not_null(),
        (pl.col("int").max() + pl.col("float").min()).max(),
    ],
)
def test_groupby_unsupported(df, expr):
    q = df.group_by("key1").agg(expr)

    assert_ir_translation_raises(q, NotImplementedError)
