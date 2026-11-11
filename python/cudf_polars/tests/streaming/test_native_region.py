# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
import datetime
from collections import defaultdict
from typing import TYPE_CHECKING

import pytest

import polars as pl

import pylibcudf as plc

from cudf_polars.dsl.ir import GroupBy, Select
from cudf_polars.dsl.translate import Translator
from cudf_polars.dsl.traversal import traversal
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming import remote_io
from cudf_polars.streaming.io import StreamingScan
from cudf_polars.streaming.native import NativeRegion
from cudf_polars.streaming.native.expressions import NoMatch, compile_expr
from cudf_polars.streaming.native.region import match_groupby
from cudf_polars.streaming.parallel import lower_ir_graph
from cudf_polars.streaming.statistics import collect_statistics
from cudf_polars.testing.io import make_partitioned_source
from cudf_polars.utils.config import ConfigOptions

scan_io = pytest.importorskip("cudf_polars_cucascade.scan_io")

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from cudf_polars.engine.core import StreamingEngine


# ---------------------------------------------------------------- expression contract


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [0.5, 1.5, None, 4.0, -2.0, 7.25],
            "b": [0.1, 0.0, 0.3, None, 0.5, 0.05],
            "c": pl.Series([1, 5, 3, None, 9, 2], dtype=pl.Int32),
            "s": ["x", "y", None, "x", "BRAZIL", "y"],
            "d": [
                datetime.date(1995, 1, 1),
                datetime.date(1996, 12, 31),
                None,
                datetime.date(1992, 6, 15),
                datetime.date(1998, 8, 2),
                datetime.date(1994, 3, 9),
            ],
        }
    )


def _compile(frame: pl.DataFrame, expr: pl.Expr) -> list[dict]:
    ir = Translator(
        frame.lazy().select(expr)._ldf.visit(), pl.GPUEngine()
    ).translate_ir()
    select = next(n for n in traversal([ir]) if isinstance(n, Select))
    nodes: list[dict] = []
    compile_expr(select.exprs[0].value, list(frame.columns), nodes)
    return nodes


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("a") * (1 - pl.col("b")),
        pl.col("a") + pl.col("b"),
        pl.col("a") / pl.col("b"),
        pl.col("c") > 3,
        (pl.col("a") > 0.5) & (pl.col("c") < 5),
        (pl.col("s") == "x") | (pl.col("c") == 9),
        pl.when(pl.col("s") == "BRAZIL").then(pl.col("a")).otherwise(0.0),
        pl.col("d").dt.year(),
        pl.col("c").cast(pl.Float64) * pl.col("a"),
    ],
    ids=[
        "volume",
        "add",
        "divide",
        "compare-int",
        "and",
        "or-string",
        "when-then",
        "year",
        "cast",
    ],
)
def test_native_expression_matches_polars(frame: pl.DataFrame, expr: pl.Expr) -> None:
    nodes = _compile(frame, expr)
    table = plc.Table.from_arrow(frame.to_arrow(compat_level=pl.CompatLevel.oldest()))
    got = pl.Series(scan_io.evaluate_expression(table, nodes).to_arrow())
    expected = frame.select(expr).to_series()
    # Native years are int16 where polars has int32; compare values and nulls.
    got = got.cast(expected.dtype)
    assert got.is_null().to_list() == expected.is_null().to_list()
    pairs = [
        (g, e)
        for g, e in zip(got.to_list(), expected.to_list(), strict=True)
        if e is not None
    ]
    if expected.dtype.is_float():
        assert [g for g, _ in pairs] == pytest.approx(
            [e for _, e in pairs], nan_ok=True
        )
    else:
        assert [g for g, _ in pairs] == [e for _, e in pairs]


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("s").str.to_uppercase(),
        pl.col("a").fill_null(0.0),
        pl.col("c") * 2,  # integer arithmetic stays on the regular path
        pl.col("d") + datetime.timedelta(days=1),
    ],
    ids=["string-function", "fill-null", "int-arithmetic", "date-arithmetic"],
)
def test_unsupported_expression_is_rejected(frame: pl.DataFrame, expr: pl.Expr) -> None:
    with pytest.raises(NoMatch):
        _compile(frame, expr)


# ---------------------------------------------------------------- matching


@pytest.fixture
def tables(tmp_path: Path) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    fact = pl.DataFrame(
        {
            "k": [i % 40 for i in range(8_000)],
            "v": [float(i) for i in range(8_000)],
            "w": [i % 7 for i in range(8_000)],
        }
    )
    dim = pl.DataFrame({"dk": range(40), "name": [f"n{i % 5}" for i in range(40)]})
    (tmp_path / "fact").mkdir()
    (tmp_path / "dim").mkdir()
    make_partitioned_source(fact, tmp_path / "fact", "parquet", n_files=6)
    make_partitioned_source(dim, tmp_path / "dim", "parquet", n_files=1)
    return pl.scan_parquet(tmp_path / "fact"), pl.scan_parquet(tmp_path / "dim")


def _lowered(
    q: pl.LazyFrame,
    engine: StreamingEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> list:
    # Native regions read remote scans only; treat these local files as remote.
    monkeypatch.setenv("CUDF_POLARS_NATIVE_REGION", "1")
    monkeypatch.setattr(remote_io, "eligible", lambda task: True)
    config = ConfigOptions.from_polars_engine(engine)
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        stats = collect_statistics(ir, config, executor)
    return list(traversal([lower_ir_graph(ir, config, stats).lowered]))


def _why_not(nodes: list) -> str:
    """The matcher's reason for each group-by left on the regular path."""
    dep_count: defaultdict = defaultdict(int)
    for node in nodes:
        for child in node.children:
            dep_count[child] += 1
    reasons = []
    for node in nodes:
        if isinstance(node, GroupBy):
            try:
                match_groupby(node, dep_count)
            except NoMatch as e:
                reasons.append(str(e))
    tasks = {
        n.tasks[0].paths[0].split("/")[-2]: len(n.tasks)
        for n in nodes
        if isinstance(n, StreamingScan)
    }
    return f"{reasons}, scan tasks {tasks}"


@pytest.fixture
def engine(
    streaming_engine_factory: Callable[..., StreamingEngine],
) -> StreamingEngine:
    # Small partitions, so the fact table reads more tasks than the dimension.
    return streaming_engine_factory(StreamingOptions(target_partition_size=4_000))


def test_join_groupby_becomes_native_region(
    tables, engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    fact, dim = tables
    q = (
        fact.join(dim, left_on="k", right_on="dk")
        .with_columns((pl.col("v") * 2.0).alias("v2"))
        .group_by("name")
        .agg(pl.col("v2").sum())
    )
    nodes = _lowered(q, engine, monkeypatch)
    regions = [n for n in nodes if isinstance(n, NativeRegion)]
    assert len(regions) == 1, _why_not(nodes)
    (region,) = regions
    assert len(region.children) == 1  # the dimension, as the build side
    assert not any(isinstance(n, GroupBy) for n in nodes)
    # The fact scan is read by the region itself, not by the network.
    assert not any(
        isinstance(n, StreamingScan) and "fact" in n.tasks[0].paths[0] for n in nodes
    )
    stages = region.program.branches[0]
    assert [s["op"] for s in stages if s["op"] != "select"] == [
        "join",
        "compute",
        "groupby",
    ]


def test_union_of_branches(tables, engine, monkeypatch: pytest.MonkeyPatch) -> None:
    fact, dim = tables
    joined = fact.join(dim, left_on="k", right_on="dk")
    q = (
        pl.concat(
            [
                joined.select("name", "v"),
                joined.select("name", (pl.col("v") * 2.0).alias("v")),
            ]
        )
        .group_by("name")
        .agg(pl.col("v").sum())
    )
    nodes = _lowered(q, engine, monkeypatch)
    regions = [n for n in nodes if isinstance(n, NativeRegion)]
    assert len(regions) == 1, _why_not(nodes)
    assert len(regions[0].program.branches) == 2


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda f, d: (
                f.join(d, left_on="k", right_on="dk")
                .group_by("name")
                .agg(pl.col("v").mean())
            ),
            id="mean",
        ),
        pytest.param(
            lambda f, d: (
                d.join(f, left_on="dk", right_on="k", how="left")
                .group_by("name")
                .agg(pl.col("v").sum())
            ),
            id="left-join",
        ),
        pytest.param(
            lambda f, d: f.group_by("w").agg(pl.col("v").sum()),
            id="no-join",
        ),
        pytest.param(
            lambda f, d: (
                f.join(f.select("k").unique(), on="k")
                .group_by("w")
                .agg(pl.col("v").sum())
            ),
            id="scan-on-both-sides",
        ),
    ],
)
def test_unsupported_shapes_stay_regular(
    tables, engine, monkeypatch: pytest.MonkeyPatch, build
) -> None:
    fact, dim = tables
    nodes = _lowered(build(fact, dim), engine, monkeypatch)
    assert not any(isinstance(n, NativeRegion) for n in nodes)


def test_shared_subplan_is_not_replaced(
    tables, engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The joined frame also feeds a second consumer (as in TPC-H Q11).
    fact, dim = tables
    joined = fact.join(dim, left_on="k", right_on="dk").with_columns(
        (pl.col("v") * 2.0).alias("v2")
    )
    per_name = joined.group_by("name").agg(pl.col("v2").sum())
    largest = joined.group_by("name").agg(pl.col("v").max().alias("largest"))
    q = per_name.join(largest, on="name")
    nodes = _lowered(q, engine, monkeypatch)
    assert not any(isinstance(n, NativeRegion) for n in nodes)
    assert "used outside the region" in _why_not(nodes)


def test_disabled_by_default(tables, engine, monkeypatch: pytest.MonkeyPatch) -> None:
    fact, dim = tables
    q = (
        fact.join(dim, left_on="k", right_on="dk")
        .group_by("name")
        .agg(pl.col("v").sum())
    )
    nodes = _lowered(q, engine, monkeypatch)
    assert any(isinstance(n, NativeRegion) for n in nodes)
    monkeypatch.delenv("CUDF_POLARS_NATIVE_REGION")
    config = ConfigOptions.from_polars_engine(engine)
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        stats = collect_statistics(ir, config, executor)
    lowered = lower_ir_graph(ir, config, stats).lowered
    assert not any(isinstance(n, NativeRegion) for n in traversal([lowered]))
