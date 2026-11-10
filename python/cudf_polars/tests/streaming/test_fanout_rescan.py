# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for rescanning a fanned-out scan instead of buffering it."""

from __future__ import annotations

import concurrent.futures
from collections import defaultdict
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars import Translator
from cudf_polars.dsl.traversal import traversal
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.actor_graph import nodes
from cudf_polars.streaming.actor_graph.core import determine_fanout_nodes
from cudf_polars.streaming.io import StreamingScan
from cudf_polars.streaming.parallel import lower_ir_graph
from cudf_polars.streaming.statistics import collect_statistics
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.testing.io import make_partitioned_source
from cudf_polars.utils.config import ConfigOptions

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pytest

    from cudf_polars.engine.core import StreamingEngine


def _source(tmp_path: Path) -> pl.LazyFrame:
    df = pl.DataFrame({"k": [i % 50 for i in range(2_000)], "v": range(2_000)})
    make_partitioned_source(df, tmp_path, "parquet", n_files=4)
    return pl.scan_parquet(tmp_path)


def _aggregate_joined_back(lf: pl.LazyFrame) -> pl.LazyFrame:
    # The shape of TPC-H Q18: a group-by over the scan, joined back to the same scan.
    big = (
        lf.group_by("k")
        .agg(pl.col("v").sum().alias("total"))
        .filter(pl.col("total") > 0)
    )
    return lf.join(big, on="k").select("k", "v", "total")


def _scan_fanouts(q: pl.LazyFrame, engine: StreamingEngine) -> list:
    config = ConfigOptions.from_polars_engine(engine)
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        stats = collect_statistics(ir, config, executor)
    lowered = lower_ir_graph(ir, config, stats)
    root, partition_info = lowered.lowered, lowered.partition_info
    dep_count: defaultdict = defaultdict(int)
    for node in traversal([root]):
        for child in node.children:
            dep_count[child] += 1
    fanouts = determine_fanout_nodes(root, partition_info, dep_count)
    return [info for node, info in fanouts.items() if isinstance(node, StreamingScan)]


def test_blocking_consumer_is_detected(
    tmp_path: Path, streaming_engine_factory: Callable[..., StreamingEngine]
) -> None:
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=500))
    lf = _source(tmp_path)
    fanouts = _scan_fanouts(_aggregate_joined_back(lf), engine)
    assert fanouts
    assert all(info.feeds_blocking_op for info in fanouts)

    # A self-join has no consumer that must see the whole scan first.
    self_join = lf.join(lf.select("k", pl.col("v").alias("w")), on="k")
    assert not any(info.feeds_blocking_op for info in _scan_fanouts(self_join, engine))


def test_rescan_gives_same_result(
    tmp_path: Path,
    streaming_engine_factory: Callable[..., StreamingEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decisions: list[bool] = []
    decide = nodes._rescan_instead_of_fanout

    def spy(*args, **kwargs):
        decision = decide(*args, **kwargs)
        decisions.append(decision)
        return decision

    monkeypatch.setattr(nodes, "RESCAN_MIN_DEVICE_FRACTION", 0.0)
    monkeypatch.setattr(nodes, "_rescan_instead_of_fanout", spy)
    engine = streaming_engine_factory(StreamingOptions(max_rows_per_partition=500))
    assert_gpu_result_equal(
        _aggregate_joined_back(_source(tmp_path)), engine=engine, check_row_order=False
    )
    assert any(decisions)
