# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Key-order inference from footer statistics (CUDF_POLARS_INFER_KEY_ORDER=1, experiment)."""

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.testing.asserts import assert_gpu_result_equal


def _write_ordered(path, name: str, files: int, rows: int, *, overlap: bool = False):
    """Files part.0 .. part.<files-1>, each a key range of its own (several row groups)."""
    path = path / name
    path.mkdir()
    for i in range(files):
        lo = i * rows - (1 if overlap and i else 0)
        keys = [lo + j // 2 for j in range(2 * rows)]  # every key twice
        pl.DataFrame(
            {"k": keys, f"{name}_v": [i * 1000 + j for j in range(2 * rows)]}
        ).write_parquet(path / f"part.{i}.parquet", row_group_size=rows // 2)
    return pl.scan_parquet(path / "*.parquet")


@pytest.fixture
def inferred(monkeypatch):
    import cudf_polars.streaming.actor_graph.scan_ordering as scan_ordering

    monkeypatch.setenv("CUDF_POLARS_INFER_KEY_ORDER", "1")
    calls: list[list[str]] = []
    real = scan_ordering._fast_local_task_bounds

    def recording(ir, tasks, *args):
        ordered, bounds = real(ir, tasks, *args)
        calls.append([task.paths[0].rsplit("/", 1)[-1] for task in ordered])
        return ordered, bounds

    monkeypatch.setattr(scan_ordering, "_fast_local_task_bounds", recording)
    return calls


@pytest.mark.engine_params(["spmd"])
@pytest.mark.parametrize("overlap", [False, True])
def test_inferred_key_order(tmp_path, streaming_engine_factory, inferred, overlap):
    # 12 files: listed part.0, part.1, part.10, part.11, part.2, ... (not in key order).
    left = _write_ordered(tmp_path, "l", 12, 40, overlap=overlap)
    right = _write_ordered(tmp_path, "r", 4, 120)
    engine = streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=80, target_partition_size=2048, broadcast_limit=1
        )
    )
    # Shaped like TPC-H Q21: a group-by on the key, its result joined back on the key.
    grouped = (
        left.group_by("k").agg(pl.col("l_v").sum(), pl.len()).filter(pl.col("len") > 1)
    )
    joined = (
        grouped.join(right, on="k")
        .group_by("k")
        .agg(pl.col("r_v").max(), pl.col("l_v").first(), pl.col("len").first())
    )
    assert_gpu_result_equal(joined, engine=engine, check_row_order=False)
    # Both scans were found ordered, and their tasks put in key order (not name order).
    for order in inferred:
        files = list(dict.fromkeys(order))
        assert files == sorted(files, key=lambda f: int(f.split(".")[1]))
    assert any(len(set(order)) == 12 for order in inferred)
