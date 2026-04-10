# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

import pytest
from rapidsmpf.streaming.cudf.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    Partitioning,
)

import polars as pl

from cudf_polars.experimental.rapidsmpf.collectives.shuffle import (
    ShuffleManager,
    _is_already_partitioned,
)
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "engine",
    [
        {
            "executor_options": {
                "max_rows_per_partition": 1,
                "broadcast_join_limit": 2,
                "shuffle_method": "rapidsmpf",
            }
        },
        {
            "executor_options": {
                "max_rows_per_partition": 5,
                "broadcast_join_limit": 2,
                "shuffle_method": "rapidsmpf",
            }
        },
    ],
    indirect=True,
)
def test_join_rapidsmpf(engine) -> None:
    left = pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )
    right = pl.LazyFrame(
        {
            "xx": range(6),
            "y": [2, 4, 3] * 2,
            "zz": [1, 2] * 3,
        }
    )
    q = left.join(right, on="y", how="inner")
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "engine",
    [
        {
            "executor_options": {
                "max_rows_per_partition": 1,
                "shuffle_method": "rapidsmpf",
            }
        },
        {
            "executor_options": {
                "max_rows_per_partition": 5,
                "shuffle_method": "rapidsmpf",
            }
        },
    ],
    indirect=True,
)
def test_sort_rapidsmpf(engine) -> None:
    df = pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )
    q = df.sort(by=["y", "z"])
    assert_gpu_result_equal(q, engine=engine, check_row_order=True)


def test_is_already_partitioned():
    # Unit test for _is_already_partitioned helper
    chunks = 4
    columns = (0, 1)
    modulus = 8
    nranks = 1

    # Exact match: should return True
    metadata_match = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, modulus),
            local="inherit",
        ),
    )
    assert _is_already_partitioned(metadata_match, columns, modulus, nranks) is True

    # Different columns: should return False
    metadata_diff_cols = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme((0,), modulus),
            local="inherit",
        ),
    )
    assert (
        _is_already_partitioned(metadata_diff_cols, columns, modulus, nranks) is False
    )

    # Different local partitioning: should return False
    metadata_diff_local = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, modulus),
            local=None,
        ),
    )
    assert (
        _is_already_partitioned(metadata_diff_local, columns, modulus, nranks) is False
    )

    # Different modulus: should return False
    metadata_diff_mod = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, 16),
            local="inherit",
        ),
    )
    assert _is_already_partitioned(metadata_diff_mod, columns, modulus, nranks) is False

    # No partitioning: should return False
    metadata_none = ChannelMetadata(chunks)
    assert _is_already_partitioned(metadata_none, columns, modulus, nranks) is False

    # Local not "inherit": should return False
    metadata_local = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, modulus),
            local=HashScheme((0,), 4),
        ),
    )
    assert _is_already_partitioned(metadata_local, columns, modulus, nranks) is False


def _make_failing_insert(method_name: str):
    """Return a replacement for `method_name` that always raises on first call."""

    def failing(*args, **kwargs):
        raise RuntimeError(f"Injected failure in {method_name}")

    return failing


@pytest.fixture
def _tracking_insert_finished():
    """Fixture that patches insert_finished to track calls while still calling the real one."""
    original_finished = ShuffleManager.insert_finished
    called = []

    async def tracking(self):
        called.append(True)
        await original_finished(self)

    with patch.object(ShuffleManager, "insert_finished", tracking):
        yield called


@pytest.mark.parametrize(
    "engine",
    [
        {
            "executor_options": {
                "max_rows_per_partition": 3,
                "dynamic_planning": None,  # force Shuffle IR node path
                "broadcast_join_limit": 1,  # right has 2 partitions > 1, force shuffle
                "shuffle_method": "rapidsmpf",
            }
        }
    ],
    indirect=True,
)
def test_shuffle_insert_finished_called_on_error(engine, _tracking_insert_finished) -> None:
    """insert_finished() must be awaited even when insert_hash raises.

    Without try/finally in _global_shuffle, an exception during insert_hash
    leaves ShufflerAsync incomplete, causing std::terminate in the C++ destructor.
    """
    left = pl.LazyFrame({"x": range(15), "y": [1, 2, 3] * 5})
    right = pl.LazyFrame({"xx": range(6), "y": [2, 4, 3] * 2})
    q = left.join(right, on="y", how="inner")

    with (
        patch.object(ShuffleManager, "insert_hash", _make_failing_insert("insert_hash")),
        pytest.raises(Exception, match="Injected failure"),
    ):
        assert_gpu_result_equal(q, engine=engine, check_row_order=False)

    assert _tracking_insert_finished, "insert_finished() must be called even when insert_hash raises"


@pytest.mark.parametrize(
    "engine",
    [
        {
            "executor_options": {
                "max_rows_per_partition": 3,
                "target_partition_size": 1,  # force shuffle strategy (data exceeds target)
            }
        }
    ],
    indirect=True,
)
def test_groupby_insert_finished_called_on_error(engine, _tracking_insert_finished) -> None:
    """insert_finished() must be awaited even when groupby insert_hash raises.

    Without try/finally in _shuffle_reduce, an exception during insert_hash
    leaves ShufflerAsync incomplete, causing std::terminate in the C++ destructor.
    """
    df = pl.LazyFrame({"key": list(range(100)) * 10, "value": range(1000)})
    q = df.group_by("key").agg(pl.col("value").sum())

    with (
        patch.object(ShuffleManager, "insert_hash", _make_failing_insert("insert_hash")),
        pytest.raises(Exception, match="Injected failure"),
    ):
        assert_gpu_result_equal(q, engine=engine, check_row_order=False)

    assert _tracking_insert_finished, "insert_finished() must be called even when insert_hash raises"


@pytest.mark.parametrize(
    "engine",
    [{"executor_options": {"max_rows_per_partition": 3, "shuffle_method": "rapidsmpf"}}],
    indirect=True,
)
def test_sort_insert_finished_called_on_error(engine, _tracking_insert_finished) -> None:
    """insert_finished() must be awaited even when sort insert_split raises.

    Without try/finally in _insert_chunks_into_shuffle, an exception during
    insert_split leaves ShufflerAsync incomplete, causing std::terminate.
    """
    df = pl.LazyFrame({"x": range(15), "y": [1, 2, 3] * 5})
    q = df.sort(by=["y"])

    with (
        patch.object(ShuffleManager, "insert_split", _make_failing_insert("insert_split")),
        pytest.raises(Exception, match="Injected failure"),
    ):
        assert_gpu_result_equal(q, engine=engine, check_row_order=True)

    assert _tracking_insert_finished, "insert_finished() must be called even when insert_split raises"
