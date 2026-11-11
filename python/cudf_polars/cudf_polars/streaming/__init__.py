# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Streaming execution layer for cudf-polars."""

from __future__ import annotations

__all__: list[str] = []


def _unspill_off_loop() -> None:
    # Experiment (CUDF_POLARS_UNSPILL_THREAD): bring spilled chunks back to the device on a worker
    # thread instead of the event loop, so an unspill does not stall every other actor.
    import asyncio
    import os

    if not os.environ.get("CUDF_POLARS_UNSPILL_THREAD"):
        return
    import cudf_streaming
    import cudf_streaming.table_chunk as table_chunk
    from rapidsmpf.memory.buffer import MemoryType
    from rapidsmpf.streaming.core.memory_reserve_or_wait import reserve_memory

    async def make_table_chunks_available_or_wait(
        ctx, chunks, *, reserve_extra, net_memory_delta, allow_overbooking=None
    ):
        single = isinstance(chunks, table_chunk.TableChunk)
        inputs = (chunks,) if single else tuple(chunks)
        size = sum(chunk.make_available_cost() for chunk in inputs)
        res = await reserve_memory(
            ctx,
            size + reserve_extra,
            net_memory_delta=net_memory_delta,
            mem_type=MemoryType.DEVICE,
            allow_overbooking=allow_overbooking,
        )
        if all(chunk.is_available() for chunk in inputs):
            out = [chunk.make_available(res) for chunk in inputs]
        else:
            out = await asyncio.to_thread(
                lambda: [chunk.make_available(res) for chunk in inputs]
            )
        return (out[0] if single else out), res

    table_chunk.make_table_chunks_available_or_wait = (
        make_table_chunks_available_or_wait
    )
    cudf_streaming.make_table_chunks_available_or_wait = (
        make_table_chunks_available_or_wait
    )


_unspill_off_loop()
