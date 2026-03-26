# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Union logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from cudf_polars.dsl.ir import Union
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.experimental.rapidsmpf.nodes import define_actor, shutdown_on_error
from cudf_polars.experimental.rapidsmpf.utils import (
    ChannelManager,
    process_children,
    recv_metadata,
    send_metadata,
)

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator


@define_actor()
async def union_node(
    context: Context,
    ir: Union,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    *chs_in: Channel[TableChunk],
) -> None:
    """
    Union node for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The Union IR node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output Channel[TableChunk].
    chs_in
        The input Channel[TableChunk]s.
    """
    ir_id = id(ir)
    n_inputs = len(chs_in)
    print(f"[DBG union_node START] ir_id={ir_id} n_inputs={n_inputs}", flush=True)
    async with shutdown_on_error(
        context, *chs_in, ch_out, trace_ir=ir, ir_context=ir_context
    ):
        # Merge and forward metadata.
        # Union loses partitioning/ordering info since sources may differ.
        # TODO: Warn users that Union does NOT preserve order?
        total_local_count = 0
        duplicated = True
        for i, ch_in in enumerate(chs_in):
            print(f"[DBG union_node RECV_META] ir_id={ir_id} ch={i}", flush=True)
            metadata = await recv_metadata(ch_in, context)
            print(f"[DBG union_node GOT_META] ir_id={ir_id} ch={i} local_count={metadata.local_count}", flush=True)
            total_local_count += metadata.local_count
            duplicated = duplicated and metadata.duplicated
        print(f"[DBG union_node SEND_META] ir_id={ir_id} total_local_count={total_local_count}", flush=True)
        await send_metadata(
            ch_out,
            context,
            ChannelMetadata(
                local_count=total_local_count,
                duplicated=duplicated,
            ),
        )
        print(f"[DBG union_node SEND_META_DONE] ir_id={ir_id}", flush=True)

        seq_num_offset = 0
        for i, ch_in in enumerate(chs_in):
            num_ch_chunks = 0
            print(f"[DBG union_node DATA_FWD_START] ir_id={ir_id} ch={i}", flush=True)
            while (msg := await ch_in.recv(context)) is not None:
                num_ch_chunks += 1
                await ch_out.send(
                    context,
                    Message(
                        msg.sequence_number + seq_num_offset,
                        TableChunk.from_message(msg).make_available_and_spill(
                            context.br(), allow_overbooking=True
                        ),
                    ),
                )
            seq_num_offset += num_ch_chunks
            print(f"[DBG union_node DATA_FWD_DONE] ir_id={ir_id} ch={i} chunks={num_ch_chunks}", flush=True)

        print(f"[DBG union_node DRAIN] ir_id={ir_id}", flush=True)
        await ch_out.drain(context)
        print(f"[DBG union_node END] ir_id={ir_id}", flush=True)


@generate_ir_sub_network.register(Union)
def _(
    ir: Union, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    # Union operation.
    # Pass-through all child chunks in channel order.

    # Process children
    nodes, channels = process_children(ir, rec)

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])

    # Add simple python node
    nodes[ir] = [
        union_node(
            rec.state["context"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            *[channels[c].reserve_output_slot() for c in ir.children],
        )
    ]
    return nodes, channels
