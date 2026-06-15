# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Core RapidsMPF streaming-engine API."""

from __future__ import annotations

import dataclasses
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.core.leaf_actor import pull_from_channel

import cudf_polars.dsl.tracing
from cudf_polars.dsl.ir import (
    DataFrameScan,
    Join,
    Union,
)
from cudf_polars.dsl.traversal import CachingVisitor, traversal
from cudf_polars.streaming.actor_graph.dispatch import FanoutInfo
from cudf_polars.streaming.actor_graph.nodes import (
    generate_ir_sub_network_wrapper,
    metadata_drain_node,
)
from cudf_polars.streaming.io import StreamingScan
from cudf_polars.streaming.over import Over
from cudf_polars.utils.config import SPMDContext

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    import polars as pl

    from cudf_streaming.streaming.channel_metadata import ChannelMetadata
    from cudf_streaming.streaming.table_chunk import TableChunk
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.core.leaf_actor import DeferredMessages

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import (
        GenState,
        SubNetGenerator,
    )
    from cudf_polars.streaming.base import PartitionInfo, StatsCollector
    from cudf_polars.streaming.parallel import ConfigOptions
    from cudf_polars.utils.config import StreamingExecutor


def evaluate_logical_plan(
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
    *,
    collect_metadata: bool = False,
) -> tuple[pl.DataFrame, list[ChannelMetadata] | None]:
    """
    Evaluate a logical plan with the RapidsMPF streaming runtime.

    Parameters
    ----------
    ir
        The IR node.
    config_options
        The configuration options.
    collect_metadata
        Whether to collect runtime metadata.

    Returns
    -------
    The output DataFrame and metadata collector.
    """
    # For default_singleton, inject the process-wide DefaultSingletonEngine instance
    # into config_options before treating it as a regular SPMDEngine.
    if config_options.executor.cluster == "default_singleton":
        from cudf_polars.engine.default_singleton_engine import (
            DefaultSingletonEngine,
        )

        engine = DefaultSingletonEngine.get_or_create()
        config_options = dataclasses.replace(
            config_options,
            executor=dataclasses.replace(
                config_options.executor,
                spmd_context=SPMDContext(
                    comm=engine.comm,
                    context=engine.context,
                    py_executor=engine.py_executor,
                ),
            ),
        )

    query_id = uuid.uuid4()
    with cudf_polars.dsl.tracing.bound_contextvars(
        cudf_polars_query_id=str(query_id),
    ):
        match config_options.executor.cluster:
            case "spmd" | "default_singleton":
                from cudf_polars.engine.spmd import (
                    evaluate_pipeline_spmd_mode,
                )

                result, metadata_collector = evaluate_pipeline_spmd_mode(
                    ir,
                    config_options,
                    collect_metadata=collect_metadata,
                    query_id=query_id,
                )
            case "ray":
                from cudf_polars.engine.ray import (
                    evaluate_pipeline_ray_mode,
                )

                result, metadata_collector = evaluate_pipeline_ray_mode(
                    ir,
                    config_options,
                    collect_metadata=collect_metadata,
                    query_id=query_id,
                )
            case "dask":
                from cudf_polars.engine.dask import (
                    evaluate_pipeline_dask_mode,
                )

                result, metadata_collector = evaluate_pipeline_dask_mode(
                    ir,
                    config_options,
                    collect_metadata=collect_metadata,
                    query_id=query_id,
                )
            case other:
                raise ValueError(f"Unknown cluster mode: {other}")

    return result, metadata_collector


def determine_fanout_nodes(
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    ir_dep_count: defaultdict[IR, int],
) -> dict[IR, FanoutInfo]:
    """
    Determine which IR nodes need fanout and what type.

    Parameters
    ----------
    ir
        The root IR node.
    partition_info
        Partition information for each IR node.
    ir_dep_count
        The number of IR dependencies for each IR node.

    Returns
    -------
    Dictionary mapping IR nodes to FanoutInfo tuples where:
    - num_consumers: number of consumers
    - unbounded: whether the node needs unbounded fanout
    Only includes nodes that need fanout (i.e., have multiple consumers).
    """
    # Determine which nodes need unbounded fanout
    unbounded: set[IR] = set()

    def _mark_children_unbounded(node: IR) -> None:
        for child in node.children:
            unbounded.add(child)

    # Traverse the graph and identify nodes that need unbounded fanout
    for node in traversal([ir]):
        if node in unbounded:
            _mark_children_unbounded(node)
        elif isinstance(node, (Union, Join, Over)):
            # Union processes children sequentially; Join may broadcast one
            # side; Over buffers (or samples-then-replays) its input before
            # producing output. In every case the input source needs
            # unbounded fanout so other consumers don't block it.
            _mark_children_unbounded(node)
        elif len(node.children) > 1:
            # Check if this node is doing any broadcasting.
            # When we move to dynamic partitioning, we will need a
            # new way to indicate that a node is broadcasting 1+ children.
            counts = [partition_info[c].count for c in node.children]
            has_broadcast = any(c == 1 for c in counts) and not all(
                c == 1 for c in counts
            )
            if has_broadcast:
                # Broadcasting operation - children need unbounded fanout
                _mark_children_unbounded(node)

    # Build result dictionary: only include nodes with multiple consumers
    fanout_nodes: dict[IR, FanoutInfo] = {}
    for node, count in ir_dep_count.items():
        if count > 1:
            fanout_nodes[node] = FanoutInfo(
                num_consumers=count,
                unbounded=node in unbounded,
            )

    return fanout_nodes


def generate_network(
    context: Context,
    comm: Communicator,
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
    stats: StatsCollector,
    *,
    ir_context: IRExecutionContext,
    collective_id_map: dict[IR, list[int]],
    metadata_collector: list[ChannelMetadata] | None,
) -> tuple[list[Any], DeferredMessages]:
    """
    Translate the IR graph to a RapidsMPF streaming network.

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator the network generation is collective over.
    ir
        The IR node.
    partition_info
        The partition information.
    config_options
        The configuration options.
    stats
        Statistics collector.
    ir_context
        The execution context for the IR node.
    collective_id_map
        The mapping of IR nodes to lists of collective IDs.
    metadata_collector
        The list to collect the final metadata.
        This list will be mutated when the network is executed.
        If None, metadata will not be collected.

    Returns
    -------
    The network nodes and output hook.
    """
    # Count the number of IO nodes and the number of IR dependencies
    num_io_nodes: int = 0
    ir_dep_count: defaultdict[IR, int] = defaultdict(int)
    for node in traversal([ir]):
        if isinstance(node, (DataFrameScan, StreamingScan)):
            num_io_nodes += 1
        for child in node.children:
            ir_dep_count[child] += 1

    # Determine which nodes need fanout
    fanout_nodes = determine_fanout_nodes(ir, partition_info, ir_dep_count)

    # Get max_io_threads from config (default: 2)
    max_io_threads_global = config_options.executor.max_io_threads
    max_io_threads_local = max(1, max_io_threads_global // max(1, num_io_nodes))

    # Generate the network
    state: GenState = {
        "context": context,
        "comm": comm,
        "config_options": config_options,
        "partition_info": partition_info,
        "fanout_nodes": fanout_nodes,
        "ir_context": ir_context,
        "max_io_threads": max_io_threads_local,
        "stats": stats,
        "collective_id_map": collective_id_map,
    }
    mapper: SubNetGenerator = CachingVisitor(
        generate_ir_sub_network_wrapper, state=state
    )
    nodes_dict, channels = mapper(ir)
    ch_out = channels[ir].reserve_output_slot()

    # Add node to drain metadata before pull_from_channel
    # (since pull_from_channel doesn't handle metadata messages)
    ch_final_data: Channel[TableChunk] = context.create_channel()
    drain_node = metadata_drain_node(
        context,
        comm,
        ir,
        ir_context,
        ch_out,
        ch_final_data,
        metadata_collector,
    )

    # Add final node to pull from the output data channel
    output_node, output = pull_from_channel(context, ch_in=ch_final_data)

    # Flatten the nodes dictionary into a list for run_actor_network
    nodes: list[Any] = [node for node_list in nodes_dict.values() for node in node_list]
    nodes.extend([drain_node, output_node])

    # Stream tracing: emit the static actor graph topology.
    from cudf_polars.streaming.actor_graph.events import (
        STREAM_TRACE_ENABLED,
        get_active_buffer,
    )

    if STREAM_TRACE_ENABLED:
        _buf = get_active_buffer()
        if _buf is not None:
            _emit_actor_graph_event(
                _buf, nodes_dict, channels, ir_context,
                actor_scan_meta=state.get("_actor_scan_meta", {}),
                fanout_wiring=state.get("_fanout_wiring", {}),
            )

    # Return network and output hook
    return nodes, output


def _emit_actor_graph_event(
    buf: Any,
    nodes_dict: dict[Any, list[Any]],
    channels: dict[Any, Any],
    ir_context: Any,
    actor_scan_meta: dict | None = None,
    fanout_wiring: dict | None = None,
) -> None:
    """
    Build and emit an ActorGraphEvent from the generate_network artifacts.

    Each IR node maps to one or more actors in nodes_dict. The first entry
    is the "real" actor; extras (index > 0) are fanout actors inserted by
    generate_ir_sub_network_wrapper. Channels are registered in the
    ChannelRegistry by ChannelManager.__init__; we look them up here to
    assign source/sink actor IDs.
    """
    import itertools as _itertools
    import time as _time

    from cudf_polars.streaming.actor_graph.channel_registry import get_registry
    from cudf_polars.streaming.actor_graph.events import ActorGraphEvent, write_graph_json

    registry = get_registry(buf.query_id)
    if registry is None:
        return

    actor_counter = _itertools.count(1)

    # First pass: assign actor_instance_ids and collect per-IR channel lists.
    ir_to_actor_ids: dict[Any, list[int]] = {}
    actor_records: list[dict] = []

    for ir_node, actor_list in nodes_dict.items():
        ir_id = ir_node.get_stable_id() if hasattr(ir_node, "get_stable_id") else None
        ir_type = type(ir_node).__name__

        for i, _actor in enumerate(actor_list):
            aid = next(actor_counter)
            if ir_node not in ir_to_actor_ids:
                ir_to_actor_ids[ir_node] = []
            ir_to_actor_ids[ir_node].append(aid)

            # Fanout actors are entries beyond index 0 for a given IR node.
            actual_ir_type = ir_type if i == 0 else f"fanout_{i}"
            actual_ir_id = ir_id if i == 0 else None

            record: dict = {
                "actor_instance_id": aid,
                "ir_id": actual_ir_id,
                "ir_type": actual_ir_type,
                "input_channel_ids": [],   # filled in second pass
                "output_channel_ids": [],  # filled in second pass
            }
            if i == 0 and actor_scan_meta:
                smeta = actor_scan_meta.get(id(ir_node))
                if smeta is not None:
                    record["scan_meta"] = smeta
            # Add IR-specific metadata for the primary actor
            if i == 0:
                try:
                    from cudf_polars.dsl.ir import Join, GroupBy, Sort, Filter
                    ir_meta = {}
                    if isinstance(ir_node, Join):
                        ir_meta["how"] = ir_node.options[0] if ir_node.options else "?"
                        ir_meta["left_on"] = [str(e.name) for e in ir_node.left_on]
                        ir_meta["right_on"] = [str(e.name) for e in ir_node.right_on]
                    elif isinstance(ir_node, GroupBy):
                        ir_meta["keys"] = [str(e.name) for e in ir_node.keys]
                        ir_meta["aggs"] = [str(e.name) for e in ir_node.agg_requests]
                    elif isinstance(ir_node, Sort):
                        import pylibcudf as plc
                        ir_meta["by"] = [str(e.name) for e in ir_node.by]
                        ir_meta["desc"] = [o == plc.types.Order.DESCENDING for o in ir_node.order]
                    elif isinstance(ir_node, Filter):
                        ir_meta["mask"] = str(ir_node.mask)
                    if ir_meta:
                        record["ir_meta"] = ir_meta
                except Exception:
                    pass
            actor_records.append(record)

    # Build actor_id -> record index for fast lookup.
    aid_to_record: dict[int, dict] = {r["actor_instance_id"]: r for r in actor_records}

    # Second pass: wire up input/output channel IDs using the channel map.
    # For fanout IR nodes, channels[ir] holds the fanout actor's output manager
    # (N slots to N consumers). The pre-fanout channel (primary→fanout) is saved
    # in fanout_wiring. We emit both edges with correct actor IDs.
    channel_records: list[dict] = []
    _fw = fanout_wiring or {}

    for ir_node, ch_manager in channels.items():
        out_actor_ids = ir_to_actor_ids.get(ir_node, [])
        if not out_actor_ids:
            continue

        primary_out_aid = out_actor_ids[0]
        fw = _fw.get(id(ir_node))

        if fw is not None and len(out_actor_ids) >= 2:
            # Fanout case: wire primary→fanout (pre-fanout channel) and
            # fanout→consumers (the N output slots now in ch_manager).
            fanout_aid = out_actor_ids[1]

            # primary → fanout edge
            for ch in fw["pre_fanout_manager"]._channel_slots:
                cid = registry.get(ch)
                if cid is None:
                    continue
                if primary_out_aid in aid_to_record:
                    aid_to_record[primary_out_aid]["output_channel_ids"].append(cid)
                if fanout_aid in aid_to_record:
                    aid_to_record[fanout_aid]["input_channel_ids"].append(cid)
                channel_records.append(
                    {
                        "channel_id": cid,
                        "source_actor_instance_id": primary_out_aid,
                        "sink_actor_instance_id": fanout_aid,
                        "is_metadata": False,
                    }
                )

            # fanout → consumer edges (sinks resolved below)
            for ch in ch_manager._channel_slots:
                cid = registry.get(ch)
                if cid is None:
                    continue
                if fanout_aid in aid_to_record:
                    aid_to_record[fanout_aid]["output_channel_ids"].append(cid)
                channel_records.append(
                    {
                        "channel_id": cid,
                        "source_actor_instance_id": fanout_aid,
                        "sink_actor_instance_id": None,  # resolved below
                        "is_metadata": False,
                    }
                )
        else:
            # Normal case: channels belong to the primary actor.
            for ch in ch_manager._channel_slots:
                cid = registry.get(ch)
                if cid is None:
                    continue
                if primary_out_aid in aid_to_record:
                    aid_to_record[primary_out_aid]["output_channel_ids"].append(cid)
                channel_records.append(
                    {
                        "channel_id": cid,
                        "source_actor_instance_id": primary_out_aid,
                        "sink_actor_instance_id": None,  # resolved below
                        "is_metadata": False,
                    }
                )

    # Resolve sink actor IDs: for each IR node, its children's output channels
    # are its input channels. For fanout children, each parent claims exactly one
    # slot (the slot it reserved, in visit order matching nodes_dict insertion order).
    from collections import defaultdict as _defaultdict

    cid_to_record: dict[int, dict] = {c["channel_id"]: c for c in channel_records}
    child_claim_index: dict[int, int] = _defaultdict(int)

    for ir_node, actor_list in nodes_dict.items():
        if not hasattr(ir_node, "children"):
            continue
        for child_ir in ir_node.children:
            if child_ir not in channels:
                continue
            ch_manager = channels[child_ir]
            sink_actor_ids = ir_to_actor_ids.get(ir_node, [])
            primary_sink_aid = sink_actor_ids[0] if sink_actor_ids else None

            # Each parent gets exactly one output slot from the child's manager,
            # in the order parents reserved slots (matching nodes_dict visit order).
            slot_idx = child_claim_index[id(child_ir)]
            child_claim_index[id(child_ir)] += 1

            if slot_idx < len(ch_manager._channel_slots):
                ch = ch_manager._channel_slots[slot_idx]
                cid = registry.get(ch)
                if cid is None:
                    continue
                if cid in cid_to_record and cid_to_record[cid]["sink_actor_instance_id"] is None:
                    cid_to_record[cid]["sink_actor_instance_id"] = primary_sink_aid
                if primary_sink_aid is not None and primary_sink_aid in aid_to_record:
                    aid_to_record[primary_sink_aid]["input_channel_ids"].append(cid)

    rank = 0
    nranks = 1
    if ir_context is not None and hasattr(ir_context, "query_id"):
        try:
            from rapidsmpf.communicator.communicator import Communicator  # type: ignore[import]
        except ImportError:
            pass

    graph_event = ActorGraphEvent(
        timestamp_ns=_time.monotonic_ns(),
        query_id=buf.query_id,
        actors=actor_records,
        channels=channel_records,
        rank=rank,
        nranks=nranks,
    )
    buf.emit(graph_event)
    write_graph_json(graph_event, buf.output_dir, buf.query_id)
