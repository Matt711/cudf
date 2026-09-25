# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU frontend core."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import socket
import threading
import time
import uuid
import weakref
from typing import TYPE_CHECKING, Any, ClassVar, Self, TypeVar

import cuda.core
import kvikio

import polars as pl

import pylibcudf as plc
from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.coll import AllGather
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.memory.pinned_memory_resource import (
    is_pinned_memory_resources_supported,
)
from rapidsmpf.statistics import Statistics
from rapidsmpf.streaming.core.actor import run_actor_network

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IRExecutionContext
from cudf_polars.dsl.utils.io import (
    attach_cached_parquet_metadata,
    prefetch_parquet_file_metadata_for_ir,
)
from cudf_polars.quent._plan import build_plan, build_quent_operator_map
from cudf_polars.streaming.actor_graph.collectives import ReserveOpIDs
from cudf_polars.streaming.actor_graph.collectives.common import reserve_op_id
from cudf_polars.streaming.actor_graph.core import generate_network
from cudf_polars.streaming.actor_graph.tracing import log_query_plan
from cudf_polars.streaming.actor_graph.utils import empty_table_chunk
from cudf_polars.streaming.base import StatsCollector
from cudf_polars.streaming.parallel import lower_ir_graph_with_node_map
from cudf_polars.streaming.statistics import collect_statistics
from cudf_polars.streaming.utils import _concat
from cudf_polars.utils.config import get_total_device_memory

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping
    from concurrent.futures import Executor, ThreadPoolExecutor

    import rapidsmpf.config
    from cudf_streaming.channel_metadata import ChannelMetadata
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.context import Context

    import cudf_polars.quent._logging
    import cudf_polars.quent._types
    from cudf_polars.dsl.ir import IR
    from cudf_polars.dsl.translate import Translator
    from cudf_polars.quent._context import LocalQuentContext
    from cudf_polars.streaming.base import PartitionInfo
    from cudf_polars.streaming.parallel import ConfigOptions
    from cudf_polars.utils.config import StreamingExecutor

# Process-level singleton IoContextRegistry.  Created on first use when
# SIRIUS_DATASOURCE=1, kept alive for the entire process.  Per the design,
# reset_caches() is only called between benchmark iterations (not between
# queries), so this must NOT be destroyed after each query.
_sirius_registry: Any = None

# Persistent thread pool for proactive background S3 prefetch.  Submits
# full-file fadvise for known dataset paths NOT in the current query so that
# cross-query data is cached during the previous query's execution, reducing
# Q+1 cache miss rate.  Only active when SIRIUS_PROACTIVE_PREFETCH=1.
_sirius_proactive_pool: Any = None  # concurrent.futures.ThreadPoolExecutor


def _get_proactive_pool() -> Any:
    """Return (creating if needed) the process-level proactive prefetch pool."""
    global _sirius_proactive_pool
    if _sirius_proactive_pool is None:
        from concurrent.futures import ThreadPoolExecutor

        _sirius_proactive_pool = ThreadPoolExecutor(
            max_workers=16, thread_name_prefix="sirius-proactive"
        )
    return _sirius_proactive_pool


def _submit_proactive_prefetch(
    registry: Any,
    all_known: dict[str, Any],
    skip_paths: set[str],
) -> None:
    """Submit background full-file fadvise for paths known but not in the current query.

    Runs concurrently with the NEXT query's planning and execution, giving the
    Sirius cache download time for data needed by future queries.  Matches the
    cross-query lookahead prefetch that Sirius native does via prepare_for_query().
    """
    pool = _get_proactive_pool()
    for path, info in list(all_known.items()):
        if path in skip_paths or info.size is None:
            continue
        file_size = info.size

        def _do(p: str = path, s: int = file_size, reg: Any = registry) -> None:
            try:
                ds = reg.open_datasource(p)
                ds.fadvise([(0, s)])
            except Exception:
                pass

        pool.submit(_do)


T = TypeVar("T")


def reset_statistics_from_options(
    statistics: Statistics, options: Options
) -> Statistics:
    """
    Reset the enabled state of a statistics object from options.

    Parameters
    ----------
    statistics
        Statistics to reset.
    options
        Options providing new enabled setting.

    Returns
    -------
    Statistics
        Reset statistics object.

    Notes
    -----
    Does not clear the statistics.
    """
    if Statistics.from_options(options).enabled:
        statistics.enable()
    else:
        statistics.disable()
    return statistics


def make_kvikio_monitor(*, enabled: bool) -> kvikio.SummaryMonitor | None:
    """
    Create a kvikio I/O monitor if ``enabled``.

    Parameters
    ----------
    enabled
        Whether to count, from the ``kvikio_statistics`` executor option.

    Returns
    -------
    kvikio.SummaryMonitor
        A monitor, already counting, if statistics are enabled.
    None
        If they are not.

    Notes
    -----
    kvikio has no enable/disable: the existence of a monitor is what turns
    counting on for the process, so "disabled" means "no monitor".

    With :class:`~cudf_polars.engine.spmd.SPMDEngine` the monitor counts every
    thread's kvikio I/O in the script's process, so user code performing kvikio
    reads is counted too. It cannot attribute I/O to a particular query.
    """
    if not enabled:
        return None
    return kvikio.SummaryMonitor()


def reset_kvikio_monitor(
    monitor: kvikio.SummaryMonitor | None, *, enabled: bool
) -> kvikio.SummaryMonitor | None:
    """
    Bring a kvikio I/O monitor into line with a new ``enabled`` setting.

    Parameters
    ----------
    monitor
        The rank's existing monitor, if it has one.
    enabled
        Whether to count, from the ``kvikio_statistics`` executor option.

    Returns
    -------
    kvikio.SummaryMonitor
        The reset or newly created monitor, if statistics are enabled.
    None
        If they are not, in which case any existing monitor has been stopped.
    """
    if not enabled:
        if monitor is not None:
            monitor.stop()
        return None
    if monitor is None:
        return kvikio.SummaryMonitor()
    monitor.reset()
    return monitor


def take_io_summary(
    monitor: kvikio.SummaryMonitor | None, *, clear: bool
) -> kvikio.Summary | None:
    """
    Read a rank's I/O totals, optionally restarting the measured span.

    Parameters
    ----------
    monitor
        The rank's monitor, or ``None`` if it is not counting.
    clear
        If ``True``, reset the monitor after reading, so the returned summary
        is the last word on the span that just ended.

    Returns
    -------
    kvikio.Summary
        The totals so far.
    None
        If ``monitor`` is ``None``.

    Notes
    -----
    ``None`` means "this rank was not counting", which is not the same as a
    zeroed summary meaning "this rank did no I/O".

    ``get()`` and ``reset()`` are two separate calls, so an operation
    completing between them is counted in the returned summary but dropped
    from the next one. Use ``kvikio.Summary.since(previous)`` if you need
    gapless differencing.
    """
    if monitor is None:
        return None
    # Read before the reset, so the returned summary still carries the span.
    summary = monitor.get()
    if clear:
        monitor.reset()
    return summary


def resolve_rapidsmpf_options(rapidsmpf_options: Options | None) -> Options:
    """
    Resolve ``rapidsmpf_options`` and apply cross-frontend defaults.

    If ``None`` is passed, constructs an ``Options`` instance from
    environment variables. Then applies defaults that should be consistent
    across SPMD, Ray, and Dask. Defaults are set via
    ``Options.insert_if_absent``, so explicit values or environment
    variables always take precedence.

    Defaults applied:

    - ``num_streaming_threads=4``: moderate worker count for the rapidsmpf
      streaming runtime, shared across frontends.
    - ``pinned_memory``, ``pinned_initial_pool_size=0``: pinned host memory
      enabled by default, but only on systems that support it (CUDA 12.6+
      with async memory pool support).

    Parameters
    ----------
    rapidsmpf_options
        Existing options to resolve, or ``None`` to construct from environment
        variables.

    Returns
    -------
    Options
        Resolved options with cross-frontend defaults applied.
    """
    if rapidsmpf_options is None:
        rapidsmpf_options = Options(get_environment_variables())

    pinned_memory_default = (
        "true" if is_pinned_memory_resources_supported() else "false"
    )
    rapidsmpf_options.insert_if_absent(
        {
            "num_streaming_threads": "4",
            "pinned_memory": pinned_memory_default,
            "pinned_initial_pool_size": "0",
        }
    )
    return rapidsmpf_options


@dataclasses.dataclass(frozen=True)
class ClusterInfo:
    """
    Diagnostic information about a single rank in the cluster.

    Attributes
    ----------
    pid
        Process ID of the current rank.
    hostname
        Hostname of the machine running this rank.
    cuda_visible_devices
        Value of ``CUDA_VISIBLE_DEVICES``, or ``None`` if unset.
    gpu_uuid
        UUID of the current CUDA device.
    device_memory
        Total device memory in bytes, or ``None`` if unknown.
    """

    pid: int
    hostname: str
    cuda_visible_devices: str | None
    gpu_uuid: str
    device_memory: int | None = None

    @classmethod
    def local(cls) -> ClusterInfo:
        """
        Build a :class:`ClusterInfo` for the current process and GPU.

        Returns
        -------
        Diagnostic information for this rank.
        """
        return cls(
            pid=os.getpid(),
            hostname=socket.gethostname(),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            gpu_uuid=cuda.core.Device().uuid,
            device_memory=get_total_device_memory(),
        )


class StreamingEngine(pl.GPUEngine):
    """
    Base class for multi-GPU Polars engines.

    The engine manages the lifecycle of a streaming execution and can
    be used as a context manager. On exit, :meth:`shutdown` is called.

    Notes
    -----
    The engine must be created and shut down on the same thread. In particular,
    destruction and context manager exit must occur on the thread that created
    the instance.

    Creating an engine sets the kvikio remote I/O backend to
    ``kvikio.RemoteIOBackend.MULTI_POLL`` by default (see the
    ``kvikio_remote_io_backend`` executor option), along with the
    ``kvikio_task_size`` executor option (16 MiB under ``MULTI_POLL``, 64 MiB
    under ``EASY_THREADPOOL``). Because kvikio's configuration is a global
    singleton, this overrides mutable prior ``kvikio.defaults.set(...)`` calls
    made in the process. The ``MULTI_POLL`` reactor settings are process-lifetime
    values: after the first remote I/O, subsequent engines must use the same
    values. When the backend is ``EASY_THREADPOOL``, engine creation
    also configures kvikio's thread pool (default 256 threads), which blocks
    any concurrent kvikio IO in the process until in-flight IO completes. Use
    the ``kvikio_nthreads`` executor option or the ``KVIKIO_NTHREADS``
    environment variable to control the thread count. Under ``MULTI_POLL``,
    cudf-polars does not resolve a thread-pool size at all (kvikio itself may
    still honor ``KVIKIO_NTHREADS`` via its own deferred default); remote I/O
    concurrency is instead controlled by the ``kvikio_reactor_count``,
    ``kvikio_reactor_dispatch``, and ``kvikio_request_ceiling`` executor
    options.

    Parameters
    ----------
    nranks
        Number of ranks (workers or GPUs) in the cluster.
    executor_options
        Executor-specific options (e.g. ``max_rows_per_partition``).
    engine_options
        Engine-specific keyword arguments (e.g. ``raise_on_fail``,
        ``parquet_options``).
    exit_stack
        A :class:`contextlib.ExitStack` whose registered contexts are closed
        when :meth:`shutdown` is called. If ``None``, an empty stack is created.
    """

    _quent_logger: cudf_polars.quent._logging.QuentLogger | None
    rapidsmpf_options: rapidsmpf.config.Options
    # Process-wide registry of every live :class:`StreamingEngine`. Used by
    # :class:`DefaultSingletonEngine` to enforce that no other engine is
    # alive when the singleton is constructed.
    _active_engines: ClassVar[weakref.WeakSet[StreamingEngine]] = weakref.WeakSet()
    _active_engines_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        *,
        nranks: int,
        executor_options: dict[str, Any],
        engine_options: dict[str, Any],
        exit_stack: contextlib.ExitStack | None = None,
    ):
        # Refuse to construct if a ``DefaultSingletonEngine`` is alive
        # (no-op for the singleton itself).
        from cudf_polars.engine.default_singleton_engine import (
            check_no_live_default_singleton,
        )

        check_no_live_default_singleton(self)
        self._nranks = nranks
        self._quent_events_raw: list[dict[str, Any]] = []  # populated on shutdown
        self._exit_stack: contextlib.ExitStack | None = (
            exit_stack or contextlib.ExitStack()
        )

        # Gather `min_device_size` from the cluster
        cluster_infos: list[ClusterInfo] = self.gather_cluster_info()
        device_memories = [info.device_memory for info in cluster_infos]
        executor_options["min_device_size"] = (
            None
            if any(dm is None for dm in device_memories)
            else min(device_memories, default=None)  # type: ignore[type-var]  # (None entries excluded by prior check)
        )

        # allow_gpu_sharing is consumed here since polars' GPUEngine doesn't
        # accept it.
        engine_options = dict(engine_options)
        allow_gpu_sharing = engine_options.pop("allow_gpu_sharing", False)
        super().__init__(
            executor="streaming",
            executor_options=executor_options,
            **engine_options,
        )
        if nranks > 1 and not allow_gpu_sharing:
            uuids = [info.gpu_uuid for info in cluster_infos]
            if len(uuids) != len(set(uuids)):
                raise RuntimeError(
                    "Multiple ranks share the same GPU (UUID collision detected). "
                    f"UUIDs: {uuids}. Set allow_gpu_sharing=True to allow this."
                )
        with StreamingEngine._active_engines_lock:
            StreamingEngine._active_engines.add(self)

    @classmethod
    def _active_engine_count(cls) -> int:
        """
        Return the number of currently-live :class:`StreamingEngine` instances.

        "Live" means constructed and not yet shut down (or garbage collected).
        The count is process-wide and shared across all subclasses.

        Returns
        -------
        Number of live engines, including ``self`` if called on a live
        instance.
        """
        with StreamingEngine._active_engines_lock:
            return len(StreamingEngine._active_engines)

    @property
    def nranks(self) -> int:
        """
        Number of ranks (for example GPUs or workers) in the cluster.

        Local execution without a cluster returns 1.

        Returns
        -------
        Number of ranks.
        """
        return self._nranks

    def gather_cluster_info(self) -> list[ClusterInfo]:
        """
        Collect diagnostic information from every rank.

        Returns
        -------
        List of :class:`ClusterInfo`, one per rank.
        """
        raise NotImplementedError

    def gather_statistics(self, *, clear: bool = False) -> list[Statistics]:
        """
        Collect statistics from every rank.

        Parameters
        ----------
        clear
            If ``True``, clear each rank's statistics after gathering.

        Returns
        -------
        List of :class:`~rapidsmpf.statistics.Statistics`, one per rank,
        ordered by rank index.
        """
        raise NotImplementedError

    def gather_io_summary(self, *, clear: bool = False) -> dict[int, kvikio.Summary]:
        """
        Collect kvikio I/O statistics from every rank.

        Parameters
        ----------
        clear
            If ``True``, restart each rank's measured span after reading, so
            the next call describes only what followed this one.

        Returns
        -------
        A :class:`kvikio.Summary` per rank, keyed by rank index and in rank
        order. A rank that is not counting is absent, so the result is empty
        unless the ``statistics`` option is enabled.

        Examples
        --------
        >>> for rank, summary in engine.gather_io_summary().items():  # doctest: +SKIP
        ...     print(f"--- rank {rank} ---")
        ...     print(summary)
        """
        raise NotImplementedError

    def global_statistics(self, *, clear: bool = False) -> Statistics:
        """
        Collect statistics from every rank and merge them into a single global statistics.

        Parameters
        ----------
        clear
            If ``True``, clear each rank's statistics after gathering.

        Returns
        -------
        A merged :class:`~rapidsmpf.statistics.Statistics`: per-stat counts
        and values are summed, maxima are reduced with ``max``. Formatters
        are taken from rank 0.
        """
        return Statistics.merge(self.gather_statistics(clear=clear))

    def _reset(
        self,
        *,
        rapidsmpf_options: Options | None = None,
        executor_options: dict[str, Any] | None = None,
        engine_options: dict[str, Any] | None = None,
    ) -> None:
        """
        Reset the engine with new options, keeping cluster resources alive.

        The following inputs are fixed at construction time and cannot change:
          - ``num_ranks``
          - ``num_py_executors`` (in ``executor_options``)
          - ``hardware_binding`` (in ``engine_options``)
          - ``memory_resource_config`` (in ``engine_options``)

        Subclasses must override this method. The override should:
          1. Raise :class:`RuntimeError` if the engine is already shut down.
          2. Call ``super()._reset(...)`` to apply the universal option validation below.
          3. Perform the backend-specific rebuild.

        Parameters
        ----------
        rapidsmpf_options
            New :class:`Options` for each rank's :class:`Context`.
            ``None`` is treated as an empty dict.
        executor_options
            New executor options for the polars ``GPUEngine`` layer.
            ``None`` is treated as an empty dict.
        engine_options
            New engine options for the polars ``GPUEngine`` layer.
            ``None`` is treated as an empty dict.

        Raises
        ------
        ValueError
            If ``executor_options`` or ``engine_options`` contains a
            construction-time-only key (see list above), or if a
            reserved key is set (via :func:`check_reserved_keys`).
        """
        executor_options = executor_options or {}
        engine_options = engine_options or {}
        check_reserved_keys(executor_options, engine_options)

        _disallowed_exec = {"num_py_executors"} & executor_options.keys()
        if _disallowed_exec:
            raise ValueError(
                f"executor_options keys {sorted(_disallowed_exec)} cannot be "
                "changed via _reset(). Construct a fresh engine instead."
            )
        _disallowed_engine = {
            "hardware_binding",
            "memory_resource_config",
        } & engine_options.keys()
        if _disallowed_engine:
            raise ValueError(
                f"engine_options keys {sorted(_disallowed_engine)} cannot be "
                "changed via _reset(). Construct a fresh engine instead."
            )

    def shutdown(self) -> None:
        """
        Shut down engine and release all owned resources.

        Idempotent: safe to call more than once. Must be called on the same
        thread that created the engine.
        """
        if self._exit_stack is None:
            return  # already shut down
        try:
            self._exit_stack.close()
        finally:
            self._exit_stack = None
            self.device = None
            self.memory_resource = None
            self.config = {}
            with StreamingEngine._active_engines_lock:
                StreamingEngine._active_engines.discard(self)

    def __enter__(self) -> Self:
        """Enter the context manager, returning ``self``."""
        return self

    def __exit__(self, *_: object) -> None:
        """Exit the context manager, calling :meth:`shutdown`."""
        self.shutdown()

    @property
    def _quent_events(self) -> list[dict[str, Any]]:
        """Return all Quent telemetry events collected during the engine's lifecycle."""
        # Not ready to make this public yet.
        return [x["event"] for x in self._quent_events_raw]

    def _run(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> list[T]:
        """
        Execute a function on all ranks.

        Parameters
        ----------
        func
            Function to execute.
        args
            Arguments to pass to the function.
        kwargs
            Keyword arguments to pass to the function.

        Returns
        -------
        List of results from calling ``func``, one per rank.
        """
        raise NotImplementedError


def _find_memory_error(exc: BaseException) -> MemoryError | None:
    """Recursively search for MemoryErrors."""
    if isinstance(exc, MemoryError):
        return exc
    elif isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            if (mem_error := _find_memory_error(sub)) is not None:
                return mem_error
    return None


def execute_ir_on_rank(
    ctx: Context,
    comm: Communicator,
    ir: IR,
    ir_context: IRExecutionContext,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions[StreamingExecutor],
    stats: StatsCollector,
    collective_id_map: dict[IR, list[int]],
    *,
    quent_operator_map: dict[IR, cudf_polars.quent._types.Operator] | None = None,
    local_quent_context: LocalQuentContext | None = None,
) -> tuple[DataFrame, list[ChannelMetadata]]:
    """
    Execute a Polars IR query on a single rank's GPU.

    Shared implementation used by the frontends. Each frontend acquires its local
    ``ctx``, ``comm``, and ``py_executor`` from its own per-rank state and delegates
    to this function for the actual execution.

    Parameters
    ----------
    ctx
        The active RapidsMPF streaming context for this rank.
    comm
        The active RapidsMPF communicator for this rank.
    ir
        Root IR node describing the query.
    ir_context
        Execution context reused across scan-task execution.
    partition_info
        Per-node partition metadata.
    config_options
        Executor configuration forwarded from the client.
    stats
        Statistics collector.
    collective_id_map
        Mapping from IR nodes to their pre-allocated collective operation IDs.
    quent_operator_map
        Mapping from IR nodes to their Quent operators, or ``None`` when tracing
        is disabled.
    local_quent_context
        The local Quent context for this rank, or ``None`` when tracing is
        disabled.

    Returns
    -------
    result
        This rank's output fragment as a GPU-resident :class:`~cudf_polars.containers.DataFrame`.
    metadata
        Collected channel metadata.
    """
    metadata_collector: list[ChannelMetadata] = []

    nodes, output = generate_network(
        ctx,
        comm,
        ir,
        partition_info,
        config_options,
        stats,
        ir_context=ir_context,
        collective_id_map=collective_id_map,
        metadata_collector=metadata_collector,
        quent_operator_map=quent_operator_map,
        local_quent_context=local_quent_context,
    )

    try:
        run_actor_network(ctx, actors=nodes)
    except (MemoryError, BaseExceptionGroup) as e:
        if (mem_error := _find_memory_error(e)) is not None:
            target_partition_size = config_options.executor.target_partition_size
            hint = (
                f"Try lowering `target_partition_size` (current {target_partition_size}) "
                f"and/or RAPIDSMPF_SPILL_DEVICE_LIMIT (default '80%') to reduce peak memory."
                f"\nSee https://docs.nvidia.com/cudf/latest/cudf_polars/memory_errors/ "
                f"for troubleshooting guidance."
                f"\nOriginal error:\n{mem_error}"
            )
            raise MemoryError(hint) from e
        else:
            raise

    messages = output.release()
    chunks = [
        TableChunk.from_message(msg, br=ctx.br()).make_available_and_spill(
            ctx.br(), allow_overbooking=True
        )
        for msg in messages
    ]
    if chunks:
        dfs = [
            DataFrame.from_table(
                chunk.table_view(),
                list(ir.schema.keys()),
                list(ir.schema.values()),
                chunk.stream,
            )
            for chunk in chunks
        ]
        df = _concat(*dfs, context=ir_context)
    else:
        stream = ir_context.get_cuda_stream()
        chunk = empty_table_chunk(ir, ctx, stream)
        df = DataFrame.from_table(
            chunk.table_view(),
            list(ir.schema.keys()),
            list(ir.schema.values()),
            stream,
        )
    return df, metadata_collector


_RESERVED_EXECUTOR_KEYS: frozenset[str] = frozenset(
    {"cluster", "spmd_context", "ray_context", "dask_context"}
)
_RESERVED_ENGINE_KEYS: frozenset[str] = frozenset({"memory_resource", "executor"})


def check_reserved_keys(
    executor_options: dict[str, Any],
    engine_options: dict[str, Any],
) -> None:
    """
    Raise :exc:`TypeError` if any reserved keys are present in the option dicts.

    Parameters
    ----------
    executor_options
        Executor-specific options to validate.
    engine_options
        Engine-specific options to validate.

    Raises
    ------
    TypeError
        If ``executor_options`` contains any reserved key.
    TypeError
        If ``engine_options`` contains any reserved key.
    """
    if bad := _RESERVED_EXECUTOR_KEYS & executor_options.keys():
        raise TypeError(f"executor_options may not contain reserved keys: {bad}")
    if bad := _RESERVED_ENGINE_KEYS & engine_options.keys():
        raise TypeError(f"engine_options may not contain reserved keys: {bad}")


def all_gather_host_data(
    comm: Communicator,
    br: BufferResource,
    op_id: int,
    data: bytes | bytearray,
) -> list[bytes]:
    """
    Gather host data from every rank using an AllGather collective.

    Each rank contributes a buffer of host bytes; every rank receives back
    an ordered list containing the contributions from all ranks (index `i`
    holds the bytes sent by rank `i`).

    This function is blocking: all ranks must call it, and each rank
    waits until the collective completes. The input buffer is copied
    and cannot be stream-ordered.

    Parameters
    ----------
    comm
        The communicator shared by all participating ranks.
    br
        Buffer resource for memory allocation.
    op_id
        Unique operation identifier for this collective.
    data
        Host-side buffer to broadcast from this rank.  Accepts any object
        that implements the buffer protocol (``bytes``, ``bytearray``,
        ``memoryview``, etc.).

    Returns
    -------
    List of bytes, one element per rank, ordered by rank index.
    """
    allgather = AllGather(comm=comm, op_id=op_id, br=br)
    # TODO: Make AllGather (bulk) a context manager so this becomes
    # with AllGather(...) as ag:
    #     ag.insert(0, PackedData.from_host_bytes(data, br))
    # results = ag.wait_and_extract(ordered=True)
    try:
        allgather.insert(0, PackedData.from_host_bytes(data, br))
    finally:
        allgather.insert_finished()
    results = allgather.wait_and_extract(ordered=True)
    return [r.to_host_bytes() for r in results]


def allgather_stats(
    comm: Communicator,
    br: BufferResource,
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
    executor: Executor,
) -> StatsCollector:
    """
    Collect scan statistics on rank 0 and distribute to all ranks.

    When ``comm.nranks == 1`` the allgather is skipped and statistics are
    collected locally.

    Parameters
    ----------
    comm
        Communicator shared by all participating ranks.
    br
        Buffer resource for the allgather allocation.
    ir
        Root of the pre-lowered IR graph (same object on every rank).
    config_options
        Executor configuration.
    executor: concurrent.futures.Executor
        Executor to use for IO operations. This function does not start
        or shutdown the executor.

    Returns
    -------
    A :class:`StatsCollector` valid for the local rank's IR node objects.
    """
    if comm.nranks == 1:
        return collect_statistics(ir, config_options, executor)

    if comm.rank == 0:
        stats = collect_statistics(ir, config_options, executor)
        data = json.dumps(stats.serialize(ir)).encode()
    else:
        data = b""

    with reserve_op_id() as op_id:
        all_data = all_gather_host_data(comm, br, op_id, data)

    if comm.rank == 0:
        return stats
    return StatsCollector.deserialize(json.loads(all_data[0]), ir)


_query_reset_t: float | None = None

# Process-level datasource cache: reuses SiriusDatasource base objects across queries.
# open_datasource() takes ~10-14ms per S3 path; caching here avoids re-paying that cost
# for files already opened in a prior query. Safe across reset_caches() calls because
# reset_caches() only evicts pinned block data, not the HTTP client/file handle state.
_global_datasource_cache: dict[str, Any] = {}


def get_query_reset_t() -> float | None:
    """Return the timestamp of the last reset_sirius_caches() call completion.

    Used by ReadaheadScanManager.start() to measure Python query setup delay
    (dead time between cache reset and first GET being issued).
    """
    return _query_reset_t


# Fine-grained timing for setup_delay decomposition (set each query, read by setup_breakdown).
_evaluate_on_rank_t: float | None = None  # start of evaluate_on_rank
_after_allgather_t: float | None = None   # after allgather_stats completes


def reset_sirius_caches() -> None:
    """Evict all prefetch-cached blocks from the process-level IoContextRegistry.

    Call this between benchmark iterations to start each iteration cold.
    Has no effect if SIRIUS_DATASOURCE=1 has never been used in this process.
    """
    global _query_reset_t
    if _sirius_registry is not None:
        from sirius_cache import reset_caches
        reset_caches()
    _query_reset_t = time.perf_counter()


def mark_query_start() -> None:
    """Record the current time as _query_reset_t without resetting caches.

    Call this just before q.collect() so setup_delay is measured in warm mode
    (where reset_sirius_caches() is not called between queries).
    """
    global _query_reset_t
    _query_reset_t = time.perf_counter()


def _get_or_create_sirius_registry(config_options: Any) -> Any:
    """Return the process-level IoContextRegistry singleton, creating it on first call.

    The registry owns the pinned memory pool and libcurl reactor threads.  It must
    live across all queries in a process (warm-mode semantics: cache accumulates
    across queries).  reset_caches() is only called between benchmark iterations by
    the benchmark runner, never here.

    All ScanManagerConfig fields are tunable via environment variables.  Defaults
    match the Sirius production config (sirius/config.yaml):

      REST_N_REACTORS                  — libcurl S3 reactor threads (default 16)
      SIRIUS_NUM_THREADS               — scan-manager worker threads (default 20)
      SIRIUS_HOST_CAPACITY_BYTES       — pinned-RAM pool size in bytes (default 200 GiB)
      SIRIUS_HOST_POOL_SIZE_MIB        — per-pool slab size in MiB (default 512)
      SIRIUS_HOST_INITIAL_NUMBER_POOLS — number of preallocated slabs (default 300)
      SIRIUS_EVICTION_THRESHOLD_FRACTION — LRU eviction trigger fraction (default 0.8)
      SIRIUS_ENDPOINT                  — S3 endpoint URL
      AWS_DEFAULT_REGION               — S3 region (default us-east-2)
      AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN — credentials
    """
    global _sirius_registry
    if _sirius_registry is None:
        from sirius_cache import IoContextRegistry, ScanManagerConfig

        _sirius_registry = IoContextRegistry(
            ScanManagerConfig(
                rest_n_reactors=int(os.environ.get("REST_N_REACTORS", "16")),
                num_threads=int(os.environ.get("SIRIUS_NUM_THREADS", "20")),
                host_capacity_bytes=int(
                    os.environ.get("SIRIUS_HOST_CAPACITY_BYTES", str(200 * 1024**3))
                ),
                host_pool_size_mib=int(os.environ.get("SIRIUS_HOST_POOL_SIZE_MIB", "512")),
                host_initial_number_pools=int(
                    os.environ.get("SIRIUS_HOST_INITIAL_NUMBER_POOLS", "300")
                ),
                eviction_threshold_fraction=float(
                    os.environ.get("SIRIUS_EVICTION_THRESHOLD_FRACTION", "0.8")
                ),
                object_store_endpoint=os.environ.get(
                    "SIRIUS_ENDPOINT", "https://s3.us-east-2.amazonaws.com"
                ),
                object_store_region=os.environ.get("AWS_DEFAULT_REGION", "us-east-2"),
                access_key=os.environ.get("AWS_ACCESS_KEY_ID", ""),
                secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
                session_token=os.environ.get("AWS_SESSION_TOKEN", ""),
            )
        )
    return _sirius_registry


def _extract_first_parquet_path(ir: IR) -> str | None:
    """Return the first parquet file path found in the IR, or None.

    Used to start warmup (libcurl connection pool) before footer metadata is
    fetched, so warmup and footer fetch run concurrently.
    """
    from cudf_polars.dsl.traversal import traversal
    from cudf_polars.streaming.io import StreamingScan

    for node in traversal([ir]):
        if isinstance(node, StreamingScan) and node.base_scan.typ == "parquet":
            for task in node.tasks:
                if task.paths:
                    return task.paths[0]
    return None


def _attach_sirius_datasources(
    ir: IR,
    config_options: Any,
    warmup_thread: threading.Thread | None = None,
) -> Any | None:
    """Set up SiriusDatasource objects and a ReadaheadScanManager for all parquet scan tasks.

    Called after attach_cached_parquet_metadata, before execute_ir_on_rank.
    Stores datasources in base_scan._sirius_datasources keyed by (tuple(paths), split_index).
    Each value is [datasources, stage_list] where datasources is list[SiriusDatasource] (one
    per file-slice) and stage_list=[0] is a shared mutable container so do_evaluate can signal
    reading/disposed back to the readahead worker.

    Returns readahead (ReadaheadScanManager) which the caller must stop() after query execution.
    Returns None only when no parquet scan nodes are found.
    The IoContextRegistry is a process-level singleton — it is NOT returned or destroyed here.
    sirius_cache import errors propagate — a missing library is a deployment error.
    """
    from cudf_polars.streaming.io import ParquetScanTask, StreamingScan
    from cudf_polars.streaming.readahead import PrefetchStrategy, ReadaheadScanManager

    registry = _get_or_create_sirius_registry(config_options)

    # Reset per-cycle stats counters without evicting cached blocks.
    # This gives per-query last_cycle numbers in cache_summary().
    registry.prepare_for_query()

    from cudf_polars.dsl.traversal import traversal

    scan_nodes = [
        node
        for node in traversal([ir])
        if isinstance(node, StreamingScan) and node.base_scan.typ == "parquet"
    ]

    if not scan_nodes:
        return None, set()

    if warmup_thread is not None:
        # Early warmup was started concurrently with footer fetch; join it now.
        # By this point footer fetch (meta phase) has already run, so warmup
        # is almost certainly complete. Timeout is a safety net only.
        warmup_thread.join(timeout=1.0)
    else:
        # Synchronous warmup: pre-establish libcurl connection pool to S3 endpoint.
        first_path = next(
            (t.paths[0] for node in scan_nodes for t in node.tasks if t.paths),
            None,
        )
        if first_path:
            registry.warmup(first_path)

    rest_n_reactors = int(os.environ.get("REST_N_REACTORS", "16"))
    readahead = ReadaheadScanManager(budget=rest_n_reactors)
    cache_datasources = os.environ.get("SIRIUS_CACHE_PARQUET_FOOTERS", "0") == "1"
    early_fadvise = os.environ.get("SIRIUS_EARLY_FADVISE", "0") == "1"
    query_paths: set[str] = set()

    # SIRIUS_EARLY_FADVISE: start the readahead worker before the registration loop so
    # GETs begin during task enumeration (matching Sirius's scan_manager background
    # threads that issue fadvise file-by-file as metadata arrives). Each task notifies
    # the worker immediately after registration instead of waiting for the full batch.
    if early_fadvise:
        readahead.start(PrefetchStrategy.eager, streaming=True)

    for operator_id, scan_node in enumerate(scan_nodes):
        base_scan = scan_node.base_scan

        if base_scan._sirius_datasources is None:
            base_scan._sirius_datasources = {}

        for task in scan_node.tasks:
            if not isinstance(task, ParquetScanTask):
                continue

            # One datasource per file-slice, matching Sirius scan_info._datasources.
            # Fadvise ranges are also per-path (per file-slice).
            per_path_ranges = _compute_fadvise_ranges_per_path(task)
            datasources: list[Any] = []
            for i, path in enumerate(task.paths):
                query_paths.add(path)
                if cache_datasources and path in _global_datasource_cache:
                    base_ds = _global_datasource_cache[path]
                else:
                    base_ds = registry.open_datasource(path)
                    if cache_datasources:
                        _global_datasource_cache[path] = base_ds
                ds = base_ds.duplicate()
                ranges = per_path_ranges[i] if i < len(per_path_ranges) else []
                if ranges:
                    ds.fadvise([(r.offset, r.size) for r in ranges])
                datasources.append(ds)

            stage_list: list[int] = [0]
            key = (tuple(task.paths), task.split_index)
            base_scan._sirius_datasources[key] = [datasources, stage_list]

            readahead.register_scan_task(task, operator_id)
            if early_fadvise:
                readahead.notify_new_task()

        readahead.mark_operator_closed(operator_id)

    if early_fadvise:
        readahead.finish_registration()
    else:
        readahead.start(PrefetchStrategy.eager)
    return readahead, query_paths


class _FullFileRange:
    """Simple byte-range descriptor for a whole-file fadvise fallback."""

    __slots__ = ("offset", "size")

    def __init__(self, size: int) -> None:
        self.offset = 0
        self.size = size


def _compute_fadvise_ranges_per_path(task: Any) -> list[list]:
    """Compute payload column-chunk byte ranges for each file-slice in a task.

    Returns a list of byte-range lists, one per path in task.paths, matching
    Sirius parquet_fadvise_entries() which computes per-file-slice ranges.
    When precise row-group byte ranges are unavailable, falls back to full-file
    fadvise so the prefetch cache still gets a chance to prefetch the file.
    """
    return _compute_fadvise_ranges_with_infos(task, task._get_cached_parquet_info())


def _compute_fadvise_ranges_with_infos(task: Any, cached_infos: list | None) -> list[list]:
    """Like _compute_fadvise_ranges_per_path but takes infos directly.

    Used by per-file streaming where infos are available before being attached
    to base_scan.cached_parquet_info.
    """
    if not cached_infos:
        return [[] for _ in task.paths]
    bounds = task._get_task_bounds(cached_infos)
    result: list[list] = []
    for i, info in enumerate(cached_infos):
        if bounds.row_groups is None or i >= len(bounds.row_groups):
            if info.size:
                result.append([_FullFileRange(info.size)])
            else:
                result.append([])
            continue
        if info._hybrid_scan_metadata is None:
            if info.size:
                result.append([_FullFileRange(info.size)])
            else:
                result.append([])
            continue
        rg_indices = bounds.row_groups[i]
        if not rg_indices:
            result.append([])
            continue
        options = info.default_reader_options()
        if task.base_scan.with_columns is not None:
            options.set_column_names(task.base_scan.with_columns)
        reader = info.hybrid_scan_reader(options)
        result.append(reader.payload_column_chunks_byte_ranges(rg_indices, options))
    return result


def _attach_sirius_datasources_streaming(
    ir: IR,
    config_options: Any,
) -> tuple[Any, set[str], dict[str, Any]]:
    """Per-file streaming variant of _attach_sirius_datasources.

    Matches Sirius scan_manager: as each file's footer is fetched, immediately
    compute byte ranges, create its datasource, fadvise, and register with the
    readahead worker. The worker starts S3 GETs for each file as soon as its
    footer is ready — without waiting for all files' footers.

    Returns (readahead, query_paths, all_cached_infos). Caller must still call
    attach_cached_parquet_metadata(ir, all_cached_infos) for the reader to use.
    """
    import concurrent.futures

    from cudf_polars.dsl.traversal import traversal
    from cudf_polars.dsl.utils.io import (
        _global_parquet_footer_cache,
        _prefetch_parquet_footers_for_paths,
    )
    from cudf_polars.streaming.io import ParquetScanTask, StreamingScan
    from cudf_polars.streaming.readahead import PrefetchStrategy, ReadaheadScanManager

    registry = _get_or_create_sirius_registry(config_options)
    registry.prepare_for_query()

    scan_nodes = [
        node
        for node in traversal([ir])
        if isinstance(node, StreamingScan) and node.base_scan.typ == "parquet"
    ]
    if not scan_nodes:
        return None, set(), {}

    rest_n_reactors = int(os.environ.get("REST_N_REACTORS", "16"))
    n_meta_threads = int(os.environ.get("SIRIUS_EARLY_FADVISE_THREADS", "16"))
    cache_datasources = os.environ.get("SIRIUS_CACHE_PARQUET_FOOTERS", "0") == "1"
    parse_hybrid = True  # always needed for fadvise ranges

    # --- Build task inventory ---
    # task_pending[task_key] = number of paths whose footers haven't arrived yet.
    # task_infos[task_key]   = {path: CachedParquetInfo} accumulated as footers arrive.
    # path_to_tasks[path]    = [(task, op_id)] so each footer result updates the right tasks.
    task_pending: dict[int, int] = {}
    task_infos: dict[int, dict[str, Any]] = {}
    path_to_tasks: dict[str, list[tuple[Any, int]]] = {}
    operator_remaining: dict[int, int] = {}  # op_id → tasks not yet registered

    for op_id, scan_node in enumerate(scan_nodes):
        if scan_node.base_scan._sirius_datasources is None:
            scan_node.base_scan._sirius_datasources = {}
        task_count = 0
        for task in scan_node.tasks:
            if not isinstance(task, ParquetScanTask):
                continue
            key = id(task)
            task_pending[key] = len(task.paths)
            task_infos[key] = {}
            for path in task.paths:
                path_to_tasks.setdefault(path, []).append((task, op_id))
            task_count += 1
        operator_remaining[op_id] = task_count

    readahead = ReadaheadScanManager(budget=rest_n_reactors)
    readahead.start(PrefetchStrategy.eager, streaming=True)

    all_cached_infos: dict[str, Any] = {}
    query_paths: set[str] = set()

    def _register_task(task: Any, op_id: int) -> None:
        """Process one fully-resolved task: ranges → datasource → fadvise → register."""
        infos = task_infos[id(task)]
        cached_info_list = [infos[p] for p in task.paths if p in infos]
        per_path_ranges = _compute_fadvise_ranges_with_infos(task, cached_info_list)

        datasources: list[Any] = []
        for i, path in enumerate(task.paths):
            query_paths.add(path)
            if cache_datasources and path in _global_datasource_cache:
                base_ds = _global_datasource_cache[path]
            else:
                base_ds = registry.open_datasource(path)
                if cache_datasources:
                    _global_datasource_cache[path] = base_ds
            ds = base_ds.duplicate()
            ranges = per_path_ranges[i] if i < len(per_path_ranges) else []
            if ranges:
                ds.fadvise([(r.offset, r.size) for r in ranges])
            datasources.append(ds)

        stage_list: list[int] = [0]
        key_ds = (tuple(task.paths), task.split_index)
        task.base_scan._sirius_datasources[key_ds] = [datasources, stage_list]

        readahead.register_scan_task(task, op_id)
        readahead.notify_new_task()

        operator_remaining[op_id] -= 1
        if operator_remaining[op_id] == 0:
            readahead.mark_operator_closed(op_id)

    def _resolve_path(path: str, info: Any) -> None:
        """Apply a fetched footer to all tasks that need it."""
        all_cached_infos[path] = info
        if cache_datasources:
            _global_parquet_footer_cache[path] = info
        for task, op_id in path_to_tasks.get(path, []):
            key = id(task)
            task_infos[key][path] = info
            task_pending[key] -= 1
            if task_pending[key] == 0:
                _register_task(task, op_id)

    # --- Warmup: start libcurl connection pool init in background ---
    # Runs concurrently with Phase 1 (cached footer registration) and the start
    # of Phase 2 (footer fetch). C++ warmup() has a built-in staleness check so
    # Q2+ calls are no-ops (µs). Only Q1 iter0 pays the ~490ms pool init cost.
    first_path = next(iter(path_to_tasks), None)
    _warmup_thread: threading.Thread | None = None
    if first_path:
        _warmup_thread = threading.Thread(
            target=registry.warmup, args=(first_path,), daemon=True, name="sirius-warmup"
        )
        _warmup_thread.start()

    # --- Phase 1: paths already in footer cache — register tasks immediately ---
    missing_paths: set[str] = set()
    for path in path_to_tasks:
        if cache_datasources and path in _global_parquet_footer_cache:
            _resolve_path(path, _global_parquet_footer_cache[path])
        else:
            missing_paths.add(path)

    # --- Phase 2: stream missing footers, registering each task as its paths resolve ---
    # Warmup runs concurrently with Phase 2 — no join needed here. The warmup daemon
    # thread will complete within ~490ms (Q1 iter0 only; Q2+ are µs no-ops), well before
    # data GETs are issued (which only start after footers are fetched and registered).
    if missing_paths:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=n_meta_threads, thread_name_prefix="sirius-meta"
        ) as pool:
            future_to_path = {
                pool.submit(_prefetch_parquet_footers_for_paths, [path], parse_hybrid_metadata=parse_hybrid): path
                for path in missing_paths
            }
            for future in concurrent.futures.as_completed(future_to_path):
                for info in future.result():
                    _resolve_path(info.path, info)

    readahead.finish_registration()
    return readahead, query_paths, all_cached_infos


def evaluate_on_rank(
    ctx: Context,
    comm: Communicator,
    py_executor: ThreadPoolExecutor,
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
    *,
    collect_metadata: bool = False,
    local_quent_context: LocalQuentContext | None = None,
    query_id: uuid.UUID,
) -> tuple[DataFrame, list[ChannelMetadata]]:
    """
    Evaluate a polars IR plan on a single rank.

    This is the main worker-side entry point for multi-rank execution.
    It performs the following steps collectively across all ranks:

    1. Collect statistics (on rank 0 and allgather)
    2. Lower the IR graph
    3. Reserve collective operation IDs
    4. Execute the lowered pipeline

    Parameters
    ----------
    ctx
        The active RapidsMPF streaming context for this rank.
    comm
        The active RapidsMPF communicator for this rank.
    py_executor
        Thread-pool executor used to drive the actor network.
    ir
        Root of the **pre-lowered** IR graph.
    config_options
        Executor configuration forwarded from the client.
    collect_metadata
        Whether to collect channel metadata during execution.
    local_quent_context
        The local Quent context for this rank, or ``None`` when tracing is
        disabled.
    query_id
        A unique identifier for the query.

    Returns
    -------
    result
        This rank's output fragment as a GPU-resident :class:`~cudf_polars.containers.DataFrame`.
    metadata
        Collected channel metadata.
    """
    global _evaluate_on_rank_t, _after_allgather_t
    _evaluate_on_rank_t = time.perf_counter()
    stats = allgather_stats(comm, ctx.br(), ir, config_options, py_executor)
    _after_allgather_t = time.perf_counter()
    # ``get_stable_plan_id`` is a deterministic function of the IR
    # structure, so every rank derives the same logical plan ID for a
    # given query (only rank 0 emits the declaration, but physical plans
    # on every rank reference it as their parent). It is *not* unique
    # across collects, though: re-running an identical query would reuse
    # the same plan ID under a different parent query. Namespacing by the
    # per-collect ``query_id`` (which is identical across ranks but unique
    # per collect) keeps the cross-rank agreement while making the plan ID
    # unique per collect.
    logical_plan_id = uuid.uuid5(query_id, str(ir.get_stable_plan_id()))

    physical_op_by_id: dict[str, cudf_polars.quent._types.Operator] | None = None
    quent_operator_map: dict[IR, cudf_polars.quent._types.Operator] | None = None

    _t_lower0 = time.perf_counter()
    lowering, node_map = lower_ir_graph_with_node_map(
        ir, config_options, stats, rank=comm.rank, nranks=comm.nranks
    )
    _t_lower1 = time.perf_counter()
    optimized = lowering.optimized
    ir = lowering.lowered
    partition_info = lowering.partition_info
    # TODO: figure out if we emit anything about optimized.
    if config_options.executor.quent_context is not None:
        assert local_quent_context is not None
        plan, ops, ports, logical_op_by_id = build_plan(
            optimized,
            config_options,
            query=local_quent_context.query,
            plan_id=logical_plan_id,
            worker=local_quent_context.worker,
            instance_name="logical",
            parent_plan=None,
            parent_operators_by_node_id=None,
        )
        if comm.rank == 0:
            local_quent_context.context._emit_plan_declarations(
                local_quent_context.logger, plan, ops, ports
            )

    if comm.rank == 0:
        log_query_plan(ir, config_options)

    if config_options.executor.quent_context is not None:
        assert local_quent_context is not None
        physical_plan_id = uuid.uuid4()
        physical_op_by_id = local_quent_context.context._emit_physical_plan_events(
            local_quent_context.logger,
            ir,
            config_options,
            plan_id=physical_plan_id,
            worker=local_quent_context.worker,
            parent_plan=plan,
            node_map=node_map,
            logical_op_by_id=logical_op_by_id,
        )
        quent_operator_map = build_quent_operator_map(ir, physical_op_by_id)
    ir_context = IRExecutionContext(
        py_executor, get_cuda_stream=ctx.br().stream_pool.get_stream, query_id=query_id
    )

    sirius_datasource_enabled = os.environ.get("SIRIUS_DATASOURCE", "0") == "1"
    early_fadvise_enabled = os.environ.get("SIRIUS_EARLY_FADVISE", "0") == "1"
    prefetch_file_metadata = config_options.parquet_options.prefetch_file_metadata
    # SIRIUS_DATASOURCE=1 requires hybrid scan metadata to compute fadvise byte ranges,
    # regardless of whether the read itself uses the hybrid scan reader.
    parse_hybrid_metadata = (
        config_options.parquet_options.use_hybrid_scan or sirius_datasource_enabled
    )

    _sirius_readahead = None
    _sirius_fadvised_paths: set[str] = set()

    if sirius_datasource_enabled and early_fadvise_enabled:
        # Per-file streaming path: warmup + footer fetch + fadvise + readahead start
        # all happen concurrently in _attach_sirius_datasources_streaming. Each file's
        # GET starts as soon as its footer is fetched — matches Sirius scan_manager.
        if os.environ.get("SIRIUS_CACHE_PRINT_STATS", "0") == "1" and _sirius_registry is not None:
            print("pre_query: " + _sirius_registry.cache_summary(), flush=True)
        _t_meta0 = time.perf_counter()
        _sirius_readahead, _sirius_fadvised_paths, _streaming_infos = (
            _attach_sirius_datasources_streaming(ir, config_options)
        )
        _t_meta1 = time.perf_counter()
        # Attach infos to IR nodes so the parquet reader can use them.
        if _streaming_infos:
            attach_cached_parquet_metadata(ir, _streaming_infos)
        elif prefetch_file_metadata is not False:
            cached_parquet_info_map = prefetch_parquet_file_metadata_for_ir(
                ir,
                ir_context.py_executor,
                stats=stats,
                parse_hybrid_metadata=parse_hybrid_metadata,
            )
            attach_cached_parquet_metadata(ir, cached_parquet_info_map)
        if os.environ.get("SIRIUS_CACHE_PRINT_STATS", "0") == "1":
            from cudf_polars.callback import get_callback_entry_t, get_nt_entry_t
            _nt_t = get_nt_entry_t()
            _cb_t = get_callback_entry_t()
            _lower_ms = (_t_lower1 - _t_lower0) * 1000
            _meta_ms = (_t_meta1 - _t_meta0) * 1000
            # Decompose nt_to_start into phases:
            #   translate = IR translation + Polars UDF dispatch (nt_entry → callback_entry)
            #   spmd      = callback → evaluate_on_rank entry (SPMD/quent setup)
            #   allgather = allgather_stats across ranks
            #   post_lower= after lowering → streaming start (quent build_plan, IRExecutionContext)
            _translate_ms = ((_cb_t - _nt_t) * 1000) if (_cb_t and _nt_t) else 0.0
            _spmd_ms = ((_evaluate_on_rank_t - _cb_t) * 1000) if (_evaluate_on_rank_t and _cb_t) else 0.0
            _allgather_ms = ((_after_allgather_t - _evaluate_on_rank_t) * 1000) if (_after_allgather_t and _evaluate_on_rank_t) else 0.0
            _post_lower_ms = ((_t_meta0 - _t_lower1) * 1000) if _t_lower1 else 0.0
            print(
                f"[sirius] setup_breakdown lower={_lower_ms:.1f}ms "
                f"meta+attach={_meta_ms:.1f}ms (streaming) "
                f"| translate={_translate_ms:.1f}ms spmd={_spmd_ms:.1f}ms "
                f"allgather={_allgather_ms:.1f}ms post_lower={_post_lower_ms:.1f}ms",
                flush=True,
            )
    else:
        _t_meta0 = time.perf_counter()
        if prefetch_file_metadata is not False or sirius_datasource_enabled:
            cached_parquet_info_map = prefetch_parquet_file_metadata_for_ir(
                ir,
                ir_context.py_executor,
                stats=stats,
                parse_hybrid_metadata=parse_hybrid_metadata,
            )
            attach_cached_parquet_metadata(ir, cached_parquet_info_map)
        _t_meta1 = time.perf_counter()

        if sirius_datasource_enabled:
            # Log cache state before prepare_for_query() resets last_cycle counters.
            if os.environ.get("SIRIUS_CACHE_PRINT_STATS", "0") == "1" and _sirius_registry is not None:
                print("pre_query: " + _sirius_registry.cache_summary(), flush=True)
            _t_attach0 = time.perf_counter()
            _sirius_readahead, _sirius_fadvised_paths = _attach_sirius_datasources(
                ir, config_options
            )
            _t_attach1 = time.perf_counter()
            if os.environ.get("SIRIUS_CACHE_PRINT_STATS", "0") == "1":
                _lower_ms = (_t_lower1 - _t_lower0) * 1000
                _meta_ms = (_t_meta1 - _t_meta0) * 1000
                _attach_ms = (_t_attach1 - _t_attach0) * 1000
                print(
                    f"[sirius] setup_breakdown lower={_lower_ms:.1f}ms "
                    f"meta={_meta_ms:.1f}ms attach={_attach_ms:.1f}ms",
                    flush=True,
                )

    try:
        with ReserveOpIDs(ir, config_options) as collective_id_map:
            return execute_ir_on_rank(
                ctx,
                comm,
                ir,
                ir_context,
                partition_info,
                config_options,
                stats,
                collective_id_map,
                quent_operator_map=quent_operator_map,
                local_quent_context=local_quent_context,
            )
    finally:
        if _sirius_readahead is not None:
            _sirius_readahead.stop()
        if sirius_datasource_enabled and os.environ.get("SIRIUS_CACHE_PRINT_STATS", "0") == "1":
            if _sirius_registry is not None:
                print(_sirius_registry.cache_summary(), flush=True)
        # After query finishes, proactively prefetch all known paths NOT in this query.
        # These downloads run concurrently with the next query's planning and early execution,
        # giving the cache a head-start on cross-query data (matches Sirius's prepare_for_query
        # lookahead that begins downloading future query data while current query is still live).
        if (
            sirius_datasource_enabled
            and os.environ.get("SIRIUS_PROACTIVE_PREFETCH", "0") == "1"
            and _sirius_registry is not None
        ):
            from cudf_polars.dsl.utils.io import _global_parquet_footer_cache

            _submit_proactive_prefetch(
                _sirius_registry, _global_parquet_footer_cache, _sirius_fadvised_paths
            )


def is_duplicated_output(metadata: list[ChannelMetadata] | None) -> bool:
    """
    Return whether a query's output is duplicated across ranks.

    A duplicated output is an identical, complete copy held on every rank (for
    example the result of a global sort/limit), signalled by the ``duplicated``
    channel flag on the final output.

    Parameters
    ----------
    metadata
        Channel metadata for the query.

    Returns
    -------
    ``True`` if the output is an identical copy held on every rank.
    """
    return bool(metadata and metadata[-1].duplicated)


def drop_if_replicated(
    df: DataFrame, rank: int, metadata: list[ChannelMetadata] | None
) -> DataFrame:
    """
    Drop a duplicated output on non-root ranks.

    Parameters
    ----------
    df
        This rank's output partition.
    rank
        This rank's index within the cluster.
    metadata
        Channel metadata for the query.

    Returns
    -------
    ``df`` or a freshly-allocated empty same-schema frame.
    """
    if rank != 0 and is_duplicated_output(metadata):
        return DataFrame.from_table(
            plc.copying.empty_like(df.table, stream=df.stream),
            df.column_names,
            df.dtypes,
            df.stream,
        )
    return df


def raise_for_translation_errors(translator: Translator) -> None:
    """
    Raise if the translator recorded unsupported operations.

    Parameters
    ----------
    translator
        The translator whose :attr:`~cudf_polars.dsl.translate.Translator.errors`
        are checked.

    Raises
    ------
    NotImplementedError
        If the query contains operations unsupported on the GPU.
    """
    error = translator.unsupported_operations_error()
    if error is not None:
        raise error
