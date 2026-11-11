# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, int64_t, uint8_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move, pair
from libcpp.vector cimport vector

from cuda.bindings.cyruntime cimport cudaStream_t
from pylibcudf.io.parquet cimport ParquetReaderOptions
from pylibcudf.io.parquet_metadata cimport FileMetaData
from pylibcudf.io.types cimport TableWithMetadata
from pylibcudf.libcudf.io.types cimport table_with_metadata
from pylibcudf.libcudf.io.parquet_schema cimport FileMetaData as cpp_FileMetaData
from rmm.librmm.device_buffer cimport device_buffer
from rmm.pylibrmm.device_buffer cimport DeviceBuffer
from rmm.pylibrmm.stream cimport Stream

from cudf_polars_cucascade.libcucascade.scan_io cimport (
    cpp_broadcast,
    cpp_chain,
    cpp_scan_source,
    evaluate_expression_on,
    scan_task,
    expr_node,
    planned_task,
    probe_pipeline as cpp_probe_pipeline,
    scan_io as cpp_scan_io,
    scan_io_config as cpp_scan_io_config,
    stage,
)
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from libcpp.memory cimport make_unique, shared_ptr
from pylibcudf.column cimport Column
from rapidsmpf.streaming.core.actor cimport CppActor, cpp_Actor
from rapidsmpf.streaming.core.channel cimport Channel, cpp_Channel
from rapidsmpf.streaming.core.context cimport Context
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.table cimport Table
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

__all__ = [
    "ProbePipeline",
    "ScanIO",
    "broadcast_actor",
    "chain_actor",
    "evaluate_expression",
    "scan_source_actor",
]


cdef class ScanIO:
    """
    Remote parquet I/O for scans: footers and column chunks of S3 objects, fetched ahead of their
    decode into one pinned host arena with cuCascade's REST engine.

    Footers: :meth:`request_footers` queues one job per file (one suffix-range request returns
    the size, the version tag and the footer); :meth:`footer` waits for one. Data: each scan opens
    a stream (:meth:`open_stream`) and submits its tasks' byte ranges in task order
    (:meth:`submit`); the reads start as soon as the task gets a block of the arena, and a decode
    copies the block to the device (:meth:`to_device`) and gives it back (:meth:`release`).

    Parameters
    ----------
    endpoint, region, access_key, secret_key, session_token : str
        Object store endpoint (``https://host`` or ``http://host``) and credentials.
    arena_bytes : int
        Pinned host staging arena, pinned when this is created.
    reactors : int, default 8
        REST reactor threads.
    connections : int, default 32
        Connections per reactor.
    stall_seconds : int, default 5
        A request slower than 64 KiB/s for this long is retried.
    footer_threads : int, default 32
        Footer requests in flight.
    inflight_bytes : int, default 0
        Bytes of reads on the network at once (0: no limit). Reads are issued in task order,
        so a limit near bandwidth x latency makes tasks land roughly in order.
    """

    cdef unique_ptr[cpp_scan_io] c_obj

    def __cinit__(
        self,
        str endpoint,
        str region,
        str access_key,
        str secret_key,
        str session_token="",
        *,
        size_t arena_bytes,
        size_t reactors=8,
        size_t connections=32,
        size_t stall_seconds=5,
        size_t footer_threads=32,
        size_t inflight_bytes=0,
    ):
        cdef cpp_scan_io_config cfg
        cfg.endpoint = endpoint.encode()
        cfg.region = region.encode()
        cfg.access_key = access_key.encode()
        cfg.secret_key = secret_key.encode()
        cfg.session_token = session_token.encode()
        cfg.arena_bytes = arena_bytes
        cfg.reactors = reactors
        cfg.connections = connections
        cfg.stall_seconds = stall_seconds
        cfg.footer_threads = footer_threads
        cfg.inflight_bytes = inflight_bytes
        with nogil:
            self.c_obj.reset(new cpp_scan_io(cfg))

    @property
    def arena_bytes(self) -> int:
        """Size of the pinned arena in bytes."""
        return self.c_obj.get().arena_bytes()

    def snapshot(self) -> list[int]:
        """[tasks waiting for a block, bytes in flight, bytes queued to issue, free arena bytes]."""
        return list(self.c_obj.get().snapshot())

    @property
    def fetched_bytes(self) -> int:
        """Data bytes requested from the object store so far."""
        return self.c_obj.get().fetched_bytes()

    def warmup(self, str url) -> None:
        """Open connections to the object store of ``url`` now."""
        cdef string c_url = url.encode()
        with nogil:
            self.c_obj.get().warmup(c_url)

    def request_footers(self, paths) -> None:
        """Queue the footer requests of ``paths`` not already requested (does not block)."""
        cdef vector[string] c_paths = [p.encode() for p in paths]
        with nogil:
            self.c_obj.get().request_footers(c_paths)

    def footer(self, str path):
        """The ``(size, FileMetaData)`` of ``path``, waiting for its request if needed."""
        cdef string c_path = path.encode()
        cdef pair[size_t, unique_ptr[cpp_FileMetaData]] result
        with nogil:
            result = move(self.c_obj.get().footer(c_path))
        return result.first, FileMetaData.from_libcudf(move(result.second))

    def projected_bytes(self, paths, columns) -> list[tuple[int, int]]:
        """
        Per path, ``(bytes of the column chunks of columns, row-group count)`` from its footer
        (waiting for it). An empty ``columns`` counts every column.
        """
        cdef vector[string] c_paths = [p.encode() for p in paths]
        cdef vector[string] c_columns = [c.encode() for c in columns]
        cdef vector[size_t] result
        with nogil:
            result = self.c_obj.get().projected_bytes(c_paths, c_columns)
        return [(result[2 * i], result[2 * i + 1]) for i in range(len(paths))]

    def footer_file(self, str path) -> bytes:
        """
        A minimal parquet file holding only the footer of ``path``.

        Enough for metadata readers (for example the physical column types), with no request.
        """
        cdef string c_path = path.encode()
        cdef vector[uint8_t] file
        with nogil:
            file = move(self.c_obj.get().footer_file(c_path))
        return (<char*>file.data())[:file.size()]

    def clear_footers(self) -> None:
        """Forget every footer and open object, so the next request fetches them again."""
        with nogil:
            self.c_obj.get().clear_footers()

    def open_stream(self) -> int:
        """Open a stream: one scan's tasks, given blocks in their own submission order."""
        return self.c_obj.get().open_stream()

    def close_stream(self, int64_t stream) -> None:
        """Close a stream whose tasks have all been released."""
        with nogil:
            self.c_obj.get().close_stream(stream)

    def submit(self, int64_t stream, paths, sources, offsets, sizes) -> int:
        """
        Queue the reads of one task; returns its id, or -1 if it is larger than the arena.

        Range ``i`` is ``sizes[i]`` bytes at ``offsets[i]`` of ``paths[sources[i]]``; the
        ranges are placed back to back in the task's block. Waits for the paths' footer requests.
        """
        cdef vector[string] c_paths = [p.encode() for p in paths]
        cdef vector[int32_t] c_sources = sources
        cdef vector[size_t] c_offsets = offsets
        cdef vector[size_t] c_sizes = sizes
        cdef int64_t task_id
        with nogil:
            task_id = self.c_obj.get().submit(stream, c_paths, c_sources, c_offsets, c_sizes)
        return task_id

    def ready(self, int64_t task_id) -> bool:
        """Whether every read of a task has completed (does not block)."""
        cdef bool result
        with nogil:
            result = self.c_obj.get().ready(task_id)
        return result

    def wait(self, int64_t task_id) -> None:
        """Block until every read of a task has completed; raises its read error."""
        with nogil:
            self.c_obj.get().wait(task_id)

    def to_device(self, int64_t task_id, Stream stream) -> DeviceBuffer:
        """Copy a task's staged bytes, back to back, to a new device buffer on ``stream``."""
        cdef unique_ptr[device_buffer] buf
        cdef cudaStream_t c_stream = stream._cuda_stream
        with nogil:
            buf = move(self.c_obj.get().to_device(task_id, c_stream))
        return DeviceBuffer.c_from_unique_ptr(move(buf), stream)

    def release(self, int64_t task_id) -> None:
        """Give a task's block back (after the work reading it has completed)."""
        with nogil:
            self.c_obj.get().release(task_id)

    def plan_task(
        self,
        int64_t stream_id,
        paths,
        int split_index,
        int total_splits,
        ParquetReaderOptions options,
        bool prune_with_stats,
        size_t merge_gap,
        Stream stream,
    ) -> tuple[int, bool, bool, bool, int, int]:
        """
        Plan a task from its footers and queue its reads (without the GIL).

        Returns ``(task_id, empty, unsupported, too_big, staged_bytes, decoded_bytes)``.
        ``options`` (the scan's columns and filter) must stay alive until the task is decoded.
        """
        cdef vector[string] c_paths = [p.encode() for p in paths]
        cdef cudaStream_t c_stream = stream._cuda_stream
        cdef planned_task r
        with nogil:
            r = self.c_obj.get().plan_task(
                stream_id,
                c_paths,
                split_index,
                total_splits,
                options.c_obj,
                prune_with_stats,
                merge_gap,
                c_stream,
            )
        return r.id, r.empty, r.unsupported, r.too_big, r.staged_bytes, r.decoded_bytes

    def materialize(self, int64_t task_id, Stream stream) -> TableWithMetadata:
        """Decode a planned task from its staged bytes (without the GIL)."""
        cdef cudaStream_t c_stream = stream._cuda_stream
        cdef table_with_metadata result
        with nogil:
            result = move(self.c_obj.get().materialize(task_id, c_stream))
        import rmm.mr

        return TableWithMetadata.from_libcudf(
            result, stream, rmm.mr.get_current_device_resource()
        )


_EXPR_OPS = {
    "column": 0,
    "literal": 1,
    "binary": 2,
    "year": 3,
    "cast": 4,
    "str_literal": 5,
    "ternary": 6,
}
_STAGE_OPS = {"join": 0, "compute": 1, "select": 2, "groupby": 3}


cdef vector[expr_node] _expr(list nodes):
    cdef vector[expr_node] out
    cdef expr_node n
    for d in nodes:
        n.op = _EXPR_OPS[d["op"]]
        n.column_index = d.get("column", 0)
        n.value = d.get("value", 0.0)
        n.text = d.get("text", "").encode()
        n.binop = int(d.get("binop", 0))
        n.dtype = int(d.get("dtype", 10))
        n.lhs = d.get("lhs", 0)
        n.rhs = d.get("rhs", 0)
        n.third = d.get("third", 0)
        out.push_back(n)
    return out


cdef stage _stage(dict d):
    cdef stage s
    s.op = _STAGE_OPS[d["op"]]
    s.build = d.get("build", 0)
    s.probe_keys = d.get("probe_keys", [])
    s.keep_probe = d.get("keep_probe", [])
    s.keep_build = d.get("keep_build", [])
    for e in d.get("exprs", []):
        s.exprs.push_back(_expr(e))
    s.columns = d.get("columns", [])
    s.keys = d.get("keys", [])
    s.sums = d.get("sums", [])
    return s


cdef class ProbePipeline:
    """
    Experiment: probe chains run natively over the decoded tasks of a scan.

    Worker threads decode each submitted task (planned on ``io``) and run every branch's stages
    on it: ``join`` (inner join against a build table: ``build``, ``probe_keys``, then the kept
    ``keep_probe`` and ``keep_build`` columns), ``compute`` (``exprs``: flattened expressions of
    ``column``, ``literal``, ``binary`` and ``year`` nodes, appended as columns), ``select``
    (``columns``) and a final ``groupby`` (``keys``, ``sums``). :meth:`finish` returns each
    branch's aggregate. The build tables must stay alive (this object keeps them).
    """

    cdef unique_ptr[cpp_probe_pipeline] c_obj
    cdef list _builds

    def __cinit__(
        self,
        ScanIO io,
        list branches,
        list builds,
        list build_keys,
        int threads,
        list input_columns=None,
    ):
        cdef vector[vector[stage]] c_branches
        cdef vector[stage] c_stages
        cdef vector[table_view] c_builds
        cdef vector[vector[int]] c_keys = build_keys
        cdef vector[string] c_columns = [c.encode() for c in (input_columns or [])]
        for stages in branches:
            c_stages.clear()
            for d in stages:
                c_stages.push_back(_stage(d))
            c_branches.push_back(c_stages)
        self._builds = list(builds)
        for t in self._builds:
            c_builds.push_back((<Table?>t).view())
        with nogil:
            self.c_obj.reset(
                new cpp_probe_pipeline(
                    io.c_obj.get()[0], c_branches, c_builds, c_keys, threads, c_columns
                )
            )

    def submit(self, int64_t task_id) -> None:
        """Queue a planned task; the pipeline decodes and releases it."""
        with nogil:
            self.c_obj.get().submit(task_id)

    def finish(self) -> list:
        """Wait for every task; returns each branch's aggregate (keys, then sums)."""
        cdef vector[unique_ptr[table]] result
        with nogil:
            result = move(self.c_obj.get().finish())
        import rmm.mr

        mr = rmm.mr.get_current_device_resource()
        stream = Stream()  # the results are complete: finish() synchronized
        return [Table.from_libcudf(move(result[i]), stream, mr) for i in range(result.size())]

    def close(self) -> None:
        """Streamed mode: no more tasks will be submitted."""
        with nogil:
            self.c_obj.get().close()

    def next_output(self):
        """
        Streamed mode: ``(index, table)`` of the next finished task (its submission index), or
        None once every task is done. Blocks without the GIL.
        """
        cdef pair[int64_t, unique_ptr[table]] item
        with nogil:
            item = move(self.c_obj.get().next_output())
        if item.first < 0:
            return None
        import rmm.mr

        return item.first, Table.from_libcudf(
            move(item.second), Stream(), rmm.mr.get_current_device_resource()
        )

    def stats(self) -> dict:
        """Counters of the work so far."""
        cdef vector[double] v = self.c_obj.get().stats()
        names = (
            "tasks", "decode_s", "join_s", "compute_s", "groupby_s", "fold_s", "wall_s",
            "joins", "rows_in", "rows_out",
        )
        return {name: v[i] for i, name in enumerate(names)}


def evaluate_expression(Table table, list nodes) -> Column:
    """Evaluate a native expression program on ``table`` (for testing the contract)."""
    cdef vector[expr_node] c_nodes = _expr(nodes)
    cdef table_view view = table.view()
    stream = Stream()
    cdef cudaStream_t c_stream = (<Stream>stream)._cuda_stream
    cdef unique_ptr[column] result
    with nogil:
        result = move(evaluate_expression_on(c_nodes, view, c_stream))
    import rmm.mr

    return Column.from_libcudf(move(result), stream, rmm.mr.get_current_device_resource())


def scan_source_actor(
    Context ctx not None,
    Channel ch_out not None,
    ScanIO io not None,
    list tasks,
    ParquetReaderOptions options not None,
    bool prune_with_stats,
    size_t merge_gap,
    size_t num_producers,
    list columns,
    list column_types,
    size_t max_ahead_bytes,
) -> CppActor:
    """
    A C++ actor reading a remote parquet scan through ``io``: one chunk per task, in order.

    ``tasks`` are ``(paths, split_index, total_splits)``. ``options`` (columns and filter)
    and ``io`` are kept alive by the actor. ``columns`` and ``column_types`` (type ids) are
    the output's, in order.
    """
    cdef vector[scan_task] c_tasks
    cdef scan_task t
    for paths, split_index, total_splits in tasks:
        t.paths = [p.encode() for p in paths]
        t.split_index = split_index
        t.total_splits = total_splits
        c_tasks.push_back(t)
    cdef vector[string] c_columns = [c.encode() for c in columns]
    cdef vector[int] c_types = column_types
    cdef cpp_Actor actor
    with nogil:
        actor = cpp_scan_source(
            ctx._handle,
            ch_out._handle,
            io.c_obj.get(),
            move(c_tasks),
            &options.c_obj,
            prune_with_stats,
            merge_gap,
            num_producers,
            move(c_columns),
            move(c_types),
            max_ahead_bytes,
        )
    return CppActor.from_handle(make_unique[cpp_Actor](move(actor)), owner=(io, options))


def chain_actor(
    Context ctx not None,
    list ch_builds,
    list build_keys,
    list build_types,
    Channel ch_in not None,
    Channel ch_out not None,
    list branches,
    size_t concurrency,
    int fold_keys,
    list out_types,
    bool forward_metadata=True,
    bool ordered=False,
) -> CppActor:
    """
    A C++ actor running ``branches`` of stages on every chunk of ``ch_in``.

    Join stages probe the tables collected from ``ch_builds`` (by index). With
    ``fold_keys < 0`` each result is sent on (in arrival order when ``ordered``); otherwise the
    results are partial sums with ``fold_keys`` key columns, folded into one chunk cast to
    ``out_types``. With ``forward_metadata`` false the caller handles both channels' metadata.
    """
    cdef vector[shared_ptr[cpp_Channel]] c_builds
    for ch in ch_builds:
        c_builds.push_back((<Channel?>ch)._handle)
    cdef vector[vector[int]] c_keys = build_keys
    cdef vector[vector[int]] c_build_types = build_types
    cdef vector[vector[stage]] c_branches
    cdef vector[stage] c_stages
    for stages in branches:
        c_stages.clear()
        for d in stages:
            c_stages.push_back(_stage(d))
        c_branches.push_back(c_stages)
    cdef vector[int] c_out_types = out_types
    cdef cpp_Actor actor
    with nogil:
        actor = cpp_chain(
            ctx._handle,
            move(c_builds),
            move(c_keys),
            move(c_build_types),
            ch_in._handle,
            ch_out._handle,
            move(c_branches),
            concurrency,
            fold_keys,
            move(c_out_types),
            forward_metadata,
            ordered,
        )
    return CppActor.from_handle(make_unique[cpp_Actor](move(actor)), owner=None)


def broadcast_actor(Context ctx not None, Channel ch_in not None, list chs_out) -> CppActor:
    """A C++ actor sending every chunk of ``ch_in``, metadata included, to each of ``chs_out``."""
    cdef vector[shared_ptr[cpp_Channel]] c_out
    for ch in chs_out:
        c_out.push_back((<Channel?>ch)._handle)
    cdef cpp_Actor actor
    with nogil:
        actor = cpp_broadcast(ctx._handle, ch_in._handle, move(c_out))
    return CppActor.from_handle(make_unique[cpp_Actor](move(actor)), owner=None)
