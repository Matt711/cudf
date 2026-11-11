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
    planned_task,
    scan_io as cpp_scan_io,
    scan_io_config as cpp_scan_io_config,
)

__all__ = ["ScanIO"]


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
