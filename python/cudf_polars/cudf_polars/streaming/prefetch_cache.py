# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine-scoped cache of prefetched parquet byte ranges."""

from __future__ import annotations

import asyncio
import bisect
import concurrent.futures
import contextlib
import dataclasses
import enum
import threading
from typing import TYPE_CHECKING

import kvikio

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.core.memory_reserve_or_wait import reserve_memory

if TYPE_CHECKING:
    from rapidsmpf.memory.buffer import Buffer
    from rapidsmpf.memory.memory_reservation import MemoryReservation
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.utils.config import ParquetOptions


class ChunkState(enum.Enum):
    """Lifecycle state of a single cached byte range."""

    EMPTY = enum.auto()
    ALLOCATED = enum.auto()
    LOADING = enum.auto()
    CACHED = enum.auto()


@dataclasses.dataclass
class CachedChunk:
    """A single chunk-aligned byte range, and its cached bytes once loaded."""

    offset: int
    size: int
    state: ChunkState = ChunkState.EMPTY
    reservation: MemoryReservation | None = None
    buffer: Buffer | None = None
    data: memoryview | None = None


@dataclasses.dataclass
class FileEntry:
    """Chunks cached for one file, sorted by offset."""

    chunks: list[CachedChunk] = dataclasses.field(default_factory=list)

    def find(self, offset: int) -> CachedChunk | None:
        """Return the chunk starting at ``offset``, if any."""
        i = bisect.bisect_left(self.chunks, offset, key=lambda c: c.offset)
        if i < len(self.chunks) and self.chunks[i].offset == offset:
            return self.chunks[i]
        return None

    def insert(self, chunk: CachedChunk) -> None:
        """Insert a new chunk, keeping ``chunks`` sorted by offset."""
        i = bisect.bisect_left(self.chunks, chunk.offset, key=lambda c: c.offset)
        self.chunks.insert(i, chunk)

    def remove(self, offset: int) -> None:
        """Drop the chunk at ``offset``, if present."""
        i = bisect.bisect_left(self.chunks, offset, key=lambda c: c.offset)
        if i < len(self.chunks) and self.chunks[i].offset == offset:
            del self.chunks[i]


@dataclasses.dataclass(frozen=True)
class PrefetchCacheConfig:
    """Tunables for a :class:`PrefetchCache`."""

    chunk_bytes: int
    num_prepare_workers: int
    num_fetch_workers: int
    inflight_chunk_budget: int
    pruning_thread_pool: bool
    pruning_num_workers: int

    @classmethod
    def from_parquet_options(
        cls, parquet_options: ParquetOptions
    ) -> PrefetchCacheConfig:
        """Build a config from the relevant ``ParquetOptions`` fields."""
        return cls(
            chunk_bytes=parquet_options.prefetch_cache_chunk_bytes,
            num_prepare_workers=parquet_options.prefetch_cache_num_prepare_workers,
            num_fetch_workers=parquet_options.prefetch_cache_num_fetch_workers,
            inflight_chunk_budget=parquet_options.prefetch_cache_inflight_chunk_budget,
            pruning_thread_pool=parquet_options.prefetch_pruning_thread_pool,
            pruning_num_workers=parquet_options.prefetch_pruning_num_workers,
        )


class PrefetchCache:
    """
    Engine-scoped cache of prefetched parquet byte ranges.

    Registration (:meth:`fadvise`) and fetching are decoupled: ``fadvise``
    only records that a range is wanted and returns immediately, while a
    small set of background workers reserve pinned memory and issue the
    actual reads. The workers run on a dedicated background thread with its
    own event loop, started once via :meth:`start` and living for the
    engine's lifetime, independent of any one query's own event loop.
    ``fadvise``/:meth:`get` are safe to call from any thread.
    """

    def __init__(self, config: PrefetchCacheConfig) -> None:
        self._config = config
        self._files: dict[str, FileEntry] = {}
        self._ctx: Context | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._prepare_queue: asyncio.Queue[tuple[str, CachedChunk]] | None = None
        self._fetch_queue: asyncio.Queue[tuple[str, CachedChunk]] | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._pruning_executor: concurrent.futures.ThreadPoolExecutor | None = None

    @property
    def chunk_bytes(self) -> int:
        """Size, in bytes, of the chunk-aligned ranges this cache tracks."""
        return self._config.chunk_bytes

    @property
    def pruning_executor(self) -> concurrent.futures.ThreadPoolExecutor | None:
        """
        Dedicated executor for ahead-of-read pruning work, if configured.

        ``None`` means callers should use their own thread pool instead
        (e.g. ``IRExecutionContext.py_executor``), see
        ``ParquetOptions.prefetch_pruning_thread_pool``.
        """
        return self._pruning_executor

    def start(self, ctx: Context) -> None:
        """Start the background loop and its workers, if not already running."""
        if self._thread is not None:
            return
        self._ctx = ctx
        if self._config.pruning_thread_pool:
            self._pruning_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._config.pruning_num_workers,
                thread_name_prefix="prefetch-pruning",
            )
        ready = threading.Event()

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._prepare_queue = asyncio.Queue()
            self._fetch_queue = asyncio.Queue()
            self._tasks = [
                loop.create_task(self._prepare_loop())
                for _ in range(self._config.num_prepare_workers)
            ] + [
                loop.create_task(self._fetch_loop())
                for _ in range(self._config.num_fetch_workers)
            ]
            ready.set()
            loop.run_forever()

        self._thread = threading.Thread(target=_run, name="prefetch-cache", daemon=True)
        self._thread.start()
        ready.wait()

    def stop(self) -> None:
        """Stop the background loop and join its thread."""
        if self._thread is None:
            return
        loop = self._loop
        assert loop is not None

        async def _cancel_and_stop() -> None:
            for task in self._tasks:
                task.cancel()
            for task in self._tasks:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await task
            loop.stop()

        asyncio.run_coroutine_threadsafe(_cancel_and_stop(), loop)
        self._thread.join()
        self._loop = None
        self._thread = None
        self._tasks = []
        # Cached reservations/buffers belong to the `Context` this cache was
        # started with; they're invalid once that context shuts down.
        self._files = {}
        self._ctx = None
        if self._pruning_executor is not None:
            self._pruning_executor.shutdown(wait=True, cancel_futures=True)
            self._pruning_executor = None

    def _entry(self, path: str) -> FileEntry:
        entry = self._files.get(path)
        if entry is None:
            entry = self._files[path] = FileEntry()
        return entry

    def fadvise(self, path: str, offset: int, size: int) -> None:
        """
        Register interest in a byte range, without blocking the caller.

        No-ops if the range is already registered, being fetched, or cached.
        Safe to call from any thread.
        """
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._fadvise_on_loop, path, offset, size)

    def _fadvise_on_loop(self, path: str, offset: int, size: int) -> None:
        aligned_offset = (offset // self._config.chunk_bytes) * self._config.chunk_bytes
        entry = self._entry(path)
        if entry.find(aligned_offset) is not None:
            return
        chunk = CachedChunk(offset=aligned_offset, size=size)
        entry.insert(chunk)
        assert self._prepare_queue is not None
        self._prepare_queue.put_nowait((path, chunk))

    def lookup(self, path: str, offset: int) -> CachedChunk | None:
        """Return the chunk covering ``offset`` in ``path``, if registered."""
        entry = self._files.get(path)
        if entry is None:
            return None
        aligned_offset = (offset // self._config.chunk_bytes) * self._config.chunk_bytes
        return entry.find(aligned_offset)

    def get(self, path: str, offset: int) -> memoryview | None:
        """
        Return the cached bytes for the chunk covering ``offset``, waiting if needed.

        Returns ``None`` if the range was never registered (or a worker gave
        up on it), so callers fall back to fetching it themselves. Blocks
        the calling thread; safe to call from any thread.
        """
        loop = self._loop
        if loop is None:
            return None
        future = asyncio.run_coroutine_threadsafe(self._get_on_loop(path, offset), loop)
        return future.result()

    async def _get_on_loop(self, path: str, offset: int) -> memoryview | None:
        # A chunk whose worker gave up (see `_prepare_loop`/`_fetch_loop`) is
        # dropped from `entry.chunks` entirely rather than left in a stuck
        # state, so a lookup miss here means give up rather than wait forever.
        while (chunk := self.lookup(path, offset)) is not None:
            if chunk.state is ChunkState.CACHED:
                return chunk.data
            await asyncio.sleep(0.001)
        return None

    async def _prepare_loop(self) -> None:
        # TODO: with more than one prepare worker, reservations can complete
        # out of submission order, since `reserve_memory` may resolve workers
        # unfairly under contention. Needs an ordering scheme before
        # `num_prepare_workers` can safely be raised above 1.
        assert self._prepare_queue is not None
        assert self._fetch_queue is not None
        assert self._ctx is not None
        while True:
            path, chunk = await self._prepare_queue.get()
            if chunk.state is not ChunkState.EMPTY:
                continue
            chunk.state = ChunkState.ALLOCATED
            try:
                chunk.reservation = await reserve_memory(
                    self._ctx,
                    chunk.size,
                    net_memory_delta=0,
                    mem_type=MemoryType.PINNED_HOST,
                )
            except asyncio.CancelledError:
                self._entry(path).remove(chunk.offset)
                raise
            except Exception:
                self._entry(path).remove(chunk.offset)
                continue
            self._fetch_queue.put_nowait((path, chunk))

    async def _fetch_loop(self) -> None:
        assert self._fetch_queue is not None
        assert self._ctx is not None
        while True:
            path, chunk = await self._fetch_queue.get()
            if chunk.state is not ChunkState.ALLOCATED:
                continue
            chunk.state = ChunkState.LOADING
            try:
                br = self._ctx.br()
                stream = br.stream_pool.get_stream()
                buffer = br.make_buffer(chunk.size, stream, chunk.reservation)
                with buffer.host_view() as view:
                    handle = kvikio.CuFile(path, "rb")
                    future = handle.pread(view, chunk.size, chunk.offset)
                    await asyncio.get_running_loop().run_in_executor(None, future.get)
                    chunk.buffer = buffer
                    chunk.data = view
                    chunk.state = ChunkState.CACHED
            except asyncio.CancelledError:
                self._entry(path).remove(chunk.offset)
                raise
            except Exception:
                self._entry(path).remove(chunk.offset)
