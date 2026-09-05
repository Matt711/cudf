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
import time
from typing import TYPE_CHECKING

import kvikio

import pylibcudf as plc
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
    registered_at: float = 0.0


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
    pool_capacity_bytes: int

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
            pool_capacity_bytes=parquet_options.prefetch_cache_pool_capacity_bytes,
        )


@dataclasses.dataclass
class PrefetchCacheStats:
    """
    Cumulative counters for measuring cache effectiveness.

    A hit means a chunk was already ``CACHED`` the moment a consumer asked
    for it. ``get`` never waits for one still in flight -- matching
    cuCascade's own ``host_read_from_cache_only`` -- so ``misses`` covers
    "never registered", "a worker gave up on it", and "still loading"
    alike, all indistinguishable to a consumer, all a fall back to the
    ordinary synchronous read.
    """

    registrations: int = 0
    hits_immediate: int = 0
    misses: int = 0
    #: Sum of (became-CACHED time - registration time), over every chunk
    #: that reached CACHED, regardless of whether anything ever consumed
    #: it. Divide by ``cached_count`` for the average head start a
    #: registered chunk actually got before it was ready.
    lead_time_seconds: float = 0.0
    cached_count: int = 0
    reservation_failures: int = 0
    fetch_failures: int = 0
    max_prepare_queue_depth: int = 0
    max_fetch_queue_depth: int = 0
    evictions: int = 0
    evicted_bytes: int = 0

    def summary(self) -> str:
        """One-line human-readable summary, e.g. for logging between passes."""
        total = self.hits_immediate + self.misses
        hit_rate = self.hits_immediate / total if total else 0.0
        avg_lead_time = (
            self.lead_time_seconds / self.cached_count if self.cached_count else 0.0
        )
        return (
            f"registrations={self.registrations} gets={total} "
            f"hit_rate={hit_rate:.1%} hits_immediate={self.hits_immediate} "
            f"misses={self.misses} "
            f"avg_lead_time={avg_lead_time * 1000:.2f}ms cached_count={self.cached_count} "
            f"reservation_failures={self.reservation_failures} "
            f"fetch_failures={self.fetch_failures} "
            f"max_prepare_queue_depth={self.max_prepare_queue_depth} "
            f"max_fetch_queue_depth={self.max_fetch_queue_depth} "
            f"evictions={self.evictions} evicted_bytes={self.evicted_bytes}"
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
        self._stats = PrefetchCacheStats()
        # One open handle per file, reused across every chunk fetched from
        # it. Only ever touched from `_fetch_loop`, on the cache's own loop,
        # so no lock needed. Opening a fresh `kvikio.RemoteFile` per chunk
        # would cost an HTTP HEAD request every time instead of once.
        self._handles: dict[str, kvikio.CuFile | kvikio.RemoteFile] = {}
        # Bytes currently held by chunks in ALLOCATED, LOADING, or CACHED
        # state (i.e. reserved, whether or not the read has finished yet).
        # Only ever touched from the cache's own loop.
        self._reserved_bytes = 0
        self._spill_function_id: int | None = None

    @property
    def chunk_bytes(self) -> int:
        """Size, in bytes, of the chunk-aligned ranges this cache tracks."""
        return self._config.chunk_bytes

    @property
    def stats(self) -> PrefetchCacheStats:
        """Snapshot of the cumulative hit/miss counters."""
        return dataclasses.replace(self._stats)

    def chunk_state_counts(self) -> dict[str, int]:
        """
        Snapshot of how many chunks are in each :class:`ChunkState` right now.

        The cumulative counters in :meth:`stats` say what happened
        overall, not where chunks are stuck at a given moment: a pile-up
        of ``ALLOCATED`` chunks with few reaching ``CACHED`` points at a
        `_fetch_loop` backlog, not a registration or reservation problem.
        Safe to call from any thread.
        """
        loop = self._loop
        if loop is None:
            return {}
        future = asyncio.run_coroutine_threadsafe(
            self._chunk_state_counts_on_loop(), loop
        )
        return future.result()

    async def _chunk_state_counts_on_loop(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self._files.values():
            for chunk in entry.chunks:
                counts[chunk.state.name] = counts.get(chunk.state.name, 0) + 1
        return counts

    def reset_stats(self) -> None:
        """Reset the hit/miss counters, e.g. between benchmark passes."""
        self._stats = PrefetchCacheStats()

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
            # Bounded so `_prepare_loop` can't reserve/register far faster
            # than `_fetch_loop` can actually drain: with no bound here,
            # chunks pile up faster than they can be fetched and get
            # evicted (oldest first) before `_fetch_loop` ever reaches
            # them, wasting the reservation entirely. Sized to roughly one
            # poolful, the same capacity concept eviction already uses,
            # rather than a second, independently-tunable number.
            fetch_queue_maxsize = max(
                1, self._config.pool_capacity_bytes // max(self._config.chunk_bytes, 1)
            )
            self._fetch_queue = asyncio.Queue(maxsize=fetch_queue_maxsize)
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
        # Registered so a reservation elsewhere in the engine that's
        # starved for pinned memory can reclaim it from cached chunks,
        # not just this cache's own `_evict_to_fit` on its own reservations.
        self._spill_function_id = ctx.br().spill_manager.add_spill_function(
            self._spill, priority=0, mem_type=MemoryType.PINNED_HOST
        )

    def stop(self) -> None:
        """Stop the background loop and join its thread."""
        if self._thread is None:
            return
        loop = self._loop
        assert loop is not None
        assert self._ctx is not None
        if self._spill_function_id is not None:
            self._ctx.br().spill_manager.remove_spill_function(self._spill_function_id)
            self._spill_function_id = None

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
        self._reserved_bytes = 0
        self._ctx = None
        if self._pruning_executor is not None:
            self._pruning_executor.shutdown(wait=True, cancel_futures=True)
            self._pruning_executor = None
        for handle in self._handles.values():
            with contextlib.suppress(Exception):
                handle.close()
        self._handles = {}

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
        chunk = CachedChunk(
            offset=aligned_offset, size=size, registered_at=time.monotonic()
        )
        entry.insert(chunk)
        self._stats.registrations += 1
        assert self._prepare_queue is not None
        self._prepare_queue.put_nowait((path, chunk))
        self._stats.max_prepare_queue_depth = max(
            self._stats.max_prepare_queue_depth, self._prepare_queue.qsize()
        )

    def lookup(self, path: str, offset: int) -> CachedChunk | None:
        """Return the chunk covering ``offset`` in ``path``, if registered."""
        entry = self._files.get(path)
        if entry is None:
            return None
        aligned_offset = (offset // self._config.chunk_bytes) * self._config.chunk_bytes
        return entry.find(aligned_offset)

    def get(self, path: str, offset: int) -> memoryview | None:
        """
        Return the cached bytes for the chunk covering ``offset``, if already resident.

        Returns ``None`` if the chunk isn't ``CACHED`` right now, whether
        it was never registered, a worker gave up on it, or it's simply
        still in flight -- this never waits for a chunk to become ready,
        matching cuCascade's own ``host_read_from_cache_only``. Callers
        fall back to fetching it themselves on any miss. Safe to call from
        any thread.
        """
        loop = self._loop
        if loop is None:
            return None
        future = asyncio.run_coroutine_threadsafe(self._get_on_loop(path, offset), loop)
        return future.result()

    async def _get_on_loop(self, path: str, offset: int) -> memoryview | None:
        chunk = self.lookup(path, offset)
        if chunk is not None and chunk.state is ChunkState.CACHED:
            self._stats.hits_immediate += 1
            return chunk.data
        self._stats.misses += 1
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
            self._evict_to_fit(chunk.size)
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
                self._stats.reservation_failures += 1
                continue
            self._reserved_bytes += chunk.size
            await self._fetch_queue.put((path, chunk))
            self._stats.max_fetch_queue_depth = max(
                self._stats.max_fetch_queue_depth, self._fetch_queue.qsize()
            )

    def _evict_to_fit(self, needed: int) -> None:
        """Evict CACHED/ALLOCATED chunks, oldest first, until ``needed`` more bytes fit."""
        capacity = self._config.pool_capacity_bytes
        over = self._reserved_bytes + needed - capacity
        if over > 0:
            self._evict_bytes(over)

    def _evict_bytes(self, amount: int) -> int:
        """
        Evict CACHED/ALLOCATED chunks, oldest first, until ``amount`` bytes are freed.

        Never touches a chunk mid-``LOADING``. Discards rather than
        preserving evicted chunks: they're a read cache, cheap to refetch,
        not the result of compute, so a later miss just falls back to the
        ordinary synchronous read. Returns the number of bytes actually
        freed, which may be less than ``amount`` if there's nothing left
        to evict.
        """
        candidates = sorted(
            (
                (path, chunk)
                for path, entry in self._files.items()
                for chunk in entry.chunks
                if chunk.state in (ChunkState.ALLOCATED, ChunkState.CACHED)
            ),
            key=lambda item: item[1].registered_at,
        )
        freed = 0
        for path, chunk in candidates:
            if freed >= amount:
                break
            self._entry(path).remove(chunk.offset)
            # An evicted ALLOCATED chunk may still be sitting in
            # `_fetch_queue`; marking it EMPTY here (instead of leaving it
            # ALLOCATED) makes `_fetch_loop`'s own state check skip it
            # when dequeued, rather than wastefully fetching a chunk no
            # consumer can reach anymore.
            chunk.state = ChunkState.EMPTY
            self._reserved_bytes -= chunk.size
            self._stats.evictions += 1
            self._stats.evicted_bytes += chunk.size
            freed += chunk.size
        return freed

    def _spill(self, amount: int) -> int:
        """
        Spill-function callback registered with the engine's ``SpillManager``.

        Lets a pinned-memory reservation anywhere else in the engine
        reclaim room from this cache under pressure, not just this
        cache's own reservations via :meth:`_evict_to_fit`. Safe to call
        from any thread; blocks the calling thread until the eviction
        runs on the cache's own loop.
        """
        loop = self._loop
        if loop is None:
            return 0
        future = asyncio.run_coroutine_threadsafe(self._spill_on_loop(amount), loop)
        return future.result()

    async def _spill_on_loop(self, amount: int) -> int:
        return self._evict_bytes(amount)

    def _get_handle(self, path: str) -> kvikio.CuFile | kvikio.RemoteFile:
        handle = self._handles.get(path)
        if handle is None:
            handle = (
                kvikio.RemoteFile.open(path)
                if plc.io.SourceInfo._is_remote_uri(path)
                else kvikio.CuFile(path, "rb")
            )
            self._handles[path] = handle
        return handle

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
                    handle = self._get_handle(path)
                    future = handle.pread(view, chunk.size, chunk.offset)
                    await asyncio.get_running_loop().run_in_executor(None, future.get)
                    chunk.buffer = buffer
                    chunk.data = view
                    chunk.state = ChunkState.CACHED
                    self._stats.lead_time_seconds += (
                        time.monotonic() - chunk.registered_at
                    )
                    self._stats.cached_count += 1
            except asyncio.CancelledError:
                self._entry(path).remove(chunk.offset)
                self._reserved_bytes -= chunk.size
                raise
            except Exception:
                self._entry(path).remove(chunk.offset)
                self._reserved_bytes -= chunk.size
                self._stats.fetch_failures += 1
