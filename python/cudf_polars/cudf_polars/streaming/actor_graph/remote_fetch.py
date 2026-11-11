# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Fetch a remote parquet scan's tasks ahead of their decode.

When the scan opens, every file's footer is requested from the remote I/O service (one request per
file, in parallel) and the scan opens a stream on the service. A feeder plans the tasks from their
footers (a few at a time, ahead of the decode) and submits each task's byte ranges, in task order;
the service reads them into its pinned arena as soon as the task gets a block, sharing the arena
fairly with the query's other scans. The scan's producers take their task once its bytes are in,
copy them to the device and decode them.

Decodes and their decoded chunks draw on one device memory budget shared by every scan: a task is
admitted in task order, holds its estimated peak while it decodes and its chunk's size until the
chunk is forwarded downstream, so producers cannot decode far ahead of the consumer and pile up
chunks that would be spilled.
"""

from __future__ import annotations

import asyncio
import atexit
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cudf_polars.streaming import remote_io
from cudf_polars.streaming.io import ParquetScanTask

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.io import ScanTask

__all__ = ["RemoteFetchAhead", "enabled", "materialize", "split_tail"]


def enabled(tasks: Sequence[ScanTask]) -> bool:
    """Whether a scan's tasks are read through the remote I/O service."""
    return bool(tasks) and all(
        isinstance(t, ParquetScanTask) and remote_io.eligible(t) for t in tasks
    )


def split_tail(tasks: Sequence[ScanTask]) -> list[ScanTask]:
    """
    Split a remote scan's last whole-file tasks into row-group pieces.

    The work left after a scan's last byte lands is its last chunks' decode and their way through
    the query, so smaller last chunks shorten it; earlier chunks stay whole (each chunk costs a
    fixed amount downstream). ``CUDF_POLARS_SCAN_TAIL_SPLIT=KxN`` splits the last K single-file
    tasks into N pieces each (off when unset).
    """
    spec = os.environ.get("CUDF_POLARS_SCAN_TAIL_SPLIT", "")
    if not spec or not enabled(tasks):
        return list(tasks)
    k, n = (int(x) for x in spec.lower().split("x"))
    result: list[ScanTask] = list(tasks)
    if len(result) < 2 * max(k, 8):
        # Small scans (a few files, few row groups) are left whole.
        return result
    start = max(0, len(result) - k)
    pieces: list[ScanTask] = []
    for task in result[start:]:
        if (
            n > 1
            and isinstance(task, ParquetScanTask)
            and len(task.paths) == 1
            and task.total_splits == 1
        ):
            pieces.extend(
                ParquetScanTask(
                    task.base_scan,
                    list(task.paths),
                    i,
                    n,
                    task.parquet_options,
                    task.hive_parts,
                    task.lake_options,
                )
                for i in range(n)
            )
        else:
            pieces.append(task)
    return result[:start] + pieces


async def resize(tasks: Sequence[ScanTask]) -> list[ScanTask]:
    """
    Regroup a remote scan's whole-file tasks by the bytes their footers say they read.

    Planning gives a remote scan one task per file, as it has no sizes then. Once the scan's
    footers are in (the service fetches them in parallel at scan start), consecutive files are
    fused into one task until it reads about ``CUDF_POLARS_SCAN_FUSE_BYTES`` (off when unset),
    and a file reading more than ``CUDF_POLARS_SCAN_SPLIT_BYTES`` (off when unset) is split into
    row-group pieces of about that size. Both are bytes of the projected column chunks.
    """
    fuse = int(os.environ.get("CUDF_POLARS_SCAN_FUSE_BYTES", "0"))
    split = int(os.environ.get("CUDF_POLARS_SCAN_SPLIT_BYTES", "0"))
    if (fuse <= 0 and split <= 0) or not enabled(tasks) or len(tasks) < 2:
        return list(tasks)
    if not all(
        isinstance(t, ParquetScanTask)
        and len(t.paths) == 1
        and t.total_splits == 1
        and t.hive_parts is None
        and t.lake_options is None
        for t in tasks
    ):
        return list(tasks)
    first = tasks[0]
    assert isinstance(first, ParquetScanTask)
    columns = list(first.base_scan.with_columns or ())
    service = remote_io.scan_io()
    paths = [t.paths[0] for t in tasks]
    service.request_footers(paths)
    t0 = time.perf_counter()
    sizes = await asyncio.to_thread(service.projected_bytes, paths, columns)
    _count("resize_wait_s", time.perf_counter() - t0)

    def _task(paths: list[str], index: int = 0, splits: int = 1) -> ParquetScanTask:
        return ParquetScanTask(
            first.base_scan,
            paths,
            index,
            splits,
            first.parquet_options,
            None,
            None,
        )

    result: list[ScanTask] = []
    group: list[str] = []
    group_bytes = 0
    for task, (nbytes, row_groups) in zip(tasks, sizes, strict=True):
        if split > 0 and nbytes > 1.5 * split and row_groups > 1:
            if group:
                result.append(_task(group))
                group, group_bytes = [], 0
            n = min(row_groups, round(nbytes / split))
            result.extend(_task(list(task.paths), i, n) for i in range(n))
            continue
        if fuse > 0 and group and group_bytes + nbytes > fuse:
            # Close the group when adding this file overshoots more than leaving it short.
            if group_bytes + nbytes - fuse > fuse - group_bytes:
                result.append(_task(group))
                group, group_bytes = [], 0
        group.append(task.paths[0])
        group_bytes += nbytes
        if fuse <= 0 or group_bytes >= fuse:
            result.append(_task(group))
            group, group_bytes = [], 0
    if group:
        result.append(_task(group))
    _count("resized_scans")
    _count("resized_tasks", len(result) - len(tasks))
    return result


_stats: dict[str, Any] = {}
_stats_lock = threading.Lock()


def _count(key: str, value: float = 1) -> None:
    with _stats_lock:
        _stats[key] = _stats.get(key, 0) + value


def _dump_stats() -> None:
    path = os.environ.get("CUDF_POLARS_REMOTE_FETCH_STATS")
    if path and _stats:
        service = remote_io.scan_io()
        if service is not None:
            _stats["fetched_bytes"] = service.fetched_bytes
        with open(path, "w") as f:
            for k, v in sorted(_stats.items()):
                f.write(f"{k}\t{v}\n")


atexit.register(_dump_stats)

_planners: ThreadPoolExecutor | None = None
_lock = threading.Lock()


def _planner_pool() -> ThreadPoolExecutor:
    global _planners  # noqa: PLW0603
    with _lock:
        if _planners is None:
            _planners = ThreadPoolExecutor(
                int(os.environ.get("CUDF_POLARS_REMOTE_PLAN_THREADS", "8")),
                thread_name_prefix="scan-plan",
            )
        return _planners


@dataclass
class Staged:
    """A planned task whose bytes the service reads into its arena."""

    plan: remote_io.RemotePlan | remote_io.NativePlan
    task_id: int


def _device_total_bytes() -> int:
    from cuda.bindings import runtime as cudart  # noqa: PLC0415

    _, _, total = cudart.cudaMemGetInfo()
    return int(total)


class _SharedBudget:
    """Device bytes held by decodes and unforwarded decoded chunks, across all scans."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held = 0
        self._waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Future]] = []

    async def acquire(
        self, cost: int, scan_idle: Callable[[], bool], budget: int
    ) -> None:
        # ``scan_idle`` is asked again on every wake-up: the scan's own chunks are released while
        # it waits, and once it holds none it must be admitted, or it could wait on a scan whose
        # consumer is waiting on it.
        loop = asyncio.get_running_loop()
        while True:
            with self._lock:
                if scan_idle() or self._held + cost <= budget:
                    self._held += cost
                    return
                waiter = loop.create_future()
                self._waiters.append((loop, waiter))
            await waiter

    def release(self, cost: int) -> None:
        with self._lock:
            self._held -= cost
            waiters, self._waiters = self._waiters, []
        for loop, waiter in waiters:
            loop.call_soon_threadsafe(_wake, waiter)


def _wake(waiter: asyncio.Future) -> None:
    if not waiter.done():
        waiter.set_result(None)


_budget = _SharedBudget()

_lag_loops: set[int] = set()


def _start_lag_monitor() -> None:
    # Experiment: how late the actors' event loop wakes a 5 ms sleep (loop congestion).
    if not os.environ.get("CUDF_POLARS_LOOP_LAG"):
        return
    loop = asyncio.get_running_loop()
    if id(loop) in _lag_loops:
        return
    _lag_loops.add(id(loop))
    import threading as _th  # noqa: PLC0415

    remote_io.trace("looptid", -4, 0, _th.get_native_id())

    async def _monitor() -> None:
        w0, c0 = time.perf_counter(), time.thread_time()
        while True:
            t0 = time.perf_counter()
            await asyncio.sleep(0.005)
            t1 = time.perf_counter()
            remote_io.trace("lag", -4, 0, int((t1 - t0 - 0.005) * 1e6))
            if t1 - w0 >= 0.25:
                # Share of wall time the loop thread spent on the CPU (running Python callbacks).
                c1 = time.thread_time()
                remote_io.trace("loopcpu", -4, 0, int(1000 * (c1 - c0) / (t1 - w0)))
                w0, c0 = t1, c1

    asyncio.ensure_future(_monitor())

    if os.environ.get("CUDF_POLARS_LOOP_SLOW"):
        import logging  # noqa: PLC0415

        handler = logging.FileHandler(os.environ["CUDF_POLARS_LOOP_SLOW"])
        logging.getLogger("asyncio").addHandler(handler)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        loop.set_debug(True)
        loop.slow_callback_duration = 0.005

    # A watchdog thread: its own wake-up lag (GIL contention) and samples of the loop thread's
    # stack (where the loop spends its wall time).
    import collections  # noqa: PLC0415
    import sys  # noqa: PLC0415
    import threading  # noqa: PLC0415
    import traceback  # noqa: PLC0415

    loop_tid = threading.get_ident()
    samples: collections.Counter = collections.Counter()

    def _watch() -> None:
        while True:
            t0 = time.perf_counter()
            time.sleep(0.002)
            remote_io.trace(
                "thrlag", -4, 0, int((time.perf_counter() - t0 - 0.002) * 1e6)
            )
            frame = sys._current_frames().get(loop_tid)
            if frame is not None:
                stack = traceback.extract_stack(frame)[-6:]
                samples[
                    " <- ".join(f"{f.name}:{f.lineno}" for f in reversed(stack))
                ] += 1

    def _dump() -> None:
        with open(os.environ["CUDF_POLARS_LOOP_LAG"], "w") as f:
            total = sum(samples.values())
            for stack, n in samples.most_common(40):
                f.write(f"{n}\t{100 * n / total:.1f}%\t{stack}\n")

    import atexit  # noqa: PLC0415

    atexit.register(_dump)
    threading.Thread(target=_watch, daemon=True, name="loop-watchdog").start()


def _plan(
    task: ParquetScanTask, ir_context: IRExecutionContext
) -> remote_io.RemotePlan:
    t0 = time.perf_counter()
    try:
        return remote_io.plan(task, ir_context.get_cuda_stream())
    finally:
        _count("plan_s", time.perf_counter() - t0)


def materialize(
    task: ParquetScanTask, staged: Staged, ir_context: IRExecutionContext
) -> DataFrame:
    """Decode a staged task (inside the caller's device memory reservation)."""
    stream = remote_io.decode_stream(ir_context)
    if isinstance(staged.plan, remote_io.NativePlan):
        df = remote_io.materialize_native(task, staged.plan, stream)
        staged.plan.reader = None
        remote_io.trace("decoded", -1, staged.task_id, 0)
        return df
    service = remote_io.scan_io()
    block = service.to_device(staged.task_id, stream)
    df = remote_io.materialize(task, staged.plan, block, stream)
    stream.synchronize()  # the copy out of the arena block has completed
    staged.plan.reader = None  # the reader holds device buffers of the decode
    return df


def _drain(task_ids: list[int]) -> None:
    service = remote_io.scan_io()
    for task_id in task_ids:
        try:
            service.wait(task_id)
        except Exception:  # noqa: BLE001
            _count("drain_read_errors")
        service.release(task_id)


class RemoteFetchAhead:
    """Plans a scan's tasks ahead and has the service fetch them, in task order."""

    def __init__(
        self, tasks: Sequence[ScanTask], ir_context: IRExecutionContext
    ) -> None:
        self._tasks = tasks
        self._ir_context = ir_context
        self._service = remote_io.scan_io()
        self._ahead = int(os.environ.get("CUDF_POLARS_REMOTE_PLAN_AHEAD", "64"))
        n = len(tasks)
        self._staged: list[Staged | None] = [None] * n
        self._launched = [asyncio.Event() for _ in range(n)]
        self._taken = 0
        self._cond = asyncio.Condition()
        self._feeder: asyncio.Task | None = None
        self._stream = 0
        self._waiter: ThreadPoolExecutor | None = None
        self._read: dict[int, asyncio.Future] = {}
        self._decode_budget = int(
            float(os.environ.get("CUDF_POLARS_DECODE_MEM_FRAC", "0.45"))
            * _device_total_bytes()
        )
        self._decoding = 0
        self._next_admit = 0
        # Plan and queue tasks in the service, without the GIL, in task order (one thread).
        self._native = os.environ.get("CUDF_POLARS_REMOTE_NATIVE_PLAN", "1") != "0"
        self._native_pool: ThreadPoolExecutor | None = None
        self._scan_options: tuple[Any, Any] | None = None

    def start(self) -> None:
        # Every footer of the scan is requested at once (the service runs them in parallel); the
        # feeder then plans each task as soon as its footers are in.
        self._service.request_footers([p for t in self._tasks for p in t.paths])
        priority = remote_io.scan_priority(len(self._tasks))
        self._stream = (
            self._service.open_stream(priority)
            if priority
            else self._service.open_stream()
        )
        remote_io.trace("scan_start", self._stream, -1, len(self._tasks))
        _start_lag_monitor()
        self._feeder = asyncio.ensure_future(self._feed())

    async def close(self) -> None:
        if self._feeder is not None:
            self._feeder.cancel()
            try:
                await self._feeder
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if self._waiter is not None:
            self._waiter.shutdown(wait=False, cancel_futures=True)
        if self._native_pool is not None:
            self._native_pool.shutdown(wait=False, cancel_futures=True)
        # Tasks staged but never taken (an early stop): give their blocks back once their reads
        # have landed.
        leftover = [s.task_id for s in self._staged if s is not None and s.task_id >= 0]
        self._staged = [None] * len(self._staged)
        if leftover:
            await asyncio.to_thread(_drain, leftover)
        if self._stream:
            self._service.close_stream(self._stream)
            self._stream = 0

    def _plan_native(self, task: ParquetScanTask) -> remote_io.NativePlan | None:
        stream = self._ir_context.get_cuda_stream()
        if self._scan_options is None:
            self._scan_options = remote_io.scan_options(task, stream)
        options, residual = self._scan_options
        t0 = time.perf_counter()
        try:
            return remote_io.plan_native(task, self._stream, options, residual, stream)
        finally:
            _count("native_plan_s", time.perf_counter() - t0)

    async def _feed_native(self) -> None:
        loop = asyncio.get_running_loop()
        self._native_pool = ThreadPoolExecutor(1, thread_name_prefix="scan-plan-native")
        for i, task in enumerate(self._tasks):
            async with self._cond:
                await self._cond.wait_for(lambda: i - self._taken < self._ahead)
            plan = await loop.run_in_executor(
                self._native_pool, self._plan_native, task
            )
            if plan is None:
                # A split the native planner does not handle: plan and queue it in Python.
                _count("python_planned")
                plan = await loop.run_in_executor(
                    self._native_pool, _plan, task, self._ir_context
                )
                task_id = self._service.submit(
                    self._stream,
                    plan.paths,
                    [s for s, _, _ in plan.requests],
                    [o for _, o, _ in plan.requests],
                    [z for _, _, z in plan.requests],
                )
                if task_id < 0:
                    _count("too_big_for_arena")
                    self._staged[i] = None
                    self._launched[i].set()
                    continue
            else:
                task_id = plan.task_id
            _count("staged")
            _count("staged_bytes", plan.nbytes)
            remote_io.trace("planned", self._stream, i, plan.nbytes)
            staged = Staged(plan, task_id)
            if task_id >= 0:
                self._read[i] = self._watch(task_id)
                self._read[i].add_done_callback(
                    lambda _, i=i, n=plan.nbytes: remote_io.trace(
                        "landed", self._stream, i, n
                    )
                )
            self._staged[i] = staged
            self._launched[i].set()

    async def _feed(self) -> None:
        if self._native:
            await self._feed_native()
            return
        loop = asyncio.get_running_loop()
        pool = _planner_pool()
        plans: dict[int, asyncio.Future] = {}
        next_plan = 0
        for i in range(len(self._tasks)):
            async with self._cond:
                await self._cond.wait_for(lambda: i - self._taken < self._ahead)
            # Plan ahead concurrently, but submit strictly in task order.
            while next_plan < min(len(self._tasks), i + self._ahead):
                plans[next_plan] = loop.run_in_executor(
                    pool, _plan, self._tasks[next_plan], self._ir_context
                )
                next_plan += 1
            plan = await plans.pop(i)
            remote_io.trace("planned", self._stream, i, plan.nbytes)
            task_id = self._service.submit(
                self._stream,
                plan.paths,
                [s for s, _, _ in plan.requests],
                [o for _, o, _ in plan.requests],
                [z for _, _, z in plan.requests],
            )
            if task_id < 0:
                _count("too_big_for_arena")
                staged = None
            else:
                _count("staged")
                _count("staged_bytes", plan.nbytes)
                staged = Staged(plan, task_id)
                self._read[i] = self._watch(task_id)
                self._read[i].add_done_callback(
                    lambda _, i=i, n=plan.nbytes: remote_io.trace(
                        "landed", self._stream, i, n
                    )
                )
            self._staged[i] = staged
            self._launched[i].set()

    def _watch(self, task_id: int) -> asyncio.Future:
        # One thread per scan blocks (without the GIL) on each task in turn.
        if self._waiter is None:
            self._waiter = ThreadPoolExecutor(1, thread_name_prefix="scan-wait")
        return asyncio.get_running_loop().run_in_executor(
            self._waiter, self._service.wait, task_id
        )

    async def take(self, i: int) -> Staged | None:
        """Task ``i`` once its bytes are in, or None to read it the regular way."""
        await self._launched[i].wait()
        staged = self._staged[i]
        self._staged[i] = None
        read = self._read.pop(i, None)
        t0 = time.perf_counter()
        if read is not None:
            await asyncio.wait(
                [read]
            )  # a read error surfaces again from the copy to device
        _count("producer_wait_s", time.perf_counter() - t0)
        remote_io.trace("taken", self._stream, i, 0)
        async with self._cond:
            self._taken += 1
            self._cond.notify_all()
        return staged

    async def acquire_decode(self, i: int, staged: Staged | None) -> int:
        """Wait until task ``i`` fits the decode budget (in task order); returns its cost."""
        if self._decode_budget <= 0:
            return 0
        cost = (
            0
            if staged is None
            else min(
                staged.plan.decoded_bytes + staged.plan.nbytes, self._decode_budget
            )
        )
        t0 = time.perf_counter()
        async with self._cond:
            await self._cond.wait_for(lambda: self._next_admit == i)
        await _budget.acquire(cost, lambda: self._decoding == 0, self._decode_budget)
        # No await between taking the budget and recording it, so a cancellation cannot leak it.
        self._decoding += cost
        self._next_admit += 1
        async with self._cond:
            self._cond.notify_all()
        _count("decode_wait_s", time.perf_counter() - t0)
        return cost

    async def shrink_decode(self, cost: int, keep: int) -> int:
        """Decode done: keep only ``keep`` bytes of ``cost`` held; returns what is held."""
        keep = min(cost, keep)
        await self.release_decode(cost - keep)
        return keep

    async def release_decode(self, cost: int) -> None:
        if cost > 0:
            self._decoding -= cost
            _budget.release(cost)
            async with self._cond:
                self._cond.notify_all()

    async def release(self, staged: Staged | None) -> None:
        """Task decoded: give its block back to the arena."""
        if staged is not None and staged.task_id >= 0:
            self._service.release(staged.task_id)
