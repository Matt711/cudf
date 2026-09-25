# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python equivalent of Sirius readahead_scan_manager.

Drives prepare_prefetch + prefetch_async for every registered ParquetScanTask
using a background thread (Python equivalent of Sirius's std::jthread) and a
semaphore gatekeeper (Sirius's gatekeeper class).

Only active when SIRIUS_DATASOURCE=1.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import enum
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from cudf_polars.streaming.io import ParquetScanTask


@dataclasses.dataclass
class ReadaheadCounters:
    """Mirrors Sirius readahead_counters struct for per-query diagnostics.

    Counter semantics match readahead_scan_manager.hpp exactly:
    - prefetched: IO completed before executor reached the split (readahead won)
    - wait_for_prefetch: IO issued but executor had to wait for completion
    - skipped_memory_pressure: prepare_prefetch failed (pinned pool full)
    - skipped_fell_behind: executor was already at split when worker tried
    - nothing_to_issue: no byte ranges / cache disabled / already in cache
    - candidates_taken: tasks dequeued by worker and attempted
    - dropped_expired: tasks dropped at queue head (datasource missing; analog of expired weak_ptr)
    - dropped_fell_behind: tasks dropped at queue head because already fallen behind
    - operators_drained: operator queues that reached end and were marked closed
    - memory_retries: prepare_prefetch retry iterations (waited for eviction)
    - idle_polls: gatekeeper acquired but no candidate found to spend it on
    - gate_timeouts: _gatekeeper.acquire(timeout=...) timed out (all slots in flight)
    - executor_reads: cold executor reads not prefetched by readahead
    Timing fields (not in Sirius, added for Python diagnostics):
    - gatekeeper_blocked_ms: wall time spent waiting for gatekeeper slots
    - prepare_retry_ms: wall time spent in prepare_prefetch retry loop
    """

    # outcomes — exactly one per attempt, matching prefetch_outcome_kind enum
    prefetched: int = 0
    wait_for_prefetch: int = 0
    skipped_memory_pressure: int = 0
    skipped_fell_behind: int = 0
    nothing_to_issue: int = 0
    # candidate selection
    candidates_taken: int = 0
    dropped_expired: int = 0
    dropped_fell_behind: int = 0
    operators_drained: int = 0
    # pacing
    memory_retries: int = 0
    idle_polls: int = 0
    gate_timeouts: int = 0
    # executor competition
    executor_reads: int = 0
    # GET completion tracking (not in Sirius — diagnose ok=True vs ok=False callbacks)
    prefetched_ok: int = 0       # callback ok=True: GET actually completed into pinned RAM
    prefetched_refused: int = 0  # callback ok=False: prefetch_async refused (consumer_ahead etc)
    wait_for_prefetch_ok: int = 0
    wait_for_prefetch_refused: int = 0
    # timing (not in Sirius — Python-only diagnostics)
    gatekeeper_blocked_ms: float = 0.0
    prepare_retry_ms: float = 0.0
    # upper-bound on time the GPU reader spent blocked waiting for in-flight GETs:
    # sum of (GET completion time - GET issue time) for all late tasks (wait_for_prefetch).
    # actual wait <= this value (reader may arrive mid-transfer); useful as a ceiling.
    wait_for_prefetch_total_ms: float = 0.0
    # prefetch issuance timeline (helps diagnose reactor idle gaps)
    start_to_first_prefetch_ms: float = 0.0   # latency from start() to first GET issued
    max_inter_prefetch_gap_ms: float = 0.0    # longest gap between consecutive GETs
    total_prefetch_window_ms: float = 0.0     # first GET → last GET wall time
    # query setup delay: time from cache reset to readahead.start() (ENA idle dead zone)
    setup_delay_ms: float = 0.0
    # time from NodeTraverser entry (first cudf-polars entry point) to readahead.start()
    nt_to_start_ms: float = 0.0

    def summary(self) -> str:
        """One-line summary matching Sirius readahead_scan_manager::summary() format.

        Sirius format:
          issued=N[prefetched=N wait_for_prefetch=N]
          skipped=N[memory_pressure=N fell_behind=N nothing_to_issue=N]
          candidates=N[dropped_expired=N dropped_fell_behind=N]
          operators_drained=N
          pacing[memory_retries=N idle_polls=N gate_timeouts=N]
          executor_reads=N[borrowed=0]
        """
        issued = self.prefetched + self.wait_for_prefetch
        skipped = self.skipped_memory_pressure + self.skipped_fell_behind + self.nothing_to_issue
        return (
            f"issued={issued}[prefetched={self.prefetched} wait_for_prefetch={self.wait_for_prefetch}] "
            f"skipped={skipped}[memory_pressure={self.skipped_memory_pressure} "
            f"fell_behind={self.skipped_fell_behind} nothing_to_issue={self.nothing_to_issue}] "
            f"candidates={self.candidates_taken}[dropped_expired={self.dropped_expired} "
            f"dropped_fell_behind={self.dropped_fell_behind}] "
            f"operators_drained={self.operators_drained} "
            f"pacing[memory_retries={self.memory_retries} idle_polls={self.idle_polls} "
            f"gate_timeouts={self.gate_timeouts}] "
            f"executor_reads={self.executor_reads}[borrowed=0]"
        )

    def timing_summary(self) -> str:
        """Additional timing line (not in Sirius) for Python worker diagnostics."""
        return (
            f"gatekeeper_blocked_ms={self.gatekeeper_blocked_ms:.1f} "
            f"prepare_retry_ms={self.prepare_retry_ms:.1f} "
            f"prefetched[ok={self.prefetched_ok} refused={self.prefetched_refused}] "
            f"wait_for_prefetch[ok={self.wait_for_prefetch_ok} refused={self.wait_for_prefetch_refused}] "
            f"wait_for_prefetch_total_ms={self.wait_for_prefetch_total_ms:.1f} "
            f"setup_delay={self.setup_delay_ms:.1f}ms "
            f"nt_to_start={self.nt_to_start_ms:.1f}ms "
            f"prefetch_timeline[start_to_first={self.start_to_first_prefetch_ms:.1f}ms "
            f"max_inter_gap={self.max_inter_prefetch_gap_ms:.1f}ms "
            f"window={self.total_prefetch_window_ms:.1f}ms]"
        )


class PrefetchStrategy(enum.IntEnum):
    """Python mirror of Sirius prefetch_strategy enum.

    Only ``eager`` is implemented. ``opportunistic`` requires executor idle
    signals that don't exist in cudf-polars.
    """

    eager = 0
    opportunistic = 1


class PrefetchOutcomeKind(enum.IntEnum):
    """Diagnostic outcome codes for one prefetch attempt."""

    prefetched = 0
    skipped_fell_behind = 1
    skipped_memory_pressure = 2
    wait_for_prefetch = 3
    nothing_to_issue = 4


class PrepareOutcome:
    """Wraps the raw ``prepare_result`` int returned by SiriusDatasource.prepare_prefetch.

    prepare_result values (from cucascade/include):
      0 = prepared
      1 = allocation_failed
      2 = nothing_to_prepare
      3 = fallen_behind   (consumer reached this split before preparation)
    """

    __slots__ = ("_code",)

    def __init__(self, code: int):
        self._code = code

    def ready(self) -> bool:
        """Return True if staging buffers were allocated and prefetch can proceed."""
        return self._code == 0

    @property
    def failed(self) -> int:
        """1 if allocation_failed (pool full), else 0."""
        return 1 if self._code == 1 else 0

    @property
    def fell_behind(self) -> int:
        """1 if fallen_behind (executor is already reading), else 0."""
        return 1 if self._code == 3 else 0


class ReadaheadScanManager:
    """Python equivalent of Sirius's readahead_scan_manager.

    Owns a background thread that drives prepare_prefetch + prefetch_async for
    all registered ParquetScanTask objects, ordered by operator ID (execution order).

    The gatekeeper is a threading.Semaphore(budget) — equivalent to the C++
    gatekeeper class. Budget = rest_n_reactors = number of concurrent in-flight S3 GETs.

    Usage::

        readahead = ReadaheadScanManager(budget=32)
        for op_id, scan_node in enumerate(scan_nodes):
            for task in scan_node.tasks:
                readahead.register_scan_task(task, op_id)
            readahead.mark_operator_closed(op_id)
        readahead.start(PrefetchStrategy.eager)
        # ... run query executor ...
        readahead.stop()
    """

    def __init__(self, budget: int):
        self._budget = budget
        self._gatekeeper = threading.Semaphore(budget)
        self._operator_id_to_queue_index: dict[int, int] = {}
        self._queue_index_to_operator_id: dict[int, int] = {}
        self._ordered_work_queues: list[deque[ParquetScanTask]] = []
        self._closed_operators: set[int] = set()
        self._cursor = 0
        self._stop_event = threading.Event()
        self._prefetch_worker: threading.Thread | None = None
        self._counters = ReadaheadCounters()
        # Lock for counters updated from C++ callback threads (prefetched/wait_for_prefetch)
        self._callback_lock = threading.Lock()
        # Track which queue indices have been permanently retired (drained + closed)
        # so operators_drained counter is only incremented once per operator.
        self._drained_queue_indices: set[int] = set()
        # Set when task registration is complete. In streaming mode (early fadvise),
        # the worker starts before all tasks are registered; _all_drained() must not
        # return True until registration is done. In normal mode, set immediately in start().
        self._registration_complete = threading.Event()
        # Pulsed each time a new task is registered in streaming mode so the worker
        # wakes up immediately instead of sleeping the full idle period.
        self._task_ready = threading.Event()
        # Timing state for prefetch-issue tracking; shared across prepare threads via _timing_lock.
        self._first_prefetch_t: float | None = None
        self._last_prefetch_t: float | None = None
        self._timing_lock = threading.Lock()

    def register_scan_task(self, task: ParquetScanTask, operator_id: int) -> None:
        """Register a scan task with the given operator ID.

        Must be called before start(). Tasks within the same operator are
        processed in registration order.
        """
        if operator_id not in self._operator_id_to_queue_index:
            q_idx = len(self._ordered_work_queues)
            self._operator_id_to_queue_index[operator_id] = q_idx
            self._queue_index_to_operator_id[q_idx] = operator_id
            self._ordered_work_queues.append(deque())
        q_idx = self._operator_id_to_queue_index[operator_id]
        self._ordered_work_queues[q_idx].append(task)

    def mark_operator_closed(self, operator_id: int) -> None:
        """Signal that no more tasks will be registered for this operator."""
        self._closed_operators.add(operator_id)

    def update_scan_state(
        self,
        operator_id: int,
        task: ParquetScanTask,
        stage: int,
    ) -> None:
        """Called by the executor when a task transitions to a new scan stage.

        Equivalent to Sirius readahead_scan_manager::update_scan_state().
        The stage value must be a ScanStage int (reading=4, disposed=5).
        """
        task._scan_stage = stage
        # Sirius only counts splits that were NOT prefetched (cold reads). We
        # count ALL reading transitions here — it over-counts but gives a useful
        # total scans-started number to compare against Sirius's executor_reads.
        if stage == 4:  # ScanStage.reading
            self._counters.executor_reads += 1

    def notify_new_task(self) -> None:
        """Wake the worker immediately when a new task is registered in streaming mode."""
        self._task_ready.set()

    def finish_registration(self) -> None:
        """Signal that all tasks have been registered (streaming mode only).

        Must be called after the last register_scan_task() / mark_operator_closed()
        when start() was called with streaming=True. Allows the worker to detect drain.
        """
        self._registration_complete.set()
        self._task_ready.set()  # wake worker in case it's idle-sleeping

    def start(
        self,
        strategy: PrefetchStrategy = PrefetchStrategy.eager,
        streaming: bool = False,
    ) -> None:
        """Launch the background prefetch worker thread.

        When streaming=True (SIRIUS_EARLY_FADVISE mode), the worker starts before
        all tasks are registered. The caller must call finish_registration() after
        the last register_scan_task() to allow drain detection.
        When streaming=False (default), registration is considered complete immediately.
        """
        self._start_t = time.perf_counter()
        try:
            from cudf_polars.engine.core import get_query_reset_t
            from cudf_polars.callback import get_nt_entry_t
            _reset_t = get_query_reset_t()
            if _reset_t is not None:
                self._counters.setup_delay_ms = (self._start_t - _reset_t) * 1000.0
            _nt_t = get_nt_entry_t()
            if _nt_t is not None:
                self._counters.nt_to_start_ms = (self._start_t - _nt_t) * 1000.0
        except ImportError:
            pass
        if not streaming:
            self._registration_complete.set()
        if strategy == PrefetchStrategy.eager:
            self._prefetch_worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="readahead-prefetch",
            )
            self._prefetch_worker.start()

    def stop(self) -> None:
        """Signal the worker to stop and wait for it to exit.

        Logs per-query readahead stats when SIRIUS_CACHE_PRINT_STATS=1, matching
        Sirius readahead_scan_manager destructor which calls summary() on teardown.
        """
        import os

        self._stop_event.set()
        if self._prefetch_worker is not None:
            self._prefetch_worker.join(timeout=2.0)
            self._prefetch_worker = None
        if os.environ.get("SIRIUS_CACHE_PRINT_STATS", "0") == "1":
            print(f"[readahead] {self._counters.summary()}", flush=True)
            print(f"[readahead] timing {self._counters.timing_summary()}", flush=True)

    @property
    def counters(self) -> ReadaheadCounters:
        """Expose counters for external inspection."""
        return self._counters

    def _all_drained(self) -> bool:
        """Return True when every operator queue is empty and closed.

        In streaming mode, also waits for registration to complete so the worker
        does not exit prematurely while tasks are still being added.
        """
        if not self._registration_complete.is_set():
            return False
        for op_id, q_idx in self._operator_id_to_queue_index.items():
            if op_id not in self._closed_operators:
                return False
            if self._ordered_work_queues[q_idx]:
                return False
        return True

    def _get_next_prefetching_candidate(self) -> ParquetScanTask | None:
        """Return the next task to prefetch, advancing the cursor when queues drain.

        Equivalent to Sirius prefetch_work_queue::get_next_candidate(). Walks
        _ordered_work_queues starting at _cursor, skipping fallen-behind tasks and
        advancing past drained+closed queues.
        """
        n = len(self._ordered_work_queues)
        if n == 0:
            return None
        operators_checked = 0
        while operators_checked < n:
            idx = self._cursor % n
            op_id = self._queue_index_to_operator_id.get(idx)
            q = self._ordered_work_queues[idx]
            while q:
                task = q[0]
                if task.has_fallen_behind():
                    # Dropped at queue head: executor already reached this task.
                    # Equivalent to Sirius dropped_fell_behind counter.
                    self._counters.dropped_fell_behind += 1
                    q.popleft()
                    continue
                if task._sirius_entry() is None:
                    # No datasource assigned: analog of Sirius's expired weak_ptr.
                    self._counters.dropped_expired += 1
                    q.popleft()
                    continue
                q.popleft()
                return task
            if op_id in self._closed_operators:
                if idx not in self._drained_queue_indices:
                    self._drained_queue_indices.add(idx)
                    self._counters.operators_drained += 1
                self._cursor += 1
            operators_checked += 1
        return None

    def _prepare_and_issue(self, task: ParquetScanTask, c: ReadaheadCounters) -> None:
        """Run prepare_prefetch() + prefetch_async() for one task.

        Called from a ThreadPoolExecutor thread so multiple tasks can be prepared
        concurrently, eliminating the serial prepare latency that stalls the first
        GET by 120-370ms per query when all footers are already cached.
        The gatekeeper slot was acquired by the main worker before this is called
        and is released in _on_done (or immediately on non-ready outcomes).
        """
        evict_on_failure = False
        while True:
            prep = task.prepare_for_prefetching(evict_on_failure)
            if prep.ready():
                _task_ref = task
                _lock = self._callback_lock
                _gk = self._gatekeeper
                _now = time.perf_counter()

                def _on_done(
                    _ok: bool,
                    _t=_task_ref,
                    _c=c,
                    _lk=_lock,
                    _gk=_gk,
                    _issue_t=_now,
                ) -> None:
                    _done_t = time.perf_counter()
                    _gk.release()
                    with _lk:
                        if _t.has_fallen_behind():
                            _c.wait_for_prefetch += 1
                            _c.wait_for_prefetch_total_ms += (_done_t - _issue_t) * 1000.0
                            if _ok:
                                _c.wait_for_prefetch_ok += 1
                            else:
                                _c.wait_for_prefetch_refused += 1
                        else:
                            _c.prefetched += 1
                            if _ok:
                                _c.prefetched_ok += 1
                            else:
                                _c.prefetched_refused += 1

                _now_ms = (_now - self._start_t) * 1000.0
                with self._timing_lock:
                    if self._first_prefetch_t is None:
                        c.start_to_first_prefetch_ms = _now_ms
                        self._first_prefetch_t = _now
                    else:
                        gap = (_now - self._last_prefetch_t) * 1000.0  # type: ignore[operator]
                        if gap > c.max_inter_prefetch_gap_ms:
                            c.max_inter_prefetch_gap_ms = gap
                    c.total_prefetch_window_ms = _now_ms
                    self._last_prefetch_t = _now

                task.prefetch(on_done=_on_done)
                return

            if task.has_fallen_behind() or prep.fell_behind > 0:
                self._gatekeeper.release()
                c.skipped_fell_behind += 1
                return
            if prep.failed == 0:
                # nothing_to_prepare (code 2): no byte ranges, already cached, or disabled.
                self._gatekeeper.release()
                c.nothing_to_issue += 1
                return
            # allocation_failed — pool full, retry with eviction.
            c.memory_retries += 1
            evict_on_failure = True
            _retry_t0 = time.perf_counter()
            time.sleep(0.025)
            c.prepare_retry_ms += (time.perf_counter() - _retry_t0) * 1000.0

    def _worker_loop(self) -> None:
        """Background worker: the core prefetch pump.

        Matches Sirius readahead_scan_manager::worker_loop().
        Tracks readahead_counters for per-query diagnostics.

        prepare_prefetch() calls are offloaded to a ThreadPoolExecutor so multiple
        tasks can be prepared concurrently. The main loop only does fast work:
        acquire gatekeeper slot, pick next candidate, submit to pool.
        """
        c = self._counters  # local alias to avoid repeated attribute lookups
        n_prepare_threads = min(self._budget, 16)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=n_prepare_threads, thread_name_prefix="readahead-prepare"
        ) as pool:
            while not self._stop_event.is_set():
                _gate_t0 = time.perf_counter()
                acquired = self._gatekeeper.acquire(timeout=0.1)
                c.gatekeeper_blocked_ms += (time.perf_counter() - _gate_t0) * 1000.0
                if not acquired:
                    c.gate_timeouts += 1
                    if self._all_drained():
                        break
                    continue

                task = self._get_next_prefetching_candidate()
                if task is None:
                    self._gatekeeper.release()
                    c.idle_polls += 1
                    if self._all_drained():
                        break
                    # In streaming mode, wake immediately when a new task arrives;
                    # fall back to a short sleep so the worker doesn't busy-spin.
                    self._task_ready.wait(timeout=0.01)
                    self._task_ready.clear()
                    continue

                if task.has_fallen_behind():
                    # Fell behind AFTER being dequeued as a candidate.
                    # Sirius: skipped_fell_behind (different from dropped_fell_behind
                    # which is caught at queue head in _get_next_prefetching_candidate).
                    self._gatekeeper.release()
                    c.skipped_fell_behind += 1
                    continue

                c.candidates_taken += 1
                pool.submit(self._prepare_and_issue, task, c)
            # pool.__exit__ waits for all in-flight prepare threads to finish.
