# Design: integrating real cuCascade into cudf-polars

## Why

This session built a from-scratch, engine-scoped byte-range prefetch cache
(`cudf_polars/streaming/prefetch_cache.py`), explicitly modeled on
cuCascade's real architecture, on the theory that reimplementing the shape
in Python would get most of the benefit without the dependency. It didn't.
Across many real SF1000 runs we found and fixed a real sequence of bugs
(byte ranges reading past EOF, no support for remote/S3 paths at all, no
eviction so pinned memory grew unbounded until OOM, no backpressure between
registration and fetching so the fetch queue grew unbounded, evicted chunks
still getting wastefully fetched) and once the pipeline was finally stable,
the result was: real, measured reuse demand (`re_registrations` climbing to
94% of registrations by the end of a 22-query run) that still produced a
near-zero hit rate, because the actual reused working set (~3,798 unique 64
MiB chunks, ~243 GiB) is far larger than any pinned pool size we could
reasonably dedicate to it without hurting the rest of the query engine.
Growing the pool from 32 GiB to 56 GiB made hit rate slightly better and
wall-clock time *worse*.

None of that is a Python-vs-C++ problem. It's evidence that a from-scratch
reimplementation, however faithful to cuCascade's design on paper, doesn't
get you cuCascade's actual production-tuned behavior: real threading,
real eviction scoring, a real memory-pressure-aware pool, and years of
tuning against exactly this kind of workload. Sirius doesn't reimplement
cuCascade's cache, it links against it. This document is about doing the
same thing here.

## What's actually available

cuCascade has real Python bindings, in an open (not yet merged) PR:
[NVIDIA/cuCascade#169](https://github.com/NVIDIA/cuCascade/pull/169). It
adds a `cucascade` Python package (`python/cucascade/`), with build and
install instructions in `python/README.md` in that PR. The API surface we
care about:

```python
class CuCascadeDatasource(pylibcudf.io.datasource.Datasource):
    def fadvise(self, ranges: list[tuple[int, int]], dev_id: int = -1) -> None: ...
    def duplicate(self) -> CuCascadeDatasource: ...
    def read_ranges_async(self, ranges, buffer) -> list[ReadFuture]: ...
    def read_all_ranges_async(self, ranges, buffer) -> ReadFuture: ...

class RestEngine:  # S3/HTTP, via libcurl
    def __init__(self, access_key_id="", secret_access_key="", session_token="", region="us-east-1",
                 endpoint="", n_reactors=4, tls_verify=True, pool_capacity=..., block_size=...,
                 max_connections=16, chunk_size=..., max_n_chunks=16, enable_cache=False): ...
    def open(self, path: str) -> CuCascadeDatasource: ...
```

We only care about `RestEngine` (remote I/O). Local files already go
through kvikio/`CuFile` fine; cuCascade is specifically for the S3/HTTP
path.

The important thing, verified directly against this repo's `main`, not
assumed: **`CuCascadeDatasource` subclasses `pylibcudf.io.datasource.Datasource`,
which already exists on `main` today**, and `pylibcudf.io.types.SourceInfo`
**already accepts a `Sequence[Datasource]` today**
(`python/pylibcudf/pylibcudf/io/types.pyx:520,572-576`). This is a real,
already-wired drop-in point, not something that needs new pylibcudf work:
anywhere cudf-polars currently builds
`plc.io.SourceInfo([FilepathSource(path, size)])` for a remote path, it can
instead build `plc.io.SourceInfo([engine.open(path)])` and every downstream
API (`HybridScanReader`, `fetch_byte_ranges_to_device`, `materialize_*`)
keeps working unchanged, because they already operate on `SourceInfo`/
`Datasource` generically.

`fadvise` is a plain, synchronous, `nogil` C++ call
(`with nogil: deref(self._ds).fadvise(c_ranges, dev_id)`), not async and
not backed by anything Python-event-loop-shaped. It's safe to call
directly from `do_evaluate`'s worker thread — no `asyncio` bridging, no
dedicated background thread of our own, no event-loop-lifetime questions
at all. That whole category of problem this session spent real effort on
(query-scoped vs engine-scoped event loops, cross-thread `asyncio.Queue`
safety, `run_coroutine_threadsafe` bridging for `get`/`fadvise`) doesn't
exist here, because cuCascade's own C++ threads do the actual work; Python
just calls into them.

`RestEngine` follows exactly the "self-contained engine, constructed once,
`open()` per file" shape this whole investigation kept converging on
independently: it owns its own pinned host staging pool (`pool_capacity`),
its own `ioctx`, and its own reactor thread pool (`n_reactors`), matching
cuCascade's real architecture (persistent OS threads, not per-query async
tasks) exactly, because it *is* cuCascade, not a reimplementation of it.

We want the prefetching cache enabled: `enable_cache=True`. That's the
whole point of doing this — real eviction, real tiered freshness scoring,
years of tuning against workloads like this one, none of which our
from-scratch version had.

## What's useful vs not, from `NVIDIA/cudf#23450`

[#23450](https://github.com/NVIDIA/cudf/pull/23450), "EXPERIMENT 2:
Prefetching parquet byte ranges into pinned host memory and use cucascade
to fill those buffers" (closed, not merged, 83 commits), is the prior
attempt at this integration. Read through in full. Almost none of it
applies to where we've ended up:

- `cudf::io::datasource::fadvise(ranges, dev_id)`
  (`cpp/include/cudf/io/datasource.hpp`) is the one real, still-needed
  piece: a new virtual method on cudf's own `datasource` base class,
  no-op by default. This is the C++-level hook `CuCascadeDatasource::fadvise`
  actually overrides. Needed on the cudf side regardless of how we wire
  the Python layer.
- `hybrid_scan_metadata` — already merged into `main`, not something this
  PR is still carrying. Nothing to do here.
- The `gpumemoryview.__getitem__` slicing this PR added was never
  adopted; `main` already has `gpumemoryview.byte_slice` for the same
  zero-copy-subview need, so nothing to carry forward there either.
- `fetch_byte_ranges_to_device` (sync, blocking) — already exists on
  `main` today (`plc.io.parquet_io_utils.fetch_byte_ranges_to_device`),
  not something gained from this PR.
- `python/cudf_polars/cudf_polars/streaming/prefetch.py` (605 lines, new
  file), and most of the diffs to `streaming/actor_graph/io.py`,
  `streaming/io.py`, and `utils/config.py` — a from-scratch Python
  prefetch pipeline (the direct ancestor of the `PrefetchCache` module we
  just spent this session building and un-building) and its supporting
  orchestration. Superseded by linking against real cuCascade instead of
  reimplementing it a second time.

So from this PR, essentially just the `datasource::fadvise` virtual-method
hook on the cudf C++ side carries forward.

## How sirius actually uses this (recap, already verified against
`~/sirius` source earlier this session)

- Row-group pruning happens first, against parquet footer statistics, in
  `duckdb_native_metadata.cpp`'s `mark_row_groups_pruned_by_filter_stats` —
  before any bytes are touched.
- Only the byte ranges for surviving row groups get computed
  (`column_chunk_ranges`, using the same `HybridScanReader.all_column_chunks_byte_ranges`
  API cudf-polars already uses) and handed to `fadvise`.
- Each split gets its own `datasource` via `duplicate()`
  (`parquet_gpu_ingestible.cpp`'s `seal_file`), specifically so one
  split's `fadvise` can't stomp another's prefetch handle on a shared
  datasource. This holds even when several small files are fused into one
  task (see below): fusion is about task granularity, each file inside a
  fused task still keeps its own datasource/`fadvise` identity.
- `fadvise` is called *before* a deliberate publication barrier
  (`sirius_gpu_scan_operator_data.cpp`): a split isn't registered as
  visible to the readahead worker (or to consumption) until its `fadvise`
  calls have already landed. That ordering guarantee is what gives a
  split's own read genuine lead time, rather than racing its own prep
  work — the exact race we found and diagnosed the hard way in the
  from-scratch version this session, where `scan_node` launched a split's
  prep task and its own consumption at effectively the same moment with
  nothing enforcing an order between them.
- Sirius also fuses small files together, the same idea as our own
  `FusedScan`: `parquet_batch_coalescer::push()` accumulates row groups
  across as many files as fit under a byte-budget/row-count cap into one
  emitted split, only sealing early on a partition-value or
  pushdown-decision mismatch.
- Byte-range computation and the `fadvise` call for a given split are
  **not** concurrent with each other — they're one synchronous, atomic
  prep step on a single thread: `emit_current()` computes
  `slice_column_chunk_ranges`/`parquet_fadvise_entries`, and the `emit()`
  that follows immediately constructs `scan_operator_input`, whose
  constructor calls `fadvise` right there
  (`sirius_gpu_scan_operator_data.cpp:140-142`). The concurrency sirius
  actually relies on is coarser-grained than that: **one sequencer task
  per pipeline** (`load_balancing_scan_batch_coalescer`'s
  `spawn_workers`/`slot_loop`), each independently draining its own
  split queue, so different pipelines' prep-and-fadvise streams run in
  parallel with each other, but splits *within* one pipeline are prepped
  one at a time, sequentially. Row-group pruning/metadata scanning (which
  feeds these per-pipeline queues) deliberately runs on a separate
  dispatcher from the sequencer tasks, specifically so the sequencer's
  blocking waits can't starve the producers and deadlock.

## FusedScan → `hybrid_scan_multifile`

cudf-polars only routes to hybrid scan for `SplitScan` (one physical file)
today; `FusedScan` (multiple small files coalesced into one task) always
falls back to a plain multi-file `Scan.do_evaluate` read. Sirius doesn't
have this limitation — it uses `hybrid_scan_multifile`, a real C++ class,
for exactly this case (`materialize_bulk()` in `parquet_materialize.cpp`
branches on source count: 1 source → `hybrid_scan_reader`, >1 source →
`hybrid_scan_multifile`). To actually mirror sirius, and to give
`fadvise`/prefetch a payoff on fused splits too (not just split-scans),
`FusedScan` needs the same treatment. The pieces already exist upstream,
in various states:

- **[NVIDIA/cudf#23953](https://github.com/NVIDIA/cudf/pull/23953)**
  ("introduce `ParquetScanTask`"), open and mergeable, cudf-polars-only
  (`streaming/io.py`, `dsl/utils/io.py`, `streaming/actor_graph/io.py`).
  It unifies `SplitScan` and `FusedScan` into one wrapper,
  `ParquetScanTask`, with one shared bounds calculation
  (`ParquetTaskBounds`, `row_groups: list[list[int]] | None` — already
  per-source, i.e. shaped for multi-file). This is purely Python
  orchestration, not new hybrid-scan capability: `ParquetScanTask.do_evaluate`
  still only dispatches to hybrid scan when `len(paths) == 1`, so fused
  tasks still fall through to the plain multi-file path. But it's exactly
  the seam where multi-file hybrid scan would plug in — the
  `len(paths) == 1` gate in that one `do_evaluate` method is the thing to
  relax, and `bounds.row_groups` is already the right per-source shape to
  hand to a multi-file reader without further plumbing.
- **[NVIDIA/cudf#22795](https://github.com/NVIDIA/cudf/pull/22795)**
  ("Add python bindings for multi-file hybrid scan reader APIs"), still a
  draft with merge conflicts (depends on #22793, which *is* merged — the
  C++ single-step materializers). Adds `pylibcudf.io.experimental.HybridScanMultifile`,
  a straight Python mirror of sirius's C++ `hybrid_scan_multifile`:
  construct from a list of per-source footer bytes (or pre-parsed
  `FileMetaData`), then `all_row_groups`/`filter_row_groups_with_stats`/
  `all_column_chunks_byte_ranges` (for `fadvise`, same role
  `column_chunk_ranges` plays in sirius) and `materialize_all_columns`
  (decode), all vectorized over sources with a parallel `source_indices`
  list telling you which byte range/row group belongs to which file.
  This is the actual multi-file hybrid-scan capability #23953 doesn't
  add.

So the shape is: #23953 gives `FusedScan` a place to plug in (unified
task/bounds), #22795 gives it something to plug into (the actual
multi-file reader binding). No structural mismatch between them —
`ParquetTaskBounds.row_groups` is already `list[list[int]]`, per source,
matching `HybridScanMultifile`'s expected input directly. Concretely, once
both land: extend `ParquetScanTask.do_evaluate`'s hybrid-scan dispatch
condition to also accept `len(paths) > 1`, and construct a
`HybridScanMultifile` instead of `HybridScanReader` in that branch — the
rest of the bounds/predicate-pushdown logic already generalizes.

**Both single-file and multi-file hybrid scan should dispatch from the
same `ParquetScanTask.do_evaluate`**, not two separate task types/dispatch
paths. This mirrors sirius exactly: `materialize_bulk()` is one entry
point that branches internally on source count (1 source →
`hybrid_scan_reader`, >1 → `hybrid_scan_multifile`), not two separate
operators. There's no structural reason to split it in cudf-polars either
— `ParquetTaskBounds.row_groups` is already shaped generically for both
cases, which is precisely what #23953 unified `SplitScan`/`FusedScan`
bounds computation for. Keeping one dispatch point also matters for
`fadvise`: the compute-ranges-then-`fadvise`-then-publish ordering needs
the same shape regardless of file count, and a single `do_evaluate`
keeps that ordering logic in one place instead of needing to stay in sync
across two paths.

## What we'd pull together into one working branch

To actually run benchmarks with real cuCascade against real multi-file
hybrid scan, a working branch needs, roughly in dependency order:

1. cuCascade's real Python bindings —
   [NVIDIA/cuCascade#169](https://github.com/NVIDIA/cuCascade/pull/169)
   (open). `RestEngine`, `CuCascadeDatasource`, `.fadvise()`, `.duplicate()`.
2. The `datasource::fadvise()` virtual-method hook on cudf's C++
   `datasource` base class, cherry-picked out of
   [NVIDIA/cudf#23450](https://github.com/NVIDIA/cudf/pull/23450) (closed)
   — the only still-relevant piece of that PR (see above).
3. [NVIDIA/cudf#22795](https://github.com/NVIDIA/cudf/pull/22795) (draft,
   has conflicts) — the `HybridScanMultifile` pylibcudf bindings. Will
   need conflicts resolved against current `main` before it applies.
4. [NVIDIA/cudf#23953](https://github.com/NVIDIA/cudf/pull/23953) (open,
   mergeable) — `ParquetScanTask` unification, as the integration point
   for both `fadvise` timing and multi-file hybrid scan dispatch.
5. The actual integration work this design doc describes: `RestEngine`
   construction per `SPMDEngine`, `SourceInfo`/`Datasource` wiring,
   `.duplicate()`/`fadvise()` calls in the split/fused prep step, and
   extending `ParquetScanTask.do_evaluate` to dispatch fused tasks to
   `HybridScanMultifile`.

#22793 (single-step materializers for `hybrid_scan_multifile` on the C++
side) is already merged into `main`, so nothing to pull there.

## Proposed shape for cudf-polars

- Construct one `cucascade.RestEngine` once per `SPMDEngine`, at
  construction time — matching how `Context`/`BufferResource` are already
  constructed once per engine, and matching sirius's own "engine owns the
  cache" framing. `SPMDEngine`-only, per the same scoping this session's
  `PrefetchCache` used. `enable_cache=True`.
- Credentials: read the standard AWS environment variables
  (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`,
  region) directly and pass them into `RestEngine`'s constructor. No new
  credential-resolution machinery needed.
- When building `SourceInfo` for a parquet read on a remote path, resolve
  the datasource through the engine (`engine.open(path)`) instead of a
  bare `FilepathSource(path, size)`. Local paths are unaffected, they
  don't go through `RestEngine` at all.
- Per split, call `.duplicate()` on the file's datasource before calling
  `.fadvise()`, matching sirius's `seal_file` pattern exactly, for the
  same stated reason (independent prefetch handles per split).
- Reuse the *shape* of `scan_node`'s eager-launch pattern already built
  this session (launch a lightweight, thread-pool-offloaded pruning task
  per split, ahead of that split's own `read_chunk`) to compute a split's
  row groups/byte ranges and call `fadvise` on them. Match sirius's actual
  concurrency granularity: within one split, range computation and
  `fadvise` should be a single synchronous prep step, not two
  independently scheduled tasks; the parallelism we want is across
  splits/pipelines (already how the thread pool dispatches split prep
  tasks today), not fanning individual `fadvise` calls out on their own.
  The call itself is a plain synchronous C++ call, not something that
  needs a dedicated background asyncio loop, queues, or an eviction/spill
  mechanism written in Python. All of that lives in cuCascade already.
- Delete `cudf_polars/streaming/prefetch_cache.py` and its engine wiring
  (`SPMDEngine.__init__`/`_reset`'s `PrefetchCache` construction,
  `IRExecutionContext.prefetch_cache`, `SPMDContext.prefetch_cache`, the
  `prefetch_cache_*`/`prefetch_byte_ranges` config surface) once this
  lands.

## API design: datasource resolution

Grounding in the current code, not assumed: today there are three real call
sites that build `FilepathSource(path, size)`, none of which distinguish
local vs. remote paths (`dsl/utils/io.py:87` and `:134-139` for
metadata/footer-prefetch, `streaming/io.py:244-246` for the actual hot
per-split hybrid-scan read). The only existing remote/local signal
anywhere in this code is `plc.io.SourceInfo._is_remote_uri(path)`, already
used in `dsl/utils/io.py` to decide whether to fetch file size via
`kvikio.RemoteFile`. There's no `storage_options`/`fsspec`/credential
handling anywhere in this part of the codebase to build on or route
around — `RestEngine`'s env-var credentials are genuinely the whole story.

The closest thing to a chokepoint that already exists is `CachedParquetInfo`
(`dsl/utils/io.py:26`), which owns a file's `.path`/`.size` and is already
threaded through both metadata prefetch and the hybrid-scan read path. It
doesn't currently own datasource *resolution* — each call site
independently wraps `.path`/`.size` in a fresh `FilepathSource`. That's the
seam to use:

- `SPMDEngine.__init__` (`engine/spmd.py:414`) constructs
  `self._cucascade_engine: cucascade.RestEngine | None`, once, alongside
  `self._ctx`/`self._py_executor`/`self._kvikio_monitor` — same lifetime
  class (survives `_reset`, torn down only at engine teardown). Read AWS
  env vars there; construct `None` if a remote path is never seen (or
  eagerly if the plan is known to touch S3 — TBD, see open questions).
- Add a resolution method on `CachedParquetInfo`, something like
  `CachedParquetInfo.datasource(engine: SPMDEngine) -> Datasource`, that
  does the `_is_remote_uri` check once and either returns
  `engine._cucascade_engine.open(self.path)` or
  `FilepathSource(self.path, self.size)`, and caches the result on
  `self` (one canonical open handle per file, not reopened per call).
  All three existing call sites switch to calling this instead of
  constructing `FilepathSource` directly.
- Per-split/per-fused-task `fadvise` calls go through
  `cached_info.datasource(engine).duplicate()` — never `fadvise` on the
  canonical handle itself, matching sirius's `seal_file` pattern exactly
  (one independent prefetch handle per split, so splits can't stomp each
  other's readahead state on a shared datasource). For local files,
  `FilepathSource` doesn't have `.duplicate()`/`.fadvise()` at all, so
  this only actually does anything on the remote path — local reads are
  unaffected either way.

This keeps the change small and localized: three `FilepathSource(...)`
call sites become one shared resolution method, `SPMDEngine` gains one
more engine-scoped resource in the same place its siblings already live,
and nothing about `HybridScanReader`/`HybridScanMultifile`/
`fetch_byte_ranges_to_device` needs to change at all, since they already
operate on `SourceInfo`/`Datasource` generically.

## `fadvise` readahead: avoiding lockstep with materialize

The producer/split-stream mapping above (one producer ≈ one sirius
pipeline) answers *where* `fadvise` ordering is scoped, but not *when*
it should fire relative to the actual read. Putting the
compute-ranges-then-`fadvise` step inside `read_chunk`, immediately
before that same call also materializes the split, would give cuCascade
essentially no lead time — prep and consumption happen back to back in
one call, which is the same race this session already diagnosed and fixed
once in the from-scratch prefetch cache (`scan_node` launching a split's
prep and its own consumption at effectively the same moment).

Sirius avoids this with a specific, checked mechanism, not just "call
`fadvise` earlier":

- The queue between a pipeline's prep/`fadvise` step and the downstream
  operator that actually decodes is `split_connector`, a lock-protected
  `std::deque`. The producer side (`push_split`, called from `emit()`)
  enqueues unconditionally — it never waits on the consumer. Multiple
  splits sit fadvised-but-not-yet-decoded in the deque at once.
- `fadvise` fires at split construction/publication time
  (`scan_operator_input`'s constructor, called inline from `emit()`), not
  at decode time. Actual GPU materialize/decode is a structurally later
  call, reached only when the downstream operator pulls via
  `get_next_split()` on its own schedule.
- The producer isn't unbounded-greedy either: it's gated by an explicit
  concurrency budget, `readahead_scan_manager`'s `_budget`/`_gatekeeper`
  semaphore, sized from `ioctx::n_max_concurrent_scans` — a named,
  tunable "how far ahead" control, not incidental slack from however many
  pipelines happen to be running.

Our current producer loop has no equivalent: `_producer` calls
`read_chunk` for one split, awaits it fully, then moves to the next —
sequential by construction, with no readahead within a single producer's
own split stream. Cross-producer concurrency (`num_producers`) gives some
incidental overlap, but it's not a deliberate lookahead mechanism — two
producers' splits happening to be at different pipeline stages at the
same moment is not the same guarantee as sirius's explicit budget.

**Proposed shape**: split `read_chunk`'s current single call into two
steps, matching the atomic prep-step principle already established
(range computation and `fadvise` stay one atomic unit — this only
changes *when* that unit runs relative to materialize, not how it's
composed internally):

- `_prepare_split(split) -> PreparedSplit`: compute row groups/byte
  ranges (via `HybridScanReader`/`HybridScanMultifile`) and call
  `.duplicate().fadvise(ranges)` on them, in one atomic step, offloaded
  to the thread pool.
- `_materialize_split(prepared: PreparedSplit) -> DataFrame`: the actual
  decode/read — today's `read_chunk` body, minus the prep work it now
  receives pre-computed.

Each producer maintains a small sliding window of in-flight
`_prepare_split` futures (bounded by a readahead-depth budget, the direct
analog of sirius's `n_max_concurrent_scans`) ahead of the split it's
currently materializing: as soon as one split starts materializing, prep
kicks off for the next one(s) already queued behind it, so `fadvise` for
split N+1 fires while split N is still being decoded, not after. The
budget should be a genuine knob (not just `num_producers`, which is about
scan-node-wide concurrency, not per-producer lookahead depth) — sizing it
too large risks the same pinned-pool-pressure problem the from-scratch
cache hit; sizing it at 1-2 splits ahead is probably enough to give
cuCascade real wall-clock lead time without meaningfully growing how much
data is in flight at once.

This readahead depth should be user-configurable, not hardcoded, the same
way `max_concurrent_io_tasks` already is (`resolve_max_concurrent_io_tasks`,
`streaming/io.py`) — e.g. a `fadvise_readahead_depth` (name TBD) config
option threaded through `GPUEngine`/`ParquetOptions` alongside it, so it
can be tuned per workload/hardware without a code change, same as the
existing IO-concurrency knobs.

## Open questions

- **Eager vs. lazy `RestEngine` construction.** `RestEngine.__init__`
  allocates a real pinned pool (`pool_capacity`, default ~2.5 GiB) and
  spins up `n_reactors` OS threads, so constructing one unconditionally
  in `SPMDEngine.__init__` costs real resources even for all-local-file
  workloads. Options: construct it unconditionally (simplest, matches how
  `_ctx`/`_py_executor` are already built regardless of workload shape),
  or construct it lazily on first `_is_remote_uri(path) == True` sighting
  (saves the cost for local-only workloads, but means the first remote
  path in a query pays construction latency inline instead of it being
  paid once at engine startup). Leaning toward unconditional for
  simplicity unless the local-only-workload cost turns out to matter in
  practice.
- **Two separate pinned pools.** `RestEngine` owns its *own* pinned host
  memory pool (`pool_capacity`), completely separate from rapidsmpf's
  `BufferResource` pinned pool (`RAPIDSMPF_PINNED_INITIAL_POOL_SIZE`), and
  the two aren't coordinated with each other right now. This is fine for
  now — just size each pool deliberately at SF1000 scale rather than
  assuming they won't compete. The longer-term fix isn't coordination
  logic on our side: rapidsmpf is expected to eventually move onto
  cuCascade's own memory reservation system, which would collapse this
  into one real accounting scheme rather than two independent pools
  guessing at each other's headroom.
- **Does real cuCascade's cache actually do better on this workload.**
  We measured the genuinely-reused working set at ~243 GiB for the
  SF1000/22-query benchmark, far larger than any pool size that made
  sense to dedicate to our from-scratch cache. Real eviction tuning and
  tiered freshness scoring might change that picture, or the
  working-set-vs-practical-pool-size mismatch might apply regardless of
  implementation quality. Worth testing empirically rather than assuming
  either way.
