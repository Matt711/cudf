# cudf-polars remote parquet I/O, rebuilt (io2): design, 10 Nov 2026

Goal: TPC-H SF1000 from S3, cold, under 100 s on g7e.8xlarge. Cold = every query's bytes come from S3 (no reuse of
cached bytes across queries); staging fetched bytes in pinned host memory is allowed.

Bases (local only, nothing pushed):
- cuDF `worktrees/cudf-io2`, branch `task/io2-base`: upstream main e2bc4a5020 + #24853 + #24854 + #25034
  (conflicts resolved: #24854 vs main's Iceberg `lake_options`, keeping lake scans off hybrid scan; one import vs #25034).
- cuCascade `worktrees/cucascade-io2`, branch `task/io2-base`: main 7f0e6cf + #229 + #231. #233 (request stats) left out:
  it sits on the pre-#231 layout and conflicts in 5 files; port its counters later if needed.
- Env `envs/io2`: nightlies of 10 Nov (rmm a38, kvikio a44, rapidsmpf a85), CUDA 13.2 (driver max), libcudf built from cudf-io2.

## What Sirius does (verified in source, Sirius b6b12730 / cuCascade 5489112)

Metadata (copy this):
- One metadata job per file, at execution time, on the scan-manager pool (ncpu-10 threads): `parquet_scan_core::next_split_provider`
  claims a file index, `build_file_scan_info` opens it with `open_hint::parquet_footer_probe` (src/op/scan/parquet_scan_core.cpp:1166-1256).
- The probe is ONE `Range: bytes=-512KiB` GET: size from Content-Range total, strong ETag, the last 512 KiB stashed; cuDF's
  trailer + footer reads are served from the stash (rest_reactor.cpp:838-923, rest_ioctx.cpp:374-404). 200/416 -> HEAD fallback.
  Footer bigger than the window -> one more ranged GET (no adaptive probe).
- Parsed with `hybrid_scan_reader` on canonical options, kept in the ioctx `metadata_store` keyed by path + ETag, single-flight.
- Not blocking: decode waits only for the first splits; later footers overlap data I/O. Claims round-robin across all scans of
  the query and are paced per scan by a byte window (`lookahead_batches(2) x batch bytes`, preparation_coordinator.cpp).
- No page index, no bloom filters on the S3 path; row-group pruning from footer min/max + null counts.

Data/readahead (learn from, then go further):
- REST transport: N reactor threads (epoll + curl multi), 64 handles each, no pipelining; GETs 4-16 MiB; per-slice, no fusing of
  separate slices; merge gap 512 KiB only in the cache planner. Scale with reactors, not sockets.
- Readahead needs cuCascade's cache (`cache.mode: cucs`; off by default). Eager, plan-order, FIFO per operator, bounded by a COUNT of
  tickets (max(8, 2 x pipeline threads)), bytes bounded only by the pinned pool. Prefetch lands in pinned chunks; H2D at decode.
- Sirius measured here: 110-129 s cold with cache + readahead. So Sirius is the template for metadata and transport, not a proof
  that its readahead reaches line rate.

## io2 architecture

One process-wide native I/O service (C++, in the cuDF-side cuCascade extension `python/cudf_polars/cucascade` from #25034), used
from cudf-polars through a thin Cython layer. No KvikIO anywhere in cudf-polars.

1. Transport: a cuCascade `rest_ioctx` created at engine start: R reactors x C connections (start 8 x 32, tune), header signing,
   endpoint from config (https default; http optional), `warmup()` per bucket at engine start. Raw `host_readv_async_io` only
   (no fs_cache): S3 bytes land directly in our pinned pages, one memcpy per byte, per-call semi_future completion.
2. Metadata, Sirius-style: a metadata job per file on a small C++ pool, started for every scan of the query when the actor network
   starts (no stage before it). Footer-probe open (1 GET), parse to FileMetaData, single-flight store keyed by path + ETag.
   Cross-query reuse of parsed footers: OFF by default (strict cold: refetched every query); switch to Sirius behaviour (keep) is
   one option. Hidden footer readers in cudf-polars (decimal predicate types, fast count, row counts) are rerouted to this store.
3. Byte ranges: per task, after footer-stats row-group pruning with the scan predicate, hybrid scan `all_column_chunks_byte_ranges`
   (projected columns only), then coalesce neighbours with gaps <= 512 KiB into fewer, larger GETs.
4. Pinned page ring: one arena pinned up front on the GPU's NUMA node (capacity configurable, e.g. 32-64 GiB), cut into fixed pages
   (e.g. 4 MiB). A task gets a list of pages; the reactor scatters into them (fragmented slices). Pages are handed out in each
   scan's task order and freed in any order (no fragmentation). Byte backpressure = free pages.
5. Fetch scheduler (the readahead), byte-bounded, query-wide: tasks of all scans in a queue per scan (task order); a scan with
   nothing staged is always granted its next task's pages (so a consumer waiting on scan B is never starved by scan A's staged
   bytes); beyond that, scans share the ring round-robin up to an equal share. Every grant immediately issues the GETs: the NIC is
   kept full from the first footers until the ring is full or the query's bytes are fetched.
6. Delivery: per scan, completions go to a C++ completion queue; one waiter per scan (nogil) turns them into asyncio results; the
   scan's producers receive (task index, staged pages) as rapidsmpf `ArbitraryChunk` messages through channels.
7. Decode (scan_node producers): device reservation sized from staged bytes; budget held until the Lineariser forwards the chunk
   (lesson from the native stager: otherwise decoded chunks pile up and spill); task-order admission; batched H2D of the task's
   pages into one device buffer (cudaMemcpyBatchAsync); `materialize_all_columns` (single pass, predicate applied); free pages.
   Lineariser, sequence numbers and scan_node's output contract unchanged.
8. Cold vs warm: the ring frees pages as soon as decode has copied them; nothing survives a query. A cache tier can later sit
   behind the same scheduler (fetch from cache instead of S3) for warm/hot.

## Delete (cudf-polars)
- Blocking metadata stage: `prefetch_parquet_file_metadata_for_ir`, `attach_cached_parquet_metadata`, `_prefetch_parquet_footers_for_paths`,
  the call in `engine/core.py` before `execute_ir_on_rank`, `ParquetOptions.prefetch_file_metadata`, `Scan.cached_parquet_info` plumbing,
  per-task lazy footer reads (`_fetch_parquet_info_for_hybrid_scan`, `_split_task_bounds_from_row_group_metadata`), remote footer sampling.
- All KvikIO: config/options/env (`kvikio_*`), `configure_kvikio`, monitors and IO summaries, benchmark env settings, the HEAD in dsl/utils/io.py.
- #25034's read-through datasource path (`wrap_sources`, `register_layouts`) and #24854's per-task `fetch_byte_ranges_to_device`
  I/O (its pruning + materialize code is kept).

## Order of work
A. Native service standalone: ioctx + footer jobs + page ring + scheduler + completion queue + Cython. Test: SF1000 lineitem/orders
   footers and data at line rate from one process; GB/s and CPU vs the KvikIO stager (9.7-10.3 GB/s, 71-75% CPU).
B. cudf-polars: fetch actor, decode path in scan_node, deletions. Correctness: TPC-H validation at SF1000.
C. Full cold runs vs control, tune (reactors/connections, ring size, page size, http/https), then profile the remaining gaps.

## Open decisions (Matt)
- Cross-query reuse of parsed footers (Sirius keeps them; strict cold refetches). Default here: refetch.
- Pinned ring capacity default.
- http vs https to S3.

## Results so far (11 Nov 2026, all local)
- Full cold run, latest main: io2 140-142 s vs same-build control 153 s (all 22 validate). Floor 90.8 s for the 1071 GB.
- Standalone service: 32 reactors x 32 connections, 4 GiB in-flight cap issued in task order: 11.6 GB/s from one process (bytes verified).
- Per-query trace (io9): head 1.6 s total; fetch span 107.9 s (floor 90.8; ~0.7 s per query of ramp/drain: ~0.35 s per 4 MiB request at
  ~12 MB/s per S3 connection, TCP slow start after idle = 1 on this host); tail after last byte 32 s (Q7 4.4, Q21 4.3, Q9 2.1, Q13 2.1 ...).
- tcp_slow_start_after_idle=1 here; turning it off needs root (Matt's call).

## Partition sizing and compute keeping up with the fetch (design input)
- Today remote scans have no planning statistics (footer sampling is off for remote), so every task is a whole file (~1 GB
  compressed, 5-8 GB decoded for lineitem); target_partition_size has no effect on S3.
- Sirius: splits decided at execution time as footers land; a coalescer packs row groups into splits by estimated decoded
  working bytes (scan_task_batch_size, 512 MiB) across file boundaries. No sampling, no planning-time I/O.
- Experiment (io10/io11): re-enabling footer sampling for remote and a 512 MB target made it much worse (Q7 11.4 -> 18.6 -> 34.9 s;
  5244 tasks). Profile ct23: the Join actor handles one chunk at a time (concurrency ~1.0 for 33 s), 2618 calls x ~12 ms with
  inner_join ~1 ms: ~10 ms per call of fixed overhead outside libcudf; GPU busy 3.7 of 35 s; the stalled consumer back-pressures
  the fetch (NIC 3.7 GB/s). With whole-file chunks (ct22) join calls take 41.8 ms during the fetch vs 16.7 ms alone (contention).
- So: finer runtime splits (Sirius-style, from the footers the scan already holds) are the right shape, but only pay once
  (1) per-chunk fixed overhead in the compute actors is small and (2) an actor can work on several chunks at once (independent
  probe-side chunks, sequence numbers restored downstream), so compute keeps up with ~11 GB/s.
- The chunk count must be fixed before a scan sends ChannelMetadata (local_count feeds union, sort, repartition, prefilter);
  all of a scan's footers arrive in ~0.2 s, so the scan actor can split row groups and send the count once they are in.
- Exact column sizes: the runtime footer is exact for cold; a persisted metadata cache would also make lowering-time estimates
  exact (warm/hot feature).
- py-spy (launched as the benchmark's parent) did not see the actor threads (attributed ~213 s to module import): need another
  way to attribute the ~10 ms per join call (NVTX ranges in the join actor, or nsys Python sampling).

## 11 Nov results (cold, latest main, all valid)
- io9 (4 GiB in-flight cap): 141.8 s. io14 (+ join concurrency 4): 136.8 s. io15 (+ 16 py executors): 135.5 s. Control 153.2 s.
- io15 breakdown: floor 90.8 s, fetch span over floor +9.1 s (fetch at 10-11.7 GB/s), tail after last byte 34.1 s, head 1.6 s.
- Tail split (last 8 tasks of big scans into 4 pieces): worse everywhere (Q7 10.7 -> 13.7). Smaller chunks, even only at the end,
  cost more downstream than they save: per-chunk cost in the actor chain dominates. Unrestricted it also hung Q7 (small scans split
  into empty pieces), not investigated.
- Conclusion: I/O is within ~10% of the floor. The remaining ~34 s is compute after the last byte: per-operator serial actors with
  a fixed per-chunk cost. Sirius avoids it structurally (pipeline tasks per split on a pool of GPU threads; partial aggregation per
  split). Design work: execution-time splitting from footers + operator concurrency governed by memory (or a pipeline task model).

## 11 Nov (later): compute tail investigation
- Best cold so far: io30 129.8 s, io32 131.6 s (native plan/decode), warm 126.9 s. Run-to-run noise ~5 s (io33 137.1 same config).
- io33 tail split: last byte -> last decode 13.1 s (decode backlog; Q21 3.1, Q18 1.6, Q1 1.1), last decode -> end 14.8 s
  (operators + final agg + output; Q7 1.9, Q13 1.9).
- Decode concurrency: 16 producers + 0.6 budget much worse (Q1 6.1 -> 10.9); 4 producers ~same as 8.
- Join call anatomy (Q7 tail): 12.8 ms per call, CUDA API 4.3 ms, 8.5 ms between CUDA calls (GIL waits; the
  biggest gap 3.7 ms in plain Python between the two gathers). Isolated join+gathers at Q7 shapes: 1.6 ms (GPU-bound,
  ~600 calls/s max on this GPU). sys.setswitchinterval(0.0005): Q7 10.1/11.2 -> 8.6/8.7; 0.0001 worse.
- Rejected: rapidsmpf statistics off (~-1 s, noise), blocking CUDA sync (worse), hash-table reuse via PR 24241 (no change),
  tail splitting (worse), sampling + 512 MB tasks (much worse), native planning (no cold change).
- So the remaining ~30 s is GIL-serialized per-chunk Python orchestration across operator actors plus per-call libcudf
  latency, not I/O and not GPU throughput. Structural options: fuse operator chains per chunk (Sirius pipeline/task model),
  or run the hot per-chunk path without the GIL (C++ actors), or free-threaded Python.

## 11 Nov (evening): where the remaining ~40 s goes (io42-io55, ct28)
- Full-run breakdown io47 (134.0 s cold): fetch window 88.9, head (query start -> first task landed) 16.8, scan backlog after
  last byte 14.8, operators after last scan chunk 13.4. Bytes are fixed: 1,040 GB minimum (only Q21 has in-query duplicate
  column reads, -20.7 GB). Network floor 89-91 s, so <100 s allows ~0.5 s overhead per query; today ~1.9 s.
- GPU is not the bottleneck: nsys ct28 (all 22 queries): kernels busy 36.4 s of 138.4 s (26%); max Q13 50%, Q1 44%.
- Rejected today: larger rapidsmpf stream pool (64; no change), plain-HTTP S3 (-30 s CPU but 39 connection resets -> 146 s),
  decode budget 0.7 (Q1 6.5 -> 11.6: device pool sits at the 80% spill limit), 90% spill limit + 0.6 budget (no change),
  in-flight cap 1-2 GB (8.2-9.5 GB/s), smaller GETs 0.5-1 MiB (ramp unchanged, peak lower), unspill and fanout copy on worker
  threads (no change).
- Decode: standalone ScanIO materialize of a Q1 task 0.060 s (0.076 s with a concurrent fetch); 3 threads -> 0.15 s each
  (~19 tasks/s, ~1.5x the landing rate). In-engine 0.14-0.26 s. Estimates: Q1 12.1 GB vs 4.5 GB actual chunk (filter x2).
- First landing ~0.7 s after submit regardless of idle time or GET size: bandwidth is shared fairly by all tasks in flight
  (4 GB needed for full rate, ~0.34 s per request at full load), so tasks land in bursts and the last ones together.
- Actor event loop: a 5 ms sleep wakes 5-25 ms late (p90 30-70 ms, max 300 ms) in busy phases; the loop thread is on CPU only
  20-40% and a watchdog thread sees no GIL contention (0.1-0.6 ms), so callbacks block in C++ calls. Slow callbacks (>5 ms)
  ~24% of loop time; rapidsmpf's py_actor wrapper hides which actor. Q21 spills 650-830 buffers (21-26 GB each way) and its
  post-shuffle fanout deep-copies 58.5 GB on device.
- New trace events (CUDF_POLARS_IO_TRACE): admitted/chunk/forwarded/reserved/mat per task, op:<IR> and op:JoinCall durations;
  CUDF_POLARS_LOOP_LAG=<file> (lag, loop CPU, watchdog stacks), CUDF_POLARS_LOOP_SLOW=<file> (asyncio slow callbacks).

## 11 Nov (night): actor fusion (io56-io70, ct29)
- CUDF_POLARS_FUSE_CHUNKWISE (commit e4125c6a41): chains of single-input chunkwise actors run as one actor. Cold 124.1,
  123.6, 122.7 s (vs 129.8-134.0 before); later control 127.8 s (network drift). Gain mostly in the fetch: compute keeps up,
  less backpressure (fetch rate 10.1 -> 11.0 GB/s; head 16.8 -> 12.8, window 88.9 -> 84.1, backlog 14.8 -> 12.5).
- Rejected (full runs): join post-chain fusion via PostChannel (136.9, 131.0), one hop per chain (123.3, neutral), 32 py
  executors (131.7 on a slow-network run), tail split 4x2 with fusion (134.3 vs control 127.8), reactor CPU pinning (verified
  applied, no change), 512 KiB min GET (small-query fetch unchanged: S3 latency ~60 ms per 8 MB, ~150 ms per 90 MB).
- Per-query fixed costs outside the network: ~0.07 s polars, 0.07-0.1 s planning/network setup, 0.015 s result: ~4 s total.
- Loop thread (ct29): on CPU 3.1 of 13.7 s, ~1,500 context switches/s, mostly <1 ms off-CPU (GIL handoffs around nogil calls).
- Run-to-run comparisons need interleaved controls: the same config moved 122.7 -> 127.8 s within ~1 h.

## 11 Nov (late night): scan post-fusion, I/O state sampling (io71-io80)
- CUDF_POLARS_FUSE_SCAN_POST (41d0e1ddb1): chunkwise successors of a scan run in its producers. Interleaved pairs vs fusion-only
  control: 121.7 vs 136.7, 131.8 vs 136.7; later 124.2/124.8/123.5 alone. Current best config = FUSE_CHUNKWISE + FUSE_SCAN_POST.
- The control of one config ranges 122.7-137 s across the night; the spread is the fetch rate (9.8-11.1 GB/s in-engine) while
  standalone S3 stays 11.6-12.0 GB/s. Always run interleaved pairs.
- ScanIO.snapshot sampling (io76): before each query's last byte the in-flight bytes are at the 4 GB cap ~85% of the time; arena-
  bound only ~2.8 s (Q8, Q21); idle ~6 s (~0.25 s/query). So the in-engine fetch is not starved by backpressure; at the same
  in-flight it gets ~10.5 vs 11.7 GB/s standalone (per-request latency ~0.41 vs 0.35 s): likely CPU (reactors ~14 cores, softirq
  ~1/3 of the machine). NIC settings (MTU/GRO) could not be inspected (ip link denied); changing them needs root anyway.
- Rejected: 6 GB in-flight (136.3 vs 123.5), group-by concurrency 4 (133.3 vs 134.0, Q18 worse).
- Current breakdown (io78, 123.5 s): head 13.3, window 83.6, backlog after last byte 13.6 (Q21 3.2, Q18 1.8, Q1 1.0, Q10 0.9,
  Q5 0.9), post 13.0 (Q7 1.9, Q13 1.7, Q9/Q20 0.9). Q18 scans the same lineitem columns twice (rescan instead of fanout) and its
  group-bys run one chunk at a time; Q21 reads 20.7 GB of lineitem columns twice.

## 12 Nov (early): shared column chunks (io81-io92)
- ScanIO share region (uncommitted, CUDF_POLARS_SCAN_IO_SHARE_MB, site cudf-ext-io2d-site): a column chunk planned by several
  tasks is fetched once into a pinned share region; every reader H2D-copies it from there. First version (copy into pageable
  heap on a stasher thread) made Q21 12.5 s (page faults, blocks held). Deciding at grant time missed the race with the twin
  scan's planning (Q18 shared 1/6.4 GB); deciding when reads are planned at issue time: Q21 shares ~19/20.7 GB, Q18 4.2-4.9/6.4.
- Full-run pairs LOST: io90 129.5 vs io89 125.0, io91 135.0 vs io92 123.1; fetch-window sums 101.5/105.8 vs 97.2/96.4 despite
  ~22 GB fewer bytes. Q21 unchanged (its fetch ends ~5.5 s; the rest is joins/groupby/spill). Isolating deferred read planning
  and the extra 24 GB pinned memory: io93/io95 (io2d, share off) vs io94/io96 (io2c).
- io85 vs io87 (same config, 126.9 vs 134.4): 7.0 of 7.5 s is fetch time spread over every big query (11.1-11.6 vs 10.3-10.6
  GB/s); ENA counters: fast run bw_in_exceeded 90 M / pps_exceeded 2.8 M, slow run 38 M / 3.8 M -> slow run more pps-shaped.
  NIC settings still unreadable (permission system denies ip/ethtool even with Matt's OK; needs an allow rule).
- Shared chunks shelved (patch in notes/scan-io-share-chunks.patch). Deferred read planning looked costly: io2d with share off
  130.1/129.9 vs io2c 123.7/130.5 (six io2d runs avg 131.0 vs 125.6).
- Partition size: one knob for scans and compute; for remote scans planning has no sizes (footer sampling is 0 for remote), so
  every remote scan is one task per file and target_partition_size only affects compute + the scan's chunk-size estimate.
  3 GB both: 146.3 vs 131.6 (Q13 +3.7, Q21 +2.9, Q10 +2.5). Compute-only 3 GB: 129.7 vs 134.4 (neutral); 0.75 GB: 136.5 vs
  123.0 (Q9 +3.6). Keep default 1.5 GB.
- Execution-time resize (CUDF_POLARS_SCAN_FUSE_BYTES / _SPLIT_BYTES, ScanIO.projected_bytes, site io2e): fusing to 512 MB made
  Q18 9.3/6.6 (vs 4.6-5.1), Q21 OOM cold / 11.5 warm, Q4 3.9 vs 3.4; waiting for all of a scan's footers costs ~0.22 s per scan
  (reads used to start as each footer landed). Sirius never waits: its coalescer packs each file as its footer lands and the
  pipeline needs no chunk count up front. Task size is near a sweet spot both ways (tail split, 512 MB tasks, fusion all lost).

## 11 Nov (afternoon): what bounds per-chunk operator calls (io106-io121)
- Join-plan trace (io106): almost every join already broadcasts one side; only Q21's first join (9.6 vs 56.5 GB) and Q4 Semi /
  Q13 Left (small side is the preserved one) shuffle. Q7's post-scan join calls are broadcast probes lagging the probe stream
  (~half of 480 calls after the last chunk). Broadcast limit 40 GB: inconclusive (Q9 made the same 106 calls).
- Join call CPU (Q7): ~9 ms thread CPU per call during the fetch vs 1.7 ms after. perf (PYTHONPERFSUPPORT=1, io111): Python
  threads 12.8% of process CPU (REST workers 81.6%); of Python-thread CPU ~30% is in libcuda, 86.5% of that cuStreamSynchronize
  spinning (ioctl); the interpreter is 3-4%. So per-call cost = GPU queueing (behind decode kernels) + synchronize latency, not
  Python.
- Blocking sync (CUDA_BLOCKING_SYNC=1): much worse, 140.1/141.8 vs 123.5/124.9 (every phase): spin wake-up latency matters.
- Priority streams (CUDF_POLARS_PRIORITY_STREAMS=16, e297b6e215: operators on greatest-priority streams, decodes on the pool's):
  126.8 vs 134.5, 121.4 vs 139.5 (controls second, slow), reversed order 122.3 vs 122.6 (warm 118.4 vs 121.4). Consistent:
  backlog -1.0 to -2.1 s, post -0.8 to -1.0 s. Adopted into the best config.
- For a C++ executor: the gain would come from stream-ordered chains with fewer synchronizes and deliberate decode vs compute
  GPU scheduling, not from removing the interpreter.

## 11 Nov (evening): native executor, Q7 vertical slice
- ProbePipeline (cpp/probe_pipeline.{hpp,cpp}, Cython in scan_io.pyx; harness notes/q7_native.py; site cudf-ext-io2f-site):
  worker threads take planned ScanIO tasks, materialize natively, run each branch's stages on the worker's stream (hash_join
  probes against prebuilt build tables, a small expression contract: column/literal/binary/year, select, partial groupby),
  fold partial aggregates; only final aggregates return to Python. No per-chunk Python.
- Q7 lineitem probe (both union branches) cold from S3, build tables prebuilt untimed: 67.4 GB at 11.1-11.7 GB/s, tail after
  the last byte 0.05-0.26 s with 2/4/8 threads, VALID vs expected q07. 360 join probes over 3.6 B rows: 0.7-1.3 s summed
  (~2-3.5 ms each) vs 8-25 ms per call in the engine. Engine Q7 tail after its last lineitem byte: ~1.8-2.7 s.
- Not yet measured: build sides fetched in the same query (engine Q7 also reads ~6 GB of orders/customer), memory admission,
  engine integration (lowering a scan -> chunkwise -> broadcast joins -> partial groupby chain to a native node).
- Engine integration (CUDF_POLARS_NATIVE_REGION=1, e5ed650513): a GroupBy fed by one remote scan through projections, simple
  expressions and inner joins becomes one native actor (build sides still engine sub-networks). Matches Q3, Q5, Q7, Q9, Q10
  (Q11 excluded: probe path shared outside the region). Fixed on the way: arena deadlock (read-ahead before builds capped,
  CUDF_POLARS_NATIVE_PREFETCH_MB), build tables viewing freed chunks, scan filter freed during decode, frees on destroyed streams.
- Subset Q3/5/7/9/10 with 20 GB read-ahead: 36.2 vs 40.7 s cold. Full pairs: 125.6 vs 128.4, 127.9 vs 136.6 (all valid);
  Q7 -1.6/-1.8, Q9 -4.7 in the second pair. Not rapidsmpf memory-aware yet (no reservations/spill in the native region).
- With Q8 (when/then, string compares; 660389b54e): full pairs 118.3 vs 123.5 and 129.1 vs 131.3 (all valid). The six native
  queries (Q3/5/7/8/9/10) gained 5.3 and 5.4 s in the two pairs; the other 16 queries moved +0.1 and +3.2 s (noise).
  118.3 s is the best cold run so far.
- Paused for design here (Matt, 11 Nov). Uncommitted, built only into cudf-ext-io2g-site: C++ streamed-output mode for the
  probe pipeline (bounded output queue, close()/next_output(), process-wide worker stream pool) for a "native probe source"
  that would emit one chunk per task into the regular engine; the Python matcher/actor for it is not written.

## 12 Nov: native regions on rapidsmpf C++ actors (io151-io166)
- Step 1 (24f2eaae01): NativeRegion IR node from a lowering pass, cudf_polars.streaming.native package, 29 unit tests. Same six
  queries match; gate pairs 117.5 vs 122.9 and 127.1 vs 140.1 (native queries -4.6 / -9.4 s), all valid.
- Step 2 (6e75013ba4): rapidsmpf C++ actors in the extension (scan_source, chain, broadcast), stage code shared in stages.cpp.
  rapidsmpf's C++ fanout forwards data only, so broadcast sends channel metadata first (found with CUDF_POLARS_NATIVE_DEBUG=1:
  every chain waited for input metadata).
- Fused vs composed vs python (Q3/5/7/8/9/10, alternating): fused 41.8 / 44.7, composed 42.7 / 41.9, python 42.6 / 43.9 s. No
  difference beyond network noise: C++ channel hops cost nothing measurable, so the speedup is from removing Python per-chunk
  handoffs, not from fusion. Next design step: lower supported operators to C++ actors individually (no region carving).
  Open: output partitioning metadata per operator, runtime join strategy, high-cardinality group-by via the existing reduction.
