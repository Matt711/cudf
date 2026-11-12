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
- Native operators (d1779338ac; CUDF_POLARS_NATIVE_JOIN/CHUNKWISE/GROUPBY): Python keeps the control plane, a C++ chain actor
  runs each operator's per-chunk loop (rapidsmpf when_all). Full pairs 134.4 vs 136.2 (Q22 +4.3 outlier) and 119.6 vs 124.5,
  all 22 valid. Q7 has no per-chunk Python left (7.2-7.5 s). Native compute now on high-priority streams (event-ordered after
  the decode; the input waits for the compute before it is freed).
- Pipeline timeline (channel metrics sampled every 10 ms with the I/O trace; notes/pipeline.py): Q7 probes start at 3.6-4.2 s
  though lineitem arrives from 1.8 s, because the build sides (orders scan sharing bandwidth) finish at 3.5-4.2 s; the bounded
  lineitem fanout waits on the slower branch 100%. Q9: orders chunks (probe side) wait 1-10 s for the lineitem-derived broadcast
  side. Q21: 4.1 s tail in Python operators (semi/anti joins, counts, shuffles). Next: build-first fetch priority.

## 13 Nov: build-first fetch priority (io179-io187)
- aa4c65dc2d, CUDF_POLARS_SCAN_PRIORITY=1: each service stream opens with priority = its scan's task count; granting arena room
  and issuing reads take the lowest first. The query tells the service how many streams it will open (set_expected_streams);
  until opened, each stream's arena share counts them. Off by default (old behaviour unchanged).
- Two bugs on the way, both visible only in the arena columns of the trace (scratchpad fetch_rate.py), not in query times:
  1) expected count held for the whole query (closed streams leave _streams): lineitem capped at arena/N -> Q7 10.6 s, Q9 14.5 s.
  2) counted every remote StreamingScan, but single-task scans (nation, region) are read directly and never open a stream:
     still 25 of 32 GB arena free and in-flight 2.8 of 4 GB -> Q7 9.2 s. Fixed by counting native region scans + scans of >1 task.
- Fixed version (io183 vs control io182, Q7/9/21): Q7 build sides ready at 2.5-3.1 s (was 3.6-4.2), in-flight at the 4 GB cap
  throughout, Q7 7.30 vs 7.52, Q9 10.96 vs 10.08 (noise), cold 28.8 vs 28.7. Q7 is now bound by lineitem's fetch (last byte
  6.4 s, done 7.1 s): moving builds earlier only helps where the tail waits on a build.

## Later (Matt, 13 Nov): the arena becomes the cache
- Stay on cold for now. When warm resumes: Matt's cuCascade integration PRs reached 88-89 s warm with a 100-130 GiB cache, so with
  today's cold gains warm should beat that by a lot. Sirius uses a 100-200 GiB pinned cache for both cold and warm, with
  readahead landing in the cache itself.
- Direction: make the arena the cache (one pinned pool; blocks in use = pinned, decoded blocks retained and evictable, a hit is a
  read that completes at once). Worth doing for cold too: a larger pool means more read-ahead room. Keep "full of in-use" and
  "full of retained" apart in any arena measurement.
- Also later (Matt, 13 Nov): rapidsmpf spills to its own pinned pool (async pool, on by default, no cap; tiers DEVICE >
  PINNED_HOST > HOST > DISK) while the arena is a separate 32 GB cudaHostAlloc: two pinned allocators, no shared budget.
  Direction: one pinned budget owned by rapidsmpf; the arena/cache allocates from it and registers as a PINNED_HOST spill
  tenant (evict retained blocks first: refetchable, no write-out). First measure spill bytes and pool size in cold runs.
- Full pairs (all valid): priority 132.3 / 123.0 vs control 128.9 / 129.5 -> neutral (-0.8 s mean; Q7 -0.6, Q18 -0.6). Kept opt-in.
  Machine slow today (control 129 vs 118-124 earlier).
- Where a cold run's time goes (control io185, 128.9 s; scratchpad budget.py): fetch spans 104 s for 1,071 GB (10.3 GB/s
  average), tails after each query's last byte 21 s, startup before the first read 1.3 s. Arena-limited time: 0 in every
  query (arena never under 2 GB free with in-flight below the cap). Of the sampled fetch time, ~87% at the 4 GB in-flight
  cap (network-bound, 10.3-11.6 GB/s) and ~11 s with in-flight below the cap and nothing queued (scans not asking): Q8 2.7,
  Q21 2.2, Q18 1.0, small queries. Biggest tails: Q21 3.6, Q18 2.6, Q13 2.4, then ~1 s in Q1/4/7/9/10/20.
  Q21 and Q18 lead tail, starvation and duplicate bytes (20.7 + 6.4 GB): next target.

## 13 Nov: key-order inference (CUDF_POLARS_INFER_KEY_ORDER=1, io188-io191, experimental)
- lineitem (60 files) and orders (15 files) are globally ordered by orderkey: ascending, non-overlapping file ranges, no key
  spanning row groups or files. But scans list files lexicographically (part.0, part.1, part.10, ...), so in task order the
  bounds are not monotonic.
- Upstream (rjzamora, story NVIDIA/cudf#22128, all merged parts in our base): footer ordering inference (#24199) runs only for
  explicit set_sorted hints; joins/groupbys send strict (hash) requests. Open: #25143 (generalized ordered groupby).
- Experiment: join/groupby leading keys become inference candidates; footers come from the scan service; tasks are reordered
  by their footer key bounds (assumes the scan's row order is not observed). Q18/Q21 valid.
- Q21: 9.6 / 9.4 s (was ~10.0): every groupby and join runs chunkwise (60 msgs) as lineitem arrives, the query ends 0.06 s
  after its scans; but inference delays the lineitem scans to 2-3 s (footers + bounds before the stream opens).
- Q18: 16.6 / 12.8 s (was ~7.1): lineitem groupby (60 chunks) ordered-joins orders (15 chunks) on different boundaries; the
  ordered join's alignment emits its first output at 11 s (last byte 6.3 s) and blocks upstream.
- To make it real: (1) inference off the critical path (footers already in flight; bounds cheap), (2) ordered join of
  mismatched chunk boundaries that streams (lineitem files nest 4:1 in orders files), (3) a correct rule for when a scan may
  reorder its tasks. Overlaps rjzamora's upstream work: coordinate before going further.

## 13 Nov: the download is CPU-bound in the I/O stack itself (io192, standalone svc_rate.py)
- Causal test (scan service alone, engine settings, 64 GB of lineitem): alone 11.7-12.0 GB/s; next to 8 busy cores 10.4-10.7;
  next to 16 busy cores 9.7-10.1 (in-engine: 10.3). CPU contention alone reproduces the in-engine rate.
- Full control run with per-thread CPU (io192, 130.2 s): engine (python + rmpf) ~2.6 cores; REST reactor threads 18.8 cores;
  machine softirq 7.9 cores; machine busy p50 27.7 / p90 31.7 of 32. The I/O stack, not the engine, saturates the box.
- Standalone service: ~19-23 cores for 11-12 GB/s. Each reactor thread tops out at ~0.8 GB/s on ~1 core; connections per reactor
  do not matter (32x8 = 32x32). TLS read-ahead (OpenSSL, via CURLOPT_SSL_CTX_FUNCTION; worktree cucascade-readahead, opt-in
  CUCASCADE_TLS_READ_AHEAD) made it worse: 256 KiB 9.7 GB/s, 1 MiB 6.0 GB/s. Rejected.
- Kernel profile (kptr_restrict=0 by Matt): copy to user 19%; page zeroing on allocation (kernel_init_pages + page alloc) ~11.5%
  of the service's CPU (ENA RX ring refill pages are zeroed: CONFIG_INIT_ON_ALLOC_DEFAULT_ON=y); ENA refill/descriptors/DMA ~6%;
  hardened usercopy checks (check_heap_object, __virt_addr_valid) ~4.3%; user memcpy (curl buffer -> ours) ~15%; AES-GCM ~5%.
  Boot cmdline has no override. Candidate: boot with init_on_alloc=0 hardened_usercopy=off (needs root + reboot).
- Streaming-store sink copy (cuCascade worktree cucascade-readahead, CUCASCADE_SINK_STREAM=1): standalone machine CPU 26.7 ->
  18.7 cores (no-copy ceiling 14.2) at the same rate. glibc's non-temporal threshold never applied (callback chunks <= one TLS
  record, under its 16 KiB minimum). Full cold pairs (io193-io196, io2m site): 120.4 / 125.4 s vs control 136.2 / 132.8 s, all
  valid; download 11.26 / 10.65 vs 9.71 / 10.16 GB/s; tails unchanged (~24 s).
- Matt: cold = cache cleared after every query; lukewarm = not cleared; hot = iteration 1 of lukewarm. The benchmark's
  --io-mode cold never touched the cuCascade cache (it only drops the OS page cache of local files), so earlier "cold" cache
  numbers were lukewarm. Added scan_cache::clear + ScanCache.clear + cucascade.clear_cache(), called by --io-mode cold.
- Matt's PR cache path (io2 tree, scan service off, 100 GiB cuCascade cache, streaming sink copy on), same day:
  strict cold (io199, cache cleared before every query): 180.5 s / 169.3 s (both iterations cold), ~1,130 GB per iteration;
  lukewarm (io200): iteration 0 90.9 s (lukewarm), iteration 1 79.2 s (hot), 509 GB for both. io2 strict cold: 120.4-125.4 s.
  So io2's scheduler is ~50-60 s faster under strict cold; the cache path already beats 88-89 s lukewarm/hot. Next: io2's
  scheduler reading through the cache (step b) for both.
- Step b (io2 scheduler reading through fs_cache: prefetch_ranges + device_read_ranges_async; cudf c3df2559f7, cuCascade 4215b3d):
  chunk size matters (Q7/Q9 cold: 256 KiB 11.3/12.1 s, 4 MiB 8.9/10.9 s; 16 MiB exhausts the reactor staging pool). Full runs,
  same day, streaming on: io2 arena cold 126.1/127.3 s (2,174 GB both iterations); io2 through cache cold 151.6/144.9 s (2,435
  GB: +12% from chunk rounding; download 122 vs 102 s); through cache lukewarm 152.1/132.8 s with 2,205 GB: no reuse (every byte
  cached, plain LRU at 100 GiB churns ~1 TB per run; no column admission; objects reopened each query). Cache mode is a loss for
  cold; lukewarm needs admission (Matt's PR path gets ~300 GB lukewarm through column_admission).
- Why step b was slow (Matt: "it makes no sense"; he was right). Standalone harness svc_cache.py (real lineitem column-chunk
  ranges, 2,820 tasks / 53 GB, 16 consumer threads, to device), arena 10.6-11.0 GB/s:
  1. Over-read: a chunk holds an edge-anchored extent, so a range inside a chunk widens to the nearer edge. 4 MiB chunks read
     22% more bytes (NIC 64.9 vs 53.6 GB); 1 MiB 3%; 256 KiB 0.5%.
  2. One GET per chunk: fs_cache::prefetch made one slice per chunk, so the REST reactor never merged chunks into its 4-16 MiB
     segments. 256 KiB chunks fetched at 8.7 GB/s. Fix (cuCascade b5e28dc): contiguous fills join one multi-chunk slice ->
     11.5 GB/s fetch at 64 KiB and 256 KiB, same as the arena.
  3. Copy to device: device_read_ranges_async retires every call through cudaStreamAddCallback + a ticket poll (2 waiter
     threads, backoff to 256 us) and blocks the caller until the copy is done. Resident-only copy rate 13.5 GB/s flat from 4 to
     64 threads (arena 25-27 in the same harness; raw H2D is 54 GB/s for cudaHostAlloc and registered THP memory alike, one copy
     or a 256 KiB batch). Fix: fs_cache::pin_resident_ranges + io2 copies the pieces itself in one cudaMemcpyBatchAsync and
     keeps the pins until release() (cudf 0f779ab5ca), the arena contract.
  After both: cache 256 KiB 10.9-11.5 GB/s end to end vs arena 10.6-10.8, same NIC bytes, same CPU.
- Full cold A/B after those fixes (io210-213): cache 132.6/136.2, 127.9/125.4 s vs arena 118.5/117.8, 126.9/122.3 s. Counters
  (CUDF_POLARS_SCAN_IO_STATS, io214) showed ~92% of prefetches "uncovered" and nearly every file falling back to the blocking
  device_read_ranges_async: plan_reads cut a big task's ranges into 64 MiB batches at arbitrary offsets, so two prefetches
  shared the chunk at each cut; the first filled it from one edge, the second skipped it (loading), and its bytes were never
  resident. The harness missed it (row-group tasks are never cut). Fix: batches part only on chunk boundaries (a range that
  starts in the chunk the last one ended in stays in the batch). Harness, one task per file: 390/420 uncovered, 0/30 pinned
  -> 0/420, 30/30.
- Runs after the batch fix (cold, 100 GiB cache, 256 KiB chunks; arena runs interleaved): cache io215 124.9/124.9, io219
  125.2/129.0, io220 120.9/117.6 s vs arena io216 128.4/133.0, io218 128.0/122.5, io222 125.0/125.5, io224 118.4/116.0 s:
  parity (means 123.8 vs 124.6). Per run ~2,750 file reads pinned, ~35 fall back (chunk shared with a neighbouring task and
  cached with an edge-anchored fill too narrow for it). Waiting for loading chunks before falling back (io221/io223) cost Q1
  +4 s: reverted. Live tasks at every clear: 0.
- OPEN: io217 (same build as io215) returned wrong Q4 results then cudaErrorIllegalAddress in materialize; 1 of 6 cache
  runs with the pinned copy path; none of 79 recent arena/other runs. Not reproduced in the harness with byte verification
  (row-group and per-file tasks, two concurrent projections of the same files, clears every 50 ms during in-flight reads).
  Ruled out: pin lifetime (stream synced before release), leftover tasks at clear, double free in reclaim (CAS), per-op
  completion publishing. Do not use cache mode for results until this is found.

### io217 follow-up (12 Nov 2026)

- dfc1..dfc8: eight more cache-mode runs (io2z: current code + `CUCASCADE_POOL_CHECK=1`), all valid; no chunk
  buffer was handed out twice or returned twice. Cache-mode total: 14 valid of 15. Times 116.7 to 137.9 s.
- No clear ever overlapped a live task (528 clears, all `live tasks at clear 0`), so the clear-wait changed nothing.
- io217 failed both Q4 iterations with a cache clear between them, while Q3 (same `orders` files, other columns)
  passed twice. Footers are cleared each query (`begin_query`), so they are not the persistent state. What
  survives a clear: fs_cache generations (strong S3 ETags keep them across queries), the buffer pool,
  completion lanes, reactor connections.
- Reviewed, no defect found: pin_resident_ranges addressing (same as the existing device path), REST fill
  mapping of coalesced slices (fills are fixed while loading), stream_copy (ends in sfence), eviction-loop
  buffer lists (cleared each round), every `release()` caller (all synchronize first).
- Tools for the next occurrence: `CUCASCADE_POOL_CHECK=1` (cheap, keep on) and `CUDF_POLARS_SCAN_IO_VERIFY=1`
  (names the file, range, chunk and whether the chunk or the copy was wrong; about 6x slower, so only for a
  targeted repro).

### Tails and a second cache-mode failure (12 Nov 2026, later)

- tcp_slow_start_after_idle=0 (Matt set it): no change. Small-query fetch rate 9.5 GB/s at both settings
  (dfc1-10 at 1, dfc12-14 at 0).
- why1 (cache mode, io2z): Q21 iteration 0 hit cudaErrorIllegalAddress in a group-by reduce_by_key; the CUDA
  context died, so Q21 it1 and Q22 failed too. Pool check silent. Cache mode: 2 failed of 22 runs (io217 Q4,
  why1 Q21). The chunk checksum check (CUCASCADE_CHUNK_CHECK) is 2-3x slower in full runs: not usable as is.
- Tail anatomy (dfc1-14, per pass): 13.6 s from last byte to last decode, 9.9 s from last decode to the end.
  Per query: Q21 3.4 + 0.2, Q18 2.1 + 0.3, Q13 0.9 + 1.7, Q1 1.3 + 0.3; queries already fully native (Q5-Q9)
  have tails of 0.3-0.6 s.
- Native group-by of counts and means (1c82a76bd8): all valid, Q1 tail unchanged. Q1's tail is the network:
  300-650 GETs stay in flight until the last 0.2-0.5 s (drain).
- S3 GETs (rq1, 787k): total p50 0.11 s, p99 0.48, p99.9 0.84, max 2.2; first byte p99 0.10. 196 over 1 s.
  Stragglers are rare; drain averages 0.2-0.5 s per query (5.8 s per pass).
- Q21 and Q18 have mid-query gaps with nothing in flight: their scans fetch a bounded window ahead of
  consumption, and the consumer is slow. Q21 profile (nsys): 4.9 s of kernels in a 10.1 s run; group-by
  hash builds 1.45 s (n_unique per l_orderkey on ~60M-row chunks), gathers 0.6 s, spill packing 0.3 s.
