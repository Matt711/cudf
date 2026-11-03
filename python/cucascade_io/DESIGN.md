# cuCascade REST backend for cudf-polars (prototype)

Reads `s3://` parquet through cuCascade's REST backend and prefetching cache (`cucascade::io::datasource`, a `cudf::io::datasource`) instead of the byte cache in `python/bytecache`. Enabled with `CUDF_POLARS_CUCASCADE_GB` (cache size); off by default. Goal: reproduce bytecache's SF1000 numbers (104 s cold, 87 s warm) with cuCascade, then upstream.

Build: `python/cucascade_io/build.sh` (needs a cuCascade build, see the script variables, and liburing). Settings (env, all `CUDF_POLARS_CUCASCADE_*`): `GB` (cache size, enables the backend), `REACTORS` (REST reactors, default 16; cuCascade's own default 2 peaks at 1.6 GB/s), `CONNECTIONS` (per reactor, default 128), `BLOCK_KB` (cache chunk size, default 256), `EVICT_BATCH` (cuCascade eviction_batch_fraction, default 0.02, needs the patch below), `BUDGET` (share of the cache the admission policy may fill, 0.9, counted in chunk-aligned bytes), `EVICT_THRESHOLD` (cuCascade eviction_threshold_fraction, 0.95; its default 0.8 caps the usable cache), `COLUMN_POLICY` (1), `ADVISE_MIN_BYTES`, `EVICTION` (lru|idle), `VERBOSE` (exit report). Glue-only env: `CC_PREFETCH=1` (separate prefetch step, off), `CC_NO_ALIGN=1` (do not widen advised ranges to whole chunks), `CC_RANK_MARGIN` (1.5), `CC_ASYNC_THREADS` (128).

## What was learned (cuCascade main b09010a, libcudf 26.12, SF300, g7e.8xlarge)
- Credentials: cuCascade's REST factory takes static credentials plus an explicit endpoint only (no env or role chain); the glue passes the AWS_* env values in memory.
- 2 REST reactors (default) peak at 1.6 GB/s (q6 cold 15.9 s); 16 reactors peak at 7.7 GB/s (q6 3.3 s).
- cuCascade only keeps chunks that a scan registered with `fadvise` (then prepare/prefetch). Plain reads are never cached, so a first integration read 632 GB at SF300 (stock behaviour) vs 57 GB for bytecache.
- libcudf reads exact column-chunk ranges without advising them, so `advising_source` (cc_glue.cpp) wraps the datasource and registers each sizeable read's range with the cache first (one duplicate datasource per read, kept until the scan's datasource dies).
- Using the separate `prefetch_async` step made cold reads about 20x slower (0.11 GB/s vs 2.4 GB/s; q6 cold 48 s). Not root-caused (suspect: demand reads wait for low-priority prefetch chunks that run only when no high-priority request waits). Off by default (`CC_PREFETCH=1` turns it on).
- About 10% of advised bytes are still re-fetched on a repeat read (cause not found; possibly partial chunks at column boundaries).

## Results (2 iterations, default tuned kvikio env, held driver state, 22/22 validated)
| SF300 | cold | warm | S3 received |
|---|---|---|---|
| bytecache | 34.7 s | 23.3 s | 57 GB |
| cuCascade, first integration (nothing cached) | 59.3 s | 60.8 s | 632 GB |
| cuCascade, advised reads, no prefetch | 47.3 s | 27.3 s | 117 GB |

## Update: what limited SF1000 (one pass, 2 iterations; same machine, held driver state, tuned-kvikio env, all 22/22 validated)
- Read throughput scales with reads in flight (one 8 MiB read is one ~110 ms round trip): 0.57 GB/s at 8 threads, 2.6 at 32, 8.2 at 128 (direct reads).
- cuCascade's host space pre-allocates only `initial_number_pools` (default 4 x 128 MiB) and grows; with the cache armed every `prepare_prefetch` then waits (231 ms average, advised reads capped at 2.9 GB/s). Pre-allocating pools for the whole cache gives 9.2 GB/s in the same test. The glue now pre-allocates (pins) the cache in `bytecache.warm_up()` at engine start, so q1 does not pay for it (q1 39 s -> 10 s).
- LRU alone over 100 GB reads about 800 to 850 GB per pass at SF1000 (like Sirius at 100 to 120 GB host: 862 to 981 GB, 130 to 146 s). Sirius' 96 to 109 s uses a 200 GB host tier.
- Column-aware admission (rank by requested/registered, advise only columns that fit `budget_fraction` of the cache) helps; the budget must stay below cuCascade's 80% eviction threshold. SF1000, 1 iteration:

| config | cold | S3 received |
|---|---|---|
| LRU, lazy pools (2 it.) | 196.6 s (cold of 2) | 1623 GB (2 it.) |
| policy, budget 0.9 | 168.8 s | 847 GB |
| policy, budget 0.6, 1 MiB chunks (default) | 152.8 s | 662 GB |
| policy, budget 0.6, 256 KiB chunks | 177.6 s | 502 GB |
| budget 0.6, 1 MiB, aligned ranges + prefetch step (CC_ALIGN=1 CC_PREFETCH=1) | 1309.5 s | 657 GB |
| same, 512 KiB chunks | 715.4 s | 553 GB |
| bytecache, tuned kvikio (for reference, 2 it. cold) | 104.2 s | 547 GB (2 it.) |

- Smaller chunks keep small columns cacheable (a read that covers part of a chunk cannot be recorded as cached) but cut throughput; the prefetch step starves demand reads (cause not shown). Both off by default.

## Update 2: sticky admission, footprint accounting, chunk size (SF1000, 2 iterations, held state, 22/22 validated)
- The prefetch step stalled because cuCascade's async read blocks the calling thread until the prefetch lands (`await_inflight_prefetch`), and libcudf submits async reads from a few threads, so each read became synchronous (q19 alone: 232 s at 0.3 GB/s). The wait now runs on a glue thread pool (q19 alone: 11.5 s at 8 to 10 GB/s). The REST reactor ignores io_priority (only the uring reactor has tiers), so priority was not the cause. With the stall fixed, prefetch + aligned ranges still did not beat plain advised reads (cold 146.6 s vs 134.2 s), so it stays off.
- Admission is sticky: a (file, column) piece that was admitted stays admitted; new pieces are admitted while the admitted set fits `BUDGET`; a column is replaced only by one that ranks 1.5x higher (rank = requested / registered). Rank-driven re-admission made LRU evict hot chunks instead of stale ones.
- The budget must be counted in chunk-aligned bytes: a 0.5 MB column chunk still occupies whole 1 MiB chunks (l_orderkey 1.8 GB raw = 7.4 GB footprint, l_discount 2.9 -> 8.5, l_shipdate 8.7 -> 14.3, l_extendedprice 33 -> 38.6). With raw sizes the cold pass evicted 328 k chunks (631 GB fetched); with aligned sizes 21.8 k (514 GB).

| config (budget 0.9, sticky, aligned) | cold | warm | S3 cold pass | S3 warm pass |
|---|---|---|---|---|
| 1 MiB chunks (default) | 125.3 to 128.4 s | 119.5 to 121.1 s | 514 GB | 485 GB |
| 512 KiB chunks | 141.8 s | 104.7 s | 517 GB | 337 GB |
| 256 KiB chunks | 164.5 s | 100.6 s | 511 GB | 293 GB |
| bytecache (tuned kvikio) | 104.2 s | 87.1 s | 332 GB | 206 GB |

- Smaller chunks keep small columns from wasting cache (warm bytes fall) but cost cold time without extra bytes: per-chunk CPU/bookkeeping in the cache (evictions 612 k at 256 KiB). The REST reactor already fuses contiguous chunk fills into GETs of up to 16 MiB, so request size is not the reason.
- Remaining gap to bytecache (cold 20 to 25 s, warm 15 to 35 s): footprint inflation of small columns (about 24 GB of the 92 GB admitted set), LRU instead of rank-driven eviction, and per-chunk overhead at small chunk sizes. Ideas for cuCascade itself: sub-chunk validity so a partly covered chunk can be cached; an eviction hook or per-request priority so a caller can say which chunks are cold; a lower per-chunk cost.

## REST tuning sweep (SF1000, 1 iteration, current defaults, held state, 22/22 validated, one run each)
| reactors | connections per reactor | cold | S3 received |
|---|---|---|---|
| 16 | 64 | 121.5 s | 539 GB |
| 8 | 64 | 148.0 s | 554 GB |
| 24 | 64 | 129.0 s | 542 GB |
| 32 | 64 | 135.1 s | 559 GB |
| 16 | 32 | 143.7 s | 542 GB |
| 16 | 128 | 136.3 s | 576 GB |

16 x 64 is the best of the six and matches Sirius' own sweep (8 reactors much slower; 32 no better). Run-to-run noise is about 5 to 7%, so the 24 and 32 results are not clearly worse. This is the counterpart of the kvikio 48 reactors / 512 ceiling setting. bytecache single pass with tuned kvikio: 104.1 to 104.6 s.

## Update 3: why small columns missed on every repeat, and aligned advising
cuCascade records the valid part of a chunk as an edge-anchored extent (prefix or suffix of the chunk) and a *cached* chunk cannot be widened (`chunk_state::merge_fill` only widens pre-load chunks; a request needing more "correctly misses and reads for itself"). Two column chunks that share a chunk therefore miss forever when the first fill covered only one edge (l_discount repeat: 38 of 52 MB on the wire, 94 misses). Single read shapes (whole, head, tail, interior, straddling) all cache fine in isolation (`shape_test.py`).
Fix in the glue: advise chunk-aligned ranges (default; `CC_NO_ALIGN=1` turns it off), so the first fill of a chunk is the whole chunk. Repeat of l_discount: 0.5 MB (the footer probe), 0 misses. Cost: edge chunks are fetched whole (first read of l_extendedprice 688 MB vs 590 MB).
SF1000, 2 iterations, 1 MiB chunks, budget 0.9, sticky: cold 125.3 s, warm 108.3 s (before: 125.3 to 128.4 / 119.5 to 121.1); S3 487 GB cold pass, 410 GB warm pass (before 514 / 485). Warm bytes are now dominated by columns the budget refuses (about 360 GB), so what remains is capacity and footprint.
Small chunks: in a pure I/O test (128 threads, 8 MiB reads, no eviction) throughput falls with chunk size (1 MiB 8.3 to 9.0 GB/s, 512 KiB 9.0, 256 KiB 3.5 to 6.6, 64 KiB 2.4); no thread is CPU-saturated. A plausible cause is bytes in flight = connections x request size (request size follows the chunk), but the microbenchmark is noisy (same setting 3.5, 5.3, 6.6 GB/s) and 256 connections per reactor was worse than 128, so this is not shown.

## Chunk size with aligned advising and 128 connections per reactor (SF1000, 2 iterations, held state, 22/22 validated, one run each)
| chunk | connections | cold | warm | S3 cold pass | S3 warm pass |
|---|---|---|---|---|---|
| 1 MiB (default) | 64 | 125.3 s | 108.3 s | 487 GB | 410 GB |
| 512 KiB | 128 | 146.9 s | 102.4 s | 529 GB | 304 GB |
| 256 KiB | 128 | 137.5 s | 98.4 s | 512 GB | 275 GB |
| bytecache (tuned kvikio) | n/a | 104.2 s | 87.1 s | 332 GB | 206 GB |

More connections cut the cold cost of 256 KiB chunks (164.5 s at 64 connections, 137.5 s at 128). Smaller chunks lower the warm pass but the cold pass fetches the same 490 to 530 GB at every chunk size, so the cold-pass gap to bytecache (about 150 GB, about 20 s) is not a footprint or capacity effect; it is which columns are resident as the pass proceeds. A rank-driven eviction (or a per-request priority on fadvise) inside cuCascade is the missing piece; with LRU plus sticky admission in the glue the cold pass converges late.

## Update 4: patch to cuCascade (eviction priority, small eviction batches) closes most of the gap
**The glue now needs a patched cuCascade**: `cucascade-evict-priority.patch` (apply to NVIDIA/cuCascade at b09010a with `git apply`; local clone only, not pushed anywhere). What it does (about 50 lines in 6 files):
- `cached_chunk` gets an `std::atomic<uint8_t> priority` (fits the existing 32-byte layout); `fadvise(..., priority)` raises (never lowers) the priority of every chunk the request names; the evictor reclaims the lowest priority first and the oldest request first within a level (it used to walk requests in arrival order, so the chunks loaded first, which are the hot ones, went first, and a hit did not move a chunk back in line).
- `cache::config::eviction_batch_fraction` (default 0.25 = old behaviour). The evictor used to free 25% of all allocated chunks in one pass whenever the pool crossed its threshold; the glue sets 0.02.
- The glue sets a column's priority to 1 + 2 x rank (reads per full column) on every advised read.

SF1000, 2 iterations, 22/22 validated, held state (S3 bytes are per pass):
| config | cold | warm | cold pass | warm pass |
|---|---|---|---|---|
| unpatched, 1 MiB chunks, 64 conn (previous best) | 125.3 s | 108.3 s | 487 GB | 410 GB |
| patched, priority + sticky admission, 1 MiB, batch 0.05 | 129.0 s | 115.6 s | 470 GB | 387 GB |
| patched, priority, admit everything (no policy), 1 MiB | 138.5 s | 117.0 s | 628 GB | 418 GB |
| **patched, 256 KiB, 128 conn, batch 0.02 (default)** | **119.1 s, 111.6 s** | **104.6 s, 95.8 s** | 371 / 370 GB | 276 / 280 GB |
| patched, 256 KiB, 128 conn, batch 0.005 | 114.5 s | 104.4 s | 366 GB | 276 GB |
| patched, 512 KiB, 128 conn, batch 0.02 | 118.1 s | 111.2 s | 418 GB | 321 GB |
| bytecache (tuned kvikio) | 104.2 s | 87.1 s | 332 GB | 206 GB |
Priority alone barely moves the numbers; admitting everything is worse (refusing low-value columns matters); the win needs small chunks (small columns stop wasting cache, and l_suppkey / l_partkey get partly cached, as in bytecache) together with small eviction batches (otherwise each pass flushes a quarter of the cache, hot chunks first, and 256 KiB chunks thrash: 611 k evictions) and 128 connections per reactor (256 KiB requests are smaller). Run-to-run spread is about 7%.

## Committed defaults, measured with no environment overrides (SF1000, held state, 22/22 validated)
| run | cold | warm | S3 received |
|---|---|---|---|
| single pass 1 | 112.1 s | n/a | 413 GB |
| single pass 2 | 116.4 s | n/a | 399 GB |
| 2 iterations | 118.2 s | 104.1 s | 648 GB |

With the earlier env-var runs of the same settings (cold 111.6 and 119.1 s, warm 95.8 and 104.6 s): single pass 112 to 116 s (bytecache 104.1 to 104.6 s), 2-iteration cold 112 to 119 s (bytecache 104.2 s), warm 96 to 105 s (bytecache 87.1 s). Run-to-run spread about 5 to 9%.

## Larger cache: 120 GiB (SF1000, 2 iterations, held state, 22/22 validated, one run)
| cache | cold | warm | S3 cold pass | S3 warm pass |
|---|---|---|---|---|
| cuCascade 100 GiB (committed defaults, 3 runs) | 111.6 to 119.1 s | 95.8 to 104.6 s | 370 to 371 GB | 276 to 280 GB |
| cuCascade 120 GiB | 112.7 s | 92.5 s | 283 GB | 165 GB |
| bytecache 100 GB | 104.2 s | 87.1 s | 332 GB | 206 GB |

Bytes fall 25 to 40% (below bytecache's) but time barely moves, so the remaining 5 to 9 s gap to bytecache is not fetch volume. Per query the cold gap is 8.5 s in total, no query more than 1.5 s slower, and queries that read nothing from S3 are also slower (warm q3 3.8 vs 3.2 s and q4 2.4 vs 1.9 s at 0 GB on the wire), which points at a per-read cost on the cached path (per-read datasource duplicate and fadvise, 256 KiB chunk bookkeeping, copy out). Not yet profiled.
Cache size units: CUDF_POLARS_CUCASCADE_GB is GiB (100 = 107.4 GB); bytecache's CUDF_POLARS_BYTECACHE_GB is GB (100 = 100 GB), so the 100 GiB cuCascade cache is 7% larger than bytecache's.
Startup: creating the cache pins it all at engine start, measured at 29.7 s for 100 GiB (about 0.3 s per GiB, the same with 1 or 8 pinning threads); available host memory falls from 160 to 59 GB. This is not in the benchmark's total time, so process launch to end of the 22 queries is about 30 s longer than the tables for cuCascade. Sirius host tiers on this machine for comparison (one pass, LRU, not validated): 100 GB 136.5 s, 120 GB 129.9 s, 160 GB 120.0 s, 180 GB 108.9 s, 200 GB 96 to 109 s (405 to 438 GB read), 220 GB 100.9 s; the 22 queries need about 186 GB of distinct columns.

## Cached read path is not the gap (warm microbenchmark, host only)

warm_bench.py: 64 threads read 12 GB of SF100 lineitem ranges through the cuCascade datasource (20 GiB cache, 256 KiB chunks, 128 connections).
Pass 1 fills from S3, passes 2 and 3 are pure cache hits (0 wire bytes).

| range | fill (S3) | warm hits |
|---|---|---|
| 8 MiB | 2.3 GB/s | 77 to 86 GB/s |
| 1 MiB | 1.0 GB/s | 17.6 to 18.0 GB/s |

The SF1000 queries need about 3.6 GB/s on average (400 GB in 110 s), so a cached read is about 5x to 20x faster than needed.
Per-read steps (duplicate, fadvise, prepare) total 2 to 3 s of thread time over a whole 22-query run, also negligible.
So the 7 to 15 s gap to bytecache is not per-read overhead on hits. Remaining suspects: miss-path concurrency (bytes in flight while a
query waits), eviction and refetch volume, and the unit difference in cache size (cuCascade 100 GiB is 7% larger than bytecache 100 GB).

## Equal cache size to bytecache (93.1 GiB = 100 GB), SF1000, held state, 22/22 validated

| run | cuCascade 93.1 GiB | cuCascade 100 GiB (earlier) | bytecache 100 GB |
|---|---|---|---|
| single pass | 129.3 s, 471 GB from S3 | 112.1 / 116.4 s, 413 / 399 GB | 104.1 to 104.6 s, 332 GB |
| 2 iterations, cold / warm | 133.6 / 115.7 s | 111.6 to 119.1 / 95.8 to 104.6 s | 104.2 / 87.1 s |

One run each, so the run-to-run spread (5 to 9%) applies. Even so, the 7% smaller cache costs about 15% in time and 14% more S3 bytes
(471 vs 413 GB), a steeper cliff than the size change alone explains. At equal size cuCascade fetches 140 GB (42%) more than bytecache
per pass, so the gap is bytes, not per-read overhead (see the warm microbenchmark above). Likely causes: usable capacity lost to
budget_fraction 0.9, the 0.95 eviction threshold and chunk-aligned footprint, plus LRU eviction of pieces bytecache keeps pinned.

## Eviction threshold 0.99 (new default), SF1000, held state, 22/22 validated

Chunk occupancy runs above the admitted footprint, so with budget 0.9 and threshold 0.95 the evictor fired on data the policy had
admitted (about 400k chunk evictions per pass). Single passes at 93.1 GiB (one run each, spread 5 to 9%; time moves with S3 noise, bytes are steadier):

| budget / threshold | time | S3 bytes | evictions |
|---|---|---|---|
| 0.9 / 0.95 (old default) | 129.3, 142.4 s | 471, 468 GB | 404k |
| 0.8 / 0.95 | 126.6 s | 430 GB | 72k |
| 0.85 / 0.95 | 116.4 s | 423 GB | 117k |
| 0.87 / 0.95 | 122.5 s | 404 GB | 72k |
| 0.9 / 0.99 | 118.3 s | 390 GB | 75k |
| 0.93 / 0.99 | 121.8 s | 399 GB | 183k |
| 0.95 / 0.995 | 123.4 s | 423 GB | 320k |
| 1.0 / 0.99 | 125.2 s | 451 GB | 529k |

Default threshold is now 0.99 (budget stays 0.9). Two-iteration runs with the new defaults:

| cache | cold / warm | S3 total | before |
|---|---|---|---|
| 93.1 GiB (= bytecache 100 GB) | 117.3 / 100.2 s | 666 GB | 133.6 / 115.7 s, 821 GB |
| 100 GiB | 119.0 / 110.3 s | 628 GB | 111.6 to 119.1 / 95.8 to 104.6 s |

Clear gain at 93.1 GiB; at 100 GiB it is neutral within noise. bytecache at 100 GB is still ahead (104.2 / 87.1 s, 332 GB per pass).
The per-column report shows the aligned footprint is 1.4 GiB above the registered size per big column (256 KiB padding per chunk edge),
about 7 GiB of the 84 GiB budget is padding. That is the next lever (smaller chunks or tail-sharing).

## Hidden 10% of the pool: host reservation_limit_fraction (default 0.9)

cuCascade's host_memory_space_config.reservation_limit_fraction defaults to 0.9, so the usable pool (and the evictor's budget) is 90% of
the configured capacity. Measured with the new pool[...] line in fs_cache::summary(): at 93.1 GiB the eviction budget was 83.0 GiB
(0.99 x 0.9 x 93.1). The glue now sets it to 1.0 (env CC_RESV overrides). Single passes at 93.1 GiB, threshold 0.99, SF1000, one run each:

| admission budget | time | S3 bytes | evictions |
|---|---|---|---|
| 0.9 (default) | 118.0 s | 368 GB | 38k |
| 0.95 | 124.1 s | 365 GB | 75k |
| 0.98 | 134.8 s | 360 GB | 121k |

Before the change the same setting gave 385 to 390 GB. Higher budgets save a few GB but evict more, which costs more time than the bytes save.
Time per S3 byte is now the same as bytecache (0.32 vs 0.31 s/GB), so the remaining gap (368 vs 332 GB per pass) is bytes only.
Upstream: the 0.9 default means any cache size asked for is 10% smaller in practice, including Sirius host tier sizes.
cucascade-evict-priority.patch now also adds pool occupancy to fs_cache::summary().

## Staging shares the cache pool: why 0.99 was unsafe, and what did not help (SF1000, held state, 100 GiB unless stated)

The REST reactor stages every uncached device read (about 290 GB per pass, the columns the policy refuses to cache) through pinned blocks taken
from the same host pool as the cache chunks, and a staging allocation that finds no free block FAILS the read ("pinned staging exhausted").
Eviction starts when cache plus staging usage passes threshold x capacity, so the threshold must leave headroom for the requests in flight
(connections x GET size). With the hidden 0.9 reservation limit removed, threshold 0.99 left about 1% free, which is why prefetch, 256
connections and merged slices each segfaulted or raised out_of_memory in 1 of 2 runs. Default is now 0.97 (3% headroom, about 12k blocks).

| single pass, 100 GiB | time | S3 bytes | CPU user + system |
|---|---|---|---|
| threshold 0.95 | 114.6 s | 360 GB | 863 s |
| threshold 0.97 (default) | 108.8 s | 347 GB | 818 s |
| threshold 0.99 | 114.1, 114.2 s | 348, 345 GB | 827 s |
| 0.97 + adjacent miss slices merged to 512 KiB | 114.0 s | 354 GB | 850 s |
| 0.97 + merged to 1 MiB | 116.4 s | 348 GB | 779 s |
| 0.99 + merged to 2 MiB or 16 MiB | crash (staging exhausted) | | |
| 256 connections (0.99) | 116.7 s, then crash | | |
| 32 reactors x 128 | 134.5 s | | |
| prefetch on (0.99) | 137.0 s, then crash | | |

Findings:
- Every demand read is split into one slice per 256 KiB chunk position, and the reactor sizes a GET per slice, so there are about 1.5M GETs per
  pass (1.1M uncached, 0.4M fills). Merging adjacent uncached slices (CUCASCADE_MERGE_MISS_KB, off by default, in the patch) cuts CPU a little and
  does not change time: GET size is not what limits the cold pass.
- The Cython datasource releases the GIL for every call; it is not a Python bottleneck.
- Cached-read microbenchmark (warm_bench.py) shows 64 KiB to 1 MiB chunks and 16 to 128 threads all land at 13 to 21 GB/s for 1 MiB reads (Python
  thread pool bound), so it cannot isolate cache cost.
- The interleaved default vs 256-connection pair (114.1, 114.2 vs 116.7) shows run-to-run spread of about 2 to 3% at fixed bytes; the earlier
  111.5 to 119 s spread was mostly S3 noise.
- Same-day bytecache (100 GB, 2 iterations): 102.4 s cold, 89.4 s warm, 546 GB over both passes. cuCascade 100 GiB (7% more memory): 115.7 / 98.1 s,
  599 GB. The 13 s and 9 s gaps match the 10% extra bytes. bytecache at 120 GB does not fit this host state (memory guard stopped it at 19 GB free).
- Per-query cold times at 120 GiB: +1 s on most queries (broad), warm gap concentrated in Q12, Q14, Q17, Q19 (they read the refused columns
  l_partkey, l_commitdate, l_receiptdate).

Upstream candidates: reservation_limit_fraction default 0.9 (silently 10% smaller cache), staging allocated from its own pool (or waiting
instead of failing when the cache pool is full), and one slice per run of contiguous chunks in the device-read path.

## Committed defaults (threshold 0.97, reservation 1.0), 2 iterations, SF1000, held state, 22/22 validated both passes

| run | cold | warm | S3 total |
|---|---|---|---|
| cuCascade 93.1 GiB (= 100 GB) | 120.1 s | 105.2 s | 652 GB |
| cuCascade 100 GiB (107 GB) | 111.8 s | 98.4 s | 605 GB |
| bytecache 100 GB, same day | 102.4 s | 89.4 s | 546 GB |

At 100 GiB the gap is 9 to 10% in time and 11% in bytes. Starting point of this task was 196.6 s cold / 132.2 s warm. The goal (90 to 100 s cold)
is not met by either engine at 100 GB; bytecache is at 102 to 104 s. Next lever: a larger cache (Sirius ran a 200 GB host tier at 96 to 109 s),
which needs a driver reload to fit in this host state.

## Larger caches after a driver reload (fresh state: 245 GB free), committed defaults, SF1000

Both runs were stopped by the memory guard (host MemAvailable under 20 GB) in the second iteration of query 17, so queries 1 to 17 of both
iterations completed; Q18 to Q22 are taken from the same-day bytecache run to estimate a full pass (estimates, not measurements).

| cache | Q1-17 cold | Q1-17 warm | estimated full cold / warm | S3 bytes through Q17 |
|---|---|---|---|---|
| cuCascade 100 GiB | 77.7 s | 70.1 s | 107 / 95 s | |
| cuCascade 150 GiB (161 GB) | 74.1 s | 58.4 s | 103 / 84 s | 248 GB |
| cuCascade 186 GiB (200 GB, Sirius host tier size) | 78.1 s | 58.6 s | 107 / 84 s | 210 GB |
| bytecache 100 GB, same day | 73.5 s | 64.1 s | 102 / 89 s | |

- A bigger cache helps the warm pass (70 to 58 s for Q1-17) and does nothing for the cold pass: a first pass reads every distinct byte once,
  so the cold time is set by how fast first-touch data moves, not by capacity. Cold is about 102 to 107 s in every configuration here, bytecache included.
- The engine needs about 30 GB of host memory beyond the cache at its peak (second iteration of q17), so on this 249 GB host the largest usable
  cache is about 140 to 145 GiB; a 186 GiB cache does not fit (Sirius fits it because it uses host memory differently).
- Each run leaves about 35 GB held by the driver (host used 3 -> 39 -> 74 GB), so a reload buys about one large run.
- Cold network profile: median 2 to 3.6 GB/s per 5 s window against a raw S3 capacity well above that, so the lever for the cold pass is
  overlapping first-touch I/O with compute, which bytecache's own read-ahead experiments did not improve either.

## Where the remaining cold gap is (same day, 100 GiB vs bytecache 100 GB)

Per-query comparison, one cold iteration each:

- Q13 alone cold: cuCascade 8.7 s, bytecache 9.8 s. Q10,12,13 in sequence: 25.1 s vs 24.6 s. Q1 three alternating repeats: 11.07 s vs 10.93 s. Q1 to Q13 in sequence: cuCascade 4.8 s slower in total, about 0.4 s per query. The cold read path is level with bytecache; the single-query gaps in earlier full runs (Q13 +4 s, Q12 +3.8 s) did not repeat.
- Full single pass, three cuCascade runs: 114.6, 110.8, 112.7 s (and 116.0 with a tight stall detector) against bytecache 107.2 s. The gap is 3 to 7 s, about the size of run-to-run noise plus one random 3 to 4 s query stall per run.
- Admitted footprint (about 90 GB) covers the same hot columns bytecache keeps, and the 107 GB cache is larger than bytecache's 100 GB, so capacity is not the explanation at this size.
- The stalls show up as seconds of 0.1 GB/s network. The REST stall detector (CC_STALL_KB, CC_STALL_S, opt-in) at 512 KiB/s for 2 s did not remove them (116.0 and 112.7 s vs 110.8 s default), so they are not slow GETs that the detector can retry.
