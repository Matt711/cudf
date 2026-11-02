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
