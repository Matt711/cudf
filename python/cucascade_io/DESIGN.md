# cuCascade REST backend for cudf-polars (prototype)

Reads `s3://` parquet through cuCascade's REST backend and prefetching cache (`cucascade::io::datasource`, a `cudf::io::datasource`) instead of the byte cache in `python/bytecache`. Enabled with `CUDF_POLARS_CUCASCADE_GB` (cache size); off by default. Goal: reproduce bytecache's SF1000 numbers (104 s cold, 87 s warm) with cuCascade, then upstream.

Build: `python/cucascade_io/build.sh` (needs a cuCascade build, see the script variables, and liburing). Settings: `CUDF_POLARS_CUCASCADE_REACTORS` (default 2; 16 is much better), `_CONNECTIONS` (per reactor, 64), `_ADVISE_MIN_BYTES`, `_EVICTION` (lru|idle), `_VERBOSE`.

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
