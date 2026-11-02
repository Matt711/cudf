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
