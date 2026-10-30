# bytecache: a cross-query byte cache for cudf-polars S3 reads (prototype)

Status: working prototype for benchmarking. It is not proposed for upstream as written. Everything below is
measured on one machine (g7e.8xlarge: 1x RTX PRO 6000 Blackwell 97 GB, 32 vCPU, 249 GB RAM, 100 Gbps NIC,
us-east-2, TPC-H SF1000 from `s3://rapids-tpch/tpch-rs/scale-1000`), one run per setting unless a range is given.
Cold S3 runs vary by about 6 to 7% between repeats, so differences smaller than that are noise.

## 1. Goal and the question this answers

Goal: run all 22 TPC-H queries at SF1000, cold from S3, in about 90 to 100 s with cudf-polars on this instance
("speed of light" on AWS). Sirius, a GPU DuckDB extension, is the reference: on this same instance its best cold
runs are 96 to 109 s (the Sirius team reports 100 to 101 s).

Where cudf-polars started (main + three open PRs, tuned kvikio settings): about 174 s cold from S3, 118 s from
local NVMe. This document describes the first big step: a cache that lets later queries reuse bytes read by
earlier ones.

## 2. What the measurements say (why a cache)

All numbers from `tasks/tpch-sf1000-baseline/results/RESULT.md` (addenda 7 to 15).

- **Re-reading dominates.** The 22 queries read 1065 GB from S3 for only 186 GB of distinct columns (5.7x).
  Reads equal the projected columns exactly (1061 GB needed vs 1065 GB read), so projection works.
- **Filters cannot skip data in this layout.** Parquet min/max statistics keep 100% of row groups for the date
  filters in q4, q5, q6, q12, q14 (each row group spans the whole date range). So the only lever is not
  re-reading the same columns.
- **Re-reading is concentrated.** `lineitem.l_extendedprice` alone is 40% of all reads (12 queries use it);
  the top four lineitem columns (85 GB) account for about 667 GB of savable reads.
- **Sirius gets there with a big host cache.** With a 200 GB host tier it reads about 410 GB and runs 96 to
  109 s cold; with 100 to 120 GB it reads 850 to 980 GB and runs 137 to 141 s.
- **Simulation of policies** (column granularity, queries in order q1..q22): at 75 GB, plain LRU reads 719 GB,
  LRU-2 / LFU about 475 GB, the oracle static pin set 438 GB. At 100 GB: LRU 398, LRU-2 322, oracle 320.
  A frequency-aware policy is within about 8% of the oracle; plain LRU is not good until about 100 GB.
- **Bytes become time.** Local NVMe, 75 GB hot column set kept in the page cache, one process per query: the sum
  of query times fell from 356.5 s to 151.3 s as bytes read fell from 1071 GB to 373 GB (0.28 s saved per GB,
  which is the NVMe's own speed; S3 is about 3x faster so compute becomes the limit).
- **The instance is healthy.** PCIe Gen5 x16 (55.6 GB/s host to device), no throttling, the NIC reached 12.5 GB/s.

## 3. Design

### 3.1 Where it plugs in

pylibcudf cannot take a Python-written data source (`SourceInfo` accepts paths, host buffers, or a C++-backed
`Datasource`), and hybrid scan only covers scans with a predicate. So the cache is a C++
`cudf::io::datasource` subclass exposed to Python through a small Cython class, with no libcudf rebuild:

- `bytecache.hpp / bytecache.cpp`: the cache and `bytecache::cached_source` (a `cudf::io::datasource`).
- `_bytecache.pyx`: Cython `CachedDatasource(Datasource)`, plus `configure()`, `get_stats()`, `reset_stats()`, `clear()`.
- `cudf_polars/utils/bytecache.py`: `wrap_sources(paths, sizes)` returns cache-backed sources when enabled.
- `cudf_polars/dsl/ir.py` (`Scan.do_evaluate`, parquet branch) and `cudf_polars/dsl/utils/io.py`
  (`_prefetch_parquet_footers_for_paths`) call `wrap_sources` for remote paths. Everything else is unchanged.
  With the environment variable unset the standard path is used.

### 3.2 The cache

- **Storage.** One anonymous `mmap` arena (`MAP_NORESERVE`, so pages are touched lazily), divided into fixed
  blocks (default 1 MiB). Key = (file id, block index). File id comes from a process-wide table of remote
  objects, each holding one `kvikio::RemoteHandle` (so a path is HEAD-ed once per process).
- **Policy.** Segmented LRU: a block seen once sits in a probation list, a block seen again moves to a protected
  list. Eviction takes the oldest probation block first, then the oldest protected block. This approximates the
  LRU-2 policy that the simulation favoured and resists scans (a single pass over a big column does not flush the
  hot columns). Blocks being read or fetched are pinned and never evicted.
- **Misses.** Consecutive missing blocks are fused into one GET of up to 16 MiB through
  `kvikio::RemoteHandle::pread`, using the MULTI_POLL reactors that cudf-polars already configures. The bytes
  land in a staging buffer, are copied into the cache slots and into the caller's destination. Concurrent
  readers of a block that is being fetched wait for that one fetch (no duplicate GETs).
- **Footers** are ordinary blocks, so parquet footers (and the HEAD for the object size) are fetched once per
  process, not once per query.
- **Assumptions.** Objects are immutable (no ETag check); one process owns the cache (no sharing between ranks).

### 3.3 Reading and overlap (the part that took several tries)

- libcudf's parquet reader overlaps remote reads only for sources that prefer device reads
  (`is_device_read_preferred`), by calling `device_read_async`. For other sources it calls the **synchronous**
  `host_read` on its small shared host worker pool, which caps concurrency (first attempt: 0.2 GB/s, q8 took
  204 s). So `cached_source` supports device reads and runs them on its own thread pool.
- Each pool job fetches, stages the bytes in a pinned buffer, copies to the destination on the caller's stream
  and synchronises before its future completes. Many jobs are in flight at once; threads are only parked in
  I/O, so the pool is large (default 512, `BYTECACHE_THREADS`).
- **Memory is bounded independently of threads.** GET staging buffers come from a counting budget
  (`BYTECACHE_GET_BUFFERS`, default 256 x 16 MiB), reserved all-or-nothing per read (no hold-and-wait, so no
  deadlock), in waves when one read needs more than the budget. Host-to-device copies go through a shared pinned
  pool (`BYTECACHE_PINNED_BUFFERS`, default 128 x 16 MiB) in 16 MiB pieces. An earlier version kept a pinned
  buffer per thread and allocated GET buffers without limit: resident memory reached 232 GB at 100 GB cache and
  1024 threads, which exhausted host RAM and took the machine down.

### 3.4 Build and ABI notes

- `build.sh` builds `_bytecache.abi3.so` against the conda env (libcudf, kvikio, pylibcudf). pylibcudf is built
  as a limited-API (`abi3`) Cython module, so this module must be too (`-DCYTHON_LIMITED_API=1
  -DPy_LIMITED_API=0x030A0000`); a regular build crashed with a type refcount underflow when instances of the
  `Datasource` subclass were freed. It also needs `-DKVIKIO_CUFILE_FOUND -DKVIKIO_CUFILE_VERSION_API_FOUND
  -DKVIKIO_LIBCURL_FOUND`, `-I$ENV/include/rapids` for the newer CCCL, and links `-lcudf -lkvikio -lcudart`.
- The Python `CachedDatasource` objects must outlive the reads that use them (`SourceInfo` holds raw pointers);
  `Scan.do_evaluate` keeps them in a local variable.
- `BYTECACHE_DEBUG=1` installs a SIGSEGV handler that prints a native backtrace.

## 4. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `CUDF_POLARS_BYTECACHE_GB` | unset (off) | Cache size in GB; `0` routes through the source with caching off |
| `CUDF_POLARS_BYTECACHE_PATH` | - | Directory containing `_bytecache*.so` |
| `CUDF_POLARS_BYTECACHE_BLOCK_MB` | 1 | Block size |
| `CUDF_POLARS_BYTECACHE_GET_MB` | 16 | Largest fused GET |
| `BYTECACHE_THREADS` | 512 | I/O threads |
| `BYTECACHE_GET_BUFFERS` | 256 | GET staging buffers (x GET size) |
| `BYTECACHE_PINNED_BUFFERS` | 128 | Pinned copy buffers (x 16 MiB) |
| `BYTECACHE_DEBUG` | unset | Native backtrace on SIGSEGV |

Recommended companions (measured to help without the cache too): `CUDF_POLARS__EXECUTOR__KVIKIO_REACTOR_COUNT=48`,
`CUDF_POLARS__EXECUTOR__KVIKIO_REQUEST_CEILING=512` (cold S3 215 s to 174 s); `KVIKIO_NTHREADS=16` for local
files (cold local 234 s to 118 s). Memory: budget about cache size + 20 GB for staging + the engine's own host
memory (57 to 131 GB RSS seen without the cache). Do not use a cache so large that the sum exceeds RAM.

## 5. Results so far (cold from S3, 22/22 queries validated)

| Setup | Total | S3 bytes |
|---|---|---|
| cudf-polars, default kvikio | 208 to 233 s | 1065 GB |
| cudf-polars, tuned kvikio (48 reactors / 512) | 171 to 181 s | 1065 GB |
| + bytecache 100 GB, 128 I/O threads (first working version) | 222 s | 524 GB |
| + bytecache 100 GB, 512 I/O threads (per-thread buffers) | 145.0 s | 520 GB |
| + bytecache 100 GB, 1024 I/O threads (per-thread buffers) | 139.0 s | - |
| Sirius main, best config (200 GB host tier) | 96 to 109 s | about 410 GB |

The 512- and 1024-thread numbers were taken with the older unbounded-staging build (before the bounded-memory
change in section 3.3); results with the bounded build are being collected. Hit rate at 100 GB was about 60%
of requested bytes (634 of 1065 GB).

## 6. Known limitations

- The network is under-used: during the cache runs it averaged about 3 GB/s (peak 9 to 12 GB/s, NIC limit
  12.5 GB/s). Reads are issued on demand, not ahead of need, so the engine still waits for data.
- Hit rate at 100 GB (about 60%) is below what a column-aware policy should reach; the block-level SLRU does
  not know which columns are reused. q13 got slower with the cache (13.8 s vs about 7.7 s) and is unexplained.
- Even a perfect cache leaves cudf-polars at about 109 s (the warm local pass), against about 88 s for Sirius,
  so the last 10 to 20 s needs compute work on the slow queries (q13, q7, q21, q18, q9).
- No ETag or version check; single process only; no direct-to-device path (data is staged in pinned memory).
- Not tested: the Ray and Dask engines, multi-GPU, anything other than the SPMD engine and parquet on S3.

## 7. Next steps

1. Repeats and a cache-size sweep with the bounded build, inside a memory guard (a run is stopped if available
   RAM falls below 20 GB).
2. **Read-ahead** (the design the DuckDB asynchronous I/O post describes, and what Sirius does): start fetching
   the byte ranges of upcoming scan tasks while the GPU works on the current ones, limited by a byte budget
   rather than a thread count, so I/O overlaps compute.
3. A policy that keeps the heavily reused columns (frequency-aware, size-aware) to reach the simulated hit
   rates with 75 to 100 GB.
4. Then compute work on the slowest queries if the goal still is not met.

## 8. How the numbers were produced

The harness is `python -m cudf_polars.streaming.benchmarks.pdsh --frontend spmd --io-mode lukewarm` with
`--validate-directory` against the expected answers; one iteration; the GPU is checked idle before every run;
the NIC is sampled once per second from `/proc/net/dev`; the bytecache counters are written to
`kvikio-settings.json` per run. The benchmark scripts and all raw results are in
`tasks/tpch-sf1000-baseline/{scripts,results}` on the benchmark host (not part of this branch). The column/policy
simulation is `tools/column_cache_sim.py` (it expects `results/needed-bytes.json`, produced by the per-query
column extraction on that host, and the local parquet copy). Integrity test:
`selftest.py` (in this directory; usage `python selftest.py <cache MiB> <threads>`, needs S3 credentials in the environment) compares cache reads with standard reads on an SF1 object, single-threaded and
with 6 concurrent readers, with and without eviction, with default and very tight staging limits.
