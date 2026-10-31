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
- **Policy (column-aware).** Every block is attributed to the column its first byte belongs to (section 3.5). A column's
  priority is how many times its full size has been requested so far (requested bytes / registered bytes): caching a
  column saves its whole size on every later read, so the saving per cached byte is (reads - 1). When the cache is full
  a new block is admitted only if its column outranks the lowest-ranked resident column, which then gives up its
  highest-offset block (so a partly cached column keeps a stable prefix and a cyclic scan of it still hits, instead of
  thrashing). Otherwise the block bypasses the cache. Bytes that belong to no known column (footers, metadata) rank
  highest. The earlier policy, a segmented LRU over blocks, is still available with `BYTECACHE_POLICY=slru`; it
  thrashes under repeated sequential scans (20% hit rate at 75 GB, versus 57% with the column policy).
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
  (`BYTECACHE_GET_BUFFERS`, default 512 x 16 MiB), reserved all-or-nothing per read (no hold-and-wait, so no
  deadlock), in waves when one read needs more than the budget. Host-to-device copies go through a shared pinned
  pool (`BYTECACHE_PINNED_BUFFERS`, default 512 x 16 MiB) in 16 MiB pieces. An earlier version kept a pinned
  buffer per thread and allocated GET buffers without limit: resident memory reached 232 GB at 100 GB cache and
  1024 threads, which exhausted host RAM and took the machine down.

- **Large reads run their GETs in parallel.** libcudf merges adjacent column chunks, so one read can be hundreds
  of MB (q2 reads the 716 MB `supplier/part.0.parquet` as a single range). An earlier version copied large
  reads through the pinned buffer 16 MiB at a time, one piece after another, so that read cost 45 sequential GET
  latencies (15 s; q2 took 18 s instead of 4.5 s). Now a read larger than one pinned buffer is fetched whole into
  a temporary buffer (all its GETs in flight together, as the standard kvikio path does with its 16 MiB tasks)
  and then copied to the device through pinned pieces. The temporary buffers are bounded by `BYTECACHE_BIG_MB`
  (default 4096 MiB); a read bigger than that budget falls back to the piece-by-piece loop.
- **The pinned pool is created at start-up.** `cudaMallocHost` takes about 5 ms on an idle GPU but about 1 s
  while kernels are running, so creating pinned buffers lazily inside a query stalled reads. `configure()` now
  pins all buffers once (about 8 GiB, a few seconds); `BYTECACHE_PREPIN=0` turns that off.
- **Diagnostics.** `get_stats()` also reports per-stage thread-time (`ns_open`, `ns_pinned_wait`, `ns_read`,
  `ns_h2d`, `ns_budget_wait`, `ns_queue_wait`), `device_reads`, `max_inflight`, `host_reads`/`ns_host_read`;
  `get_timeline()` gives per-second submitted/completed device reads; `get_slow_reads()` lists device reads whose
  whole lifetime exceeded 1 s, split into queue/pinned/read/h2d milliseconds. These found the large-read problem:
  the timeline showed 3123 of 3124 reads finishing in 2 s and one straggler finishing at 13 s.

### 3.5 Metadata store (column layout)

pylibcudf's hybrid reader returns the exact compressed byte range of every column chunk for chosen columns and row
groups (`HybridScanReader.all_column_chunks_byte_ranges`), computed from the footer the engine already parsed (about
0.01 s per file, so no footer parser of our own). `cudf_polars/utils/bytecache.py` registers each remote file once
(`register_layouts`, or `register_layouts_from_sources` when the engine did not prefetch footers: it then reads the footer
through the cache-backed source, so the footer blocks stay cached). The C++ cache keeps, per file, the sorted
(start, end, column) segments and, per column ("<table>|<column>"), counters: registered, requested, hit and fetched
bytes and resident blocks (`get_column_stats()`). This is the per-column hit-rate summary used in section 5, and it is
the input the read-ahead needs (which byte ranges an upcoming scan task will read). In the SF1000 runs 99.6% of
requested bytes are attributed to a known column.

### 3.6 Read-ahead (implemented, off by default: it did not help)

`prefetch_columns(path, table, columns)` fetches the registered byte ranges of the chosen columns into the cache on its own
thread pool (`BYTECACHE_PREFETCH_THREADS`, default 256, one job per range piece of at most 64 MiB) with its own staging
budget (`BYTECACHE_PREFETCH_BUFFERS`, default 256), skipping blocks the policy would not admit and not counting towards request
or hit statistics. `cudf_polars/utils/bytecache.py:ScanPrefetcher` runs per scan node: it opens and registers every file of
the node in parallel, then stays a window (`CUDF_POLARS_BYTECACHE_AHEAD_TASKS`, default 24; `..._AHEAD_GB`, default 12)
ahead of the newest task a producer has started (`scan_node` calls `advance(task_idx)`). Enable with
`CUDF_POLARS_BYTECACHE_READAHEAD=1`.

Measured (100 GB cache, cold S3, SF1000): q1 alone fetches 58 GB either way; with read-ahead 37.5 GB of it is prefetched and
hit, q1 goes from 17.7 s to 15.3 s. Over all 22 queries read-ahead did not help: 131.3 s (98 GB prefetched, 81%
hit rate, same 335 GB fetched) against 128 to 134 s without in later repeats (the 116.7 s that it was first compared with was a low outlier; the conclusion that it does not help stands) (an earlier version that registered files one at a time prefetched
only 5 GB: 126.0 s). Prefetching earlier moves the same bytes sooner but makes demand reads compete for the same
network, CPU and threads. Scan concurrency cannot be raised instead: 12 or 16 concurrent remote scan tasks made the run
145.7 s and 176.8 s, because each task allocates its device buffers when its read starts and holds them while it waits.

Where the limit is (fetch microbenchmark, `scripts/fetch_microbench.py` on the benchmark host, no GPU work): plain kvikio
reading 64 GB of 4 MB ranges from 60 lineitem objects reaches 2.0 GB/s with 64 threads, 7.8 with 256, 10.9 with 512 and
11.3 with 1024; this cache's prefetch path fetched q1's 57 GB in 6.3 s (9.0 GB/s). So the network and the cache can move
q1's bytes in about 6 s, but inside the engine q1 takes 15 s at about 5 GB/s. The stage timers show device reads waiting
about 56 ms in the 512-thread pool queue and then 81 ms in the read (3 ms for the GPU copy), so the pool of blocking
reads is saturated and per-read latency inside the engine is about 1.6x the microbenchmark's.

### 3.7 Concurrent footer warming (the largest single gain after the cache)

Before any data moves, the engine opens every file of a scan (an HTTP HEAD for the size) and reads and parses its footer.
That ran one file at a time: about 2 s per 60-file table on every query with the standard path, and about 5 s the first time
with the cache. `bytecache.wrap_sources` now first calls `warm_footers(paths)` (C++, GIL released, on the prefetch pool): it
opens all not-yet-seen files and reads their last 1 MiB into the cache concurrently. Later queries then find the footer
bytes cached (0.3 s per 60-file table). Cold S3, 100 GB cache, SF1000, three runs: **106.9, 108.1, 107.3 s (mean 107.4 s)**
against 131.3, 128.0, 133.7 s (mean 131.0 s) without it (q1 16.4 -> 10.9 s, q3 6.5 -> 4.7 s). Also in this change: the column
layout of a file is now computed with one hybrid-reader call (chunks come back row-group-major), 14x faster than one call per column.

Decomposition (cold S3, SF1000, 22/22 validated): tuned no-cache 171 to 181 s; cache read path with capacity 0 (nothing cached, 1078 GB
fetched) 179.9 s; **1 GB cache (footers cached, 1% data hit rate, 1144 GB fetched) 161.3 s**; 100 GB cache without footer warming 128 to 134 s; 100 GB cache with
footer warming 107.4 s. So footer/open work is worth about 20 s with or without a data cache (180 -> 161 at 1 GB, 131 -> 107 at 100 GB) and the data cache about
50 s (161 -> 107 with footers warmed): roughly additive, 180 -> 107 s in total. Read-ahead on top of the new build: 113.2 s (one run), still no gain.

Cache size with the footer-warming build (100 GB: 107.4 s, 72% hit rate, 336 GB fetched; 75 GB: 135.1 s, 59%, 489 GB; 50 GB: 136.4 s, 43%, 672 GB) shows a
cliff between 75 and 100 GB; process peak memory is 155, 176 and 205 GB. Raising the engine's device spill soft limit to 90% or 95% (`RAPIDSMPF_SPILL_DEVICE_LIMIT`)
removed most spilling in q7, q9 and q13 but the total got worse (117.4 s, 118.1 s), so spilling is not what limits those queries.

Why the reads are bursty (code): `reader_impl::read_compressed_data()` issues the async reads for every column chunk of a pass, then blocks on
`read_chunks_tasks.get()`, and only then decodes; cudf-polars defaults `pass_read_limit` and `chunk_read_limit` to 0 (no limit), so one pass is the whole
scan task (all row groups of its file(s), about 660 column chunks for a q1 task), and nothing issues the next task's reads while this one decodes.
Setting `CUDF_POLARS__PARQUET_OPTIONS__PASS_READ_LIMIT=2GiB` makes q1's I/O smooth (no idle seconds, steady 5.5 to 8.5 GB/s) but the full run is worse
(122.7 s; 128.5 s with 8 GiB; 107.4 s default), because smaller passes slow the other queries (q4 4.2 -> 6.0 s, q13 6.2 -> 7.8 s, q21 9.9 -> 11.9 s). The limit is global.

Hybrid scan (`CUDF_POLARS__PARQUET_OPTIONS__USE_HYBRID_SCAN=true`, used only for split tasks of scans with a pushed-down predicate) now reads through the cache
(`_read_with_hybrid_scan` uses `bytecache.wrap_sources`) and validates 22/22 at SF1000, but is slower: 148.5 s with the 100 GB cache (107.4 s standard), 225.5 s with
no cache (171 to 181 s standard), 133.9 s with the cache and the internal stats-pruning step off. Reads, spill volumes and memory waits are close to the standard path's,
so the cost is per-task overhead in the hybrid path (the stats step alone is about 15 s on this data, where it can never prune a row group). It would need that
overhead removed before it could serve as the hook for a scan manager. Gated read-ahead (prefetch only while at most 64 or 256 demand reads are running or queued,
`BYTECACHE_PREFETCH_GATE`) also did not help: 116.2 s and 117.5 s against 107.4 s (86 and 100 GB prefetched).

What the q1 profile showed (per-second cache counters): reads are issued in bursts of 1000 to 1900 (one scan task's reads), which
saturate the 512-thread pool for about a second, and the pool then drains to zero running reads while the tasks decode, so
the network is idle roughly half the time even though a plain kvikio microbenchmark reaches 10.9 GB/s at the same
concurrency. Read-ahead fills those gaps (pool saturated from t=4 s on, q1 14.1 s vs 16.1 s) but at that point started
about 4 s late, because registering the layouts of 60 files took that long (footer parsing in `read_parquet_footers` holds the GIL).

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
| `CUDF_POLARS_BYTECACHE_BLOCK_MB` | 0.25 | Block size in MiB (see section 5: smaller blocks fetch fewer bytes) |
| `CUDF_POLARS_BYTECACHE_GET_MB` | 16 | Largest fused GET |
| `BYTECACHE_THREADS` | 512 | I/O threads |
| `BYTECACHE_GET_BUFFERS` | 512 | GET staging buffers (x GET size) |
| `BYTECACHE_PINNED_BUFFERS` | 512 | Pinned copy buffers (x 16 MiB) |
| `CUDF_POLARS_BYTECACHE_READAHEAD` | 0 | `1` enables read-ahead (section 3.6) |
| `BYTECACHE_PREFETCH_THREADS` / `_BUFFERS` | 256 / 256 | Read-ahead pool and staging budget |
| `BYTECACHE_BIG_MB` | 4096 | Budget for temporary buffers of reads larger than one pinned buffer |
| `BYTECACHE_PREPIN` | 1 | Pin the staging pool at start-up (`0` = lazily) |
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
| + bytecache 100 GB, 512 threads, bounded memory, pre-pinned, parallel large reads (block-LRU policy) | 135.2 to 150.1 s (four runs, mean 140.6) | 512 to 521 GB |
| same, 75 GB cache (block-LRU policy) | 158.4 and 172.9 s | 998 to 1023 GB |
| **+ column-aware policy, 100 GB** | **125.6 s** (72% hit rate) | **459 GB** |
| + column-aware policy, 75 GB | 146.6 s (57% hit rate) | 692 GB |
| column policy, 100 GB, **256 KiB blocks** (now the default) | 116.7, 131.3, 128.0, 133.7 s (four runs, **mean 127.4 s**, spread 117 to 134 s) | **336 GB** |
| column policy, 100 GB, 128 KiB blocks | 125.0 s (73% hit rate) | 309 GB |
| **+ concurrent footer warming (256 KiB blocks, 100 GB)** | **106.9, 108.1, 107.3 s (mean 107.4)** | 336 GB |

The three block sizes (1 MiB, 256 KiB, 128 KiB) fetch 459, 336, 309 GB with the same hit rate (the rest was over-read: whole
blocks fetched for sparse reads). The time effect is not established: the first 256 KiB run (116.7 s) was a low outlier,
three repeats gave 128.0 to 133.7 s, 1 MiB blocks gave 125.6 s in one run and 128 KiB 125.0 s in one run. Cold runs vary
by about +/-7%, so only differences larger than that mean anything; fewer bytes from S3 is a clear gain, but it has not
yet shown up as a clear gain in time.
| Sirius main, best config (200 GB host tier) | 96 to 109 s | about 410 GB |

The 145.0 s and 139.0 s rows were taken with the older unbounded-staging build. With the bounded build: 100 GB,
512 threads, 512 GET / 512 pinned buffers: 150.1 s (peak RSS 201 GB); 1024 threads, 1024/1024: 157.5 s (215 GB);
with too-small limits (256 GET / 128 pinned) it took 248 to 250 s because reads in flight were capped, and a
75 GB cache with those limits took 371 s (hit rate 24%). After the pre-pinned pool and the parallel
large-read fix (section 3.3) the 100 GB run takes 140.1 s (22/22 validated, 60% hit rate, 521 GB from S3,
NIC average 3.5 and peak 9.0 GB/s, peak RSS 202 GB); q2 fell from 14 s to 1.9 s. Hit rate at 100 GB was about 60%
of requested bytes (634 of 1065 GB).

## 6. Known limitations

- The network is under-used: during the cache runs it averaged about 3 GB/s (peak 9 to 12 GB/s, NIC limit
  12.5 GB/s). Reads are issued on demand, not ahead of need, so the engine still waits for data.
- Remaining waste is over-reading: some columns are fetched far more than requested (at 100 GB `l_comment` 25 GB
  requested, 83 GB fetched; `ps_comment` 2 GB, 7 GB), probably sparse page reads against 1 MiB blocks. q13 took 13.8 s in one early run and about 7 s in later ones (not investigated).
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

## 9. Update 2026-10-31: the machine state matters, and the scan-level ideas did not help

### 9.1 The 107 s result depends on the driver holding pinned host memory

The same code and configuration (100 GB cache, 512 threads, 48/512 kvikio, 8 producers) gives different totals depending on
how much pinned host memory the NVIDIA driver already holds when the run starts:

| state at start | full 22-query run |
|---|---|
| fresh (right after an NVIDIA driver reload, about 3 GB host used) | 115.1 s (29f0c8dca9), 118.6 s (c27344af70) |
| held (80 to 104 GB held by the driver from earlier runs) | **104.5 s** (two runs, 104.52 s each, 22/22 validated, 0 GB growth) |

The difference is the queries that spill into rapidsmpf's pinned host pool (q18 10.8 s vs 18.8 s, q7 6.7 vs 9.8 s, q13). The
earlier 107.4 s (106.9, 108.1, 107.3 s) was measured in the held state. Getting fresh pinned memory costs about 0.3 s per GiB
(16 GiB took 4.9 to 5.4 s). Total spill time was about 60% higher in the fresh state, with identical cache counters.

What is held: after a clean exit, about 80 GB of host memory (50 to 80 GB per run from a fresh state) stays allocated by the driver. It shows as
"used" in `free`, belongs to no process, is not in any kernel accounting category (MemFree drops, Slab/AnonPages/Cached/Mapped do not
rise, `nr_foll_pin_acquired` equals `nr_foll_pin_released`) and is released only by reloading the NVIDIA kernel modules
(`rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia; modprobe nvidia`). Stopping `nvidia-persistenced` or unloading only
`nvidia_uvm` freed nothing (2 GB). `gc.collect()` and `cudaDeviceReset()` before exit freed nothing (117.4 to 117.2 GB in-process, 80 GB
still held after exit). It is not caused by the byte cache: with the cache off, a full run (185.3 s, 22/22 validated) left the same
80 GB. A pool-only rapidsmpf test (allocate, touch and free 20 GiB, four exit modes) did not reproduce it, and q1 with the cache on or
with rapidsmpf pinned memory off leaks 0 to 1 GB; the heavy cases are the queries that spill (q13, q18, q21). Capping
`RAPIDSMPF_PINNED_MAX_POOL_SIZE` bounds the amount (q18: 4 GiB cap, 0 GB held, 40 s instead of 23.7 s). Not found: why the pages are not
returned at exit. Related external report (different platform and driver, not shown to be the same bug): a clean-exit memory hold on
driver 590.48.01 cleared by unloading the module. Untested: `RAPIDSMPF_PINNED_INITIAL_POOL_SIZE` pre-sizing, which would move the commit cost to
engine startup.

### 9.2 Scan-level ideas tried against the 104.5 s held-state baseline (all validated 22/22, none helped)

| change | total |
|---|---|
| baseline, 8 producers | 104.5 s |
| scan manager (fetch ahead, decode after data is host-resident), 6 producers | 114.6 s |
| scan manager, 4 producers | 121.5 s |
| 12 producers | q13 out of memory (not valid); q1 25 s, q9 22 s |
| decode credits K=4 | 106.2 s |
| decode credits K=2 | 108.9 s |

Fewer producers help only when compute is the bottleneck (warm q1 4.0 s at 4 producers vs about 10 s at 8; q7, q13) and lose 1 to 2 s on
the scan-bound queries. All of the above are off by default: `CUDF_POLARS_SCAN_MANAGER=1` (with
`CUDF_POLARS_SCAN_MANAGER_PRODUCERS`) and `CUDF_POLARS_SCAN_CREDITS=K` (a producer may start decoding only while fewer than K decoded chunks wait
for the downstream actor). q1 alone with credits K=2: cold 11.1 s, warm 6.2 s vs 8.6 s without. An earlier "manager slower" result
(128 s) was partly my error (16 cache threads); the measurements above use 512.

### 9.3 Per-query I/O profile (one process per query, 2 iterations, 100 GB cache)

Cold sums to 273.6 s because each query starts with an empty cache; the warm second pass (no S3 fetch) sums to 84.0 s (fresh state).
That is the no-S3-I/O estimate. The read, copy and decode of 57 GB of q1 columns from the host cache runs at about 15 GB/s (3.7 s); the
rest of q1's warm 9.5 s is compute and contention. Quent traces (`cudf_polars_quent`, built from `quent/bridge` into its own env)
show warm q1 is compute-bound: GroupBy and HStack take 3.5 to 6 times longer per task when 8 decodes run beside them.
