# bytecache: a cross-query byte-range cache for cudf-polars S3 reads (prototype)

Goal: cold TPC-H SF1000 from S3 in about 90 to 100 s on a g7e.8xlarge. Measurements that drive the design
(see tasks/tpch-sf1000-baseline/results/RESULT.md, addenda 7 to 14):

- cudf-polars reads 1065 GB per 22-query run for 186 GB of distinct columns; Sirius with a 200 GB host cache reads about 410 GB and runs 96 to 109 s.
- Re-reading is concentrated: l_extendedprice alone is 40% of reads. Simulation at column granularity: an online frequency-aware policy
  (LRU-2 / LFU) with 75 GB reads about 475 GB, with 100 GB about 322 GB; plain LRU reads 719 GB at 75 GB.
- pylibcudf cannot take a Python-written data source, hybrid scan only covers scans with a predicate, so the cache is a C++ `cudf::io::datasource`
  plugged into `SourceInfo([Datasource])` through a small Cython class. No libcudf rebuild.

Design (v1):
- Process-wide block cache: anonymous mmap arena, fixed-size blocks (default 1 MiB), key (file id, block index).
- Policy: segmented LRU (probation list for blocks seen once, protected list for blocks seen again). Evict probation first. This approximates LRU-2.
- Misses: consecutive missing blocks are fused into GETs of up to 16 MiB through kvikio `RemoteHandle::pread` (the MULTI_POLL reactors cudf-polars already configures),
  into a thread-local staging buffer, then copied into cache slots and into the caller's buffer. Concurrent readers of a block wait for the single fetch.
- Footers are ordinary blocks, so file metadata is also cached across queries.
- Objects are assumed immutable (no ETag check). Counters are exported for experiments.

Not in v1: pinned arena, direct device reads, readahead/prefetch, ETag validation, multi-GPU.
