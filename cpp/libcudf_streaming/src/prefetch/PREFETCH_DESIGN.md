# cudf Prefetch Cache — Design Notes

## What we built

A port of Sirius PR #1661 (`aminaramoon:feature/frugal_caching_and_dynamic_io`, commit `31c1b7c8`)
into cudf at `cudf_streaming::prefetch::` namespace. All source lives under
`cpp/libcudf_streaming/src/prefetch/sirius/`.

## Overall architecture

```
cudf-polars (Python)
    │  creates scan_context with ordered split list
    ▼
scan_context (C++ façade — scan_context.cpp, Stage 5)
    │  background readahead thread: stays 16 splits ahead
    │  get_datasource(i) → sirius_datasource*
    ▼
cudf_streaming::prefetch::io::sirius_datasource
    │  implements cudf::io::datasource
    │  serves reads from prefetching_cache (hit) or reactor (miss)
    ▼
cudf_streaming::prefetch::io::cache::prefetching_cache
    │  direct-mapped slot cache (slot_i = offset / chunk_size)
    │  batches H2D copies via device_copy_batch → cudaMemcpyBatchAsync
    ▼
io_uring reactor pool  →  O_DIRECT reads from NVMe
```

## How prefetch-ahead works (Sirius PR #1661 model)

The fadvise → prepare_prefetch → prefetch_async sequence:

1. `sirius_datasource::fadvise(byte_ranges, gpu_id)` — registers ranges in the
   cache's `file_entry::slots[]` (direct-mapped, O(1) lookup by offset/chunk_size).
   Returns a `prefetching_handle` with two atomic state machines:
   `producer_stage` (IO progress) and `consumer_stage` (scan progress).

2. `prepare_prefetch(wait_for_eviction=true)` — allocates pinned host-memory
   staging buffers from `buffer_pool`. Readahead waits; live reads don't.

3. `prefetch_async(on_done)` — issues async O_DIRECT io_uring reads into the
   pinned staging buffers. `on_done` fires on IO completion.

On `device_read_async`: checks slot state.
- Hit (`ready`): accumulate into `device_copy_batch`, issue as one `cudaMemcpyBatchAsync`.
- In-flight (`loading`): `wait_until_ready()`, then hit path.
- Miss: fall through to backend read.

## Readahead driver design (scan_context.cpp — Stage 5)

Sirius uses an event-driven `readahead_scan_manager`. We don't need that because
cudf-polars knows the full ordered split list at plan time.

Our driver is a semaphore-gated background thread (eager mode, budget=16):

```
construction:
  receives: ordered list of splits [(path, byte_ranges[])]  ← from Python planner
  creates:  io_context + prefetching_cache
  spawns:   readahead thread

readahead thread:
  cursor = 0
  loop:
    gatekeeper.acquire()               ← blocks when 16 IOs in flight
    ds = open(split[cursor])
    ds.fadvise(ranges, gpu_id)
    ds.prepare_prefetch(wait=true)
    ds.prefetch_async(on_done: gatekeeper.release())
    stash ds in slot[cursor]
    cursor++

get_datasource(i):
  wait for slot[i] ready
  ds.update(scan_stage::reading)
  return ds*

release_datasource(i):
  ds.update(scan_stage::disposed)     ← evictor can reclaim chunks
```

### C++ API surface (target)

```cpp
scan_context ctx(scan_manager_config, splits);
cudf::io::datasource* ds = ctx.get_datasource(i);   // blocks until prefetched
ctx.release_datasource(i);
```

### Python/Cython API (Stage 6–7)

```python
# cudf_polars/io/parquet.py (streaming scan path)
if prefetch_cache_enabled():
    ctx = ScanContext(config, splits)        # starts eager readahead
    for i, split in enumerate(splits):
        ds = ctx.get_datasource(i)           # sirius_datasource*
        cudf.read_parquet(..., datasource=ds)
        ctx.release_datasource(i)
```

## Sirius benchmark settings (SF300 reference)

These are the settings Sirius uses for their benchmark numbers:

```yaml
scan_manager:
  max_readahead_scans: 16      # gatekeeper budget — 16 IOs in flight
  num_threads: 20              # reactor pool threads
  rest_n_reactors: 16          # REST reactor pool (S3)
  backend: sirius              # io_uring path, not kvikio

cache:
  mode: sirius
  eviction: lru
  eviction_threshold_fraction: 0.8   # evict at 80% pool occupancy

host memory:
  capacity_bytes: 200GB        # total pinned staging pool
  pool_size: 512               # chunk size (likely MiB)
  downgrade_trigger_fraction: 0.95
  downgrade_stop_fraction: 0.9
```

Key mapping to our `scan_manager_config`:
- `max_readahead_scans: 16` → gatekeeper budget (semaphore depth)
- `num_threads: 20` → reactor pool size
- `backend: sirius` → `io_backend::sirius`
- `eviction_threshold_fraction: 0.8` → `cache::config::eviction_threshold_fraction`

## Source layout

```
src/prefetch/
  scan_context.cpp              ← Stage 5 (TODO): readahead driver + public C++ façade
  prefetch_config.hpp           ← scan_manager_config aggregate
  CMakeLists.txt

  sirius/
    error.hpp                   ← CUDA/logic error macros
    prefetch_config.hpp         ← (see above)
    log/logging.hpp             ← SIRIUS_LOG_* → rapids_logger
    exec/                       ← semi_future, thread_pool, invocable, ...
    memory/                     ← ported from cuCascade (cucascade::memory:: → our ns)
    cuda/
      event.cpp/hpp             ← cuda event wrapper
      device_copy_batch.hpp     ← batched H2D copies → cudaMemcpyBatchAsync
    io/
      types.hpp                 ← host_buffer, device_buffer, prepared_io_slice
      io_request.hpp            ← grouped_coordinator, grouped_io_request
      io_context.hpp/cpp        ← ioctx base + io_context_registry (datasource_factory)
      sirius_datasource.hpp/cpp ← cudf::io::datasource impl
      datasource_factory.hpp/cpp
      templated_ioctx.hpp
      cache/
        types.hpp/cpp           ← cached_chunk, file_entry, buffer_pool, chunk_fill
        prefetching_cache.hpp/cpp ← the main cache
        metadata_store.hpp/cpp
        config.hpp
      uring/                    ← guarded #ifdef CUDF_STREAMING_HAS_URING
      kvikio/                   ← fallback backend
      rest/                     ← HTTP/S3 backend
```

## Build

```bash
cd /raid/mmurray/cudf/cpp/libcudf_streaming/build_gcc13
ninja cudf_streaming_prefetch -j8
```

Target artifact: `libcudf_streaming_prefetch.a`
