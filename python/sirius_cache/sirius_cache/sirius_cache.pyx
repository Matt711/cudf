# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Cython extension: sirius_cache
#
# Exposes the Sirius prefetch-cache IO stack to Python.  One IoContextRegistry
# per process is enough; create it once at session start and keep it alive for
# the whole benchmark run.
#
# Public API (what Agent 3 imports):
#   ScanStage            — enum of prefetch pipeline stages
#   ScanManagerConfig    — config bundle
#   IoContextRegistry    — owns the IO stack (session-level singleton)
#   SiriusDatasource     — one per file per Polars task
#   PrefetchingHandle    — opaque handle (kept for backward compat)
#   reset_caches()       — flush the prefetch cache between benchmark iterations

import enum

from cpython.ref cimport PyObject
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector
from libc.stddef cimport size_t
from libc.stdint cimport int64_t, uintptr_t

from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.pylibrmm.device_buffer cimport DeviceBuffer as RmmDeviceBuffer
from rmm.pylibrmm.stream cimport Stream as RmmStream

# pylibcudf base class so SourceInfo accepts SiriusDatasource instances
from pylibcudf.io.datasource cimport Datasource as PlcDatasource
from pylibcudf.libcudf.io.datasource cimport datasource as cudf_datasource
from pylibcudf.gpumemoryview cimport gpumemoryview
from pylibcudf.io.text cimport ByteRangeInfo
from pylibcudf.utils cimport _get_stream

# The companion sirius_cache.pxd is automatically included by Cython — all
# cdef extern declarations and cdef class declarations are available without
# an explicit cimport.

# Module-level reference to the active registry (set by IoContextRegistry.__init__).
# reset_caches() uses it so callers don't need to pass the registry explicitly.
cdef IoContextRegistry _active_registry = None

__all__ = [
    "ScanStage",
    "ScanManagerConfig",
    "IoContextRegistry",
    "SiriusDatasource",
    "PrefetchingHandle",
    "reset_caches",
]


# ─── ScanStage ───────────────────────────────────────────────────────────────

class ScanStage(enum.IntEnum):
    """Python mirror of ``cudf_streaming::io::cache::scan_stage``.

    Passed to :meth:`SiriusDatasource.update` to advance the consumer stage
    on the datasource's prefetching handle.  Drives the fallen-behind detector
    and tells the evictor when a block's reader has passed it.

    Values match the C++ enum's implicit ordering (``none=0``…``disposed=5``).
    """
    none        = 0
    initialized = 1
    queued      = 2
    preparing   = 3
    reading     = 4
    disposed    = 5


# ─── ScanManagerConfig ────────────────────────────────────────────────────────

cdef class ScanManagerConfig:
    """Configuration for the Sirius IO stack.

    Parameters
    ----------
    num_threads : int
        Thread count for the kvikIO thread pool (default 20).
    rest_n_reactors : int
        REST reactor count — drives S3 parallelism (default 16).
    uring_n_reactors : int
        io_uring reactor count — for local backends (default 4).
    host_capacity_bytes : int
        Maximum bytes of pinned host memory for the prefetch cache pool.
    host_pool_size_mib : int
        Number of 1-MiB blocks per memory pool slab.
    host_initial_number_pools : int
        Number of pool slabs to pre-allocate on startup.
    eviction_threshold_fraction : float
        Fraction of pool capacity at which LRU eviction kicks in (default 0.8).
    object_store_endpoint : str
        S3 endpoint URL (e.g. "https://s3.us-east-2.amazonaws.com").
    object_store_region : str
        AWS region string (e.g. "us-east-2").
    access_key : str
        AWS access key ID.
    secret_key : str
        AWS secret access key.
    session_token : str
        STS session token (empty for long-lived credentials).
    use_prefetch_cache : bool
        Whether to arm the Sirius prefetch cache (default True).
    """

    def __init__(
        self,
        *,
        size_t num_threads                 = 20,
        size_t rest_n_reactors             = 16,
        size_t uring_n_reactors            = 4,
        size_t host_capacity_bytes         = 200 * 1024 ** 3,
        size_t host_pool_size_mib          = 512,
        size_t host_initial_number_pools   = 300,
        double eviction_threshold_fraction = 0.8,
        str object_store_endpoint          = "",
        str object_store_region            = "",
        str access_key                     = "",
        str secret_key                     = "",
        str session_token                  = "",
        bool use_prefetch_cache            = True,
    ):
        self._cfg.num_threads                = num_threads
        self._cfg.rest_n_reactors            = rest_n_reactors
        self._cfg.uring_n_reactors           = uring_n_reactors
        self._cfg.host_capacity_bytes        = host_capacity_bytes
        self._cfg.host_pool_size_mib         = host_pool_size_mib
        self._cfg.host_initial_number_pools  = host_initial_number_pools
        self._cfg.eviction_threshold_fraction = eviction_threshold_fraction
        self._cfg.object_store_endpoint      = object_store_endpoint.encode()
        self._cfg.object_store_region        = object_store_region.encode()
        self._cfg.access_key                 = access_key.encode()
        self._cfg.secret_key                 = secret_key.encode()
        self._cfg.session_token              = session_token.encode()
        self._cfg.use_prefetch_cache         = use_prefetch_cache


# ─── IoContextRegistry ───────────────────────────────────────────────────────

cdef class IoContextRegistry:
    """Owns the Sirius IO stack: the pinned host-memory pool, the kvikio
    backend, and (optionally) the prefetch cache.

    Construct exactly once per process (or once per benchmark run) and keep
    alive for the duration.  Destroying it tears down background threads and
    returns all pinned memory.

    Parameters
    ----------
    cfg : ScanManagerConfig
        Configuration bundle.

    Examples
    --------
    >>> cfg = ScanManagerConfig(num_threads=20, host_capacity_bytes=200*1024**3)
    >>> registry = IoContextRegistry(cfg)
    >>> ds = registry.open_datasource("/data/file.parquet")
    """

    def __init__(self, ScanManagerConfig cfg):
        global _active_registry
        with nogil:
            self._session = make_unique[SiriusSession](cfg._cfg)
        # Register as the process-wide active registry for reset_caches().
        _active_registry = self

    def __dealloc__(self):
        global _active_registry
        if _active_registry is self:
            _active_registry = None
        # unique_ptr destructor shuts down reactors and frees pinned memory.

    def cache_summary(self) -> str:
        """Return the prefetch cache hit/miss/eviction counters as a string.

        The string format is::

            prefetching_cache: global[reads=N hits=N h2d=N miss=N evictions=N]
                               last_cycle[reads=N hits=N h2d=N miss=N evictions=N]

        Returns an empty string if the cache is not active.
        """
        return deref(self._session).cache_summary().decode()

    def warmup(self, str bucket_url) -> None:
        """Pre-warm libcurl connections to the S3 endpoint.

        Issues a lightweight request against ``bucket_url`` (e.g.
        ``"s3://my-bucket"``) so DNS lookup, TCP connect, and TLS handshake
        are already done before the first query fires.  Best-effort: never
        throws and never blocks the caller.

        Parameters
        ----------
        bucket_url : str
            Container URL, not an object path (e.g. ``"s3://my-bucket"``).
            A full object URL is accepted; the key is ignored.
        """
        cdef string c_url = bucket_url.encode()
        with nogil:
            deref(self._session).warmup(c_url)

    def prepare_for_query(self) -> None:
        """Reset per-query epoch counters without evicting cached blocks.

        Call this at the start of each query to reset the ``last_cycle``
        counters so the cache's per-cycle statistics start fresh.  Unlike
        :func:`reset_caches`, this does **not** evict any cached data.
        """
        with nogil:
            deref(self._session).prepare_for_query()

    def open_datasource(self, str path) -> SiriusDatasource:
        """Open one file and return a SiriusDatasource bound to this registry.

        The returned datasource may be used to call fadvise() to hint the IO
        layer about upcoming reads.  The datasource must not outlive this
        registry.

        Parameters
        ----------
        path : str
            Local file path or s3://bucket/key URI.

        Returns
        -------
        SiriusDatasource
        """
        cdef string c_path = path.encode()
        cdef unique_ptr[sirius_datasource] ds_ptr

        with nogil:
            ds_ptr = deref(self._session).open(c_path)

        cdef SiriusDatasource py_ds = SiriusDatasource.__new__(SiriusDatasource)
        py_ds._ds = move(ds_ptr)
        return py_ds


# ─── SiriusDatasource ────────────────────────────────────────────────────────

cdef class SiriusDatasource(PlcDatasource):
    """Wraps one ``cudf_streaming::io::sirius_datasource``.

    Subclasses :class:`pylibcudf.io.datasource.Datasource` so it can be
    passed directly to ``plc.io.SourceInfo([ds])`` — pylibcudf's
    ``SourceInfo`` checks ``isinstance(src, Datasource)``.

    Opened via :meth:`IoContextRegistry.open_datasource`.  Do not construct
    directly.

    Notes
    -----
    One datasource per file per Polars task.  Call :meth:`fadvise` to register
    upcoming byte ranges, then :meth:`prepare_prefetch` and
    :meth:`prefetch_async` to issue background S3 reads.  Use
    :meth:`duplicate` when the same file is split across multiple scans so
    each scan has its own handle.
    """

    cdef cudf_datasource* get_datasource(self) except * nogil:
        """Return the underlying ``cudf::io::datasource*`` for pylibcudf."""
        return <cudf_datasource*> self._ds.get()

    def duplicate(self) -> SiriusDatasource:
        """Return a new datasource sharing this file's io_object.

        The duplicate points at the same underlying file (shared io_object)
        but carries an independent, empty prefetching handle.  Use this when
        a single file is split across multiple scans so each scan can call
        :meth:`fadvise` without stomping on another scan's handle.

        Returns
        -------
        SiriusDatasource
        """
        if not self._ds:
            raise RuntimeError("SiriusDatasource: datasource is not open")
        cdef sirius_datasource* raw_ptr
        with nogil:
            raw_ptr = call_duplicate(deref(self._ds))
        cdef SiriusDatasource py_ds = SiriusDatasource.__new__(SiriusDatasource)
        py_ds._ds.reset(raw_ptr)
        return py_ds

    def fadvise(self, list byte_ranges, int device_id = -1) -> None:
        """Hint the prefetch cache about upcoming reads from this file.

        Registers ``byte_ranges`` with the prefetching cache and stores the
        resulting handle inside this datasource.  The handle is released when
        this datasource is garbage-collected or when the C++ side decides to
        reclaim it.

        Parameters
        ----------
        byte_ranges : list of (int, int)
            ``(offset, size)`` pairs identifying the byte ranges this task
            will read.
        device_id : int, optional
            CUDA device ID that will consume the prefetched data.  Pass -1
            (the default) to let the cache choose.
        """
        if not self._ds:
            raise RuntimeError("SiriusDatasource: datasource is not open")

        cdef vector[byte_range_info] c_ranges
        cdef int64_t off, sz

        for item in byte_ranges:
            off, sz = item[0], item[1]
            c_ranges.push_back(byte_range_info(off, sz))

        with nogil:
            call_fadvise(deref(self._ds), c_ranges, device_id)

    def update(self, int stage) -> None:
        """Advance the consumer stage on this datasource's prefetching handle.

        Drives the prefetch pipeline's fallen-behind detector and tells the
        evictor when a block's reader has advanced past it.  Pass a
        :class:`ScanStage` value (or its integer equivalent).

        Parameters
        ----------
        stage : int
            One of the :class:`ScanStage` values.
        """
        if not self._ds:
            return
        with nogil:
            call_update(deref(self._ds), stage)

    def prepare_prefetch(self, bool wait_for_eviction=True) -> int:
        """Allocate staging buffers in the pinned host pool for this request.

        Must be called after :meth:`fadvise` and before
        :meth:`prefetch_async`.

        Parameters
        ----------
        wait_for_eviction : bool
            When ``True``, block until LRU eviction frees enough space rather
            than failing immediately on a full pool.

        Returns
        -------
        int
            Raw ``prepare_result`` enum value:
            ``0`` = prepared, ``1`` = allocation_failed,
            ``2`` = nothing_to_prepare.
        """
        if not self._ds:
            return 2  # nothing_to_prepare
        cdef int result
        with nogil:
            result = call_prepare_prefetch(deref(self._ds), wait_for_eviction)
        return result

    def prefetch_async(self, callback=None) -> None:
        """Dispatch background S3 → host-pool reads.

        Must be called after :meth:`prepare_prefetch` returns ``0``
        (prepared).

        Parameters
        ----------
        callback : callable or None
            If provided, called exactly once with a single ``bool`` argument
            indicating whether the prefetch succeeded.  The callback fires on
            the IO reactor thread; the implementation re-acquires the GIL
            before invoking it.  Pass ``None`` for fire-and-forget behaviour.
        """
        if not self._ds:
            return
        if callback is None:
            with nogil:
                call_prefetch_async(deref(self._ds))
        else:
            call_prefetch_async_cb(deref(self._ds), <void*>(<PyObject*>callback))

    def await_inflight_prefetch(self) -> None:
        """Block until the in-flight S3 → host-pool transfer completes.

        Call this from a worker thread after :meth:`prefetch_async` when you
        need the data to be resident in pinned host memory before the main
        read loop starts.  Releases the GIL while waiting.
        """
        if not self._ds:
            return
        with nogil:
            call_await_inflight_prefetch(deref(self._ds))

    def fetch_byte_ranges_vectored(
        self, list byte_ranges, object stream=None
    ) -> list:
        """Vectored device read: all byte-range H2D copies in one batched call.

        Replaces N × ``fetch_byte_ranges_to_device`` calls (serial H2D) with a
        single ``device_read_ranges_async`` dispatch.  Requires the prefetch
        cache to be active; if the datasource has no active handle the call
        still works but falls back to sequential reads internally.

        Parameters
        ----------
        byte_ranges : list of ByteRangeInfo
            As returned by HybridScanReader.all_column_chunks_byte_ranges().
        stream : pylibcudf.Stream or None
            CUDA stream for the async copies.

        Returns
        -------
        list of gpumemoryview
            One view per byte range, all backed by a single RMM DeviceBuffer.
            Same format as ``fetch_byte_ranges_to_device()``.
        """
        if not self._ds:
            raise RuntimeError("SiriusDatasource: datasource is not open")
        if not byte_ranges:
            return []

        cdef RmmStream _stream = _get_stream(stream)
        cdef cudaStream_t raw_stream = _stream.view().get()

        cdef vector[size_t] c_offsets
        cdef vector[size_t] c_sizes
        cdef size_t total_size = 0
        cdef ByteRangeInfo bri

        for item in byte_ranges:
            bri = <ByteRangeInfo>item
            c_offsets.push_back(<size_t>bri.c_obj.offset())
            c_sizes.push_back(<size_t>bri.c_obj.size())
            total_size += <size_t>bri.c_obj.size()

        if total_size == 0:
            return []

        # Allocate one contiguous device buffer for all ranges combined.
        cdef RmmDeviceBuffer buf = RmmDeviceBuffer(size=total_size, stream=_stream)
        cdef uintptr_t base_ptr = buf.ptr

        # Build destination pointer vector: each range gets a sub-slice of buf.
        cdef vector[uintptr_t] c_dst_ptrs
        cdef size_t cum = 0
        for i in range(c_sizes.size()):
            c_dst_ptrs.push_back(base_ptr + cum)
            cum += c_sizes[i]

        # Issue the vectored device read (blocks until the future is ready).
        with nogil:
            call_device_read_ranges_async(
                deref(self._ds),
                c_offsets,
                c_sizes,
                c_dst_ptrs,
                raw_stream,
            )

        # Build gpumemoryview list matching fetch_byte_ranges_to_device() output.
        cdef gpumemoryview owner_gv = gpumemoryview(buf)
        result = []
        cdef size_t off = 0
        for i in range(c_sizes.size()):
            n = c_sizes[i]
            result.append(owner_gv.byte_slice(slice(off, off + n)))
            off += n
        return result

    @property
    def size(self) -> int:
        """Total size of the underlying file in bytes."""
        if not self._ds:
            raise RuntimeError("SiriusDatasource: datasource is not open")
        cdef size_t sz
        with nogil:
            sz = deref(self._ds).size()
        return sz

    def __bool__(self):
        return <bool> self._ds


# ─── PrefetchingHandle ───────────────────────────────────────────────────────

cdef class PrefetchingHandle:
    """Opaque handle kept for backward compatibility.

    The C++ ``sirius_datasource`` owns the underlying ``prefetching_handle``
    internally.  This Python object simply holds a reference to the parent
    :class:`SiriusDatasource`, ensuring the datasource is not garbage-collected
    while the handle is live.

    .. deprecated::
        :meth:`SiriusDatasource.fadvise` no longer returns this object.
        Construct it manually if backward compatibility is required.
    """

    def release(self):
        """Explicitly release the handle early (same as letting it go out of scope)."""
        self._datasource = None

    def __bool__(self):
        return self._datasource is not None


# ─── Module-level functions ───────────────────────────────────────────────────

def reset_caches():
    """Reset the prefetch cache between benchmark iterations.

    Requests immediate eviction of all cached data and resets the per-query
    epoch counter so the next benchmark iteration starts cold.

    Operates on the last :class:`IoContextRegistry` that was constructed in
    this process.

    Raises
    ------
    RuntimeError
        If no :class:`IoContextRegistry` has been constructed yet.
    """
    global _active_registry
    if _active_registry is None:
        raise RuntimeError(
            "sirius_cache.reset_caches(): no IoContextRegistry active in this process. "
            "Construct IoContextRegistry(cfg) before calling reset_caches()."
        )
    cdef IoContextRegistry reg = _active_registry
    with nogil:
        deref(reg._session).reset_caches()
