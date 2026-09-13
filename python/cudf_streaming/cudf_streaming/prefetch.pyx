# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python bindings for the cudf_streaming prefetch scan context."""

from cython.operator cimport dereference as deref
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from libcpp cimport bool
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.optional cimport nullopt, optional
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rapidsmpf._detail.exception_handling cimport ex_handler

from pylibcudf.libcudf.io.datasource cimport datasource
from pylibcudf.libcudf.io.text cimport byte_range_info
from pylibcudf.io.datasource cimport Datasource

__all__ = ["BorrowedSiriusDatasource", "ScanContext", "SiriusDatasource"]


# ---------------------------------------------------------------------------
# C++ declarations
# ---------------------------------------------------------------------------

cdef extern from "<cudf_streaming/experimental/prefetch_context.hpp>" \
        namespace "cudf_streaming::prefetch" nogil:

    cdef cppclass cpp_scan_context_config \
            "cudf_streaming::prefetch::scan_context_config":
        size_t host_memory_capacity
        size_t host_block_size
        int gpu_device_id
        double gpu_reservation_fraction
        string s3_endpoint
        string s3_region
        string s3_access_key_id
        string s3_secret_access_key
        string s3_session_token
        bool use_sirius_datasource_for_local

    cdef cppclass cpp_prefetch_datasource \
            "cudf_streaming::prefetch::PrefetchDatasource"(datasource):
        void fadvise(
            const vector[byte_range_info]& ranges,
            optional[int] dev_id
        ) except +ex_handler
        unique_ptr[cpp_prefetch_datasource] duplicate() except +ex_handler

    cdef struct cpp_scan_split "cudf_streaming::prefetch::scan_split":
        string path
        vector[byte_range_info] ranges

    cdef cppclass cpp_scan_context \
            "cudf_streaming::prefetch::scan_context":
        @staticmethod
        unique_ptr[cpp_scan_context] create(
            const cpp_scan_context_config& cfg
        ) except +ex_handler
        unique_ptr[cpp_prefetch_datasource] open_datasource(
            string path
        ) except +ex_handler
        void prepare_for_query(uint64_t query_id) nogil
        void start_readahead(
            vector[cpp_scan_split] splits,
            optional[int] gpu_id,
            size_t budget
        ) except +ex_handler
        cpp_prefetch_datasource* get_datasource(size_t i) except +ex_handler
        void release_datasource(size_t i) noexcept nogil
        void stop_readahead() noexcept nogil


# ---------------------------------------------------------------------------
# SiriusDatasource (Python class)
# ---------------------------------------------------------------------------

cdef class SiriusDatasource(Datasource):
    """A cudf datasource backed by the prefetch cache.

    Subclasses :class:`pylibcudf.io.datasource.Datasource` so it can be
    passed directly to :class:`pylibcudf.io.SourceInfo`.

    Do not construct directly; use :meth:`ScanContext.open_datasource`.
    """

    cdef unique_ptr[cpp_prefetch_datasource] _ds

    # pylibcudf.io.datasource.Datasource.get_datasource override
    cdef datasource* get_datasource(self) except * nogil:
        return self._ds.get()

    def fadvise(self, byte_ranges, gpu_id=None):
        """Hint the cache that these byte ranges will be read soon.

        Parameters
        ----------
        byte_ranges : list of pylibcudf.io.text.ByteRangeInfo
            The byte ranges to prefetch.
        gpu_id : int, optional
            GPU device to stage the data near (preferred NUMA node).
        """
        cdef vector[byte_range_info] cpp_ranges
        cdef optional[int] cpp_dev_id
        for br in byte_ranges:
            cpp_ranges.push_back(byte_range_info(br.offset, br.size))
        if gpu_id is not None:
            cpp_dev_id = <int>gpu_id
        with nogil:
            self._ds.get().fadvise(cpp_ranges, cpp_dev_id)

    def duplicate(self):
        """Return a fresh datasource sharing this file's I/O context.

        Each split scan should hold its own datasource (own prefetch handle)
        but shares the same underlying file object and cache bucket.
        """
        cdef SiriusDatasource dup = SiriusDatasource.__new__(SiriusDatasource)
        with nogil:
            dup._ds = self._ds.get().duplicate()
        return dup


# ---------------------------------------------------------------------------
# ScanContext (Python class)
# ---------------------------------------------------------------------------

cdef class ScanContext:
    """Per-engine prefetch scan context.

    Holds the background memory pools, I/O reactors, and prefetching cache.
    Construct once at engine startup; destroy at engine shutdown.

    Parameters
    ----------
    host_memory_gb : float
        Pinned host memory (GiB) to reserve for the prefetch staging pool.
        Default: 8 GiB.
    gpu_device_id : int
        GPU device to create a GPU-tier memory space for.  Pass -1 to disable.
        Default: 0.
    s3_endpoint : str, optional
        S3 endpoint URL (e.g. "https://s3.amazonaws.com").
    s3_region : str, optional
        AWS region (e.g. "us-east-1").
    s3_access_key_id : str, optional
        AWS access key.
    s3_secret_access_key : str, optional
        AWS secret key.
    s3_session_token : str, optional
        Temporary session token (STS).
    use_sirius_datasource_for_local : bool
        Route local files through io_uring (requires CUDF_STREAMING_HAS_URING).
        Default: False.
    """

    cdef unique_ptr[cpp_scan_context] _ctx

    def __cinit__(
        self,
        double host_memory_gb=8.0,
        int gpu_device_id=0,
        str s3_endpoint="",
        str s3_region="",
        str s3_access_key_id="",
        str s3_secret_access_key="",
        str s3_session_token="",
        bool use_sirius_datasource_for_local=False,
    ):
        cdef cpp_scan_context_config cfg
        cfg.host_memory_capacity = <size_t>(host_memory_gb * 1024.0 * 1024.0 * 1024.0)
        cfg.gpu_device_id = gpu_device_id
        cfg.s3_endpoint = s3_endpoint.encode()
        cfg.s3_region = s3_region.encode()
        cfg.s3_access_key_id = s3_access_key_id.encode()
        cfg.s3_secret_access_key = s3_secret_access_key.encode()
        cfg.s3_session_token = s3_session_token.encode()
        cfg.use_sirius_datasource_for_local = use_sirius_datasource_for_local
        with nogil:
            self._ctx = cpp_scan_context.create(cfg)

    def open_datasource(self, str path):
        """Open a prefetch-backed datasource for the given path.

        Parameters
        ----------
        path : str
            Local filesystem path or S3 URI (e.g. "s3://bucket/key.parquet").

        Returns
        -------
        SiriusDatasource
            A datasource usable with :class:`pylibcudf.io.SourceInfo`.
        """
        cdef SiriusDatasource ds = SiriusDatasource.__new__(SiriusDatasource)
        cdef string cpp_path = path.encode()
        with nogil:
            ds._ds = self._ctx.get().open_datasource(cpp_path)
        return ds

    def prepare_for_query(self, uint64_t query_id=0):
        """Signal the start of a new query epoch for cache telemetry."""
        with nogil:
            self._ctx.get().prepare_for_query(query_id)

    def start_readahead(self, list splits, int gpu_id=-1, size_t budget=16):
        """Start background readahead for an ordered list of (path, byte_ranges) splits.

        Parameters
        ----------
        splits : list of (str, list of (int, int))
            Each entry is (path, [(offset, size), ...]).  Pass an empty
            range list to open the file without pre-issuing range reads.
        gpu_id : int, optional
            GPU device to stage data near.  -1 means no preference.
        budget : int
            Maximum number of IOs in flight at once (default 16).
        """
        cdef vector[cpp_scan_split] cpp_splits
        cdef cpp_scan_split s
        cdef optional[int] cpp_gpu_id
        cdef string cpp_path

        for path, ranges in splits:
            s.path = (<str>path).encode()
            s.ranges.clear()
            for offset, size in ranges:
                s.ranges.push_back(byte_range_info(<size_t>offset, <size_t>size))
            cpp_splits.push_back(s)

        if gpu_id >= 0:
            cpp_gpu_id = gpu_id

        with nogil:
            self._ctx.get().start_readahead(cpp_splits, cpp_gpu_id, budget)

    def get_datasource(self, size_t i):
        """Return the prefetched datasource for split *i* (blocks until ready).

        Returns a :class:`BorrowedSiriusDatasource` — a non-owning view
        backed by the ScanContext's readahead slot.  The ScanContext must
        remain alive while the datasource is in use.  Call
        :meth:`release_datasource` when done.
        """
        cdef BorrowedSiriusDatasource bds = BorrowedSiriusDatasource.__new__(
            BorrowedSiriusDatasource
        )
        cdef cpp_prefetch_datasource* ptr
        with nogil:
            ptr = self._ctx.get().get_datasource(i)
        bds._ptr = ptr
        return bds

    def release_datasource(self, size_t i):
        """Signal that the reader has finished with split *i*."""
        with nogil:
            self._ctx.get().release_datasource(i)

    def stop_readahead(self):
        """Stop the background readahead thread and join it."""
        with nogil:
            self._ctx.get().stop_readahead()


# ---------------------------------------------------------------------------
# BorrowedSiriusDatasource — non-owning view of a readahead slot
# ---------------------------------------------------------------------------

cdef class BorrowedSiriusDatasource(Datasource):
    """Non-owning reference to a prefetch-cache datasource slot.

    Returned by :meth:`ScanContext.get_datasource`.  The underlying memory
    is owned by the :class:`ScanContext`; this object must not outlive it.
    Call :meth:`ScanContext.release_datasource` when done reading.
    """

    cdef cpp_prefetch_datasource* _ptr  # raw pointer — NOT owned

    cdef datasource* get_datasource(self) except * nogil:
        return self._ptr
