# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport make_unique
from libc.stddef cimport size_t
from libc.stdint cimport uint8_t

# The .pxd is auto-included and provides: PlcDatasource, datasource,
# PrefetchDatasource_cpp, and the PrefetchDatasource cdef class shell.


cdef class PrefetchDatasource(PlcDatasource):
    """
    Pylibcudf datasource backed by pinned host RAM.

    Two construction modes:

    1. ``from_buffer(file_size, base_offset, data)``
       Single-segment: covers [base_offset, base_offset+len(data)).

    2. ``from_segments(file_size, segments)``
       Multi-segment sparse: ``segments`` is a list of ``(offset, data_bytes)``
       pairs covering only the byte ranges the parquet reader will actually
       access (bloom filters + column chunks + footer).  Drastically reduces
       pinned RAM compared to full-file downloads.
    """

    @staticmethod
    def from_buffer(size_t file_size, size_t base_offset, bytes data):
        """
        Construct from a single contiguous byte range.

        Parameters
        ----------
        file_size : int
            Total size of the parquet file (returned by datasource.size()).
        base_offset : int
            Absolute byte offset within the file where ``data`` starts.
        data : bytes
            Raw bytes covering [base_offset, base_offset+len(data)).
        """
        cdef const uint8_t* ptr = data
        cdef size_t n = len(data)
        cdef PrefetchDatasource obj = PrefetchDatasource.__new__(PrefetchDatasource)
        obj._ds = make_unique[PrefetchDatasource_cpp](file_size, base_offset, ptr, n)
        return obj

    @staticmethod
    def from_segments(size_t file_size, list segments):
        """
        Construct from a list of sparse byte ranges.

        Parameters
        ----------
        file_size : int
            Total size of the parquet file (returned by datasource.size()).
        segments : list of (int, bytes)
            Each entry is ``(file_offset, data_bytes)`` covering one non-overlapping
            byte range of the parquet file.  Ranges need not be in order.
        """
        cdef PrefetchDatasource obj = PrefetchDatasource.__new__(PrefetchDatasource)
        obj._ds = make_unique[PrefetchDatasource_cpp](file_size)
        cdef const uint8_t* ptr
        cdef size_t off, n
        for off, data in segments:
            ptr = <const uint8_t*> data
            n = len(data)
            if n > 0:
                obj._ds.get().add_segment(off, ptr, n)
        # Sort once on the construction thread before handing to concurrent readers.
        obj._ds.get().finalize_segments()
        return obj

    cdef datasource* get_datasource(self) except * nogil:
        return <datasource*> self._ds.get()

    def stats(self):
        """Return (reads, hits, misses, bytes_served)."""
        cdef PrefetchDatasource_cpp* p = self._ds.get()
        return (p.n_reads(), p.n_hits(), p.n_misses(), p.bytes_served())
