# SPDX-License-Identifier: Apache-2.0
# distutils: language = c++
"""Cython glue: expose bytecache::cached_source as a pylibcudf Datasource (see DESIGN.md)."""
from libc.stdint cimport uint64_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from pylibcudf.io.datasource cimport Datasource
from pylibcudf.libcudf.io.datasource cimport datasource

cdef extern from "bytecache.hpp" namespace "bytecache" nogil:
    cdef cppclass config:
        size_t capacity_bytes
        size_t block_size
        size_t max_get_bytes
    cdef cppclass stats:
        uint64_t bytes_requested
        uint64_t bytes_hit
        uint64_t bytes_fetched
        uint64_t blocks_evicted
        uint64_t blocks_resident
        uint64_t gets_issued
        uint64_t waits_on_inflight
    void c_configure "bytecache::configure"(const config&) except +
    stats c_get_stats "bytecache::get_stats"()
    void c_reset_stats "bytecache::reset_stats"()
    void c_clear "bytecache::clear"() except +
    cdef cppclass cached_source(datasource):
        cached_source(string path, size_t size) except +
        size_t size() const


def configure(size_t capacity_bytes, size_t block_size=1 << 20, size_t max_get_bytes=16 << 20):
    """Set the process-wide cache. capacity_bytes=0 disables caching (reads still go through kvikio)."""
    cdef config c
    c.capacity_bytes = capacity_bytes
    c.block_size = block_size
    c.max_get_bytes = max_get_bytes
    c_configure(c)


def get_stats():
    cdef stats s = c_get_stats()
    return {
        "bytes_requested": s.bytes_requested,
        "bytes_hit": s.bytes_hit,
        "bytes_fetched": s.bytes_fetched,
        "blocks_evicted": s.blocks_evicted,
        "blocks_resident": s.blocks_resident,
        "gets_issued": s.gets_issued,
        "waits_on_inflight": s.waits_on_inflight,
    }


def reset_stats():
    c_reset_stats()


def clear():
    c_clear()


cdef class CachedDatasource(Datasource):
    """A remote (s3://) object read through the process-wide byte cache."""
    cdef unique_ptr[cached_source] c_src

    def __cinit__(self, str path, size_t size=0):
        self.c_src.reset(new cached_source(path.encode(), size))

    cdef datasource* get_datasource(self) except * nogil:
        return <datasource*> self.c_src.get()

    def nbytes(self):
        return self.c_src.get().size()
