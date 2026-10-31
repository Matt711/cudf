# SPDX-License-Identifier: Apache-2.0
# distutils: language = c++
"""Cython glue: expose bytecache::cached_source as a pylibcudf Datasource (see DESIGN.md)."""
from libc.stdint cimport uint32_t, uint64_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

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
        uint64_t ns_open
        uint64_t ns_pinned_wait
        uint64_t ns_read
        uint64_t ns_h2d
        uint64_t ns_budget_wait
        uint64_t device_reads
        uint64_t ns_queue_wait
        uint64_t max_inflight
        uint64_t inflight_now
        uint64_t queued_now
        uint64_t bytes_prefetched
        uint64_t prefetch_jobs
        uint64_t host_reads
        uint64_t ns_host_read
    cdef cppclass column_stats:
        string name
        uint64_t registered_bytes
        uint64_t requested_bytes
        uint64_t hit_bytes
        uint64_t fetched_bytes
        uint64_t resident_blocks
    void c_register_layout "bytecache::register_layout"(const string&, const string&, const vector[string]&, const vector[uint32_t]&, const vector[uint64_t]&, const vector[uint64_t]&) except +
    void c_warm "bytecache::warm_footers"(const vector[string]&, bint) nogil
    uint64_t c_prefetch "bytecache::prefetch_columns"(const string&, const string&, const vector[string]&, uint64_t) except +
    uint64_t c_new_group "bytecache::new_group"()
    bint c_wait_group "bytecache::wait_group"(uint64_t, uint64_t) nogil
    void c_release_group "bytecache::release_group"(uint64_t)

    vector[column_stats] c_get_column_stats "bytecache::get_column_stats"()
    void c_configure "bytecache::configure"(const config&) except +
    stats c_get_stats "bytecache::get_stats"()
    void c_reset_stats "bytecache::reset_stats"()
    void c_clear "bytecache::clear"() except +
    vector[uint64_t] c_tl_submit "bytecache::timeline_submit"()
    vector[uint64_t] c_tl_done "bytecache::timeline_done"()
    vector[string] c_slow "bytecache::slow_reads"()
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
        "ns_open": s.ns_open,
        "ns_pinned_wait": s.ns_pinned_wait,
        "ns_read": s.ns_read,
        "ns_h2d": s.ns_h2d,
        "ns_budget_wait": s.ns_budget_wait,
        "device_reads": s.device_reads,
        "ns_queue_wait": s.ns_queue_wait,
        "max_inflight": s.max_inflight,
        "inflight_now": s.inflight_now,
        "queued_now": s.queued_now,
        "bytes_prefetched": s.bytes_prefetched,
        "prefetch_jobs": s.prefetch_jobs,
        "host_reads": s.host_reads,
        "ns_host_read": s.ns_host_read,
    }


def register_layout(str path, str table, list columns, list col_index, list offsets, list sizes):
    """Tell the cache where each column chunk of an object lives (offset, size), so blocks can be attributed to columns."""
    cdef vector[string] cols
    for c in columns:
        cols.push_back((<str>c).encode())
    c_register_layout(path.encode(), table.encode(), cols, col_index, offsets, sizes)


def warm_footers(list paths, bint urgent=True):
    """Read the footers of many objects into the cache concurrently (GIL released). urgent: a scan is waiting for them
    (demand pool, no read-ahead gate); pass False from speculative callers."""
    cdef vector[string] ps
    cdef bint u = urgent
    for p in paths:
        ps.push_back((<str>p).encode())
    with nogil:
        c_warm(ps, u)


def new_prefetch_group():
    """A handle for 'everything submitted for one scan task'."""
    return c_new_group()


def wait_prefetch_group(uint64_t group, uint64_t timeout_ms):
    """Wait (GIL released) until all prefetch jobs of the group finished or the timeout passed. True if complete."""
    cdef bint done
    with nogil:
        done = c_wait_group(group, timeout_ms)
    return done


def release_prefetch_group(uint64_t group):
    c_release_group(group)


def prefetch_columns(str path, str table, list columns, uint64_t group=0):
    """Start read-ahead of the named columns of a registered file. Returns the bytes of the ranges (0 if unknown)."""
    cdef vector[string] cols
    for c in columns:
        cols.push_back((<str>c).encode())
    return c_prefetch(path.encode(), table.encode(), cols, group)


def get_column_stats():
    """Per-column accounting: registered/requested/hit/fetched bytes and resident blocks."""
    cdef vector[column_stats] v = c_get_column_stats()
    return [
        {
            "name": s.name.decode(),
            "registered_bytes": s.registered_bytes,
            "requested_bytes": s.requested_bytes,
            "hit_bytes": s.hit_bytes,
            "fetched_bytes": s.fetched_bytes,
            "resident_blocks": s.resident_blocks,
        }
        for s in v
    ]


def get_timeline():
    """(submitted, completed): per-second counts of device reads since the first one."""
    return list(c_tl_submit()), list(c_tl_done())


def get_slow_reads():
    """Device reads that took longer than 1 s: 'path offset size milliseconds'."""
    return [s.decode() for s in c_slow()]


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
