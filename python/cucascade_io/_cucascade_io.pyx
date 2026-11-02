# SPDX-License-Identifier: Apache-2.0
# distutils: language = c++
"""Cython glue: expose cuCascade's REST/prefetching-cache datasource as a pylibcudf Datasource."""
from libc.stdint cimport uint32_t, uint64_t
from libcpp cimport bool as cbool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libc.stdint cimport uint8_t

from pylibcudf.io.datasource cimport Datasource
from pylibcudf.libcudf.io.datasource cimport datasource

cdef extern from "cc_glue.hpp" namespace "ccglue" nogil:
    cdef cppclass options:
        size_t cache_bytes
        size_t block_bytes
        size_t initial_pools
        size_t pool_threads
        size_t rest_reactors
        size_t max_connections
        string endpoint
        string region
        string access_key
        string secret_key
        string session_token
        size_t advise_min_bytes
        double evict_threshold
        double budget_fraction
        cbool column_policy
        cbool lru
    void c_init "ccglue::init"(const options&) except +
    unique_ptr[datasource] c_open "ccglue::open"(const string&, uint64_t) except +
    int c_fadvise "ccglue::fadvise"(datasource*, const vector[pair[uint64_t, uint64_t]]&, cbool) except +
    void c_register_layout "ccglue::register_layout"(const string&, const string&, const vector[string]&, const vector[uint32_t]&, const vector[uint64_t]&, const vector[uint64_t]&) except +
    string c_column_report "ccglue::column_report"() except +
    string c_cache_summary "ccglue::cache_summary"() except +
    void c_shutdown "ccglue::shutdown"() except +


cdef extern from "cudf/io/datasource.hpp" namespace "cudf::io" nogil:
    cdef cppclass sized_datasource "cudf::io::datasource":
        size_t size()
        size_t host_read(size_t, size_t, uint8_t*) except +


def init(*, size_t cache_bytes, str endpoint, str region, str access_key, str secret_key, str session_token="",
         size_t rest_reactors=16, size_t max_connections=64, size_t initial_pools=0, size_t pool_threads=8, size_t block_bytes=1048576, size_t advise_min_bytes=262144, double evict_threshold=0.95, double budget_fraction=0.9, bint column_policy=True, bint lru=True):
    cdef options o
    o.cache_bytes = cache_bytes
    o.initial_pools = initial_pools
    o.pool_threads = pool_threads
    o.block_bytes = block_bytes
    o.rest_reactors = rest_reactors
    o.max_connections = max_connections
    o.endpoint = endpoint.encode()
    o.region = region.encode()
    o.access_key = access_key.encode()
    o.secret_key = secret_key.encode()
    o.session_token = session_token.encode()
    o.advise_min_bytes = advise_min_bytes
    o.evict_threshold = evict_threshold
    o.budget_fraction = budget_fraction
    o.column_policy = column_policy
    o.lru = lru
    with nogil:
        c_init(o)


def register_layout(str path, str directory, list names, list col_index, list offsets, list sizes):
    cdef vector[string] n = [x.encode() for x in names]
    c_register_layout(path.encode(), directory.encode(), n, col_index, offsets, sizes)


def column_report():
    return c_column_report().decode()


def cache_summary():
    return c_cache_summary().decode()


def shutdown():
    with nogil:
        c_shutdown()


cdef class CucascadeDatasource(Datasource):
    """An s3:// object read through cuCascade's REST backend and prefetching cache."""
    cdef unique_ptr[datasource] c_src

    def __cinit__(self, str path, uint64_t size=0):
        cdef string p = path.encode()
        cdef uint64_t s = size
        with nogil:
            self.c_src = c_open(p, s)

    cdef datasource* get_datasource(self) except * nogil:
        return self.c_src.get()

    def nbytes(self):
        return (<sized_datasource*> self.c_src.get()).size()

    def fadvise(self, offsets, sizes, bint start=True):
        """Tell the cache which byte ranges this scan will read (so the chunks are kept for later scans)."""
        cdef vector[pair[uint64_t, uint64_t]] r
        for o, n in zip(offsets, sizes):
            r.push_back(pair[uint64_t, uint64_t](<uint64_t>o, <uint64_t>n))
        cdef int rc
        with nogil:
            rc = c_fadvise(self.c_src.get(), r, start)
        return rc

    def read_into(self, uint64_t offset, uint64_t size, uint64_t dst_addr):
        """Host read of [offset, offset+size) into the buffer at dst_addr (pure I/O, used by the throughput benchmarks)."""
        cdef uint8_t* d = <uint8_t*> dst_addr
        cdef size_t n
        with nogil:
            n = (<sized_datasource*> self.c_src.get()).host_read(offset, size, d)
        return n
