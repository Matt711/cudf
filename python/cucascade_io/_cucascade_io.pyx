# SPDX-License-Identifier: Apache-2.0
# distutils: language = c++
"""Cython glue: expose cuCascade's REST/prefetching-cache datasource as a pylibcudf Datasource."""
from libc.stdint cimport uint64_t
from libcpp cimport bool as cbool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport pair
from libcpp.vector cimport vector

from pylibcudf.io.datasource cimport Datasource
from pylibcudf.libcudf.io.datasource cimport datasource

cdef extern from "cc_glue.hpp" namespace "ccglue" nogil:
    cdef cppclass options:
        size_t cache_bytes
        size_t initial_pools
        size_t rest_reactors
        size_t max_connections
        string endpoint
        string region
        string access_key
        string secret_key
        string session_token
        size_t advise_min_bytes
        cbool lru
    void c_init "ccglue::init"(const options&) except +
    unique_ptr[datasource] c_open "ccglue::open"(const string&, uint64_t) except +
    int c_fadvise "ccglue::fadvise"(datasource*, const vector[pair[uint64_t, uint64_t]]&, cbool) except +
    string c_cache_summary "ccglue::cache_summary"() except +
    void c_shutdown "ccglue::shutdown"() except +


cdef extern from "cudf/io/datasource.hpp" namespace "cudf::io" nogil:
    cdef cppclass sized_datasource "cudf::io::datasource":
        size_t size()


def init(*, size_t cache_bytes, str endpoint, str region, str access_key, str secret_key, str session_token="",
         size_t rest_reactors=2, size_t max_connections=64, size_t initial_pools=4, size_t advise_min_bytes=262144, bint lru=True):
    cdef options o
    o.cache_bytes = cache_bytes
    o.initial_pools = initial_pools
    o.rest_reactors = rest_reactors
    o.max_connections = max_connections
    o.endpoint = endpoint.encode()
    o.region = region.encode()
    o.access_key = access_key.encode()
    o.secret_key = secret_key.encode()
    o.session_token = session_token.encode()
    o.advise_min_bytes = advise_min_bytes
    o.lru = lru
    with nogil:
        c_init(o)


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
