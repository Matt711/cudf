# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libc.stddef cimport size_t
from libc.stdint cimport uint8_t

from pylibcudf.io.datasource cimport Datasource as PlcDatasource
from pylibcudf.libcudf.io.datasource cimport datasource


cdef extern from "prefetch_cache/prefetch_datasource.hpp" nogil:
    cdef cppclass PrefetchDatasource_cpp "PrefetchDatasource"(datasource):
        # Single-segment constructor (original API)
        PrefetchDatasource_cpp(size_t file_size,
                               size_t base_offset,
                               const uint8_t* data_bytes,
                               size_t data_size) except +
        # Multi-segment: empty-init constructor
        PrefetchDatasource_cpp(size_t file_size) except +
        # Add one segment to a multi-segment datasource
        void add_segment(size_t file_offset,
                         const uint8_t* data_bytes,
                         size_t data_size) except +
        # Sort segments once after all add_segment() calls (thread-safety: call
        # only from the construction thread before any concurrent reads).
        void finalize_segments() except +
        size_t n_reads()      except +
        size_t n_hits()       except +
        size_t n_misses()     except +
        size_t bytes_served() except +


cdef class PrefetchDatasource(PlcDatasource):
    cdef unique_ptr[PrefetchDatasource_cpp] _ds
    cdef datasource* get_datasource(self) except * nogil
