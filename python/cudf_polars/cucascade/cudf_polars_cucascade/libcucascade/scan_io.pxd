# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, int64_t, uint8_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport pair
from libcpp.vector cimport vector

from cuda.bindings.cyruntime cimport cudaStream_t
from pylibcudf.libcudf.io.parquet_schema cimport FileMetaData
from rmm.librmm.device_buffer cimport device_buffer

from cudf_polars_cucascade.exception_handler cimport libcucascade_exception_handler


cdef extern from "scan_io.hpp" namespace "cudf_polars_cucascade" nogil:

    cdef cppclass scan_io_config:
        scan_io_config() except +libcucascade_exception_handler
        string endpoint
        string region
        string access_key
        string secret_key
        string session_token
        size_t reactors
        size_t connections
        size_t stall_seconds
        size_t arena_bytes
        size_t footer_threads
        size_t inflight_bytes

    cdef cppclass scan_io:
        scan_io(const scan_io_config& cfg) except +libcucascade_exception_handler
        void warmup(const string& url) except +libcucascade_exception_handler
        void request_footers(
            const vector[string]& paths
        ) except +libcucascade_exception_handler
        pair[size_t, unique_ptr[FileMetaData]] footer(
            const string& path
        ) except +libcucascade_exception_handler
        vector[uint8_t] footer_file(const string& path) except +libcucascade_exception_handler
        void clear_footers() except +libcucascade_exception_handler
        size_t arena_bytes()
        int64_t open_stream() except +libcucascade_exception_handler
        void close_stream(int64_t stream) except +libcucascade_exception_handler
        int64_t submit(
            int64_t stream,
            const vector[string]& paths,
            const vector[int32_t]& sources,
            const vector[size_t]& offsets,
            const vector[size_t]& sizes,
        ) except +libcucascade_exception_handler
        bool ready(int64_t id) except +libcucascade_exception_handler
        void wait(int64_t id) except +libcucascade_exception_handler
        unique_ptr[device_buffer] to_device(
            int64_t id, cudaStream_t stream
        ) except +libcucascade_exception_handler
        void release(int64_t id) except +libcucascade_exception_handler
        size_t fetched_bytes()
