# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, int64_t, uint8_t
from libcpp cimport bool
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport pair
from libcpp.vector cimport vector

from cuda.bindings.cyruntime cimport cudaStream_t
from pylibcudf.libcudf.io.parquet cimport parquet_reader_options
from pylibcudf.libcudf.io.parquet_schema cimport FileMetaData
from pylibcudf.libcudf.io.types cimport table_with_metadata
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from rmm.librmm.device_buffer cimport device_buffer

from cudf_polars_cucascade.exception_handler cimport libcucascade_exception_handler
from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.streaming.core.actor cimport cpp_Actor
from rapidsmpf.streaming.core.channel cimport cpp_Channel
from rapidsmpf.streaming.core.context cimport cpp_Context


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
        size_t cache_bytes
        size_t cache_chunk_bytes

    cdef cppclass planned_task "cudf_polars_cucascade::scan_io::planned_task":
        int64_t id
        bool empty
        bool unsupported
        bool too_big
        size_t staged_bytes
        size_t decoded_bytes

    cdef cppclass scan_io:
        scan_io(const scan_io_config& cfg) except +libcucascade_exception_handler
        void warmup(const string& url) except +libcucascade_exception_handler
        void request_footers(
            const vector[string]& paths
        ) except +libcucascade_exception_handler
        pair[size_t, unique_ptr[FileMetaData]] footer(
            const string& path
        ) except +libcucascade_exception_handler
        vector[size_t] projected_bytes(
            const vector[string]& paths, const vector[string]& columns
        ) except +libcucascade_exception_handler
        vector[uint8_t] footer_file(const string& path) except +libcucascade_exception_handler
        void clear_footers() except +libcucascade_exception_handler
        size_t clear_cache() except +libcucascade_exception_handler
        size_t arena_bytes()
        int64_t open_stream(int64_t priority) except +libcucascade_exception_handler
        void set_expected_streams(size_t count) except +libcucascade_exception_handler
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
        vector[size_t] snapshot() except +
        planned_task plan_task(
            int64_t stream_id,
            const vector[string]& paths,
            int split_index,
            int total_splits,
            const parquet_reader_options& options,
            bool prune_with_stats,
            size_t merge_gap,
            cudaStream_t stream,
        ) except +libcucascade_exception_handler
        table_with_metadata materialize(
            int64_t id, cudaStream_t stream
        ) except +libcucascade_exception_handler


cdef extern from "probe_pipeline.hpp" namespace "cudf_polars_cucascade" nogil:

    cdef cppclass expr_node:
        int op
        int column_index
        double value
        string text
        int binop
        int dtype
        int lhs
        int rhs
        int third

    cdef cppclass stage:
        int op
        int build
        vector[int] probe_keys
        vector[int] keep_probe
        vector[int] keep_build
        vector[vector[expr_node]] exprs
        vector[int] columns
        vector[int] keys
        vector[int] sums

    unique_ptr[column] evaluate_expression_on(
        const vector[expr_node]& nodes, table_view table, cudaStream_t stream
    ) except +libcucascade_exception_handler

    cdef cppclass probe_pipeline:
        probe_pipeline(
            scan_io& io,
            vector[vector[stage]] branches,
            vector[table_view] builds,
            vector[vector[int]] build_keys,
            int threads,
            vector[string] input_columns,
        ) except +libcucascade_exception_handler
        void submit(int64_t task_id) except +libcucascade_exception_handler
        vector[unique_ptr[table]] finish() except +libcucascade_exception_handler
        void close() except +libcucascade_exception_handler
        pair[int64_t, unique_ptr[table]] next_output() except +libcucascade_exception_handler
        vector[double] stats() except +libcucascade_exception_handler


cdef extern from "native_actors.hpp" namespace "cudf_polars_cucascade::actor" nogil:

    cdef cppclass scan_task:
        vector[string] paths
        int split_index
        int total_splits

    cpp_Actor cpp_scan_source "cudf_polars_cucascade::actor::scan_source"(
        shared_ptr[cpp_Context] ctx,
        shared_ptr[cpp_Channel] ch_out,
        scan_io* io,
        vector[scan_task] tasks,
        const parquet_reader_options* options,
        bool prune_with_stats,
        size_t merge_gap,
        size_t num_producers,
        vector[string] columns,
        vector[int] column_types,
        size_t max_ahead_bytes,
    ) except +ex_handler

    cpp_Actor cpp_broadcast "cudf_polars_cucascade::actor::broadcast"(
        shared_ptr[cpp_Context] ctx,
        shared_ptr[cpp_Channel] ch_in,
        vector[shared_ptr[cpp_Channel]] chs_out,
    ) except +ex_handler

    cpp_Actor cpp_chain "cudf_polars_cucascade::actor::chain"(
        shared_ptr[cpp_Context] ctx,
        vector[shared_ptr[cpp_Channel]] ch_builds,
        vector[vector[int]] build_keys,
        vector[vector[int]] build_types,
        shared_ptr[cpp_Channel] ch_in,
        shared_ptr[cpp_Channel] ch_out,
        vector[vector[stage]] branches,
        size_t concurrency,
        int fold_keys,
        vector[int] out_types,
        bool forward_metadata,
        bool ordered,
    ) except +ex_handler
