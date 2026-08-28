# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint8_t
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.vector cimport vector

from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.device_buffer cimport device_buffer
from rmm.librmm.memory_resource cimport device_async_resource_ref

from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.io.datasource cimport datasource
from pylibcudf.libcudf.io.text cimport byte_range_info
from pylibcudf.libcudf.utilities.span cimport device_span, host_span

ctypedef const uint8_t const_uint8_t
ctypedef const byte_range_info const_byte_range_info

cdef extern from * nogil:
    """
    #include <future>
    #include <memory>
    #include <cudf/io/parquet_io_utils.hpp>

    static std::pair<std::vector<rmm::device_buffer>,
                     std::vector<cudf::device_span<uint8_t const>>>
    cpp_fetch_byte_ranges_to_device(
        cudf::io::datasource& datasource,
        cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
        rmm::cuda_stream_view stream,
        rmm::device_async_resource_ref mr)
    {
        auto [buffers, spans, fut] =
            cudf::io::parquet::fetch_byte_ranges_to_device_async(
                datasource, byte_ranges, stream, mr);
        // Block until the async fetch completes so the returned buffers/spans
        // are fully populated. Kept alongside the genuinely async
        // cpp_fetch_byte_ranges_to_device_async below for callers that don't
        // need to overlap this fetch with anything else.
        fut.get();
        return {std::move(buffers), std::move(spans)};
    }

    // Opaque handle for a std::future<void> from fetch_byte_ranges_to_device_async,
    // so callers can start a fetch, do other work, and wait on it later instead of
    // blocking immediately.
    struct cpp_fetch_future {
        std::future<void> fut;
    };

    static std::pair<std::vector<rmm::device_buffer>,
                     std::vector<cudf::device_span<uint8_t const>>>
    cpp_fetch_byte_ranges_to_device_async(
        cudf::io::datasource& datasource,
        cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
        rmm::cuda_stream_view stream,
        rmm::device_async_resource_ref mr,
        std::unique_ptr<cpp_fetch_future>& out_future)
    {
        auto [buffers, spans, fut] =
            cudf::io::parquet::fetch_byte_ranges_to_device_async(
                datasource, byte_ranges, stream, mr);
        out_future = std::make_unique<cpp_fetch_future>(cpp_fetch_future{std::move(fut)});
        return {std::move(buffers), std::move(spans)};
    }

    static void cpp_wait_fetch_future(cpp_fetch_future& f)
    {
        f.fut.get();
    }
    """
    pair[vector[device_buffer], vector[device_span[const_uint8_t]]] \
        cpp_fetch_byte_ranges_to_device(
            datasource& source,
            host_span[const_byte_range_info] byte_ranges,
            cuda_stream_view stream,
            device_async_resource_ref mr,
        ) except +libcudf_exception_handler

    cdef cppclass cpp_fetch_future:
        pass

    pair[vector[device_buffer], vector[device_span[const_uint8_t]]] \
        cpp_fetch_byte_ranges_to_device_async(
            datasource& source,
            host_span[const_byte_range_info] byte_ranges,
            cuda_stream_view stream,
            device_async_resource_ref mr,
            unique_ptr[cpp_fetch_future]& out_future,
        ) except +libcudf_exception_handler

    void cpp_wait_fetch_future(cpp_fetch_future& f) except +libcudf_exception_handler

cdef extern from "cudf/io/parquet_io_utils.hpp" \
        namespace "cudf::io::parquet" nogil:

    unique_ptr[datasource.buffer] fetch_page_index_to_host(
        datasource& ds,
        byte_range_info page_index_bytes,
    ) except +libcudf_exception_handler
