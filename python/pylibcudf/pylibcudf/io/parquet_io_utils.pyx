# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IO utilities for Parquet."""

from libc.stddef cimport size_t
from libc.stdint cimport uint8_t, uintptr_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector
from cython.operator cimport dereference

from rmm.librmm.device_buffer cimport device_buffer
from rmm.pylibrmm.device_buffer cimport DeviceBuffer
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from pylibcudf.gpumemoryview cimport gpumemoryview
from pylibcudf.io.text cimport ByteRangeInfo
from pylibcudf.io.types cimport SourceInfo
from pylibcudf.libcudf.io.datasource cimport datasource, make_datasources
from pylibcudf.libcudf.io.parquet_io_utils cimport (
    const_byte_range_info,
    const_uint8_t,
    cpp_fetch_byte_ranges_to_device,
    cpp_fetch_byte_ranges_to_device_async,
    cpp_fetch_future,
    cpp_wait_fetch_future,
    fetch_page_index_to_host as cpp_fetch_page_index_to_host,
)

from pylibcudf.libcudf.io.text cimport byte_range_info
from pylibcudf.libcudf.utilities.span cimport device_span, host_span
from pylibcudf.utils cimport _get_memory_resource, _get_stream
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pylibcudf.typing import CudaStreamLike

__all__ = [
    "FetchFuture",
    "fetch_byte_ranges_to_device",
    "fetch_byte_ranges_to_device_async",
    "fetch_page_index_to_host",
]


cdef class FetchFuture:
    """Handle for an in-flight :func:`fetch_byte_ranges_to_device_async` fetch.

    The fetch is already running in the background by the time this is
    returned; call :meth:`wait` when the caller actually needs the data in
    the buffers this fetch is filling.

    Holds the datasource, stream, and memory resource the fetch was issued
    against, so they stay alive for as long as this future does: the fetch
    may still be reading through the datasource, or writing to the stream,
    after :func:`fetch_byte_ranges_to_device_async` has already returned.
    """

    cdef unique_ptr[cpp_fetch_future] c_obj
    cdef vector[unique_ptr[datasource]] _sources
    cdef Stream _stream
    cdef DeviceMemoryResource _mr

    def wait(self) -> None:
        """Block until the fetch this future represents has completed."""
        with nogil:
            cpp_wait_fetch_future(dereference(self.c_obj))


cpdef list fetch_byte_ranges_to_device(
    SourceInfo source_info,
    list byte_ranges,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    """Fetch byte ranges from a Parquet source into device memory.

    Parameters
    ----------
    source_info : SourceInfo
        Source describing a single Parquet file.
    byte_ranges : list[ByteRangeInfo]
        Byte ranges to fetch, as returned by
        :meth:`~pylibcudf.io.experimental.HybridScanReader.filter_column_chunks_byte_ranges`,
        :meth:`~pylibcudf.io.experimental.HybridScanReader.payload_column_chunks_byte_ranges`,
        or
        :meth:`~pylibcudf.io.experimental.HybridScanReader.all_column_chunks_byte_ranges`.
    stream : Stream, optional
        CUDA stream.
    mr : DeviceMemoryResource, optional
        Device memory resource.

    Returns
    -------
    list[gpumemoryview]
        One view per byte range. Each view holds a reference to the
        :class:`~rmm.DeviceBuffer` that owns its memory, keeping the
        allocation alive for as long as the view is referenced.

    Raises
    ------
    ValueError
        If ``source_info`` does not describe exactly one source.
    """
    cdef Stream _stream = _get_stream(stream)
    cdef DeviceMemoryResource _mr = _get_memory_resource(mr)
    cdef vector[unique_ptr[datasource]] sources = make_datasources(source_info.c_obj)
    if sources.size() != 1:
        raise ValueError(
            f"fetch_byte_ranges_to_device requires exactly one source, "
            f"got {sources.size()}"
        )

    cdef vector[byte_range_info] ranges_vec
    cdef ByteRangeInfo bri
    for bri in byte_ranges:
        ranges_vec.push_back(bri.c_obj)

    cdef pair[vector[device_buffer], vector[device_span[const_uint8_t]]] fetched
    with nogil:
        fetched = cpp_fetch_byte_ranges_to_device(
            dereference(sources[0]),
            host_span[const_byte_range_info](ranges_vec.data(), ranges_vec.size()),
            _stream.view(),
            _mr.get_mr(),
        )

    if fetched.first.size() != 1:
        raise RuntimeError(
            f"Expected exactly one device buffer, got {fetched.first.size()}"
        )
    cdef DeviceBuffer owner = DeviceBuffer.c_from_unique_ptr(
        make_unique[device_buffer](move(fetched.first[0])),
        _stream,
        _mr,
    )
    cdef gpumemoryview owner_gv = gpumemoryview(owner)
    cdef uintptr_t base = owner_gv.ptr
    cdef uintptr_t ptr
    cdef size_t n
    result = []
    for i in range(fetched.second.size()):
        ptr = <uintptr_t>fetched.second[i].data()
        n = fetched.second[i].size()
        result.append(owner_gv.byte_slice(slice(ptr - base, ptr - base + n)))
    return result


cpdef tuple fetch_byte_ranges_to_device_async(
    SourceInfo source_info,
    list byte_ranges,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    """Start fetching byte ranges from a Parquet source into device memory.

    Returns as soon as the fetch has been submitted, not once it has
    completed. Device buffers are allocated and returned immediately so
    their addresses are known, but their contents are only valid once the
    returned future's :meth:`~FetchFuture.wait` has been called.

    Parameters
    ----------
    source_info : SourceInfo
        Source describing a single Parquet file.
    byte_ranges : list[ByteRangeInfo]
        Byte ranges to fetch, as returned by
        :meth:`~pylibcudf.io.experimental.HybridScanReader.filter_column_chunks_byte_ranges`,
        :meth:`~pylibcudf.io.experimental.HybridScanReader.payload_column_chunks_byte_ranges`,
        or
        :meth:`~pylibcudf.io.experimental.HybridScanReader.all_column_chunks_byte_ranges`.
    stream : Stream, optional
        CUDA stream.
    mr : DeviceMemoryResource, optional
        Device memory resource.

    Returns
    -------
    tuple[list[gpumemoryview], FetchFuture]
        One view per byte range, and a future to wait on before reading them.

    Raises
    ------
    ValueError
        If ``source_info`` does not describe exactly one source.
    """
    cdef Stream _stream = _get_stream(stream)
    cdef DeviceMemoryResource _mr = _get_memory_resource(mr)
    cdef FetchFuture future = FetchFuture.__new__(FetchFuture)
    future._sources = make_datasources(source_info.c_obj)
    if future._sources.size() != 1:
        raise ValueError(
            f"fetch_byte_ranges_to_device_async requires exactly one source, "
            f"got {future._sources.size()}"
        )
    future._stream = _stream
    future._mr = _mr

    cdef vector[byte_range_info] ranges_vec
    cdef ByteRangeInfo bri
    for bri in byte_ranges:
        ranges_vec.push_back(bri.c_obj)

    cdef pair[vector[device_buffer], vector[device_span[const_uint8_t]]] fetched
    with nogil:
        fetched = cpp_fetch_byte_ranges_to_device_async(
            dereference(future._sources[0]),
            host_span[const_byte_range_info](ranges_vec.data(), ranges_vec.size()),
            _stream.view(),
            _mr.get_mr(),
            future.c_obj,
        )

    if fetched.first.size() != 1:
        raise RuntimeError(
            f"Expected exactly one device buffer, got {fetched.first.size()}"
        )
    cdef DeviceBuffer owner = DeviceBuffer.c_from_unique_ptr(
        make_unique[device_buffer](move(fetched.first[0])),
        _stream,
        _mr,
    )
    cdef gpumemoryview owner_gv = gpumemoryview(owner)
    cdef uintptr_t base = owner_gv.ptr
    cdef uintptr_t ptr
    cdef size_t n
    result = []
    for i in range(fetched.second.size()):
        ptr = <uintptr_t>fetched.second[i].data()
        n = fetched.second[i].size()
        result.append(owner_gv.byte_slice(slice(ptr - base, ptr - base + n)))
    return result, future


cpdef bytes fetch_page_index_to_host(
    SourceInfo source_info,
    ByteRangeInfo page_index_range,
):
    """Fetch parquet page index bytes to host memory.

    Parameters
    ----------
    source_info : SourceInfo
        Source describing a single Parquet file.
    page_index_range : ByteRangeInfo
        Byte range of the page index, as returned by
        :meth:`~pylibcudf.io.experimental.HybridScanReader.page_index_byte_range`.

    Returns
    -------
    bytes
        Raw page index bytes copied to Python host memory.

    Raises
    ------
    ValueError
        If ``source_info`` does not describe exactly one source.
    """
    cdef vector[unique_ptr[datasource]] sources = make_datasources(source_info.c_obj)
    if sources.size() != 1:
        raise ValueError(
            f"fetch_page_index_to_host requires exactly one source, "
            f"got {sources.size()}"
        )

    cdef unique_ptr[datasource.buffer] buf
    with nogil:
        buf = move(cpp_fetch_page_index_to_host(
            dereference(sources[0]),
            (<ByteRangeInfo>page_index_range).c_obj,
        ))

    if buf.get() is NULL:
        raise RuntimeError("fetch_page_index_to_host returned no buffer")
    cdef const uint8_t* ptr = buf.get().data()
    cdef size_t n = buf.get().size()
    return bytes(ptr[:n])
