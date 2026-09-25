// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "prefetch_datasource.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

// NVTX3 is header-only (ships with CUDA 11+/CCCL).  If unavailable we skip.
#ifdef __has_include
#  if __has_include(<nvtx3/nvtx3.hpp>)
#    include <nvtx3/nvtx3.hpp>
#    define PREFETCH_HAVE_NVTX3 1
#  endif
#endif

// ---------------------------------------------------------------------------
// Single-segment constructor (existing API, unchanged).
// ---------------------------------------------------------------------------
PrefetchDatasource::PrefetchDatasource(std::size_t file_size,
                                       std::size_t base_offset,
                                       const uint8_t* data_bytes,
                                       std::size_t data_size)
    : _file_size(file_size)
{
    cudaStreamCreateWithFlags(&_h2d_stream, cudaStreamNonBlocking);
    void* buf = nullptr;
    cudaError_t err = cudaMallocHost(&buf, data_size);
    if (err != cudaSuccess) {
        cudaStreamDestroy(_h2d_stream);
        throw std::runtime_error(std::string("cudaMallocHost failed: ") +
                                 cudaGetErrorString(err));
    }
    std::memcpy(buf, data_bytes, data_size);
    _segments.push_back({base_offset, buf, data_size});
    _sorted = true;
}

// ---------------------------------------------------------------------------
// Multi-segment constructor: start empty, then call add_segment().
// ---------------------------------------------------------------------------
PrefetchDatasource::PrefetchDatasource(std::size_t file_size)
    : _file_size(file_size)
{
    cudaStreamCreateWithFlags(&_h2d_stream, cudaStreamNonBlocking);
}

void PrefetchDatasource::add_segment(std::size_t file_offset,
                                     const uint8_t* data_bytes,
                                     std::size_t data_size)
{
    void* buf = nullptr;
    cudaError_t err = cudaMallocHost(&buf, data_size);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMallocHost failed in add_segment: ") +
                                 cudaGetErrorString(err));
    }
    std::memcpy(buf, data_bytes, data_size);
    _segments.push_back({file_offset, buf, data_size});
    _sorted = false;  // may need re-sort
}

// ---------------------------------------------------------------------------
// Destructor: free all pinned buffers.
// ---------------------------------------------------------------------------
PrefetchDatasource::~PrefetchDatasource()
{
    for (auto& seg : _segments) {
        if (seg.pinned_buf) {
            cudaFreeHost(seg.pinned_buf);
            seg.pinned_buf = nullptr;
        }
    }
    if (_h2d_stream) {
        cudaStreamSynchronize(_h2d_stream);
        cudaStreamDestroy(_h2d_stream);
    }
}

// ---------------------------------------------------------------------------
// Sort segments by file_offset.
// _ensure_sorted() is NOT thread-safe and must only be called from a single
// thread.  finalize_segments() is the public API for callers to sort once
// after construction before handing the datasource to concurrent readers.
// ---------------------------------------------------------------------------
void PrefetchDatasource::_ensure_sorted()
{
    if (!_sorted) {
        std::sort(_segments.begin(), _segments.end(),
                  [](const Segment& a, const Segment& b) {
                      return a.file_offset < b.file_offset;
                  });
        _sorted = true;
    }
}

void PrefetchDatasource::finalize_segments()
{
    _ensure_sorted();
}

// ---------------------------------------------------------------------------
// Binary-search for the segment covering [offset, offset+size).
// Returns a pointer to the start of the requested region, or nullptr on miss.
// ---------------------------------------------------------------------------
const uint8_t* PrefetchDatasource::_find(std::size_t offset, std::size_t size)
{
    _ensure_sorted();

    // Find the last segment whose file_offset <= offset.
    auto it = std::upper_bound(
        _segments.begin(), _segments.end(), offset,
        [](std::size_t off, const Segment& seg) { return off < seg.file_offset; }
    );
    if (it == _segments.begin()) return nullptr;
    --it;

    // Check that the entire [offset, offset+size) range is within this segment.
    if (offset + size <= it->file_offset + it->size) {
        return static_cast<const uint8_t*>(it->pinned_buf) + (offset - it->file_offset);
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// device_read_async: cudaMemcpyAsync from pinned RAM to device.
//
// Handles multi-segment reads: the requested [offset, offset+size) range may
// span multiple cached segments (e.g. adjacent column chunks). We walk all
// overlapping segments and issue one cudaMemcpyAsync per overlap. Genuine
// gaps (ranges never downloaded) are zero-filled via a leading memset.
// ---------------------------------------------------------------------------
std::future<std::size_t> PrefetchDatasource::device_read_async(
    std::size_t offset, std::size_t size,
    uint8_t* dst, cuda::stream_ref stream)
{
#ifdef PREFETCH_HAVE_NVTX3
    nvtx3::scoped_range rng{"PrefetchDS::device_read_async"};
#endif
    _n_reads.fetch_add(1, std::memory_order_relaxed);

    if (size == 0) {
        std::promise<std::size_t> p;
        p.set_value(0);
        return p.get_future();
    }

    _ensure_sorted();

    const std::size_t end = offset + size;

    // Binary-search for the first segment that could overlap [offset, end).
    // We want the last segment with file_offset <= offset.
    auto it = std::upper_bound(
        _segments.begin(), _segments.end(), offset,
        [](std::size_t off, const Segment& seg) { return off < seg.file_offset; }
    );
    if (it != _segments.begin()) --it;

    // First pass (CPU only): count how many bytes we can cover.
    std::size_t covered = 0;
    for (auto jt = it; jt != _segments.end() && jt->file_offset < end; ++jt) {
        const std::size_t seg_end = jt->file_offset + jt->size;
        if (seg_end <= offset) continue;
        const std::size_t lo = std::max(jt->file_offset, offset);
        const std::size_t hi = std::min(seg_end, end);
        if (hi > lo) covered += (hi - lo);
    }

    if (covered < size) {
        // Genuine gaps — zero on the private H2D stream so the fence below covers it.
        cudaMemsetAsync(dst, 0, size, _h2d_stream);
    }

    // Second pass: issue one cudaMemcpyAsync per overlapping segment on the
    // private H2D stream (not the caller's compute stream).  This keeps H2D
    // transfers off the compute stream, allowing other reactors' compute kernels
    // to run concurrently on the SMs — matching kvikio_source compat-mode behavior.
    std::size_t actually_copied = 0;
    for (; it != _segments.end() && it->file_offset < end; ++it) {
        const std::size_t seg_end = it->file_offset + it->size;
        if (seg_end <= offset) continue;
        const std::size_t lo = std::max(it->file_offset, offset);
        const std::size_t hi = std::min(seg_end, end);
        if (hi <= lo) continue;
        const std::size_t chunk = hi - lo;
        const uint8_t* src = static_cast<const uint8_t*>(it->pinned_buf) + (lo - it->file_offset);
        cudaMemcpyAsync(dst + (lo - offset), src, chunk, cudaMemcpyHostToDevice, _h2d_stream);
        actually_copied += chunk;
    }

    // Record a CUDA event on the H2D stream and make the caller's compute stream
    // wait for it.  cudaEventDisableTiming avoids profiling overhead.
    // Safe to destroy the event after cudaStreamWaitEvent — CUDA defers the
    // actual release until the stream has processed the wait operation.
    cudaEvent_t h2d_done;
    cudaEventCreateWithFlags(&h2d_done, cudaEventDisableTiming);
    cudaEventRecord(h2d_done, _h2d_stream);
    cudaStreamWaitEvent(stream.get(), h2d_done, 0);
    cudaEventDestroy(h2d_done);

    if (actually_copied == size) {
        _n_hits.fetch_add(1, std::memory_order_relaxed);
        _bytes_served.fetch_add(size, std::memory_order_relaxed);
    } else {
        _n_misses.fetch_add(1, std::memory_order_relaxed);
        _bytes_served.fetch_add(actually_copied, std::memory_order_relaxed);
    }

    std::promise<std::size_t> p;
    p.set_value(size);
    return p.get_future();
}

// ---------------------------------------------------------------------------
// host_read (buffer-returning): zero-copy view into pinned RAM.
// ---------------------------------------------------------------------------
namespace {
struct pinned_view_buffer final : public cudf::io::datasource::buffer {
    const uint8_t* _ptr;
    std::size_t    _sz;
    pinned_view_buffer(const uint8_t* ptr, std::size_t sz) : _ptr(ptr), _sz(sz) {}
    const uint8_t* data() const override { return _ptr; }
    std::size_t    size() const override { return _sz; }
};
} // namespace

// ---------------------------------------------------------------------------
// _host_read_multi: assemble [offset, offset+size) from multiple segments.
// Returns true if all bytes were covered, false on genuine gaps.
// ---------------------------------------------------------------------------
bool PrefetchDatasource::_host_read_multi(
    std::size_t offset, std::size_t size, uint8_t* dst)
{
    _ensure_sorted();
    const std::size_t end = offset + size;
    auto it = std::upper_bound(
        _segments.begin(), _segments.end(), offset,
        [](std::size_t off, const Segment& seg) { return off < seg.file_offset; });
    if (it != _segments.begin()) --it;

    std::size_t covered = 0;
    for (auto jt = it; jt != _segments.end() && jt->file_offset < end; ++jt) {
        const std::size_t seg_end = jt->file_offset + jt->size;
        if (seg_end <= offset) continue;
        const std::size_t lo = std::max(jt->file_offset, offset);
        const std::size_t hi = std::min(seg_end, end);
        if (hi <= lo) continue;
        const uint8_t* src = static_cast<const uint8_t*>(jt->pinned_buf) + (lo - jt->file_offset);
        std::memcpy(dst + (lo - offset), src, hi - lo);
        covered += (hi - lo);
    }
    return covered == size;
}

std::unique_ptr<cudf::io::datasource::buffer> PrefetchDatasource::host_read(
    std::size_t offset, std::size_t size)
{
    _n_reads.fetch_add(1, std::memory_order_relaxed);

    // Fast path: single-segment read (zero-copy view into pinned RAM).
    const uint8_t* src = _find(offset, size);
    if (src != nullptr) {
        _n_hits.fetch_add(1, std::memory_order_relaxed);
        _bytes_served.fetch_add(size, std::memory_order_relaxed);
        return std::make_unique<pinned_view_buffer>(src, size);
    }

    // Slow path: assemble from multiple segments into a contiguous buffer.
    auto buf_data = std::vector<uint8_t>(size, 0);
    if (_host_read_multi(offset, size, buf_data.data())) {
        _n_hits.fetch_add(1, std::memory_order_relaxed);
        _bytes_served.fetch_add(size, std::memory_order_relaxed);
    } else {
        _n_misses.fetch_add(1, std::memory_order_relaxed);
    }
    return std::make_unique<cudf::io::datasource::owning_buffer<std::vector<uint8_t>>>(
        std::move(buf_data));
}

// ---------------------------------------------------------------------------
// host_read (in-place): memcpy from pinned RAM into caller's buffer.
// ---------------------------------------------------------------------------
std::size_t PrefetchDatasource::host_read(
    std::size_t offset, std::size_t size, uint8_t* dst)
{
    _n_reads.fetch_add(1, std::memory_order_relaxed);

    // Fast path: single-segment read.
    const uint8_t* src = _find(offset, size);
    if (src != nullptr) {
        _n_hits.fetch_add(1, std::memory_order_relaxed);
        _bytes_served.fetch_add(size, std::memory_order_relaxed);
        std::memcpy(dst, src, size);
        return size;
    }

    // Slow path: assemble from multiple segments.
    std::memset(dst, 0, size);
    if (_host_read_multi(offset, size, dst)) {
        _n_hits.fetch_add(1, std::memory_order_relaxed);
        _bytes_served.fetch_add(size, std::memory_order_relaxed);
    } else {
        _n_misses.fetch_add(1, std::memory_order_relaxed);
    }
    return size;
}
