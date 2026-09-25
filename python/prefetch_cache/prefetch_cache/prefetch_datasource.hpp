#pragma once
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// Must come before any cudf headers to stub out cuda::mul_overflow on older CCCL.
#include "prefetch_cache/cuda_compat.hpp"

#include <cudf/io/datasource.hpp>
#include <cuda/stream_ref>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <stdexcept>
#include <vector>

class PrefetchDatasource : public cudf::io::datasource {
public:
    // -----------------------------------------------------------------------
    // Single-segment constructor (original API, unchanged).
    // Downloads [base_offset, base_offset+data_size) into pinned RAM.
    // -----------------------------------------------------------------------
    PrefetchDatasource(std::size_t file_size,
                       std::size_t base_offset,
                       const uint8_t* data_bytes,
                       std::size_t data_size);

    // -----------------------------------------------------------------------
    // Multi-segment constructor: start with an empty segment list.
    // Call add_segment() to populate, then the datasource is ready for use.
    // -----------------------------------------------------------------------
    explicit PrefetchDatasource(std::size_t file_size);

    // Add one sparse segment.  data_bytes is copied into a new cudaMallocHost
    // buffer.  Segments need not be added in order.
    void add_segment(std::size_t file_offset,
                     const uint8_t* data_bytes,
                     std::size_t data_size);

    // Sort segments by file_offset.  MUST be called once after all add_segment()
    // calls complete, before any concurrent host_read/device_read_async calls.
    // _ensure_sorted() is NOT thread-safe; this public method exists so callers
    // can sort on the construction thread before handing the datasource to readers.
    void finalize_segments();

    ~PrefetchDatasource() override;

    // Return true so libcudf calls device_read_async directly (our cudaMemcpyAsync
    // on a private H2D stream + event fence on the compute stream).  Allows the
    // H2D DMA copy engine to overlap with SM compute across the 32 reactor streams.
    bool supports_device_read() const override { return true; }
    bool is_device_read_preferred(std::size_t) const override { return true; }
    std::size_t size() const override { return _file_size; }

    // Fire cudaMemcpyAsync from pinned RAM.
    std::future<std::size_t> device_read_async(
        std::size_t offset, std::size_t size,
        uint8_t* dst, cuda::stream_ref stream) override;

    // Host reads (serve from whichever segment covers the range).
    std::unique_ptr<buffer> host_read(std::size_t offset, std::size_t size) override;
    std::size_t host_read(std::size_t offset, std::size_t size, uint8_t* dst) override;

    // Stats accessors.
    std::size_t n_reads()      const { return _n_reads.load(std::memory_order_relaxed); }
    std::size_t n_hits()       const { return _n_hits.load(std::memory_order_relaxed); }
    std::size_t n_misses()     const { return _n_misses.load(std::memory_order_relaxed); }
    std::size_t bytes_served() const { return _bytes_served.load(std::memory_order_relaxed); }

private:
    struct Segment {
        std::size_t file_offset;  // absolute offset within the parquet file
        void*       pinned_buf;   // cudaMallocHost allocation
        std::size_t size;         // bytes stored
    };

    std::vector<Segment> _segments;   // sorted by file_offset after first access
    std::size_t          _file_size;
    bool                 _sorted{false};

    // Private H2D copy stream: copies are issued here (not on the caller's
    // compute stream), matching kvikio_source behavior.  A per-call CUDA event
    // then makes the compute stream wait for copies before proceeding.
    cudaStream_t _h2d_stream{nullptr};

    std::atomic<std::size_t> _n_reads{0};
    std::atomic<std::size_t> _n_hits{0};
    std::atomic<std::size_t> _n_misses{0};
    std::atomic<std::size_t> _bytes_served{0};

    // Ensure _segments is sorted; called lazily on first read.
    void _ensure_sorted();

    // Returns pointer to byte at `offset` (size bytes available) or nullptr on miss.
    const uint8_t* _find(std::size_t offset, std::size_t size);

    // Assembles [offset, offset+size) from multiple segments into dst.
    // Returns true if all bytes were covered; false on genuine gaps.
    bool _host_read_multi(std::size_t offset, std::size_t size, uint8_t* dst);
};
