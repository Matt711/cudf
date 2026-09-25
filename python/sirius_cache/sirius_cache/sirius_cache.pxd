# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stddef cimport size_t
from libc.stdint cimport int64_t, uintptr_t
from cuda.bindings.cyruntime cimport cudaStream_t

# pylibcudf Datasource base class (so SourceInfo accepts SiriusDatasource)
from pylibcudf.io.datasource cimport Datasource as PlcDatasource
from pylibcudf.libcudf.io.datasource cimport datasource as cudf_datasource


# ---------------------------------------------------------------------------
# C++ declarations
# ---------------------------------------------------------------------------

cdef extern from "cudf/io/text/byte_range_info.hpp" namespace "cudf::io::text" nogil:
    cdef cppclass byte_range_info:
        byte_range_info() except +
        byte_range_info(int64_t offset, int64_t size) except +
        int64_t offset() const
        int64_t size() const


cdef extern from "cudf/io/streaming/io/sirius_datasource.hpp" \
        namespace "cudf_streaming::io" nogil:
    cdef cppclass sirius_datasource(cudf_datasource):
        size_t size() const


# ---------------------------------------------------------------------------
# Inline C++ bridge (change A: replaces sirius_cache/_impl.hpp)
# ---------------------------------------------------------------------------

cdef extern from * namespace "sirius_cache_impl" nogil:
    """
    #include <Python.h>

    #include <cucascade/memory/config.hpp>
    #include <cucascade/memory/memory_reservation_manager.hpp>
    #include <cudf/io/streaming/io/cache/config.hpp>
    #include <cudf/io/streaming/io/cache/prefetching_cache.hpp>
    #include <cudf/io/streaming/io/cache/types.hpp>
    #include <cudf/io/streaming/io/datasource_factory.hpp>
    #include <cudf/io/streaming/io/io_context.hpp>
    #include <cudf/io/streaming/io/object_store_config.hpp>
    #include <cudf/io/streaming/io/sirius_datasource.hpp>
    #include <cudf/io/streaming/scan_manager/config.hpp>
    #include <cudf/io/text/byte_range_info.hpp>

    #include <cstdint>
    #include <memory>
    #include <optional>
    #include <span>
    #include <stdexcept>
    #include <string>
    #include <vector>

    namespace sirius_cache_impl {

    // ─── Configuration bundle passed from Python ─────────────────────────────

    struct session_config {
      std::size_t num_threads{20};
      std::size_t rest_n_reactors{16};
      std::size_t uring_n_reactors{4};
      std::size_t host_capacity_bytes{200UL * 1024 * 1024 * 1024};
      std::size_t host_pool_size_mib{512};
      std::size_t host_initial_number_pools{300};
      double eviction_threshold_fraction{0.8};
      std::string object_store_endpoint;
      std::string object_store_region;
      std::string access_key;
      std::string secret_key;
      std::string session_token;
      bool use_prefetch_cache{true};
    };

    // ─── Helpers ─────────────────────────────────────────────────────────────

    inline std::vector<cucascade::memory::memory_space_config> make_rm_configs(
      const session_config& cfg)
    {
      cucascade::memory::host_memory_space_config hcfg;
      hcfg.numa_id              = -1;
      hcfg.memory_capacity      = cfg.host_capacity_bytes;
      hcfg.block_size           = 1UL << 20;
      hcfg.pool_size            = cfg.host_pool_size_mib;
      hcfg.initial_number_pools = cfg.host_initial_number_pools;
      hcfg.make_portable        = true;
      return {hcfg};
    }

    inline cudf_streaming::scan_manager::scan_manager_config make_scan_cfg(
      const session_config& cfg)
    {
      cudf_streaming::scan_manager::scan_manager_config sc;
      sc.uring_n_reactors = cfg.uring_n_reactors;
      sc.rest_n_reactors  = cfg.rest_n_reactors;
      if (cfg.num_threads > 0) {
        sc.kvikio.nthreads = static_cast<unsigned int>(cfg.num_threads);
      }
      sc.object_store.endpoint      = cfg.object_store_endpoint;
      sc.object_store.region        = cfg.object_store_region;
      sc.object_store.access_key    = cfg.access_key;
      sc.object_store.secret_key    = cfg.secret_key;
      sc.object_store.session_token = cfg.session_token;
      return sc;
    }

    // ─── SiriusSession ───────────────────────────────────────────────────────

    class SiriusSession {
     public:
      explicit SiriusSession(const session_config& cfg)
        : _rm(make_rm_configs(cfg))
        , _registry(make_scan_cfg(cfg), _rm)
      {
        using namespace cudf_streaming::io;
        _ioctx = _registry.make_ioctx(io_context_type::restful);
        if (!_ioctx) {
          throw std::runtime_error("sirius_cache: failed to build restful ioctx");
        }
        _ioctx->start();
        if (cfg.use_prefetch_cache) {
          cache::config ccfg;
          ccfg.mode                        = cache::cache_mode::sirius;
          ccfg.eviction                    = cache::eviction_policy::lru;
          ccfg.eviction_threshold_fraction = cfg.eviction_threshold_fraction;
          ccfg.apply_mode();
          _ioctx->initialize_cache(_rm, ccfg, nullptr);
        }
      }

      ~SiriusSession()
      {
        if (_ioctx) {
          _ioctx->shutdown_cache();
          _ioctx->shutdown();
        }
      }

      SiriusSession(const SiriusSession&)            = delete;
      SiriusSession& operator=(const SiriusSession&) = delete;
      SiriusSession(SiriusSession&&)                 = delete;
      SiriusSession& operator=(SiriusSession&&)      = delete;

      std::unique_ptr<cudf_streaming::io::sirius_datasource> open(const std::string& path)
      {
        if (!_ioctx) { throw std::runtime_error("sirius_cache: session not initialized"); }
        return _ioctx->open_datasource(path);
      }

      // Pre-warm libcurl connections to @p bucket_url (e.g. "s3://my-bucket").
      void warmup(const std::string& bucket_url) noexcept
      {
        if (_ioctx) { _ioctx->warmup(bucket_url); }
      }

      // Flush and reset the prefetch cache (call between benchmark iterations).
      void reset_caches() noexcept
      {
        if (!_ioctx) { return; }
        auto* c = _ioctx->cache();
        if (!c) { return; }
        c->evict(c->claimed_bytes());
        c->prepare_for_query();
      }

      // Reset per-query epoch counters without evicting cached blocks.
      void prepare_for_query() noexcept
      {
        if (!_ioctx) { return; }
        auto* c = _ioctx->cache();
        if (c) { c->prepare_for_query(); }
      }

      std::string cache_summary() const noexcept
      {
        if (!_ioctx) { return ""; }
        auto* c = _ioctx->cache();
        if (!c) { return ""; }
        return c->summary();
      }

     private:
      cucascade::memory::memory_reservation_manager _rm;
      cudf_streaming::io::io_context_registry       _registry;
      std::shared_ptr<cudf_streaming::io::ioctx>    _ioctx;
    };

    // ─── Free-function wrappers ───────────────────────────────────────────────

    inline void call_fadvise(cudf_streaming::io::sirius_datasource& ds,
                             const std::vector<cudf::io::text::byte_range_info>& ranges,
                             int device_id_or_neg1)
    {
      std::optional<int> dev_id;
      if (device_id_or_neg1 >= 0) { dev_id = device_id_or_neg1; }
      ds.fadvise(
        std::span<const cudf::io::text::byte_range_info>(ranges.data(), ranges.size()),
        dev_id);
    }

    // Returns a raw pointer; caller wraps it in unique_ptr (takes ownership).
    inline cudf_streaming::io::sirius_datasource* call_duplicate(
        const cudf_streaming::io::sirius_datasource& ds)
    {
      return ds.duplicate().release();
    }

    // Returns the raw prepare_result integer so Python can interpret it.
    inline int call_prepare_prefetch(cudf_streaming::io::sirius_datasource& ds,
                                     bool wait_for_eviction)
    {
      return static_cast<int>(ds.prepare_prefetch(wait_for_eviction));
    }

    // Fire-and-forget prefetch with no callback.
    inline void call_prefetch_async(cudf_streaming::io::sirius_datasource& ds)
    {
      ds.prefetch_async([](bool) noexcept {});
    }

    // Prefetch with a GIL-safe Python callback (py_cb_raw is a borrowed PyObject*).
    inline void call_prefetch_async_cb(cudf_streaming::io::sirius_datasource& ds,
                                        void* py_cb_raw) noexcept
    {
      auto* py_cb = static_cast<PyObject*>(py_cb_raw);
      Py_XINCREF(py_cb);
      ds.prefetch_async([py_cb](bool ok) noexcept {
        PyGILState_STATE gstate = PyGILState_Ensure();
        PyObject* arg = PyBool_FromLong(ok ? 1 : 0);
        if (arg) {
          PyObject* res = PyObject_CallOneArg(py_cb, arg);
          Py_XDECREF(res);
          Py_DECREF(arg);
        }
        Py_DECREF(py_cb);
        PyGILState_Release(gstate);
      });
    }

    inline void call_await_inflight_prefetch(cudf_streaming::io::sirius_datasource& ds) noexcept
    {
      ds.prefetch_async([](bool) noexcept {});
      ds.await_inflight_prefetch();
    }

    // Advance the consumer stage on the datasource's prefetching handle.
    inline void call_update(cudf_streaming::io::sirius_datasource& ds, int stage) noexcept
    {
      ds.update(static_cast<cudf_streaming::io::cache::scan_stage>(stage));
    }

    inline std::size_t call_device_read_ranges_async(
        cudf_streaming::io::sirius_datasource& ds,
        const std::vector<std::size_t>& file_offsets,
        const std::vector<std::size_t>& sizes,
        const std::vector<std::uintptr_t>& dst_ptrs,
        cudaStream_t raw_stream)
    {
      if (file_offsets.empty()) { return 0; }
      rmm::cuda_stream_view stream{raw_stream};
      std::vector<cudf_streaming::io::slice> slices;
      slices.reserve(file_offsets.size());
      for (std::size_t i = 0; i < file_offsets.size(); ++i) {
        if (sizes[i] == 0) { continue; }
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        slices.emplace_back(
          file_offsets[i],
          sizes[i],
          reinterpret_cast<std::uint8_t*>(dst_ptrs[i]));
      }
      if (slices.empty()) { return 0; }
      return ds.device_read_ranges_async(
                 std::span<const cudf_streaming::io::slice>(slices.data(), slices.size()),
                 stream)
             .get();
    }

    }  // namespace sirius_cache_impl
    """

    cdef cppclass session_config:
        size_t num_threads
        size_t rest_n_reactors
        size_t uring_n_reactors
        size_t host_capacity_bytes
        size_t host_pool_size_mib
        size_t host_initial_number_pools
        double eviction_threshold_fraction
        string object_store_endpoint
        string object_store_region
        string access_key
        string secret_key
        string session_token
        bool use_prefetch_cache

    cdef cppclass SiriusSession:
        SiriusSession(const session_config& cfg) except +
        unique_ptr[sirius_datasource] open(const string& path) except +
        void warmup(const string& bucket_url) noexcept
        void reset_caches() noexcept
        void prepare_for_query() noexcept
        string cache_summary() noexcept

    void call_fadvise(
        sirius_datasource& ds,
        const vector[byte_range_info]& ranges,
        int device_id_or_neg1,
    ) except +

    sirius_datasource* call_duplicate(const sirius_datasource& ds) except +

    int call_prepare_prefetch(sirius_datasource& ds, bool wait_for_eviction) except +

    void call_prefetch_async(sirius_datasource& ds) noexcept

    void call_prefetch_async_cb(sirius_datasource& ds, void* py_cb) noexcept

    void call_await_inflight_prefetch(sirius_datasource& ds) noexcept

    void call_update(sirius_datasource& ds, int stage) noexcept

    size_t call_device_read_ranges_async(
        sirius_datasource& ds,
        const vector[size_t]& file_offsets,
        const vector[size_t]& sizes,
        const vector[uintptr_t]& dst_ptrs,
        cudaStream_t raw_stream,
    ) except +


# ---------------------------------------------------------------------------
# Python extension types (declared here so other pyx files can cimport them)
# ---------------------------------------------------------------------------

cdef class ScanManagerConfig:
    cdef session_config _cfg


cdef class IoContextRegistry:
    cdef unique_ptr[SiriusSession] _session


cdef class SiriusDatasource(PlcDatasource):
    cdef unique_ptr[sirius_datasource] _ds
    cdef cudf_datasource* get_datasource(self) except * nogil


cdef class PrefetchingHandle:
    cdef object _datasource   # Python ref → keeps SiriusDatasource alive
