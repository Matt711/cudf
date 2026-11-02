// SPDX-License-Identifier: Apache-2.0
#include "cc_glue.hpp"

#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/cache/fs_cache.hpp>
#include <cucascade/io/config.hpp>
#include <cucascade/io/datasource_factory.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

#include <cudf/io/text/byte_range_info.hpp>

#include <cstdlib>
#include <mutex>
#include <optional>
#include <stdexcept>

namespace ccglue {
namespace {
struct state {
  std::unique_ptr<cucascade::memory::memory_reservation_manager> mgr;
  std::unique_ptr<cucascade::io::io_context_registry> reg;
  std::shared_ptr<cucascade::io::ioctx> ctx;
};
std::mutex g_m;
std::unique_ptr<state> g_s;
std::size_t g_min_advise = 0;
bool g_prefetch = std::getenv("CC_PREFETCH") != nullptr;  // the separate low-priority prefetch starves demand reads (see DESIGN notes); off by default

/// Wraps a cuCascade datasource so every sizeable read is first registered with the prefetching cache: cuCascade only keeps chunks that a
/// scan advised (fadvise + prepare + prefetch), and libcudf reads exact column-chunk ranges without advising them. Each read gets its
/// own duplicate datasource (its own cache handle); the duplicates live until this datasource is destroyed, so async reads stay valid.
class advising_source : public cudf::io::datasource {
 public:
  advising_source(std::unique_ptr<cucascade::io::datasource> base, std::size_t min_bytes) : base_(std::move(base)), min_bytes_(min_bytes) {}

  size_t size() const override { return base_->size(); }
  bool supports_device_read() const override { return base_->supports_device_read(); }
  bool is_device_read_preferred(size_t n) const override { return base_->is_device_read_preferred(n); }

  size_t host_read(size_t off, size_t n, uint8_t* dst) override { return pick(off, n).host_read(off, n, dst); }
  std::unique_ptr<buffer> host_read(size_t off, size_t n) override { return pick(off, n).host_read(off, n); }
  std::future<size_t> host_read_async(size_t off, size_t n, uint8_t* dst) override { return pick(off, n).host_read_async(off, n, dst); }
  std::future<std::unique_ptr<buffer>> host_read_async(size_t off, size_t n) override { return pick(off, n).host_read_async(off, n); }
  std::unique_ptr<buffer> device_read(size_t off, size_t n, cucascade::io::cudf_stream_type s) override { return pick(off, n).device_read(off, n, s); }
  size_t device_read(size_t off, size_t n, uint8_t* dst, cucascade::io::cudf_stream_type s) override { return pick(off, n).device_read(off, n, dst, s); }
  std::future<size_t> device_read_async(size_t off, size_t n, uint8_t* dst, cucascade::io::cudf_stream_type s) override
  {
    return pick(off, n).device_read_async(off, n, dst, s);
  }

 private:
  cucascade::io::datasource& pick(size_t off, size_t n)
  {
    if (n < min_bytes_) return *base_;
    auto dup = base_->duplicate();
    std::vector<cudf::io::text::byte_range_info> r;
    r.emplace_back(static_cast<int64_t>(off), static_cast<int64_t>(n));
    dup->fadvise(r, std::nullopt);
    auto const pr = dup->prepare_prefetch(true);
    if (pr == cucascade::io::prepare_result::prepared && g_prefetch) dup->prefetch_async([](bool) noexcept {});
    std::lock_guard l(m_);
    dups_.push_back(std::move(dup));
    return *dups_.back();
  }
  std::unique_ptr<cucascade::io::datasource> base_;
  std::size_t min_bytes_;
  std::mutex m_;
  std::vector<std::unique_ptr<cucascade::io::datasource>> dups_;
};
}  // namespace

void init(options const& o)
{
  std::lock_guard l(g_m);
  if (g_s) return;
  g_min_advise = o.advise_min_bytes;
  auto s = std::make_unique<state>();
  cucascade::memory::host_memory_space_config host;
  host.memory_capacity     = o.cache_bytes;
  host.initial_number_pools = o.initial_pools;
  std::vector<cucascade::memory::memory_space_config> spaces;
  spaces.emplace_back(host);
  s->mgr = std::make_unique<cucascade::memory::memory_reservation_manager>(std::move(spaces));

  cucascade::io::io_config cfg;
  cfg.backend                  = cucascade::io::io_backend::native;
  cfg.rest_n_reactors          = o.rest_reactors;
  cfg.rest.max_connections     = o.max_connections;
  cfg.cache.mode               = cucascade::io::cache::cache_mode::cucs;
  cfg.cache.eviction           = o.lru ? cucascade::io::cache::eviction_policy::lru : cucascade::io::cache::eviction_policy::idle;
  cfg.object_store.endpoint      = o.endpoint;
  cfg.object_store.region        = o.region;
  cfg.object_store.access_key    = o.access_key;
  cfg.object_store.secret_key    = o.secret_key;
  cfg.object_store.session_token = o.session_token;
  cfg.object_store.s3_signing_mode = cucascade::io::object_store_config::signing_mode::header;
  cfg.apply_cache_mode();

  s->reg = std::make_unique<cucascade::io::io_context_registry>(cfg, *s->mgr);
  s->ctx = s->reg->make_ioctx(cucascade::io::io_context_type::restful);
  if (!s->ctx) throw std::runtime_error("cucascade: could not create the REST io context (check endpoint, region and credentials)");
  s->ctx->start();
  if (cfg.cache.use_fs_cache())
    s->ctx->initialize_cache(*s->mgr, cfg.cache, nullptr);
  g_s = std::move(s);
}

std::unique_ptr<cudf::io::datasource> open(std::string const& path, std::uint64_t known_size)
{
  std::shared_ptr<cucascade::io::ioctx> ctx;
  {
    std::lock_guard l(g_m);
    if (!g_s) throw std::runtime_error("cucascade: init() has not been called");
    ctx = g_s->ctx;
  }
  auto base = known_size != 0 ? cucascade::io::open_datasource(ctx, path, known_size)
                              : cucascade::io::open_datasource(ctx, path, cucascade::io::open_hint::parquet_footer_probe);
  if (g_min_advise == 0) return base;
  return std::make_unique<advising_source>(std::move(base), g_min_advise);
}

int fadvise(cudf::io::datasource* ds, std::vector<std::pair<std::uint64_t, std::uint64_t>> const& ranges, bool start)
{
  auto* d = dynamic_cast<cucascade::io::datasource*>(ds);
  if (d == nullptr) throw std::runtime_error("cucascade: fadvise on a datasource that is not a cuCascade datasource");
  std::vector<cudf::io::text::byte_range_info> v;
  v.reserve(ranges.size());
  for (auto const& r : ranges) v.emplace_back(static_cast<int64_t>(r.first), static_cast<int64_t>(r.second));
  d->fadvise(v, std::nullopt);
  if (!start) return -1;
  auto const pr = d->prepare_prefetch(true);
  if (pr != cucascade::io::prepare_result::prepared) return static_cast<int>(pr) * 10 + 9;
  auto const rf = d->prefetch_async([](bool) noexcept {});
  return static_cast<int>(pr) * 10 + static_cast<int>(rf);
}

std::string cache_summary()
{
  std::lock_guard l(g_m);
  if (!g_s || g_s->ctx->cache() == nullptr) return {};
  return g_s->ctx->cache()->summary();
}

void shutdown()
{
  std::lock_guard l(g_m);
  if (!g_s) return;
  g_s->ctx->shutdown_cache();
  g_s.reset();
}

}  // namespace ccglue
