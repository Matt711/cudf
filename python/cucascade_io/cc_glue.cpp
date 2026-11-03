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
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
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
std::atomic<std::uint64_t> g_ns_dup{0}, g_ns_fadvise{0}, g_ns_prepare{0}, g_ns_lock{0}, g_n_pick{0};
inline std::uint64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
std::size_t g_block = std::size_t{1} << 20;
bool g_align = std::getenv("CC_NO_ALIGN") == nullptr;  // advise whole chunks: a cached chunk cannot be widened later, so a partly filled one makes the neighbouring range miss on every repeat
bool g_prefetch = std::getenv("CC_PREFETCH") != nullptr;  // the separate low-priority prefetch starves demand reads (see DESIGN notes); off by default


// ---- column-aware admission (same ranking idea as python/bytecache: a column's rank is how often its full size was requested) ----
struct segment { std::uint64_t start, end; std::uint32_t col; };
struct col_stat { std::atomic<std::uint64_t> registered{0}, requested{0}, admitted{0}, refused{0}; std::uint64_t seen{0}; std::string name; };
std::mutex g_pm;
std::unordered_map<std::string, std::shared_ptr<const std::vector<segment>>> g_layouts;
std::unordered_map<std::string, std::uint32_t> g_col_ids;
std::vector<std::unique_ptr<col_stat>> g_cols;
std::size_t g_budget = 0;
bool g_column_policy = true;
std::uint64_t g_seen_seq = 0;

std::shared_ptr<const std::vector<segment>> layout_of(std::string const& path)
{
  std::lock_guard l(g_pm);
  auto it = g_layouts.find(path);
  return it == g_layouts.end() ? nullptr : it->second;
}

/// Sticky admission. A (file, column) piece that was admitted stays admitted (cuCascade's LRU would otherwise evict hot chunks instead of stale
/// ones whenever the policy changes its mind). New pieces are admitted while the admitted set fits the budget; when it does not, a column that
/// ranks clearly higher than the lowest-ranked admitted column (rank = requested/registered) replaces that column entirely (its chunks then age out).
struct admitted_state {
  std::unordered_set<std::uint64_t> pieces;                    // key = col << 40 ^ hash(path)
  std::unordered_map<std::uint32_t, std::vector<std::uint64_t>> by_col;
  std::unordered_map<std::uint32_t, std::uint64_t> col_bytes;  // admitted bytes per column
  std::uint64_t total{0};
} g_adm;
std::unordered_map<std::uint64_t, std::uint64_t> g_fc_bytes;  // total bytes of one column in one file, from the layout
inline std::uint64_t piece_key(std::uint32_t col, std::string const& path) { return (static_cast<std::uint64_t>(col) << 40) ^ (std::hash<std::string>{}(path) & ((std::uint64_t{1} << 40) - 1)); }
double g_margin = std::getenv("CC_RANK_MARGIN") ? std::atof(std::getenv("CC_RANK_MARGIN")) : 1.5;

double col_rank(std::uint32_t c)
{
  auto const reg = g_cols[c]->registered.load();
  return reg ? static_cast<double>(g_cols[c]->requested.load()) / static_cast<double>(reg) : 0.0;
}

bool g_priority = std::getenv("CC_NO_PRIORITY") == nullptr;  // eviction priority = column rank (cuCascade evicts lowest first)
/// Eviction priority of a column's chunks: 1 + 2 x rank (reads per full column), capped; 0 stays reserved for unranked data.
std::uint8_t col_priority(std::uint32_t c)
{
  std::lock_guard l(g_pm);
  return static_cast<std::uint8_t>(std::min(250.0, 1.0 + 2.0 * col_rank(c)));
}

bool admit_range(std::uint32_t col, std::string const& path, std::uint64_t bytes)
{
  std::lock_guard l(g_pm);
  auto const key = piece_key(col, path);
  if (g_adm.pieces.count(key)) return true;
  if (auto it = g_fc_bytes.find(key); it != g_fc_bytes.end()) bytes = it->second;  // account the whole column chunk set of this file
  if (g_adm.total + bytes > g_budget) {
    // find the lowest-ranked admitted column; replace it only if this column ranks clearly higher
    std::uint32_t low = col;
    double lr         = 1e300;
    for (auto const& [c, b] : g_adm.col_bytes) {
      if (b == 0 || c == col) continue;
      double const r = col_rank(c);
      if (r < lr) { lr = r; low = c; }
    }
    if (low == col || col_rank(col) <= lr * g_margin) return false;
    for (auto k : g_adm.by_col[low]) g_adm.pieces.erase(k);
    g_adm.total -= g_adm.col_bytes[low];
    g_adm.col_bytes[low] = 0;
    g_adm.by_col[low].clear();
    if (g_adm.total + bytes > g_budget) return false;
  }
  g_adm.pieces.insert(key);
  g_adm.by_col[col].push_back(key);
  g_adm.col_bytes[col] += bytes;
  g_adm.total += bytes;
  return true;
}

/// Fixed thread pool for the waiting half of an advised async read: with the prefetch step cuCascade blocks the *calling* thread until the
/// prefetch lands, but libcudf submits its async reads from a handful of threads, so without this each read would be synchronous.
class wait_pool {
 public:
  explicit wait_pool(std::size_t n)
  {
    for (std::size_t i = 0; i < n; ++i) threads_.emplace_back([this] { loop(); });
  }
  ~wait_pool()
  {
    { std::lock_guard l(m_); stop_ = true; }
    cv_.notify_all();
    for (auto& t : threads_) t.join();
  }
  template <class F>
  auto submit(F&& f) -> std::future<decltype(f())>
  {
    using R = decltype(f());
    auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
    auto fut  = task->get_future();
    { std::lock_guard l(m_); q_.emplace_back([task] { (*task)(); }); }
    cv_.notify_one();
    return fut;
  }
 private:
  void loop()
  {
    for (;;) {
      std::function<void()> job;
      {
        std::unique_lock l(m_);
        cv_.wait(l, [this] { return stop_ || !q_.empty(); });
        if (q_.empty()) return;
        job = std::move(q_.front());
        q_.pop_front();
      }
      job();
    }
  }
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> q_;
  std::vector<std::thread> threads_;
  bool stop_{false};
};
wait_pool& async_pool()
{
  static wait_pool p(static_cast<std::size_t>(std::getenv("CC_ASYNC_THREADS") ? std::atoi(std::getenv("CC_ASYNC_THREADS")) : 128));
  return p;
}

/// Wraps a cuCascade datasource so every sizeable read is first registered with the prefetching cache: cuCascade only keeps chunks that a
/// scan advised (fadvise + prepare + prefetch), and libcudf reads exact column-chunk ranges without advising them. Each read gets its
/// own duplicate datasource (its own cache handle); the duplicates live until this datasource is destroyed, so async reads stay valid.
class advising_source : public cudf::io::datasource {
 public:
  advising_source(std::unique_ptr<cucascade::io::datasource> base, std::size_t min_bytes, std::string path)
    : base_(std::move(base)), min_bytes_(min_bytes), path_(std::move(path)) {}

  size_t size() const override { return base_->size(); }
  bool supports_device_read() const override { return base_->supports_device_read(); }
  bool is_device_read_preferred(size_t n) const override { return base_->is_device_read_preferred(n); }

  size_t host_read(size_t off, size_t n, uint8_t* dst) override { return pick(off, n).host_read(off, n, dst); }
  std::unique_ptr<buffer> host_read(size_t off, size_t n) override { return pick(off, n).host_read(off, n); }
  std::future<size_t> host_read_async(size_t off, size_t n, uint8_t* dst) override
  {
    auto& d = pick(off, n);
    if (!g_prefetch || &d == base_.get()) return d.host_read_async(off, n, dst);
    return async_pool().submit([&d, off, n, dst] { return d.host_read(off, n, dst); });  // the read waits for its prefetch: not on the caller's thread
  }
  std::future<std::unique_ptr<buffer>> host_read_async(size_t off, size_t n) override
  {
    auto& d = pick(off, n);
    if (!g_prefetch || &d == base_.get()) return d.host_read_async(off, n);
    return async_pool().submit([&d, off, n] { return d.host_read(off, n); });
  }
  std::unique_ptr<buffer> device_read(size_t off, size_t n, cucascade::io::cudf_stream_type s) override { return pick(off, n).device_read(off, n, s); }
  size_t device_read(size_t off, size_t n, uint8_t* dst, cucascade::io::cudf_stream_type s) override { return pick(off, n).device_read(off, n, dst, s); }
  std::future<size_t> device_read_async(size_t off, size_t n, uint8_t* dst, cucascade::io::cudf_stream_type s) override
  {
    auto& d = pick(off, n);
    if (!g_prefetch || &d == base_.get()) return d.device_read_async(off, n, dst, s);
    return async_pool().submit([&d, off, n, dst, s] { return d.device_read(off, n, dst, s); });
  }

 private:
  cucascade::io::datasource& pick(size_t off, size_t n)
  {
    if (n < min_bytes_) return *base_;
    std::vector<cudf::io::text::byte_range_info> r;
    std::uint8_t prio = 0;
    if (auto lay = layout_of(path_)) {  // registered file: account the read per column, set the eviction priority, and (policy on) admit by column
      std::uint64_t const lo = off, hi = off + n;
      auto it = std::upper_bound(lay->begin(), lay->end(), lo, [](std::uint64_t v, segment const& sg) { return v < sg.end; });
      for (; it != lay->end() && it->start < hi; ++it) {
        std::uint64_t const a = std::max(lo, it->start), b = std::min(hi, it->end);
        g_cols[it->col]->requested += b - a;
        bool const ok = !g_column_policy || admit_range(it->col, path_, b - a);
        if (ok) {
          r.emplace_back(static_cast<int64_t>(a), static_cast<int64_t>(b - a));
          g_cols[it->col]->admitted += b - a;
          if (g_priority) prio = std::max(prio, col_priority(it->col));
        } else {
          g_cols[it->col]->refused += b - a;
        }
      }
      if (r.empty()) return *base_;
    }
    if (r.empty()) r.emplace_back(static_cast<int64_t>(off), static_cast<int64_t>(n));
    if (g_align) {  // widen each range to whole cache chunks (clipped to the object) and merge overlaps
      std::vector<cudf::io::text::byte_range_info> m;
      std::int64_t const B = static_cast<std::int64_t>(g_block), fsz = static_cast<std::int64_t>(base_->size());
      for (auto const& x : r) {
        std::int64_t a = (x.offset() / B) * B, b = std::min(((x.offset() + x.size() + B - 1) / B) * B, fsz);
        if (!m.empty() && a <= m.back().offset() + m.back().size()) b = std::max(b, m.back().offset() + m.back().size()), a = m.back().offset(), m.pop_back();
        m.emplace_back(a, b - a);
      }
      r = std::move(m);
    }
    auto t0 = now_ns();
    auto dup = base_->duplicate();
    auto t1 = now_ns();
    dup->fadvise(r, std::nullopt, prio);
    auto t2 = now_ns();
    auto const pr = dup->prepare_prefetch(true);
    auto t3 = now_ns();
    g_ns_dup += t1 - t0; g_ns_fadvise += t2 - t1; g_ns_prepare += t3 - t2; g_n_pick += 1;
    if (pr == cucascade::io::prepare_result::prepared && g_prefetch) dup->prefetch_async([](bool) noexcept {});
    auto t4 = now_ns();
    std::lock_guard l(m_);
    g_ns_lock += now_ns() - t4;
    dups_.push_back(std::move(dup));
    return *dups_.back();
  }
  std::unique_ptr<cucascade::io::datasource> base_;
  std::size_t min_bytes_;
  std::string path_;
  std::mutex m_;
  std::vector<std::unique_ptr<cucascade::io::datasource>> dups_;
};
}  // namespace

void init(options const& o)
{
  std::lock_guard l(g_m);
  if (g_s) return;
  g_min_advise = o.advise_min_bytes;
  g_block = o.block_bytes;
  g_column_policy = o.column_policy;
  g_budget = static_cast<std::size_t>(static_cast<double>(o.cache_bytes) * o.budget_fraction);
  auto s = std::make_unique<state>();
  cucascade::memory::host_memory_space_config host;
  host.memory_capacity     = o.cache_bytes;
  // cuCascade's default 0.9 silently makes only 90% of the configured capacity usable; the cache size we ask for should be usable.
  host.reservation_limit_fraction = std::getenv("CC_RESV") != nullptr ? std::atof(std::getenv("CC_RESV")) : 1.0;
  host.pool_size            = (std::size_t{128} << 20) / o.block_bytes;
  host.initial_number_pools = o.initial_pools != 0 ? o.initial_pools : (o.cache_bytes + (std::size_t{128} << 20) - 1) / (std::size_t{128} << 20);
  host.initial_pool_threads = o.pool_threads;
  host.block_size           = o.block_bytes;
  host.pool_size            = (std::size_t{128} << 20) / o.block_bytes;  // 128 MiB per pinned pool, as with the default block size
  std::vector<cucascade::memory::memory_space_config> spaces;
  spaces.emplace_back(host);
  s->mgr = std::make_unique<cucascade::memory::memory_reservation_manager>(std::move(spaces));

  cucascade::io::io_config cfg;
  cfg.backend                  = cucascade::io::io_backend::native;
  cfg.rest_n_reactors          = o.rest_reactors;
  cfg.rest.max_connections     = o.max_connections;
  if (auto const* b = std::getenv("CC_STALL_B")) cfg.rest.stall_speed_limit_bytes = std::atol(b);  // a GET below this many bytes/s for CC_STALL_S seconds (1 B/s: no byte at all) is aborted and retried
  if (auto const* s = std::getenv("CC_STALL_S")) cfg.rest.stall_time_s = std::atol(s);
  cfg.cache.mode               = cucascade::io::cache::cache_mode::cucs;
  cfg.cache.eviction_threshold_fraction = o.evict_threshold;
  cfg.cache.eviction_batch_fraction     = o.evict_batch;
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
  return std::make_unique<advising_source>(std::move(base), g_min_advise, path);
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

void register_layout(std::string const& path, std::string const& dir, std::vector<std::string> const& names, std::vector<std::uint32_t> const& col_index,
                     std::vector<std::uint64_t> const& offsets, std::vector<std::uint64_t> const& sizes)
{
  std::lock_guard l(g_pm);
  if (g_layouts.count(path)) return;
  auto segs = std::make_shared<std::vector<segment>>();
  std::vector<std::uint32_t> ids(names.size());
  for (std::size_t i = 0; i < names.size(); ++i) {
    auto key = dir + "/" + names[i];
    auto [it, fresh] = g_col_ids.emplace(key, static_cast<std::uint32_t>(g_cols.size()));
    if (fresh) { g_cols.push_back(std::make_unique<col_stat>()); g_cols.back()->seen = ++g_seen_seq; g_cols.back()->name = key; }
    ids[i] = it->second;
  }
  for (std::size_t k = 0; k < col_index.size(); ++k) {
    segs->push_back({offsets[k], offsets[k] + sizes[k], ids[col_index[k]]});
    {  // budget accounting uses the cache footprint: the chunk-aligned span of the range (a 0.5 MB column chunk still occupies whole chunks)
      std::uint64_t const B = g_block, lo = (offsets[k] / B) * B, hi = ((offsets[k] + sizes[k] + B - 1) / B) * B;
      g_fc_bytes[piece_key(ids[col_index[k]], path)] += hi - lo;
    }
    g_cols[ids[col_index[k]]]->registered += sizes[k];
  }
  std::sort(segs->begin(), segs->end(), [](segment const& a, segment const& b) { return a.start < b.start; });
  g_layouts.emplace(path, std::move(segs));
}

std::string column_report()
{
  std::lock_guard l(g_pm);
  std::string out;
  for (std::uint32_t id = 0; id < g_cols.size(); ++id) {
    auto const& c = g_cols[id];
    auto it       = g_adm.col_bytes.find(id);
    out += "[adm_aligned=" + std::to_string((it == g_adm.col_bytes.end() ? 0 : it->second) >> 20) + "MiB rank=" + std::to_string(col_rank(id)) + "] ";
    out += c->name + " reg=" + std::to_string(c->registered.load() >> 20) + "MiB req=" + std::to_string(c->requested.load() >> 20) + "MiB admitted=" +
           std::to_string(c->admitted.load() >> 20) + "MiB refused=" + std::to_string(c->refused.load() >> 20) + "MiB\n";
  }
  return out;
}

std::string cache_summary()
{
  std::lock_guard l(g_m);
  if (!g_s || g_s->ctx->cache() == nullptr) return {};
  return g_s->ctx->cache()->summary() + " | picks=" + std::to_string(g_n_pick.load()) + " ms: dup=" + std::to_string(g_ns_dup.load() / 1000000) + " fadvise=" + std::to_string(g_ns_fadvise.load() / 1000000) + " prepare=" + std::to_string(g_ns_prepare.load() / 1000000) + " lock=" + std::to_string(g_ns_lock.load() / 1000000);
}

void shutdown()
{
  std::lock_guard l(g_m);
  if (!g_s) return;
  g_s->ctx->shutdown_cache();
  g_s.reset();
}

}  // namespace ccglue
