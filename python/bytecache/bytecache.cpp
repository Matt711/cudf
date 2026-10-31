// SPDX-License-Identifier: Apache-2.0
// Prototype: cross-query byte-range cache exposed as a cudf::io::datasource (see DESIGN.md).
#include "bytecache.hpp"

#include <kvikio/remote_handle.hpp>

#include <execinfo.h>
#include <pthread.h>
#include <signal.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <future>
#include <list>
#include <mutex>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace bytecache {
namespace {

enum class block_state : std::uint8_t { loading, ready, failed };

struct segment {
  std::uint64_t start, end;
  std::uint32_t col;
};

struct column_info {
  std::string name;
  std::atomic<std::uint64_t> registered{0}, requested{0}, hit{0}, fetched{0}, resident{0};
  std::set<std::uint64_t> keys;  ///< keys of resident blocks, ordered (guarded by the cache mutex)
};

struct block {
  std::uint32_t col{0};
  std::uint64_t key{};
  std::uint32_t slot{};
  std::uint32_t len{};   // valid bytes in the slot
  std::uint32_t pins{};  // readers/fetchers currently using the block
  block_state st{block_state::loading};
  bool is_protected{false};
  bool in_list{false};
  std::list<block*>::iterator it{};
};

struct remote_file {
  std::string path;
  std::shared_ptr<const std::vector<segment>> layout;  ///< sorted by start; null until registered
  std::uint32_t id{};
  std::size_t size{};
  std::unique_ptr<kvikio::RemoteHandle> handle;
};

std::size_t env_size(char const* name, std::size_t def);

/// Reusable staging buffers for fused GETs (avoids page-faulting a fresh 16 MiB buffer per request).
class staging_pool {
 public:
  using buf = std::unique_ptr<std::uint8_t[]>;
  staging_pool() : retain_(env_size("BYTECACHE_STAGING_RETAIN", 256)) {}
  buf acquire(std::size_t n)
  {
    {
      std::lock_guard l(m_);
      if (!free_.empty() && size_ == n) {
        auto b = std::move(free_.back());
        free_.pop_back();
        return b;
      }
      size_ = n;
    }
    return buf(new std::uint8_t[n]);
  }
  void release(buf b)
  {
    std::lock_guard l(m_);
    if (free_.size() < retain_) free_.push_back(std::move(b));
  }

 private:
  std::mutex m_;
  std::vector<buf> free_;
  std::size_t size_{0};
  std::size_t retain_;
};

/// Fixed-size pool for blocking reads. Never destroyed (threads detach) so shutdown order cannot crash the process.
class io_pool {
 public:
  explicit io_pool(std::size_t n, char const* name = "bc-io")
  {
    for (std::size_t i = 0; i < n; ++i) {
      std::thread([this, name] {
        pthread_setname_np(pthread_self(), name);
        for (;;) {
          std::function<void()> job;
          {
            std::unique_lock l(m_);
            cv_.wait(l, [this] { return !q_.empty(); });
            job = std::move(q_.front());
            q_.pop_front();
          }
          job();
        }
      }).detach();
    }
  }
  template <class F>
  auto submit(F&& f) -> std::future<decltype(f())>
  {
    using R   = decltype(f());
    auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
    auto fut  = task->get_future();
    {
      std::lock_guard l(m_);
      q_.emplace_back([task] { (*task)(); });
    }
    cv_.notify_one();
    return fut;
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> q_;
};

io_pool& pool()
{
  static io_pool* p = [] {
    char const* e = std::getenv("BYTECACHE_THREADS");
    return new io_pool(e ? std::max(1, std::atoi(e)) : 512);
  }();
  return *p;
}

std::atomic<std::uint64_t> g_ns_open{0}, g_ns_pinned_wait{0}, g_ns_read{0}, g_ns_h2d{0}, g_ns_budget_wait{0}, g_device_reads{0}, g_ns_queue_wait{0}, g_inflight{0}, g_max_inflight{0}, g_host_reads{0}, g_ns_host_read{0}, g_bytes_prefetched{0}, g_prefetch_jobs{0}, g_queued{0};

constexpr std::size_t kMaxColumns = 4096;
constexpr int kBuckets = 128;
std::atomic<std::uint64_t> g_hist_submit[kBuckets], g_hist_done[kBuckets];
std::atomic<std::int64_t> g_t0_ns{0};

int bucket_now()
{
  auto const now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
  std::int64_t expected = 0;
  g_t0_ns.compare_exchange_strong(expected, now);  // first caller sets t0
  auto const b = (now - g_t0_ns.load()) / 1000000000;
  return static_cast<int>(std::min<std::int64_t>(b, kBuckets - 1));
}

struct job_times { double queue_ms{0}, pinned_ms{0}, read_ms{0}, h2d_ms{0}; };
thread_local job_times t_job;
std::mutex g_slow_m;
std::vector<std::string> g_slow;

struct stage_timer {
  std::atomic<std::uint64_t>& sink;
  std::chrono::steady_clock::time_point t0{std::chrono::steady_clock::now()};
  explicit stage_timer(std::atomic<std::uint64_t>& s) : sink(s) {}
  ~stage_timer()
  {
    sink += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
  }
};

std::size_t env_size(char const* name, std::size_t def)
{
  char const* e = std::getenv(name);
  return e ? static_cast<std::size_t>(std::max<long long>(1, std::atoll(e))) : def;
}

/// Counting budget with all-or-nothing acquisition (a reader reserves everything it needs up front, so there is no hold-and-wait).
class counting_budget {
 public:
  explicit counting_budget(std::size_t cap) : cap_(cap) {}
  std::size_t capacity() const { return cap_; }
  void acquire(std::size_t k)
  {
    if (k > cap_) throw std::runtime_error("bytecache: a single read needs more staging buffers than the budget allows");
    std::unique_lock l(m_);
    cv_.wait(l, [&] { return used_ + k <= cap_; });
    used_ += k;
  }
  void release(std::size_t k)
  {
    {
      std::lock_guard l(m_);
      used_ -= k;
    }
    cv_.notify_all();
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  std::size_t cap_, used_{0};
};

/// Bound on concurrently held GET staging buffers (BYTECACHE_GET_BUFFERS x max_get_bytes, default 512 x 16 MiB = 8 GiB).
counting_budget& get_budget()
{
  static counting_budget* b = new counting_budget(env_size("BYTECACHE_GET_BUFFERS", 512));
  return *b;
}

/// Bound on temporary buffers for reads larger than one pinned buffer (BYTECACHE_BIG_MB, default 4096 MiB), in 16 MiB units.
counting_budget& big_budget()
{
  static counting_budget* b = new counting_budget(env_size("BYTECACHE_BIG_MB", 4096) / 16);
  return *b;
}

/// Fixed-size pool of pinned buffers for host-to-device copies, shared by all threads and bounded
/// (BYTECACHE_PINNED_BUFFERS x 16 MiB, default 512 x 16 MiB = 8 GiB). Buffers are created lazily and never freed.
class pinned_pool {
 public:
  pinned_pool(std::size_t n, std::size_t size) : n_(n), size_(size) {}
  std::size_t buffer_size() const { return size_; }
  /// Create every pinned buffer now. cudaMallocHost blocks while kernels are running (about 1 s each under load
  /// versus 5 ms on an idle GPU), so doing this lazily inside a query stalled the reads that needed a new buffer.
  void prefill()
  {
    std::vector<std::uint8_t*> made;
    for (std::size_t i = 0; i < n_; ++i) {
      void* p = nullptr;
      if (cudaMallocHost(&p, size_) != cudaSuccess) break;
      made.push_back(static_cast<std::uint8_t*>(p));
    }
    std::lock_guard l(m_);
    for (auto* p : made) free_.push_back(p);
    created_ += made.size();
  }
  std::uint8_t* acquire()
  {
    std::unique_lock l(m_);
    cv_.wait(l, [&] { return !free_.empty() || created_ < n_; });
    if (!free_.empty()) {
      auto* p = free_.back();
      free_.pop_back();
      return p;
    }
    ++created_;
    l.unlock();
    void* p = nullptr;
    if (cudaMallocHost(&p, size_) != cudaSuccess) {
      std::lock_guard g(m_);
      --created_;
      cv_.notify_one();
      throw std::runtime_error("bytecache: cudaMallocHost failed");
    }
    return static_cast<std::uint8_t*>(p);
  }
  void release(std::uint8_t* p)
  {
    {
      std::lock_guard l(m_);
      free_.push_back(p);
    }
    cv_.notify_one();
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  std::vector<std::uint8_t*> free_;
  std::size_t n_, size_, created_{0};
};

pinned_pool& pinned()
{
  static pinned_pool* p = new pinned_pool(env_size("BYTECACHE_PINNED_BUFFERS", 512), 16 << 20);
  return *p;
}

/// Read-ahead runs on its own threads and its own staging budget so it can never starve demand reads.
io_pool& prefetch_pool()
{
  static io_pool* p = new io_pool(env_size("BYTECACHE_PREFETCH_THREADS", 256), "bc-pf");
  return *p;
}

counting_budget& prefetch_budget()
{
  static counting_budget* b = new counting_budget(env_size("BYTECACHE_PREFETCH_BUFFERS", 256));
  return *b;
}

class cache {
 public:
  static cache& get()
  {
    static cache* c = new cache;  // never destroyed: detached I/O and read-ahead threads may still be running at exit
    return *c;
  }

  void configure(config const& c)
  {
    std::lock_guard l(m_);
    if (configured_) {
      if (c.capacity_bytes != cfg_.capacity_bytes || c.block_size != cfg_.block_size ||
          c.max_get_bytes != cfg_.max_get_bytes) {
        throw std::runtime_error("bytecache: already configured with different settings");
      }
      return;
    }
    if (c.block_size == 0 || c.max_get_bytes < c.block_size || c.max_get_bytes % c.block_size != 0) {
      throw std::invalid_argument("bytecache: max_get_bytes must be a multiple of block_size");
    }
    cfg_        = c;
    configured_ = true;
    {
      char const* pol = std::getenv("BYTECACHE_POLICY");
      column_policy_  = !(pol != nullptr && std::string(pol) == "slru");
    }
    if (cols_.empty()) {
      cols_.reserve(kMaxColumns);  // never reallocated: readers index cols_ without the lock
      column_id("(unknown)");
    }
    if (c.capacity_bytes >= c.block_size) {
      nslots_ = c.capacity_bytes / c.block_size;
      void* p = mmap(nullptr, nslots_ * c.block_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
      if (p == MAP_FAILED) throw std::runtime_error("bytecache: mmap of the arena failed");
      arena_ = static_cast<std::uint8_t*>(p);
      free_slots_.reserve(nslots_);
      for (std::size_t i = nslots_; i-- > 0;) free_slots_.push_back(static_cast<std::uint32_t>(i));
    }
  }

  std::shared_ptr<remote_file> open(std::string const& path, std::size_t known_size)
  {
    {
      std::lock_guard l(m_);
      auto it = files_.find(path);
      if (it != files_.end()) return it->second;
    }
    // Open outside the lock (may issue a HEAD). A duplicate open race is harmless: the first insert wins.
    auto rf = std::make_shared<remote_file>();
    rf->path = path;
    if (known_size > 0) {
      rf->handle = std::make_unique<kvikio::RemoteHandle>(kvikio::RemoteHandle::open(
        path, kvikio::infer_remote_endpoint_type(path), std::nullopt, known_size));
    } else {
      rf->handle = std::make_unique<kvikio::RemoteHandle>(kvikio::RemoteHandle::open(path));
    }
    rf->size = rf->handle->nbytes();
    std::lock_guard l(m_);
    auto [it, inserted] = files_.emplace(path, rf);
    if (inserted) rf->id = next_file_id_++;
    return it->second;
  }

  std::size_t read(remote_file& f, std::size_t off, std::size_t n, std::uint8_t* dst, bool prefetch = false)
  {
    if (n == 0 || off >= f.size) return 0;
    n = std::min(n, f.size - off);
    if (!prefetch) st_.bytes_requested += n;
    counting_budget& budget = prefetch ? prefetch_budget() : get_budget();
    if (prefetch) {
      // Demand never waits for read-ahead: only fetch speculatively while few demand reads are running or queued
      // (BYTECACHE_PREFETCH_GATE, 0 = no gating). During a burst the prefetch backs off and fills the gaps between bursts.
      static std::size_t const gate = env_size("BYTECACHE_PREFETCH_GATE", 64);
      if (std::getenv("BYTECACHE_PREFETCH_GATE") == nullptr || gate > 0) {
        for (int spins = 0; static_cast<std::size_t>(g_inflight.load() + g_queued.load()) > gate && spins < 20000; ++spins) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      }
    }
    if (nslots_ == 0) {  // caching disabled: straight to the remote handle
      if (dst == nullptr) return 0;  // read-ahead / footer warming has nothing to populate
      f.handle->pread(dst, n, off).get();
      st_.bytes_fetched += n;
      st_.gets_issued += 1;
      return n;
    }
    std::size_t const B  = cfg_.block_size;
    std::size_t const b0 = off / B, b1 = (off + n - 1) / B;
    std::size_t const nb = b1 - b0 + 1;

    enum class kind : std::uint8_t { hit, owner, uncached };
    struct item {
      block* blk{nullptr};
      kind k{kind::uncached};
      std::uint32_t col{0};
    };
    std::vector<item> items(nb);

    // Phase 1: classify every block under the lock (hit / claimed by us / cannot be cached).
    {
      std::unique_lock l(m_);
      auto const* layout = f.layout.get();
      for (std::size_t i = 0; i < nb; ++i) {
        std::uint64_t const key = (static_cast<std::uint64_t>(f.id) << 32) | (b0 + i);
        std::size_t const bstart = (b0 + i) * B;
        std::size_t const overlap = std::min(off + n, bstart + B) - std::max(off, bstart);
        auto it                 = map_.find(key);
        if (it != map_.end()) {
          block* b = it->second;
          b->pins++;
          items[i] = {b, kind::hit, b->col};
          if (!prefetch) cols_[b->col]->requested += overlap;
          if (b->st == block_state::ready) touch(b);
          continue;
        }
        std::uint32_t const col = col_of(layout, bstart);
        if (!prefetch) cols_[col]->requested += overlap;
        std::uint32_t slot;
        if (!alloc_slot(col, slot)) {
          items[i] = {nullptr, kind::uncached, col};
          continue;
        }
        auto* b = new block;
        b->col  = col;
        b->key  = key;
        b->slot = slot;
        b->len  = static_cast<std::uint32_t>(std::min<std::size_t>(B, f.size - (b0 + i) * B));
        b->pins = 1;
        map_.emplace(key, b);
        items[i] = {b, kind::owner, col};
      }
    }

    auto copy_block_to_dst = [&](std::size_t bi, std::uint8_t const* src) -> std::size_t {
      if (dst == nullptr) return 0;  // read-ahead: only populate the cache
      std::size_t const bstart = bi * B;
      std::size_t const lo = std::max(off, bstart), hi = std::min(off + n, bstart + B);
      if (hi <= lo) return 0;
      std::memcpy(dst + (lo - off), src + (lo - bstart), hi - lo);
      return hi - lo;
    };

    // Phase 2: fused remote reads for blocks we own or that cannot be cached.
    struct piece {
      std::size_t first, last;  // block indices into items (inclusive)
      std::size_t flen{0};
      staging_pool::buf buf;
      std::future<std::size_t> fut;
    };
    std::vector<piece> pieces;
    std::size_t const max_blocks = cfg_.max_get_bytes / B;
    std::size_t held             = 0;  // GET staging buffers reserved from the shared budget and not yet released
    try {
      std::vector<std::pair<std::size_t, std::size_t>> runs;
      for (std::size_t i = 0; i < nb;) {
        auto skip = [&](std::size_t k) { return items[k].k == kind::hit || (prefetch && items[k].k == kind::uncached); };
        if (skip(i)) {
          ++i;
          continue;
        }
        std::size_t j = i;
        while (j + 1 < nb && !skip(j + 1) && (j + 1 - i) < max_blocks) ++j;
        runs.emplace_back(i, j);
        i = j + 1;
      }
      // Process the runs in waves no larger than the budget; each wave reserves its buffers all at once.
      std::size_t const wave_max = budget.capacity();
      for (std::size_t w = 0; w < runs.size(); w += wave_max) {
        std::size_t const wn = std::min(wave_max, runs.size() - w);
        {
          stage_timer t(g_ns_budget_wait);
          budget.acquire(wn);
        }
        held = wn;
        pieces.clear();
        for (std::size_t r = w; r < w + wn; ++r) {
          auto const [i, j] = runs[r];
          piece p;
          p.first = i;
          p.last  = j;
          p.buf   = pool_.acquire(cfg_.max_get_bytes);
          std::size_t const foff = (b0 + i) * B;
          std::size_t const flen = std::min((b0 + j + 1) * B, f.size) - foff;
          p.flen                 = flen;
          p.fut                  = f.handle->pread(p.buf.get(), flen, foff);
          st_.gets_issued += 1;
          st_.bytes_fetched += flen;
          if (prefetch) g_bytes_prefetched += flen;
          pieces.push_back(std::move(p));
        }
        // Phase 3: wait for the fetches, publish owned blocks, copy into dst.
        for (auto& p : pieces) {
          if (p.fut.get() != p.flen) throw std::runtime_error("bytecache: short remote read");
          std::size_t const foff = (b0 + p.first) * B;
          for (std::size_t i = p.first; i <= p.last; ++i) {
            std::size_t const bi = b0 + i;
            std::uint8_t const* src = p.buf.get() + (bi * B - foff);
            copy_block_to_dst(bi, src);
            cols_[items[i].col]->fetched += std::min<std::size_t>(B, f.size - bi * B);
            if (items[i].k == kind::owner) {
              block* b = items[i].blk;
              std::memcpy(arena_ + static_cast<std::size_t>(b->slot) * B, src, b->len);
              std::lock_guard l(m_);
              b->st = block_state::ready;
              b->pins--;
              add_to_probation(b);
              cv_.notify_all();
              items[i].blk = nullptr;
            }
          }
          pool_.release(std::move(p.buf));
          budget.release(1);
          --held;
        }
      }
    } catch (...) {
      if (held > 0) budget.release(held);
      fail_owned(items.data(), nb, [](item const& it) { return it.k == kind::owner; });
      release_hits(items.data(), nb, [](item const& it) { return it.k == kind::hit; });
      throw;
    }

    // Phase 4: blocks that were resident or fetched by another thread.
    std::vector<std::size_t> retry;
    for (std::size_t i = 0; i < nb; ++i) {
      if (items[i].k != kind::hit) continue;
      block* b = items[i].blk;
      if (prefetch) {  // read-ahead only populates the cache: nothing to copy, nothing to wait for
        std::lock_guard l(m_);
        unpin(b);
        items[i].blk = nullptr;
        continue;
      }
      {
        std::unique_lock l(m_);
        if (b->st == block_state::loading) {
          st_.waits_on_inflight += 1;
          cv_.wait(l, [&] { return b->st != block_state::loading; });
        }
      }
      if (b->st == block_state::ready) {
        auto const got = copy_block_to_dst(b0 + i, arena_ + static_cast<std::size_t>(b->slot) * B);
        st_.bytes_hit += got;
        cols_[items[i].col]->hit += got;
      } else {
        retry.push_back(i);
      }
      std::lock_guard l(m_);
      unpin(b);
      items[i].blk = nullptr;
    }
    for (std::size_t i : retry) {  // the fetching thread failed: read the block range directly
      std::size_t const bstart = (b0 + i) * B;
      std::size_t const lo = std::max(off, bstart), hi = std::min({off + n, bstart + B, f.size});
      if (hi > lo) f.handle->pread(dst + (lo - off), hi - lo, lo).get();
    }
    return n;
  }

  static std::uint32_t col_of(std::vector<segment> const* layout, std::uint64_t offset)
  {
    if (layout == nullptr || layout->empty()) return 0;
    auto it = std::upper_bound(layout->begin(), layout->end(), offset, [](std::uint64_t o, segment const& s) { return o < s.start; });
    if (it == layout->begin()) return 0;
    --it;
    return offset < it->end ? it->col : 0;
  }

  std::uint32_t column_id(std::string const& key)  // requires m_ held
  {
    auto it = col_ids_.find(key);
    if (it != col_ids_.end()) return it->second;
    if (cols_.size() >= kMaxColumns) return 0;  // table full: attribute to "(unknown)"
    auto id = static_cast<std::uint32_t>(cols_.size());
    cols_.push_back(std::make_unique<column_info>());
    cols_.back()->name = key;
    col_ids_.emplace(key, id);
    return id;
  }

  void register_layout(std::string const& path, std::string const& table, std::vector<std::string> const& columns,
                       std::vector<std::uint32_t> const& col_index, std::vector<std::uint64_t> const& offsets,
                       std::vector<std::uint64_t> const& sizes)
  {
    std::lock_guard l(m_);
    auto fit = files_.find(path);
    if (fit == files_.end() || fit->second->layout) return;  // unknown file, or already registered
    std::vector<std::uint32_t> ids(columns.size());
    for (std::size_t i = 0; i < columns.size(); ++i) ids[i] = column_id(table + "|" + columns[i]);
    auto segs = std::make_shared<std::vector<segment>>();
    segs->reserve(offsets.size());
    for (std::size_t i = 0; i < offsets.size(); ++i) {
      std::uint32_t const col = ids[col_index[i]];
      segs->push_back({offsets[i], offsets[i] + sizes[i], col});
      cols_[col]->registered += sizes[i];
    }
    std::sort(segs->begin(), segs->end(), [](segment const& a, segment const& b) { return a.start < b.start; });
    fit->second->layout = std::move(segs);
  }

  /// Submit read-ahead of the given columns of a file. Returns the bytes of the ranges (0 if the file/layout is unknown).
  std::uint64_t prefetch_columns(std::string const& path, std::string const& table, std::vector<std::string> const& columns)
  {
    std::shared_ptr<remote_file> file;
    std::shared_ptr<const std::vector<segment>> layout;
    std::vector<std::uint32_t> ids;
    {
      std::lock_guard l(m_);
      if (nslots_ == 0) return 0;
      auto fit = files_.find(path);
      if (fit == files_.end() || !fit->second->layout) return 0;
      file   = fit->second;
      layout = file->layout;
      for (auto const& name : columns) {
        auto it = col_ids_.find(table + "|" + name);
        if (it != col_ids_.end()) ids.push_back(it->second);
      }
    }
    if (ids.empty()) return 0;
    std::sort(ids.begin(), ids.end());
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;  // coalesced byte ranges, in file order
    std::uint64_t total = 0;
    for (auto const& s : *layout) {
      if (!std::binary_search(ids.begin(), ids.end(), s.col)) continue;
      total += s.end - s.start;
      if (!ranges.empty() && s.start <= ranges.back().second + (1 << 20)) {
        ranges.back().second = std::max(ranges.back().second, s.end);
      } else {
        ranges.emplace_back(s.start, s.end);
      }
    }
    if (ranges.empty()) return 0;
    g_prefetch_jobs += 1;
    // One pool job per range piece (at most 64 MiB), so the ranges of one file are fetched concurrently instead of one
    // GET at a time (a single S3 stream delivers only about 20 to 80 MB/s).
    for (auto const& [start, end] : ranges) {
      for (std::uint64_t pos = start; pos < end; pos += (64ull << 20)) {
        std::uint64_t const len = std::min<std::uint64_t>(64ull << 20, end - pos);
        prefetch_pool().submit([this, file, pos, len] {
          try {
            read(*file, pos, len, nullptr, true);
          } catch (std::exception const& e) {
            // read-ahead is best effort: a failed fetch is simply read again on demand (but record it for diagnosis)
            std::lock_guard l(g_slow_m);
            if (g_slow.size() < 64) g_slow.push_back(std::string("prefetch error: ") + e.what());
          } catch (...) {
          }
          return 0;
        });
      }
    }
    return total;
  }

  std::vector<column_stats> column_snapshot()
  {
    std::lock_guard l(m_);
    std::vector<column_stats> out;
    for (auto const& c : cols_) {
      column_stats s;
      s.name = c->name;
      s.registered_bytes = c->registered;
      s.requested_bytes  = c->requested;
      s.hit_bytes        = c->hit;
      s.fetched_bytes    = c->fetched;
      s.resident_blocks  = c->resident;
      out.push_back(std::move(s));
    }
    return out;
  }

  stats snapshot()
  {
    stats s;
    s.bytes_requested   = st_.bytes_requested;
    s.bytes_hit         = st_.bytes_hit;
    s.bytes_fetched     = st_.bytes_fetched;
    s.blocks_evicted    = st_.blocks_evicted;
    s.gets_issued       = st_.gets_issued;
    s.waits_on_inflight = st_.waits_on_inflight;
    s.ns_open = g_ns_open; s.ns_pinned_wait = g_ns_pinned_wait; s.ns_read = g_ns_read; s.ns_h2d = g_ns_h2d;
    s.ns_budget_wait = g_ns_budget_wait; s.device_reads = g_device_reads;
    s.ns_queue_wait = g_ns_queue_wait; s.max_inflight = g_max_inflight; s.host_reads = g_host_reads; s.ns_host_read = g_ns_host_read;
    s.bytes_prefetched = g_bytes_prefetched; s.prefetch_jobs = g_prefetch_jobs;
    s.inflight_now = g_inflight; s.queued_now = g_queued;
    std::lock_guard l(m_);
    s.blocks_resident = probation_.size() + protected_.size();
    return s;
  }

  void reset_stats()
  {
    st_.bytes_requested = st_.bytes_hit = st_.bytes_fetched = st_.blocks_evicted = 0;
    st_.gets_issued = st_.waits_on_inflight = 0;
  }

  void clear()
  {
    std::lock_guard l(m_);
    std::vector<block*> victims;
    for (auto* lst : {&probation_, &protected_}) {
      for (block* b : *lst) {
        if (b->pins == 0) victims.push_back(b);
      }
    }
    for (block* b : victims) {
      free_slots_.push_back(b->slot);
      evict_block(b);
    }
  }

 private:
  struct atomic_stats {
    std::atomic<std::uint64_t> bytes_requested{0}, bytes_hit{0}, bytes_fetched{0}, blocks_evicted{0},
      gets_issued{0}, waits_on_inflight{0};
  };

  // --- all of the following require m_ held ---
  void add_to_probation(block* b)
  {
    probation_.push_back(b);
    cols_[b->col]->resident += 1;
    cols_[b->col]->keys.insert(b->key);
    b->it         = std::prev(probation_.end());
    b->in_list    = true;
    b->is_protected = false;
  }

  void touch(block* b)
  {
    if (!b->in_list) return;
    if (b->is_protected) {
      protected_.splice(protected_.end(), protected_, b->it);
    } else {
      protected_.splice(protected_.end(), probation_, b->it);
      b->is_protected = true;
    }
    b->it = std::prev(protected_.end());
  }

  /// Remove a resident, unpinned block from every structure and free its memory object (the caller takes its slot).
  void evict_block(block* b)
  {
    (b->is_protected ? protected_ : probation_).erase(b->it);
    cols_[b->col]->keys.erase(b->key);
    cols_[b->col]->resident -= 1;
    map_.erase(b->key);
    delete b;
  }

  /// How many times the column's full size has been requested so far. Re-read columns rank high: caching a column
  /// saves its whole size on every later read, so the saving per cached byte is (reads - 1).
  /// Bytes that belong to no known column (footers, metadata) rank highest because they are tiny and every query re-reads them.
  double priority(std::uint32_t col) const
  {
    if (col == 0) return cols_.size() > 1 ? 1e18 : 0.0;
    auto const reg = cols_[col]->registered.load();
    return reg ? static_cast<double>(cols_[col]->requested.load()) / static_cast<double>(reg) : 0.0;
  }

  /// Find a slot for a block of column `incoming`. Column policy: a block is only admitted if its column ranks above the
  /// lowest-ranked resident column, which then gives up its highest-offset block (so a partly cached column keeps a stable
  /// prefix instead of thrashing under repeated sequential scans); otherwise the block bypasses the cache.
  bool alloc_slot(std::uint32_t incoming, std::uint32_t& slot)
  {
    if (!free_slots_.empty()) {
      slot = free_slots_.back();
      free_slots_.pop_back();
      return true;
    }
    if (column_policy_) {
      std::uint32_t victim = 0;
      double pv            = 1e300;
      bool have            = false;
      for (std::uint32_t id = 0; id < cols_.size(); ++id) {
        if (cols_[id]->keys.empty()) continue;
        double const p = priority(id);
        if (p < pv) {
          pv     = p;
          victim = id;
          have   = true;
        }
      }
      if (!have || pv >= priority(incoming)) return false;
      auto& ks = cols_[victim]->keys;
      for (auto it = ks.rbegin(); it != ks.rend(); ++it) {
        block* b = map_.find(*it)->second;
        if (b->pins != 0) continue;
        slot = b->slot;
        evict_block(b);
        st_.blocks_evicted += 1;
        return true;
      }
      return false;
    }
    for (auto* lst : {&probation_, &protected_}) {
      for (auto it = lst->begin(); it != lst->end(); ++it) {
        block* b = *it;
        if (b->pins != 0) continue;
        slot = b->slot;
        evict_block(b);
        st_.blocks_evicted += 1;
        return true;
      }
    }
    return false;
  }

  void unpin(block* b)
  {
    if (--b->pins == 0 && b->st == block_state::failed) {
      map_.erase(b->key);
      free_slots_.push_back(b->slot);
      delete b;
    }
  }

  template <class Item, class Pred>
  void fail_owned(Item* items, std::size_t n, Pred is_owner)
  {
    std::lock_guard l(m_);
    for (std::size_t i = 0; i < n; ++i) {
      if (!is_owner(items[i]) || items[i].blk == nullptr) continue;
      block* b = items[i].blk;
      b->st    = block_state::failed;
      cv_.notify_all();
      unpin(b);
      items[i].blk = nullptr;
    }
  }

  template <class Item, class Pred>
  void release_hits(Item* items, std::size_t n, Pred is_hit)
  {
    std::lock_guard l(m_);
    for (std::size_t i = 0; i < n; ++i) {
      if (!is_hit(items[i]) || items[i].blk == nullptr) continue;
      unpin(items[i].blk);
      items[i].blk = nullptr;
    }
  }

  std::mutex m_;
  std::condition_variable cv_;
  std::vector<std::unique_ptr<column_info>> cols_;
  std::unordered_map<std::string, std::uint32_t> col_ids_;
  bool column_policy_{true};
  config cfg_{};
  bool configured_{false};
  std::uint8_t* arena_{nullptr};
  std::size_t nslots_{0};
  std::vector<std::uint32_t> free_slots_;
  std::unordered_map<std::uint64_t, block*> map_;
  std::list<block*> probation_, protected_;
  std::unordered_map<std::string, std::shared_ptr<remote_file>> files_;
  std::uint32_t next_file_id_{1};
  staging_pool pool_;
  atomic_stats st_;
};

}  // namespace

namespace {
void segv_handler(int sig)
{
  void* frames[64];
  int n = backtrace(frames, 64);
  backtrace_symbols_fd(frames, n, 2);
  signal(sig, SIG_DFL);
  raise(sig);
}
}  // namespace

void register_layout(std::string const& path, std::string const& table, std::vector<std::string> const& columns,
                     std::vector<std::uint32_t> const& col_index, std::vector<std::uint64_t> const& offsets,
                     std::vector<std::uint64_t> const& sizes)
{
  cache::get().register_layout(path, table, columns, col_index, offsets, sizes);
}
std::vector<column_stats> get_column_stats() { return cache::get().column_snapshot(); }
void warm_footers(std::vector<std::string> const& paths)
{
  std::vector<std::future<int>> futs;
  futs.reserve(paths.size());
  for (auto const& p : paths) {
    futs.push_back(prefetch_pool().submit([p] {
      try {
        auto f                = cache::get().open(p, 0);
        std::size_t const n   = std::min<std::size_t>(f->size, 1 << 20);
        cache::get().read(*f, f->size - n, n, nullptr, true);
      } catch (...) {
        // best effort: the footer is read on demand later
      }
      return 0;
    }));
  }
  for (auto& f : futs) f.get();
}
std::uint64_t prefetch_columns(std::string const& path, std::string const& table, std::vector<std::string> const& columns)
{
  return cache::get().prefetch_columns(path, table, columns);
}

void configure(config const& cfg)
{
  if (std::getenv("BYTECACHE_DEBUG")) signal(SIGSEGV, segv_handler);  // debug aid: native backtrace on a crash
  cache::get().configure(cfg);
  // Pin the staging pool at start-up, while the GPU is idle (BYTECACHE_PREPIN=0 turns this off).
  char const* pre = std::getenv("BYTECACHE_PREPIN");
  if (pre == nullptr || std::atoi(pre) != 0) {
    cudaFree(nullptr);  // make sure a CUDA context exists
    pinned().prefill();
  }
}
stats get_stats() { return cache::get().snapshot(); }
void reset_stats()
{
  cache::get().reset_stats();
  for (int i = 0; i < kBuckets; ++i) { g_hist_submit[i] = 0; g_hist_done[i] = 0; }
  g_t0_ns = 0;
}
std::vector<std::string> slow_reads()
{
  std::lock_guard l(g_slow_m);
  return g_slow;
}
std::vector<std::uint64_t> timeline_submit() { return {g_hist_submit, g_hist_submit + kBuckets}; }
std::vector<std::uint64_t> timeline_done() { return {g_hist_done, g_hist_done + kBuckets}; }
void clear() { cache::get().clear(); }

struct cached_source::impl {
  std::shared_ptr<remote_file> file;
};

cached_source::cached_source(std::string path, std::size_t size) : p_(std::make_unique<impl>())
{
  stage_timer t(g_ns_open);
  p_->file = cache::get().open(path, size);
}

cached_source::~cached_source() = default;

std::size_t cached_source::size() const { return p_->file->size; }

std::size_t cached_source::host_read(std::size_t offset, std::size_t size, std::uint8_t* dst)
{
  g_host_reads += 1;
  stage_timer t(g_ns_host_read);
  return cache::get().read(*p_->file, offset, size, dst);
}

std::future<std::size_t> cached_source::host_read_async(std::size_t offset, std::size_t size, std::uint8_t* dst)
{
  auto file = p_->file;
  return pool().submit([file, offset, size, dst] { return cache::get().read(*file, offset, size, dst); });
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>> cached_source::host_read_async(
  std::size_t offset, std::size_t size)
{
  auto file = p_->file;
  return pool().submit([file, offset, size]() -> std::unique_ptr<cudf::io::datasource::buffer> {
    std::size_t const avail = offset < file->size ? std::min(size, file->size - offset) : 0;
    std::vector<std::uint8_t> out(avail);
    cache::get().read(*file, offset, avail, out.data());
    return cudf::io::datasource::buffer::create(std::move(out));
  });
}

static std::size_t read_to_device(remote_file& f, std::size_t offset, std::size_t size, std::uint8_t* dst,
                                  cuda::stream_ref stream)
{
  std::size_t const avail = offset < f.size ? std::min(size, f.size - offset) : 0;
  auto s                  = static_cast<cudaStream_t>(stream.get());
  auto& pp                = pinned();
  std::size_t const S     = pp.buffer_size();

  auto copy_piece = [&](std::uint8_t const* src, std::size_t pos, std::size_t len) {
    std::uint8_t* stage;
    {
      auto const t0 = std::chrono::steady_clock::now();
      stage_timer t(g_ns_pinned_wait);
      stage = pp.acquire();
      t_job.pinned_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    struct release_guard {
      pinned_pool& pool;
      std::uint8_t* p;
      ~release_guard() { pool.release(p); }
    } guard{pp, stage};
    std::memcpy(stage, src, len);
    auto const t0 = std::chrono::steady_clock::now();
    stage_timer t(g_ns_h2d);
    struct h2d_acc {
      std::chrono::steady_clock::time_point t0;
      ~h2d_acc() { t_job.h2d_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
    } acc{t0};
    if (cudaMemcpyAsync(dst + pos, stage, len, cudaMemcpyHostToDevice, s) != cudaSuccess || cudaStreamSynchronize(s) != cudaSuccess) {
      throw std::runtime_error("bytecache: host-to-device copy failed");
    }
  };

  // Large reads: fetch the whole range at once so its GETs run in parallel (reading it 16 MiB at a time, one
  // after another, made a 716 MB column range take 15 s), then copy to the device through pinned pieces.
  std::size_t const units = (avail + S - 1) / S;
  if (units > 1 && units <= big_budget().capacity()) {
    big_budget().acquire(units);
    struct big_release {
      std::size_t n;
      ~big_release() { big_budget().release(n); }
    } br{units};
    std::unique_ptr<std::uint8_t[]> big(new std::uint8_t[avail]);
    {
      auto const t0 = std::chrono::steady_clock::now();
      stage_timer t(g_ns_read);
      cache::get().read(f, offset, avail, big.get());
      t_job.read_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    for (std::size_t done = 0; done < avail;) {
      std::size_t const len = std::min(S, avail - done);
      copy_piece(big.get() + done, done, len);
      done += len;
    }
    return avail;
  }

  // Small reads (and reads larger than the big-read budget): one pinned buffer at a time.
  for (std::size_t done = 0; done < avail;) {
    std::size_t const len = std::min(S, avail - done);
    std::uint8_t* stage;
    {
      auto const t0 = std::chrono::steady_clock::now();
      stage_timer t(g_ns_pinned_wait);
      stage = pp.acquire();
      t_job.pinned_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    struct release_guard {
      pinned_pool& pool;
      std::uint8_t* p;
      ~release_guard() { pool.release(p); }
    } guard{pp, stage};
    {
      auto const t0 = std::chrono::steady_clock::now();
      stage_timer t(g_ns_read);
      cache::get().read(f, offset + done, len, stage);
      t_job.read_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    auto const t0 = std::chrono::steady_clock::now();
    stage_timer t(g_ns_h2d);
    struct h2d_acc {
      std::chrono::steady_clock::time_point t0;
      ~h2d_acc() { t_job.h2d_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
    } acc{t0};
    if (cudaMemcpyAsync(dst + done, stage, len, cudaMemcpyHostToDevice, s) != cudaSuccess || cudaStreamSynchronize(s) != cudaSuccess) {
      throw std::runtime_error("bytecache: host-to-device copy failed");
    }
    done += len;
  }
  return avail;
}

std::size_t cached_source::device_read(std::size_t offset, std::size_t size, std::uint8_t* dst,
                                       cuda::stream_ref stream)
{
  return read_to_device(*p_->file, offset, size, dst, stream);
}

std::future<std::size_t> cached_source::device_read_async(std::size_t offset, std::size_t size,
                                                          std::uint8_t* dst, cuda::stream_ref stream)
{
  auto file = p_->file;
  g_device_reads += 1;
  g_hist_submit[bucket_now()] += 1;
  g_queued += 1;
  auto const t_submit = std::chrono::steady_clock::now();
  return pool().submit([file, offset, size, dst, stream, t_submit] {
    t_job = job_times{};
    g_queued -= 1;
    auto const t_start = std::chrono::steady_clock::now();
    t_job.queue_ms     = std::chrono::duration<double, std::milli>(t_start - t_submit).count();
    g_ns_queue_wait += static_cast<std::uint64_t>(t_job.queue_ms * 1e6);
    auto const now = ++g_inflight;
    auto prev      = g_max_inflight.load();
    while (now > prev && !g_max_inflight.compare_exchange_weak(prev, now)) {}
    struct dec { ~dec() { --g_inflight; g_hist_done[bucket_now()] += 1; } } d;
    auto const n = read_to_device(*file, offset, size, dst, stream);
    double const total = t_job.queue_ms + std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    if (total > 1000) {
      std::lock_guard l(g_slow_m);
      if (g_slow.size() < 64)
        g_slow.push_back(file->path + " off=" + std::to_string(offset) + " size=" + std::to_string(size) + " total_ms=" + std::to_string((long)total) +
                         " queue_ms=" + std::to_string((long)t_job.queue_ms) + " pinned_ms=" + std::to_string((long)t_job.pinned_ms) +
                         " read_ms=" + std::to_string((long)t_job.read_ms) + " h2d_ms=" + std::to_string((long)t_job.h2d_ms));
    }
    return n;
  });
}

std::unique_ptr<cudf::io::datasource::buffer> cached_source::host_read(std::size_t offset,
                                                                       std::size_t size)
{
  g_host_reads += 1;
  stage_timer t(g_ns_host_read);
  std::size_t const avail = offset < p_->file->size ? std::min(size, p_->file->size - offset) : 0;
  std::vector<std::uint8_t> out(avail);
  cache::get().read(*p_->file, offset, avail, out.data());
  return cudf::io::datasource::buffer::create(std::move(out));
}

}  // namespace bytecache
