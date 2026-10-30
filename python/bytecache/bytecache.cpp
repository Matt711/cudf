// SPDX-License-Identifier: Apache-2.0
// Prototype: cross-query byte-range cache exposed as a cudf::io::datasource (see DESIGN.md).
#include "bytecache.hpp"

#include <kvikio/remote_handle.hpp>

#include <execinfo.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <future>
#include <list>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace bytecache {
namespace {

enum class block_state : std::uint8_t { loading, ready, failed };

struct block {
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
  std::uint32_t id{};
  std::size_t size{};
  std::unique_ptr<kvikio::RemoteHandle> handle;
};

/// Reusable staging buffers for fused GETs (avoids page-faulting a fresh 16 MiB buffer per request).
class staging_pool {
 public:
  using buf = std::unique_ptr<std::uint8_t[]>;
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
    if (free_.size() < 256) free_.push_back(std::move(b));
  }

 private:
  std::mutex m_;
  std::vector<buf> free_;
  std::size_t size_{0};
};

class cache {
 public:
  static cache& get()
  {
    static cache c;
    return c;
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

  std::size_t read(remote_file& f, std::size_t off, std::size_t n, std::uint8_t* dst)
  {
    if (n == 0 || off >= f.size) return 0;
    n = std::min(n, f.size - off);
    st_.bytes_requested += n;
    if (nslots_ == 0) {  // caching disabled: straight to the remote handle
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
    };
    std::vector<item> items(nb);

    // Phase 1: classify every block under the lock (hit / claimed by us / cannot be cached).
    {
      std::unique_lock l(m_);
      for (std::size_t i = 0; i < nb; ++i) {
        std::uint64_t const key = (static_cast<std::uint64_t>(f.id) << 32) | (b0 + i);
        auto it                 = map_.find(key);
        if (it != map_.end()) {
          block* b = it->second;
          b->pins++;
          items[i] = {b, kind::hit};
          if (b->st == block_state::ready) touch(b);
          continue;
        }
        std::uint32_t slot;
        if (!alloc_slot(slot)) {
          items[i] = {nullptr, kind::uncached};
          continue;
        }
        auto* b = new block;
        b->key  = key;
        b->slot = slot;
        b->len  = static_cast<std::uint32_t>(std::min<std::size_t>(B, f.size - (b0 + i) * B));
        b->pins = 1;
        map_.emplace(key, b);
        items[i] = {b, kind::owner};
      }
    }

    auto copy_block_to_dst = [&](std::size_t bi, std::uint8_t const* src) -> std::size_t {
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
    try {
      for (std::size_t i = 0; i < nb;) {
        if (items[i].k == kind::hit) {
          ++i;
          continue;
        }
        std::size_t j = i;
        while (j + 1 < nb && items[j + 1].k != kind::hit && (j + 1 - i) < max_blocks) ++j;
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
        pieces.push_back(std::move(p));
        i = j + 1;
      }
      // Phase 3: wait for the fetches, publish owned blocks, copy into dst.
      for (auto& p : pieces) {
        if (p.fut.get() != p.flen) throw std::runtime_error("bytecache: short remote read");
        std::size_t const foff = (b0 + p.first) * B;
        for (std::size_t i = p.first; i <= p.last; ++i) {
          std::size_t const bi = b0 + i;
          std::uint8_t const* src = p.buf.get() + (bi * B - foff);
          copy_block_to_dst(bi, src);
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
      }
    } catch (...) {
      fail_owned(items.data(), nb, [](item const& it) { return it.k == kind::owner; });
      release_hits(items.data(), nb, [](item const& it) { return it.k == kind::hit; });
      throw;
    }

    // Phase 4: blocks that were resident or fetched by another thread.
    std::vector<std::size_t> retry;
    for (std::size_t i = 0; i < nb; ++i) {
      if (items[i].k != kind::hit) continue;
      block* b = items[i].blk;
      {
        std::unique_lock l(m_);
        if (b->st == block_state::loading) {
          st_.waits_on_inflight += 1;
          cv_.wait(l, [&] { return b->st != block_state::loading; });
        }
      }
      if (b->st == block_state::ready) {
        st_.bytes_hit += copy_block_to_dst(b0 + i, arena_ + static_cast<std::size_t>(b->slot) * B);
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

  stats snapshot()
  {
    stats s;
    s.bytes_requested   = st_.bytes_requested;
    s.bytes_hit         = st_.bytes_hit;
    s.bytes_fetched     = st_.bytes_fetched;
    s.blocks_evicted    = st_.blocks_evicted;
    s.gets_issued       = st_.gets_issued;
    s.waits_on_inflight = st_.waits_on_inflight;
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
    for (auto* lst : {&probation_, &protected_}) {
      for (auto it = lst->begin(); it != lst->end();) {
        block* b = *it;
        if (b->pins == 0) {
          it = lst->erase(it);
          map_.erase(b->key);
          free_slots_.push_back(b->slot);
          delete b;
        } else {
          ++it;
        }
      }
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

  bool alloc_slot(std::uint32_t& slot)
  {
    if (!free_slots_.empty()) {
      slot = free_slots_.back();
      free_slots_.pop_back();
      return true;
    }
    for (auto* lst : {&probation_, &protected_}) {
      for (auto it = lst->begin(); it != lst->end(); ++it) {
        block* b = *it;
        if (b->pins != 0) continue;
        slot = b->slot;
        lst->erase(it);
        map_.erase(b->key);
        delete b;
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

void configure(config const& cfg)
{
  if (std::getenv("BYTECACHE_DEBUG")) signal(SIGSEGV, segv_handler);  // debug aid: native backtrace on a crash
  cache::get().configure(cfg);
}
stats get_stats() { return cache::get().snapshot(); }
void reset_stats() { cache::get().reset_stats(); }
void clear() { cache::get().clear(); }

struct cached_source::impl {
  std::shared_ptr<remote_file> file;
};

cached_source::cached_source(std::string path, std::size_t size) : p_(std::make_unique<impl>())
{
  p_->file = cache::get().open(path, size);
}

cached_source::~cached_source() = default;

std::size_t cached_source::size() const { return p_->file->size; }

std::size_t cached_source::host_read(std::size_t offset, std::size_t size, std::uint8_t* dst)
{
  return cache::get().read(*p_->file, offset, size, dst);
}

std::unique_ptr<cudf::io::datasource::buffer> cached_source::host_read(std::size_t offset,
                                                                       std::size_t size)
{
  std::size_t const avail = offset < p_->file->size ? std::min(size, p_->file->size - offset) : 0;
  std::vector<std::uint8_t> out(avail);
  cache::get().read(*p_->file, offset, avail, out.data());
  return cudf::io::datasource::buffer::create(std::move(out));
}

}  // namespace bytecache
