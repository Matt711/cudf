/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scan_io.hpp"

#include <cucascade/io/config.hpp>
#include <cucascade/io/details/std_future.hpp>
#include <cucascade/io/types.hpp>
#include <cucascade/memory/config.hpp>

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/stream>

#include <algorithm>
#include <array>
#include <cstring>
#include <future>
#include <numeric>
#include <span>
#include <stdexcept>

namespace cudf_polars_cucascade {

namespace {
constexpr std::size_t align = 4096;

std::size_t aligned(std::size_t size) { return std::max(align, (size + align - 1) / align * align); }

// The REST engine stages only device reads; every read here lands in host memory we own, so its
// pinned staging pool can stay small.
constexpr std::size_t staging_bytes = 256UL << 20;
constexpr std::size_t staging_block = 1UL << 20;

constexpr std::size_t cache_pool_bytes = 128UL << 20;  // one pinned pool of the cache

constexpr std::size_t parquet_trailer_bytes = 8;  // footer length (4) + "PAR1" (4)
}  // namespace

scan_io::scan_io(scan_io_config const& cfg)
  : _arena_bytes{aligned(cfg.arena_bytes)},
    _cached{cfg.cache_bytes > 0},
    _inflight_limit{cfg.inflight_bytes}
{
  cucascade::memory::host_memory_space_config host;
  host.reservation_limit_fraction = 1.0;
  if (_cached) {
    // The cache: pinned now, in pools of cache_pool_bytes, cut into chunks.
    host.memory_capacity      = cfg.cache_bytes;
    host.block_size           = cfg.cache_chunk_bytes;
    host.pool_size            = cache_pool_bytes / cfg.cache_chunk_bytes;
    host.initial_number_pools = (cfg.cache_bytes + cache_pool_bytes - 1) / cache_pool_bytes;
    host.initial_pool_threads = 8;
  } else {
    host.memory_capacity      = staging_bytes;
    host.block_size           = staging_block;
    host.pool_size            = staging_bytes / staging_block;
    host.initial_number_pools = 1;
  }
  std::vector<cucascade::memory::memory_space_config> spaces;
  spaces.emplace_back(host);
  _memory = std::make_unique<cucascade::memory::memory_reservation_manager>(std::move(spaces));

  cucascade::io::io_config config;
  config.backend                        = cucascade::io::io_backend::native;
  config.rest_n_reactors                = cfg.reactors;
  config.rest.max_connections           = cfg.connections;
  config.rest.stall_time_s              = cfg.stall_seconds;
  config.cache.mode = _cached ? cucascade::io::cache::cache_mode::cucs
                              : cucascade::io::cache::cache_mode::none;
  config.cache.eviction = cucascade::io::cache::eviction_policy::lru;
  config.object_store.endpoint          = cfg.endpoint;
  config.object_store.region            = cfg.region;
  config.object_store.access_key        = cfg.access_key;
  config.object_store.secret_key        = cfg.secret_key;
  config.object_store.session_token     = cfg.session_token;
  config.object_store.s3_signing_mode   = cucascade::io::object_store_config::signing_mode::header;
  config.apply_cache_mode();

  _registry = std::make_unique<cucascade::io::io_context_registry>(config, *_memory);
  _context  = _registry->make_ioctx(cucascade::io::io_context_type::restful);
  if (!_context) {
    throw std::runtime_error(
      "scan_io: could not create the REST io context (check endpoint, region and credentials)");
  }
  _context->start();
  if (_cached) { _context->initialize_cache(*_memory, config.cache, nullptr); }

  if (_arena_bytes > 0) {
    if (!_cached) {
      void* ptr = nullptr;
      CUDF_CUDA_TRY(cudaHostAlloc(&ptr, _arena_bytes, cudaHostAllocPortable));
      _arena = static_cast<std::uint8_t*>(ptr);
    }
    _free.emplace_back(0, _arena_bytes);  // with a cache, a budget: offsets but no memory
  }

  for (std::size_t i = 0; i < std::max<std::size_t>(cfg.footer_threads, 1); ++i) {
    _footer_threads.emplace_back(&scan_io::footer_main, this);
  }
  _scheduler = std::thread(&scan_io::scheduler_main, this);
}

scan_io::~scan_io()
{
  {
    std::lock_guard lock(_footer_mutex);
    _footer_stop = true;
  }
  _footer_cv.notify_all();
  for (auto& t : _footer_threads) {
    t.join();
  }
  {
    std::lock_guard lock(_mutex);
    _stop = true;
  }
  _cv.notify_all();
  _scheduler.join();
  // Reads in flight write into the arena: wait for them before freeing it. Reads never issued
  // will not complete, so they are dropped first.
  {
    std::unique_lock lock(_mutex);
    for (auto& q : _to_issue) {
      if (--q.t->pending == 0) { q.t->done = true; }
    }
    _to_issue.clear();
    _cv.wait(lock, [&] {
      return std::all_of(
        _tasks.begin(), _tasks.end(), [](auto const& kv) { return kv.second->pending == 0; });
    });
  }
  if (_arena != nullptr) { cudaFreeHost(_arena); }
  _tasks.clear();  // their cache requests go before the cache
  if (_cached) { _context->shutdown_cache(); }
  _context->shutdown();
}

void scan_io::warmup(std::string const& url) { _context->warmup(url); }

// ---------------------------------------------------------------- footers

std::shared_ptr<scan_io::footer_entry> scan_io::footer_entry_for(std::string const& path,
                                                                 bool request)
{
  std::lock_guard lock(_footer_mutex);
  auto it = _footers.find(path);
  if (it != _footers.end()) { return it->second; }
  if (!request) { return nullptr; }
  auto entry = std::make_shared<footer_entry>();
  _footers.emplace(path, entry);
  _footer_queue.push_back(path);
  _footer_cv.notify_one();
  return entry;
}

void scan_io::request_footers(std::vector<std::string> const& paths)
{
  for (auto const& path : paths) {
    (void)footer_entry_for(path, true);
  }
}

void scan_io::load_footer(std::string const& path, footer_entry& entry)
{
  // One suffix-range GET resolves the size and the version tag and keeps the file's last bytes;
  // the trailer and the footer are then read from those bytes (a footer larger than the window
  // costs one more ranged GET).
  auto object = _context->open_io_object(path, cucascade::io::open_hint::parquet_footer_probe);
  auto const size = object->size();
  if (size < 2 * parquet_trailer_bytes) {
    throw std::runtime_error("scan_io: " + path + " is too small to be a parquet file");
  }
  std::array<std::uint8_t, parquet_trailer_bytes> trailer{};
  _context->host_read_io(*object, size - parquet_trailer_bytes, trailer.size(), trailer.data());
  if (std::memcmp(trailer.data() + 4, "PAR1", 4) != 0) {
    throw std::runtime_error("scan_io: " + path + " has no parquet trailer");
  }
  std::uint32_t footer_len = 0;
  std::memcpy(&footer_len, trailer.data(), sizeof(footer_len));
  if (footer_len + parquet_trailer_bytes > size) {
    throw std::runtime_error("scan_io: " + path + " has a corrupt footer length");
  }
  std::vector<std::uint8_t> footer(footer_len);
  _context->host_read_io(
    *object, size - parquet_trailer_bytes - footer_len, footer_len, footer.data());
  // Parsed the way the hybrid scan reader does (it also prepares the schema the readers expect).
  cudf::io::parquet::experimental::hybrid_scan_reader reader(
    std::span<std::uint8_t const>(footer.data(), footer.size()),
    cudf::io::parquet_reader_options{});
  auto metadata = std::make_unique<cudf::io::parquet::FileMetaData>(reader.parquet_metadata());
  std::lock_guard lock(entry.mutex);
  entry.object   = std::move(object);
  entry.metadata = std::move(metadata);
  entry.footer   = std::move(footer);
}

void scan_io::footer_main()
{
  while (true) {
    std::string path;
    std::shared_ptr<footer_entry> entry;
    {
      std::unique_lock lock(_footer_mutex);
      _footer_cv.wait(lock, [&] { return _footer_stop || !_footer_queue.empty(); });
      if (_footer_stop) { return; }
      path = std::move(_footer_queue.front());
      _footer_queue.pop_front();
      auto it = _footers.find(path);
      if (it == _footers.end()) { continue; }  // cleared meanwhile
      entry = it->second;
    }
    try {
      load_footer(path, *entry);
    } catch (...) {
      std::lock_guard lock(entry->mutex);
      entry->error = std::current_exception();
    }
    {
      std::lock_guard lock(entry->mutex);
      entry->done = true;
    }
    entry->cv.notify_all();
  }
}

std::shared_ptr<scan_io::footer_entry> scan_io::wait_footer(std::string const& path)
{
  auto entry = footer_entry_for(path, true);
  std::unique_lock lock(entry->mutex);
  entry->cv.wait(lock, [&] { return entry->done; });
  if (entry->error) { std::rethrow_exception(entry->error); }
  return entry;
}

std::pair<std::size_t, std::unique_ptr<cudf::io::parquet::FileMetaData>> scan_io::footer(
  std::string const& path)
{
  auto entry = wait_footer(path);
  return {entry->object->size(),
          std::make_unique<cudf::io::parquet::FileMetaData>(*entry->metadata)};
}

std::vector<std::uint8_t> scan_io::footer_file(std::string const& path)
{
  auto entry = wait_footer(path);
  auto const& footer = entry->footer;
  std::vector<std::uint8_t> file;
  file.reserve(footer.size() + 12);
  file.insert(file.end(), {'P', 'A', 'R', '1'});
  file.insert(file.end(), footer.begin(), footer.end());
  auto const len = static_cast<std::uint32_t>(footer.size());
  auto const* len_bytes = reinterpret_cast<std::uint8_t const*>(&len);
  file.insert(file.end(), len_bytes, len_bytes + sizeof(len));
  file.insert(file.end(), {'P', 'A', 'R', '1'});
  return file;
}

std::vector<std::size_t> scan_io::projected_bytes(std::vector<std::string> const& paths,
                                                  std::vector<std::string> const& columns)
{
  std::vector<std::size_t> result;
  result.reserve(2 * paths.size());
  for (auto const& path : paths) {
    auto entry       = wait_footer(path);
    std::size_t sum  = 0;
    for (auto const& group : entry->metadata->row_groups) {
      for (auto const& chunk : group.columns) {
        auto const& name = chunk.meta_data.path_in_schema;
        if (columns.empty() ||
            (!name.empty() &&
             std::find(columns.begin(), columns.end(), name.front()) != columns.end())) {
          sum += static_cast<std::size_t>(chunk.meta_data.total_compressed_size);
        }
      }
    }
    result.push_back(sum);
    result.push_back(entry->metadata->row_groups.size());
  }
  return result;
}

void scan_io::clear_footers()
{
  std::lock_guard lock(_footer_mutex);
  _footer_queue.clear();
  _footers.clear();
}

// ---------------------------------------------------------------- data

std::int64_t scan_io::open_stream(std::int64_t priority)
{
  std::lock_guard lock(_mutex);
  auto const id = _next_stream++;
  if (_pending_streams > 0) { --_pending_streams; }
  stream_state s;
  s.priority = priority;
  _streams.emplace(id, std::move(s));
  return id;
}

void scan_io::set_expected_streams(std::size_t count)
{
  std::lock_guard lock(_mutex);
  _pending_streams = count;
}

void scan_io::close_stream(std::int64_t stream)
{
  {
    std::lock_guard lock(_mutex);
    _streams.erase(stream);
  }
  _cv.notify_all();
}

std::int64_t scan_io::submit(std::int64_t stream,
                             std::vector<std::string> const& paths,
                             std::vector<std::int32_t> const& sources,
                             std::vector<std::size_t> const& offsets,
                             std::vector<std::size_t> const& sizes)
{
  CUDF_EXPECTS(sources.size() == offsets.size() && offsets.size() == sizes.size(),
               "sources, offsets and sizes differ in length");
  auto t     = std::make_shared<task>();
  t->stream  = stream;
  t->sources = sources;
  t->offsets = offsets;
  t->sizes   = sizes;
  t->size    = std::accumulate(sizes.begin(), sizes.end(), std::size_t{0});
  if (aligned(t->size) > _arena_bytes) { return -1; }
  // The objects opened by the footer jobs: data requests need no further round trip and are
  // checked against the version tag the footer request saw.
  t->objects.reserve(paths.size());
  for (auto const& path : paths) {
    t->objects.push_back(wait_footer(path)->object);
  }
  return enqueue(std::move(t));
}

std::int64_t scan_io::enqueue(std::shared_ptr<task> t)
{
  if (aligned(t->size) > _arena_bytes) { return -1; }
  auto const stream = t->stream;
  std::int64_t id;
  {
    std::lock_guard lock(_mutex);
    auto it = _streams.find(stream);
    CUDF_EXPECTS(it != _streams.end(), "scan_io: unknown stream");
    id = _next_id++;
    _tasks.emplace(id, t);
    it->second.queue.push_back(id);
    _largest_task = std::max(_largest_task, aligned(t->size));
  }
  _cv.notify_all();
  return id;
}

scan_io::planned_task scan_io::plan_task(std::int64_t stream_id,
                                         std::vector<std::string> const& paths,
                                         int split_index,
                                         int total_splits,
                                         cudf::io::parquet_reader_options const& options,
                                         bool prune_with_stats,
                                         std::size_t merge_gap,
                                         cudaStream_t stream)
{
  planned_task result;
  std::vector<cudf::io::parquet::FileMetaData> metadata;
  std::vector<std::shared_ptr<cucascade::io::io_object>> objects;
  metadata.reserve(paths.size());
  for (auto const& path : paths) {
    auto entry = wait_footer(path);
    metadata.push_back(*entry->metadata);
    objects.push_back(entry->object);
  }

  // The task's row groups: every row group of every file, or the stride of one split of a
  // single file (the last split reads to the end), as the scan task bounds do.
  std::vector<std::vector<cudf::size_type>> row_groups(metadata.size());
  if (total_splits > 1) {
    auto const n = static_cast<int>(metadata.size() == 1 ? metadata[0].row_groups.size() : 0);
    if (metadata.size() != 1 || total_splits > n) {
      result.unsupported = true;
      return result;
    }
    auto const stride = n / total_splits;
    auto const start  = stride * split_index;
    auto const stop   = split_index == total_splits - 1 ? n : start + stride;
    for (auto rg = start; rg < stop; ++rg) {
      row_groups[0].push_back(rg);
    }
  } else {
    for (std::size_t f = 0; f < metadata.size(); ++f) {
      for (std::size_t rg = 0; rg < metadata[f].row_groups.size(); ++rg) {
        row_groups[f].push_back(static_cast<cudf::size_type>(rg));
      }
    }
  }

  // Estimated decoded bytes: per projected column chunk, the larger of 8 bytes per row and its
  // uncompressed size (strings), doubled when the decode applies a filter (a filtered copy).
  auto const columns = options.get_column_names();
  auto selected      = [&](cudf::io::parquet::ColumnChunk const& chunk) {
    if (!columns.has_value()) { return true; }
    auto const& path = chunk.meta_data.path_in_schema;
    return !path.empty() &&
           std::find(columns->begin(), columns->end(), path.front()) != columns->end();
  };
  std::size_t decoded = 0;
  for (std::size_t f = 0; f < metadata.size(); ++f) {
    for (auto rg : row_groups[f]) {
      auto const& group = metadata[f].row_groups[rg];
      for (auto const& chunk : group.columns) {
        if (!selected(chunk)) { continue; }
        decoded += std::max<std::size_t>(chunk.meta_data.total_uncompressed_size,
                                         static_cast<std::size_t>(group.num_rows) * 8);
      }
    }
  }
  auto const filtered = options.get_filter().has_value();
  result.decoded_bytes = decoded * (filtered ? 2 : 1);

  auto decode     = std::make_shared<decode_state>();
  decode->options = options;
  decode->reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
      std::move(metadata), decode->options);
  if (prune_with_stats && filtered) {
    row_groups = decode->reader->filter_row_groups_with_stats(
      row_groups, decode->options, ::cuda::stream_ref{stream});
  }
  if (decode->reader->total_rows_in_row_groups(row_groups) == 0) {
    result.empty = true;
    return result;
  }
  auto [ranges, sources] =
    decode->reader->all_column_chunks_byte_ranges(row_groups, decode->options);
  decode->row_groups = std::move(row_groups);

  // Merge neighbouring ranges of a file into fewer requests, placed back to back in the block.
  std::vector<std::size_t> order(ranges.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](auto a, auto b) {
    return std::pair{sources[a], ranges[a].offset()} < std::pair{sources[b], ranges[b].offset()};
  });
  auto t    = std::make_shared<task>();
  t->stream = stream_id;
  t->objects = std::move(objects);
  decode->positions.assign(ranges.size(), 0);
  decode->sizes.resize(ranges.size());
  std::size_t block = 0;
  std::int64_t cur_source = -1, cur_offset = 0, cur_end = 0;
  std::size_t cur_pos = 0;
  auto flush = [&] {
    if (cur_source >= 0) {
      t->sources.push_back(static_cast<std::int32_t>(cur_source));
      t->offsets.push_back(static_cast<std::size_t>(cur_offset));
      t->sizes.push_back(static_cast<std::size_t>(cur_end - cur_offset));
    }
  };
  for (auto i : order) {
    auto const src    = static_cast<std::int64_t>(sources[i]);
    auto const offset = ranges[i].offset();
    auto const end    = offset + ranges[i].size();
    decode->sizes[i]  = static_cast<std::size_t>(ranges[i].size());
    if (src == cur_source && offset <= cur_end + static_cast<std::int64_t>(merge_gap)) {
      if (end > cur_end) {
        block += static_cast<std::size_t>(end - cur_end);
        cur_end = end;
      }
    } else {
      flush();
      cur_source = src;
      cur_offset = offset;
      cur_end    = end;
      cur_pos    = block;
      block += static_cast<std::size_t>(ranges[i].size());
    }
    decode->positions[i] = cur_pos + static_cast<std::size_t>(offset - cur_offset);
  }
  flush();
  t->size   = block;
  t->decode = std::move(decode);
  result.staged_bytes = block;
  result.id           = enqueue(std::move(t));
  result.too_big      = result.id < 0;
  return result;
}

cudf::io::table_with_metadata scan_io::materialize(std::int64_t id, cudaStream_t stream)
{
  wait(id);
  auto t        = find(id);
  auto& decode  = *t->decode;
  auto const sv = ::cuda::stream_ref{stream};
  rmm::device_buffer block(t->size, sv);
  copy_to_device(*t, static_cast<std::uint8_t*>(block.data()), stream);
  std::vector<cudf::device_span<std::uint8_t const>> spans;
  spans.reserve(decode.positions.size());
  auto const* base = static_cast<std::uint8_t const*>(block.data());
  for (std::size_t i = 0; i < decode.positions.size(); ++i) {
    spans.emplace_back(base + decode.positions[i], decode.sizes[i]);
  }
  auto table = decode.reader->materialize_all_columns(
    decode.row_groups, spans, decode.options, sv, cudf::get_current_device_resource_ref());
  // The block (and the arena region it was copied from) must outlive the decode's kernels.
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream));
  t->decode.reset();  // the reader holds device buffers of the decode
  return table;
}

bool scan_io::try_allocate(task& t)
{
  auto const size = aligned(t.size);
  for (auto it = _free.begin(); it != _free.end(); ++it) {
    if (it->second >= size) {
      t.block_offset = it->first;
      if (it->second == size) {
        _free.erase(it);
      } else {
        *it = {it->first + size, it->second - size};
      }
      t.allocated = true;
      return true;
    }
  }
  return false;
}

std::pair<std::int64_t, std::shared_ptr<scan_io::task>> scan_io::next_grant()
{
  if (_streams.empty()) { return {-1, nullptr}; }
  std::size_t free_bytes = 0;
  for (auto const& [off, sz] : _free) {
    free_bytes += sz;
  }
  std::size_t holding_none = 0;
  for (auto const& [id, s] : _streams) {
    if (s.held == 0) { ++holding_none; }
  }
  auto const share = _arena_bytes / (_streams.size() + _pending_streams);
  std::vector<std::int64_t> ids;
  ids.reserve(_streams.size());
  for (auto const& [id, s] : _streams) {
    ids.push_back(id);
  }
  std::sort(ids.begin(), ids.end());
  // Round robin from the stream after the last one served, then by priority (stable).
  auto const start = std::upper_bound(ids.begin(), ids.end(), _rr) - ids.begin();
  std::vector<std::int64_t> order;
  order.reserve(ids.size());
  for (std::size_t k = 0; k < ids.size(); ++k) {
    order.push_back(ids[(start + k) % ids.size()]);
  }
  std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
    return _streams.at(a).priority < _streams.at(b).priority;
  });
  for (auto const sid : order) {
    auto& s        = _streams.at(sid);
    if (s.queue.empty()) { continue; }
    auto const id   = s.queue.front();
    auto t          = _tasks.at(id);
    auto const size = aligned(t->size);
    if (s.held > 0) {
      // Beyond its first block a stream keeps to its share and leaves room for the first block of
      // every other stream that holds none.
      auto const reserve = holding_none * _largest_task;
      if (s.held + size > share || free_bytes < size + reserve) { continue; }
    }
    if (!try_allocate(*t)) { continue; }
    s.queue.pop_front();
    s.held += size;
    _rr = sid;
    return {id, t};
  }
  return {-1, nullptr};
}

std::vector<std::size_t> scan_io::snapshot()
{
  std::lock_guard lock(_mutex);
  std::size_t waiting = 0;
  for (auto const& [_, st] : _streams) {
    waiting += st.queue.size();
  }
  std::size_t to_issue = 0;
  for (auto const& q : _to_issue) {
    to_issue += q.batch.bytes;
  }
  std::size_t free_bytes = 0;
  for (auto const& [_, size] : _free) {
    free_bytes += size;
  }
  return {waiting, _inflight, to_issue, free_bytes};
}

void scan_io::scheduler_main()
{
  while (true) {
    std::vector<queued_batch> issue_now;
    bool finished_empty = false;
    {
      std::unique_lock lock(_mutex);
      _cv.wait(lock, [&] {
        if (_stop) { return true; }
        // Give blocks to every task that fits; their reads join the issue queue in that order.
        while (true) {
          auto [id, t] = next_grant();
          if (!t) { break; }
          _fetched_bytes += t->size;
          auto batches = plan_reads(*t);
          t->pending   = batches.size();
          if (batches.empty()) {
            t->done        = true;
            finished_empty = true;
          }
          for (auto& b : batches) {
            _to_issue.push_back({id, t, std::move(b)});
          }
        }
        // Issue in that order while the bytes in flight stay under the limit.
        while (!_to_issue.empty() &&
               (_inflight_limit == 0 || _inflight == 0 || _inflight < _inflight_limit)) {
          // The oldest batch of the stream of lowest priority.
          auto priority_of = [&](queued_batch const& q) {
            auto it = _streams.find(q.t->stream);
            return it == _streams.end() ? std::int64_t{0} : it->second.priority;
          };
          auto best = _to_issue.begin();
          for (auto it = _to_issue.begin(); it != _to_issue.end(); ++it) {
            if (priority_of(*it) < priority_of(*best)) { best = it; }
          }
          _inflight += best->batch.bytes;
          issue_now.push_back(std::move(*best));
          _to_issue.erase(best);
        }
        return !issue_now.empty() || finished_empty;
      });
      if (_stop) { return; }
    }
    if (finished_empty) { _cv.notify_all(); }
    for (auto& q : issue_now) {
      issue(std::move(q));
    }
  }
}

std::vector<scan_io::read_batch> scan_io::plan_reads(task const& t) const
{
  // Vectored reads straight into the task's block. The REST engine gives one call to at most two
  // reactors, so a task's ranges are cut into calls of about batch_bytes each: a scan with one big
  // task in flight then still uses every reactor. Each range is split into requests of a few MiB
  // by the reactors, which write the bytes where the slices point.
  std::vector<read_batch> batches;
  std::vector<std::size_t> open(t.objects.size(), SIZE_MAX);  // file -> its open batch
  std::size_t pos = t.block_offset;
  for (std::size_t i = 0; i < t.sizes.size(); ++i) {
    auto offset    = t.offsets[i];
    auto remaining = t.sizes[i];
    auto const f   = static_cast<std::size_t>(t.sources[i]);
    while (remaining > 0) {
      if (open[f] == SIZE_MAX || batches[open[f]].bytes >= batch_bytes) {
        open[f] = batches.size();
        batches.push_back({f, {}, 0});
      }
      auto& b          = batches[open[f]];
      auto const piece = std::min(remaining, batch_bytes - b.bytes);
      // With a cache the slices only carry the ranges (the reads land in its chunks).
      static std::uint8_t no_destination;
      b.slices.emplace_back(offset, piece, _arena != nullptr ? _arena + pos : &no_destination);
      b.bytes += piece;
      offset += piece;
      pos += piece;
      remaining -= piece;
    }
  }
  return batches;
}

void scan_io::issue(queued_batch q)
{
  auto const bytes = q.batch.bytes;
  if (_cached) {
    std::vector<cucascade::io::byte_range> ranges;
    ranges.reserve(q.batch.slices.size());
    for (auto const& sl : q.batch.slices) {
      ranges.emplace_back(static_cast<std::int64_t>(sl.offset()), static_cast<std::int64_t>(sl.size()));
    }
    // A prefetch that loads nothing (already resident, or refused) is not an error: the decode
    // reads the ranges through the cache either way.
    auto handle = _context->cache()->prefetch_ranges(
      *q.t->objects[q.batch.file], ranges, [this, t = q.t, bytes](auto const&) noexcept {
        finish_read(t, bytes, nullptr);
      });
    if (handle) {
      std::lock_guard lock(_mutex);
      q.t->handles.push_back(std::move(handle));
    }
    return;
  }
  auto future      = _context->host_readv_async_io(*q.t->objects[q.batch.file], q.batch.slices);
  // Runs on a reactor thread: only bookkeeping, then the scheduler issues what comes next.
  std::move(future).install_callback(
    [this, t = std::move(q.t), bytes](cucascade::exec::try_t<std::size_t>&& result) mutable noexcept {
      finish_read(t, bytes, result.has_exception() ? result.exception() : nullptr);
    });
}

void scan_io::finish_read(std::shared_ptr<task> const& t,
                          std::size_t bytes,
                          std::exception_ptr error)
{
  {
    std::lock_guard lock(_mutex);
    if (error && !t->error) { t->error = error; }
    _inflight -= bytes;
    if (--t->pending == 0) { t->done = true; }
  }
  _cv.notify_all();
}

std::shared_ptr<scan_io::task> scan_io::find(std::int64_t id)
{
  std::lock_guard lock(_mutex);
  auto it = _tasks.find(id);
  CUDF_EXPECTS(it != _tasks.end(), "scan_io: unknown task id");
  return it->second;
}

bool scan_io::ready(std::int64_t id)
{
  auto t = find(id);
  std::lock_guard lock(_mutex);
  return t->done;
}

void scan_io::wait(std::int64_t id)
{
  auto t = find(id);
  std::unique_lock lock(_mutex);
  _cv.wait(lock, [&] { return t->done; });
  if (t->error) { std::rethrow_exception(t->error); }
}

std::unique_ptr<rmm::device_buffer> scan_io::to_device(std::int64_t id, cudaStream_t stream)
{
  wait(id);
  auto t   = find(id);
  auto buf = std::make_unique<rmm::device_buffer>(t->size, ::cuda::stream_ref{stream});
  copy_to_device(*t, static_cast<std::uint8_t*>(buf->data()), stream);
  return buf;
}

void scan_io::copy_to_device(task const& t, std::uint8_t* dst, cudaStream_t stream)
{
  if (t.size == 0) { return; }
  if (!_cached) {
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(dst, _arena + t.block_offset, t.size, cudaMemcpyHostToDevice, stream));
    return;
  }
  // Each range to its place in the block, per file, read through the cache: from its chunks where
  // resident, from the object store otherwise. Waits for the copies.
  std::vector<std::vector<cucascade::io::slice>> per_file(t.objects.size());
  std::size_t pos = 0;
  for (std::size_t i = 0; i < t.sizes.size(); ++i) {
    if (t.sizes[i] > 0) {
      per_file[static_cast<std::size_t>(t.sources[i])].emplace_back(
        t.offsets[i], t.sizes[i], dst + pos);
    }
    pos += t.sizes[i];
  }
  auto* cache = _context->cache();
  std::vector<std::future<std::size_t>> reads;
  for (std::size_t f = 0; f < per_file.size(); ++f) {
    if (per_file[f].empty()) { continue; }
    reads.push_back(cucascade::io::bridge_semi_to_std([&] {
      return cache->device_read_ranges_async(
        *t.objects[f], per_file[f], ::cuda::stream_ref{stream});
    }));
  }
  for (auto& r : reads) {
    r.get();
  }
}

std::size_t scan_io::clear_cache()
{
  auto* cache = _cached ? _context->cache() : nullptr;
  return cache == nullptr ? 0 : cache->reclaim(cucascade::io::cache::reclaim_mode::forced);
}

void scan_io::release(std::int64_t id)
{
  {
    std::unique_lock lock(_mutex);
    auto it = _tasks.find(id);
    if (it == _tasks.end()) { return; }
    auto task_ptr = it->second;
    auto& t       = *task_ptr;
    // Reads in flight write into the block: never hand it out again before they land.
    _cv.wait(lock, [&] { return t.pending == 0; });
    it = _tasks.find(id);
    if (t.allocated) {
      if (auto s = _streams.find(t.stream); s != _streams.end()) { s->second.held -= aligned(t.size); }
      _free.emplace_back(t.block_offset, aligned(t.size));
      std::sort(_free.begin(), _free.end());
      std::vector<std::pair<std::size_t, std::size_t>> merged;
      for (auto const& [off, sz] : _free) {
        if (!merged.empty() && merged.back().first + merged.back().second == off) {
          merged.back().second += sz;
        } else {
          merged.emplace_back(off, sz);
        }
      }
      _free = std::move(merged);
    }
    _tasks.erase(it);
  }
  _cv.notify_all();
}

}  // namespace cudf_polars_cucascade
