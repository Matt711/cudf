// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Top-level configuration aggregate for the cudf prefetch cache port.
// Replaces Sirius's DuckDB-coupled scan_manager/config.hpp.

#pragma once

#include "io/cache/config.hpp"
#include "io/kvikio/config.hpp"
#include "io/object_store_config.hpp"
#include "io/rest/config.hpp"
#include "io/uring/config.hpp"

#include <algorithm>
#include <cstddef>
#include <thread>

namespace cudf_streaming::prefetch::scan_manager {

/// Number of io_uring reactors to spin up (default 1).
inline constexpr std::size_t default_uring_n_reactors = 1;

/// Number of REST reactors to spin up (default 4).
inline constexpr std::size_t default_rest_n_reactors = 4;

/// IO backend that serves managed reads.
enum class io_backend {
  /// Sirius's own IO stack: uring for local paths, REST for s3:// URLs.
  sirius,
  /// The kvikIO backend (drives kvikio::FileHandle directly).
  kvikio,
};

/**
 * @brief Top-level configuration for the prefetch I/O layer.
 *
 * Aggregates per-backend configs (uring, rest, object-store) and cache-level
 * tuning.  Construct with defaults and override specific fields before
 * passing to @c io_context_registry.
 */
struct scan_manager_config {
  /// IO backend that serves managed reads.
  io_backend backend{io_backend::sirius};

  io::uring::config        uring{};           ///< io_uring (local NVMe) backend config
  io::kvikio_config        kvikio{};          ///< kvikIO (fallback) backend config
  io::rest::config         rest{};            ///< REST / S3 reactor config
  io::object_store_config  object_store{};    ///< S3 endpoint / credentials
  io::cache::config        cache{};           ///< Prefetching cache sizing

  std::size_t uring_n_reactors{default_uring_n_reactors};
  std::size_t rest_n_reactors{default_rest_n_reactors};
};

}  // namespace cudf_streaming::prefetch::scan_manager
