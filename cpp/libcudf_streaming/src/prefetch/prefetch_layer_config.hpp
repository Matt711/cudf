// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sirius/io/cache/config.hpp"
#include "sirius/io/kvikio/config.hpp"
#include "sirius/io/object_store_config.hpp"
#include "sirius/io/rest/config.hpp"
#include "sirius/io/uring/config.hpp"

#include <cstddef>

namespace cudf_streaming::prefetch {

enum class io_backend { uring, kvikio };

/// Top-level configuration for the prefetch layer.
/// Aggregates per-backend configs consumed by datasource_factory.
struct prefetch_config {
  io_backend backend{io_backend::uring};

  io::uring::config uring{};
  std::size_t uring_n_reactors{2};

  io::rest::config rest{};
  std::size_t rest_n_reactors{4};

  io::object_store_config object_store{};

  io::cache::config cache{};

  // kvikio has no separate config struct; backend=kvikio selects the fallback
};

}  // namespace cudf_streaming::prefetch
