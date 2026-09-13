// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright 2026, Sirius Contributors.
 * (Ported from aminaramoon:feature/frugal_caching_and_dynamic_io, commit 31c1b7c8)
 */

#pragma once

#include <kvikio/compat_mode.hpp>

#include <cstddef>
#include <optional>

// NOTE ON NAMESPACING: lives in cudf_streaming::prefetch::io, not
// cudf_streaming::prefetch::io::kvikio, deliberately — a kvikio sub-namespace
// would shadow the upstream ::kvikio namespace for unqualified uses inside io/.
namespace cudf_streaming::prefetch::io {

struct kvikio_config {
  std::size_t n_max_concurrent_scans{0};
  std::optional<unsigned int>   nthreads;
  std::optional<std::size_t>    task_size;
  std::optional<std::size_t>    gds_threshold;
  std::optional<std::size_t>    bounce_buffer_size;
  std::optional<bool>           auto_direct_io_read;
  std::optional<bool>           auto_direct_io_read_overread;
  std::optional<bool>           thread_pool_per_block_device;
  std::optional<kvikio::CompatMode> compat_mode;
};

void apply_kvikio_defaults(kvikio_config const& cfg);

}  // namespace cudf_streaming::prefetch::io
