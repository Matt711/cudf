// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace cudf_streaming::prefetch::io::uring {

struct config {
  std::size_t bounce_size{1UL << 20};
  /// When false, every prep path except the BYO-device-buffer read
  /// (prep_device_rx_request) reads through the buffered (page-cache) file
  /// handle instead of the O_DIRECT one.  Defaults to O_DIRECT.
  bool use_odirect{true};

  // max number of contiguous segments to fuse into one readv SQE.
  std::size_t max_n_chunks{1};

  // Maximum number of simultaneous logical scans (matches the reactor's slot count).
  std::size_t n_max_concurrent_scans{64};

  // Minimum file-offset and buffer-pointer alignment required for O_DIRECT reads.
  // With O_DIRECT enabled this is 4 KiB; without it, any alignment is acceptable.
  [[nodiscard]] std::size_t min_alignment_requirement() const noexcept
  {
    return use_odirect ? 4096 : 1;
  }

  // Maximum byte gap between two adjacent ranges that will be merged into a
  // single readv SQE.  Zero means no cross-gap merging.
  [[nodiscard]] std::size_t merge_gap_size() const noexcept { return 0; }
};

}  // namespace cudf_streaming::prefetch::io::uring
