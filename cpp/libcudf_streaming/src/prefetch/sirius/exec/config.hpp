// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Ported from Sirius PR #1661 (aminaramoon:feature/frugal_caching_and_dynamic_io)
// src/include/exec/config.hpp

#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <vector>

namespace cudf_streaming::prefetch::exec {

inline constexpr int default_gpu_pipeline_num_threads = 4;
inline constexpr int default_downgrade_num_threads    = 1;

struct thread_pool_config {
  int num_threads{0};
  std::string thread_name_prefix{"thread"};
  std::vector<int> cpu_affinity_list;
};

}  // namespace cudf_streaming::prefetch::exec
