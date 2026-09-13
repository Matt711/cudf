// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <rmm/cuda_stream_view.hpp>

#include <exception>
#include <functional>
#include <memory>
#include <string>

namespace cudf_streaming {
namespace prefetch {
namespace memory {

class oom_handling_policy {
 public:
  virtual ~oom_handling_policy() = default;

  using RetryFunc = std::function<void*(std::size_t, rmm::cuda_stream_view)>;

  void* handle_oom(std::size_t bytes,
                   rmm::cuda_stream_view stream,
                   std::exception_ptr eptr,
                   RetryFunc retry_function)
  {
    return do_handle_oom(bytes, stream, eptr, std::move(retry_function));
  }

  virtual std::string get_policy_name() const noexcept = 0;

 protected:
  virtual void* do_handle_oom(std::size_t bytes,
                              rmm::cuda_stream_view stream,
                              std::exception_ptr eptr,
                              RetryFunc retry_function) = 0;
};

class throw_on_oom_policy final : public oom_handling_policy {
 protected:
  void* do_handle_oom(std::size_t bytes,
                      rmm::cuda_stream_view stream,
                      std::exception_ptr eptr,
                      RetryFunc retry_function) final;

  std::string get_policy_name() const noexcept override;
};

std::unique_ptr<oom_handling_policy> make_default_oom_policy();

}  // namespace memory
}  // namespace prefetch
}  // namespace cudf_streaming
