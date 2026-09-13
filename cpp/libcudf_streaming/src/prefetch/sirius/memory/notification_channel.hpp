// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <utility>

namespace cudf_streaming {
namespace prefetch {
namespace memory {

struct notification_channel : std::enable_shared_from_this<notification_channel> {
  struct event_notifier {
    explicit event_notifier(notification_channel& channel);
    ~event_notifier();

    void post();

   private:
    std::shared_ptr<notification_channel> _channel;
  };

  enum class wait_status { IDLE, NOTIFIED, SHUTDOWN };

  ~notification_channel();

  wait_status wait();

  std::unique_ptr<event_notifier> get_notifier();

  void shutdown();

 private:
  void notify();

  void release_notifier();

  void acquire_notifier();

  mutable std::mutex _mutex;
  std::condition_variable _cv;
  bool _has_been_notified{false};
  std::size_t _n_active_notifiers{0};
  bool _is_running{true};
};

using event_notifier = notification_channel::event_notifier;

struct notify_on_exit {
  explicit notify_on_exit(std::unique_ptr<event_notifier> notifier);
  ~notify_on_exit() noexcept;

 private:
  std::unique_ptr<event_notifier> _notifier;
};

}  // namespace memory
}  // namespace prefetch
}  // namespace cudf_streaming
