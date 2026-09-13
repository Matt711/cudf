// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright 2025, NVIDIA CORPORATION & AFFILIATES.
 * (Ported from cuCascade, commit 1b0e7b6c28dafa43bfe2c48011a3657c0dd6f127)
 */

#include "notification_channel.hpp"

namespace cudf_streaming {
namespace prefetch {
namespace memory {

//===----------------------------------------------------------------------===//
// notification_channel::event_notifier
//===----------------------------------------------------------------------===//

notification_channel::event_notifier::event_notifier(notification_channel& channel)
  : _channel(channel.shared_from_this())
{
  _channel->acquire_notifier();
}

notification_channel::event_notifier::~event_notifier() { _channel->release_notifier(); }

void notification_channel::event_notifier::post() { _channel->notify(); }

void notification_channel::acquire_notifier()
{
  std::lock_guard lock(_mutex);
  _n_active_notifiers++;
}

//===----------------------------------------------------------------------===//
// notification_channel
//===----------------------------------------------------------------------===//

notification_channel::~notification_channel() { shutdown(); }

notification_channel::wait_status notification_channel::wait()
{
  std::unique_lock lock(_mutex);
  bool notified = false;
  _cv.wait(lock, [&, self = shared_from_this()] {
    notified = std::exchange(_has_been_notified, false);
    return notified || (_n_active_notifiers == 0) || not _is_running;
  });
  return !_is_running ? wait_status::SHUTDOWN
         : (notified) ? wait_status::NOTIFIED
                      : wait_status::IDLE;
}

std::unique_ptr<notification_channel::event_notifier> notification_channel::get_notifier()
{
  return std::make_unique<event_notifier>(*this);
}

void notification_channel::shutdown()
{
  std::lock_guard lock(_mutex);
  _is_running = false;
  _cv.notify_one();
}

void notification_channel::notify()
{
  std::lock_guard lock(_mutex);
  _has_been_notified = true;
  _cv.notify_one();
}

void notification_channel::release_notifier()
{
  std::lock_guard lock(_mutex);
  _n_active_notifiers--;
  if (_n_active_notifiers == 0) { _cv.notify_all(); }
}

//===----------------------------------------------------------------------===//
// notify_on_exit
//===----------------------------------------------------------------------===//

notify_on_exit::notify_on_exit(std::unique_ptr<event_notifier> notifier)
  : _notifier(std::move(notifier))
{
}

notify_on_exit::~notify_on_exit() noexcept
{
  try {
    if (_notifier) _notifier->post();
  } catch (...) {
  }
}

}  // namespace memory
}  // namespace prefetch
}  // namespace cudf_streaming
