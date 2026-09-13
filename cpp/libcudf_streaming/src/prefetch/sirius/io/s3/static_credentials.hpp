// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "io/object_store_config.hpp"

#include <chrono>
#include <optional>
#include <string>

namespace cudf_streaming::prefetch::io::s3 {

/**
 * @brief Credentials snapshot consumed by @c sirius_sigv4_presigned_authorizer's
 *        constructor (and other static-creds-based authorizers).
 *
 * Long-lived creds (typical SET s3_access_key / s3_secret_key flow): leave
 * @c session_token empty and @c expires_at @c nullopt.
 *
 * Temporary creds (STS AssumeRole / IMDS / SSO): populate @c session_token;
 * set @c expires_at when the source can report it. The static provider treats
 * @c expires_at as informational; future refresh-aware providers (downstream)
 * may use it to schedule re-acquisition.
 *
 * This is a pure POD by design: integration code constructs it from
 * @c object_store_config string fields, hands it to the authorizer's
 * constructor, and the authorizer holds an internal copy. There is no public
 * accessor on @c s3_request_authorizer to read these back — the seam exposes
 * only @c authorize() (see @c s3_request_authorizer.hpp).
 */
struct static_credentials {
  std::string access_key_id;
  std::string secret_access_key;
  std::string session_token;
  std::optional<std::chrono::system_clock::time_point> expires_at;
};

/// Map an @c object_store_config's static-credential fields into a
/// @c static_credentials snapshot: access key, secret, and (for STS temporary
/// credentials) session token. @c expires_at is left @c nullopt —
/// @c object_store_config carries no expiry.
inline static_credentials static_credentials_from(object_store_config const& cfg)
{
  static_credentials creds;
  creds.access_key_id     = cfg.access_key;
  creds.secret_access_key = cfg.secret_key;
  creds.session_token     = cfg.session_token;
  return creds;
}

}  // namespace cudf_streaming::prefetch::io::s3
