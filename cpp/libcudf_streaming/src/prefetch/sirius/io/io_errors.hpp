// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdexcept>

namespace cudf_streaming::prefetch::io {

/**
 * @brief Raised by @c s3_request_authorizer implementations when credential
 *        acquisition or signing fails.
 *
 * Surfaces from:
 *   - missing or malformed static credentials (caught at provider construction
 *     or first call),
 *   - misconfigured endpoint / region,
 *   - underlying signing-library failure (HMAC / SHA256),
 *   - upstream credential broker errors (in future refresh-aware impls).
 *
 * Backends translate this into the broader IO error path of the caller.
 * Future IO-layer error types share this header.
 */
class credential_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

}  // namespace cudf_streaming::prefetch::io
