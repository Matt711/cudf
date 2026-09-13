// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace cudf_streaming::prefetch::io::s3 {

/**
 * @brief Object reference passed to the credential / authorizer seam.
 *
 * @c bucket carries the object-store bucket name (no scheme, no trailing
 * slashes). @c key carries the RAW, literal object key bytes (a `%`, `?` or
 * `#` in the key is a literal byte, not an escape) — the provider / authorizer
 * RFC3986-encodes it once for canonical URI construction, so a literal `%`
 * becomes `%25` on the wire.
 *
 * Used by @c s3_request_authorizer (the presigned/header signing seam) as the
 * object-identity type.
 */
struct s3_object_ref {
  std::string bucket;
  std::string key;
};

}  // namespace cudf_streaming::prefetch::io::s3
