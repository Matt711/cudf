// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace cudf_streaming::prefetch::io::s3 {

/// Default cap on the number of entries a whole-listing call will accumulate.
/// Exceeding it throws (never truncates) — a partial key set would resolve a
/// glob to a silently incomplete table.
inline constexpr std::size_t default_max_list_objects = 100'000;

/// Default cap on the total number of entries a paged listing will scan across
/// all pages. Bounds the scanned-object count and request cost on a prefix
/// whose population dwarfs the matches.
inline constexpr std::size_t default_max_scanned_objects = 1'000'000;

/// One object from a ListObjectsV2 page: full key + object size in bytes.
/// The size rides the LIST response for free, so downstream opens need no
/// size-discovery round-trip.
struct list_entry {
  std::string key;
  std::uint64_t size = 0;
};

/// One parsed page of an S3 ListObjectsV2 response.
struct list_objects_v2_page {
  /// Every `<Contents>` object in document order, XML-entity-unescaped.
  /// Excludes `<CommonPrefixes><Prefix>` entries (directory rollups, not keys).
  std::vector<list_entry> entries;
  /// `<IsTruncated>` — true when another page follows.
  bool is_truncated = false;
  /// `<NextContinuationToken>` — the cursor for the next page; empty when absent.
  std::string next_continuation_token;
};

/**
 * @brief Parse one complete ListObjectsV2 response.
 *
 * Hand-rolled (no XML dependency). Accepts an optional XML declaration and root
 * attributes, and unescapes the five predefined XML entities. Fails closed so a
 * malformed body never parses as a silently-incomplete listing.
 *
 * @throws std::runtime_error for malformed roots, entries, sizes, or paging
 *         fields.
 */
list_objects_v2_page parse_list_objects_v2(std::string_view xml);

}  // namespace cudf_streaming::prefetch::io::s3
