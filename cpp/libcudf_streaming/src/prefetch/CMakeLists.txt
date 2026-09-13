# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Private C++20 prefetch target linked into cudf_streaming.
# Sources are ported from Sirius PR #1661 (pinned at 31c1b7c) and cuCascade
# (pinned at 1b0e7b6). See sirius/PROVENANCE.json for exact revision records.

# Placeholder: sources will be added in later stages.
# Stage 1 establishes the build skeleton and dependency closure.
set(CUDF_STREAMING_PREFETCH_SOURCES
  # Stage 2: memory substrate (ported from cuCascade 1b0e7b6)
  sirius/cuda/event.cpp
  sirius/memory/common.cpp
  sirius/memory/topology_discovery.cpp
  sirius/memory/disk_access_limiter.cpp
  sirius/memory/error.cpp
  sirius/memory/fixed_size_host_memory_resource.cpp
  sirius/memory/memory_reservation.cpp
  sirius/memory/memory_reservation_manager.cpp
  sirius/memory/memory_space.cpp
  sirius/memory/notification_channel.cpp
  sirius/memory/null_device_memory_resource.cpp
  sirius/memory/oom_handling_policy.cpp
  sirius/memory/reservation_aware_resource_adaptor.cpp
  sirius/memory/small_pinned_host_memory_resource.cpp
  # Stage 3: transport and completion (ported from Sirius PR #1661)
  sirius/io/uri_parser.cpp
  sirius/io/io_context.cpp
  sirius/io/sirius_datasource.cpp
  sirius/io/datasource_factory.cpp
  sirius/io/kvikio/kvikio_context.cpp
  sirius/io/rest/curl_handle.cpp
  sirius/io/rest/rest_ioctx.cpp
  sirius/io/rest/rest_reactor.cpp
  sirius/io/rest/s3/list_parser.cpp
  sirius/io/rest/s3/sigv4.cpp
  sirius/io/rest/s3/sigv4_authorizer.cpp
  # Stage 4: cache and metadata
  sirius/io/cache/metadata_store.cpp
  sirius/io/cache/types.cpp
  sirius/io/cache/prefetching_cache.cpp
  # Stage 5: public C++ facade
  scan_context.cpp
)

if(NOT CUDF_STREAMING_PREFETCH_SOURCES)
  # Nothing to compile yet; skip target creation in Stage 1.
  return()
endif()

add_library(cudf_streaming_prefetch STATIC ${CUDF_STREAMING_PREFETCH_SOURCES})

target_include_directories(cudf_streaming_prefetch
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/sirius
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor
    ${CUDF_STREAMING_SOURCE_DIR}/include
)

set_target_properties(cudf_streaming_prefetch
  PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    COMPILE_DEFINITIONS "CUDF_STREAMING_PREFETCH_BUILDING"
)

target_link_libraries(cudf_streaming_prefetch
  PRIVATE
    cudf::cudf
    kvikio::kvikio
    rapids_logger::rapids_logger
    absl::any_invocable
    absl::cleanup
    CURL::libcurl
    OpenSSL::SSL
    OpenSSL::Crypto
    Threads::Threads
    CUDA::nvml_static
)

if(CUDF_STREAMING_HAS_URING)
  target_sources(cudf_streaming_prefetch PRIVATE
    sirius/io/uring/uring_ioctx.cpp
    sirius/io/uring/uring_reactor.cpp
  )
  target_link_libraries(cudf_streaming_prefetch PRIVATE liburing::liburing)
  target_compile_definitions(cudf_streaming_prefetch PRIVATE CUDF_STREAMING_HAS_URING=1)
endif()

if(CUDF_STREAMING_HAS_NUMA)
  target_sources(cudf_streaming_prefetch PRIVATE
    sirius/memory/numa_region_pinned_host_allocator.cpp
  )
  target_link_libraries(cudf_streaming_prefetch PRIVATE libnuma::libnuma)
  target_compile_definitions(cudf_streaming_prefetch PRIVATE CUDF_STREAMING_HAS_NUMA=1)
endif()
