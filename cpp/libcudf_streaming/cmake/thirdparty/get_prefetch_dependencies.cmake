# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Discover native dependencies for the libcudf_streaming prefetch extension.
# No Sirius or cuCascade network checkout is required; sources are vendored.

function(find_and_configure_prefetch_dependencies)
  # KvikIO (for cuFile/GDS and remote I/O backends)
  find_package(kvikio REQUIRED)

  # libcurl (for REST/S3 I/O)
  find_package(CURL REQUIRED)

  # OpenSSL (for S3 SigV4 signing)
  find_package(OpenSSL REQUIRED)

  # liburing (for io_uring local backend)
  # Search order: explicit cache override → conda/micromamba prefix → system paths.
  # Install via: `mamba install -c conda-forge liburing` (adds headers + .so to $CONDA_PREFIX)
  # or `apt-get install liburing-dev` on Debian/Ubuntu.
  find_path(LIBURING_INCLUDE_DIR NAMES liburing.h
    PATHS
      "$ENV{CONDA_PREFIX}/include"
      "$ENV{CONDA_PREFIX}/include/liburing"
      /usr/include
      /usr/local/include
      /usr/include/liburing
    NO_DEFAULT_PATH
  )
  find_library(LIBURING_LIBRARY NAMES uring liburing
    PATHS
      "$ENV{CONDA_PREFIX}/lib"
      /usr/lib
      /usr/local/lib
      /usr/lib/x86_64-linux-gnu
      /usr/lib/aarch64-linux-gnu
    NO_DEFAULT_PATH
  )
  if(NOT LIBURING_INCLUDE_DIR OR NOT LIBURING_LIBRARY)
    message(STATUS "liburing not found; io_uring backend will be disabled")
    message(STATUS "  To enable: mamba install -c conda-forge liburing  (or apt-get install liburing-dev)")
    set(CUDF_STREAMING_HAS_URING OFF PARENT_SCOPE)
  else()
    message(STATUS "liburing found: ${LIBURING_LIBRARY} (headers: ${LIBURING_INCLUDE_DIR})")
    set(CUDF_STREAMING_HAS_URING ON PARENT_SCOPE)
    add_library(liburing::liburing UNKNOWN IMPORTED)
    set_target_properties(liburing::liburing PROPERTIES
      IMPORTED_LOCATION "${LIBURING_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${LIBURING_INCLUDE_DIR}"
    )
  endif()

  # libnuma (for NUMA topology discovery)
  find_path(LIBNUMA_INCLUDE_DIR NAMES numa.h
    PATHS
      "$ENV{CONDA_PREFIX}/include"
      /usr/include
      /usr/local/include
    NO_DEFAULT_PATH
  )
  find_library(LIBNUMA_LIBRARY NAMES numa libnuma
    PATHS
      "$ENV{CONDA_PREFIX}/lib"
      /usr/lib
      /usr/local/lib
      /usr/lib/x86_64-linux-gnu
      /usr/lib/aarch64-linux-gnu
    NO_DEFAULT_PATH
  )
  if(NOT LIBNUMA_INCLUDE_DIR OR NOT LIBNUMA_LIBRARY)
    message(STATUS "libnuma not found; NUMA topology disabled")
    set(CUDF_STREAMING_HAS_NUMA OFF PARENT_SCOPE)
  else()
    set(CUDF_STREAMING_HAS_NUMA ON PARENT_SCOPE)
    add_library(libnuma::libnuma UNKNOWN IMPORTED)
    set_target_properties(libnuma::libnuma PROPERTIES
      IMPORTED_LOCATION "${LIBNUMA_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${LIBNUMA_INCLUDE_DIR}"
    )
  endif()

  # Abseil (absl::AnyInvocable)
  find_package(absl QUIET)
  if(absl_FOUND)
    set(CUDF_STREAMING_HAS_ABSL ON PARENT_SCOPE)
  else()
    message(STATUS "Abseil not found via find_package; trying pkg-config")
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(ABSL QUIET absl_base)
    endif()
    set(CUDF_STREAMING_HAS_ABSL OFF PARENT_SCOPE)
  endif()

  # Threads
  find_package(Threads REQUIRED)

  set(CUDF_STREAMING_PREFETCH_DEPS_FOUND ON PARENT_SCOPE)
endfunction()

find_and_configure_prefetch_dependencies()
