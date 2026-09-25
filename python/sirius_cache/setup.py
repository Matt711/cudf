#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the sirius_cache Cython extension.
#
# Path resolution (first match wins):
#   1. Environment variables CUDF_SRC, CUCASCADE_SRC (set these in Docker)
#   2. $CONDA_PREFIX (installed headers/libs)
#   3. Local dev fallbacks under /raid/mmurray/
#
# Docker usage:
#   CUDF_SRC=/build/cudf CUCASCADE_SRC=/build/cucascade pip install --no-build-isolation .
#
# Local dev:
#   pip install --no-build-isolation -e .

import os
import sys
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup

# ---------------------------------------------------------------------------
# Root directories — overridable via environment variables for Docker builds
# ---------------------------------------------------------------------------

CONDA = Path(os.environ.get("CONDA_PREFIX", ""))
CUDF_SRC = Path(os.environ.get("CUDF_SRC", "/raid/mmurray/cudf"))
CUCASCADE_SRC = Path(os.environ.get("CUCASCADE_SRC", "/raid/mmurray/sirius/cucascade"))
DOCKER_LIBS = Path("/raid/mmurray/cloud-benchmarking-infra/docker/libs")
PREFETCH_BUILD = Path("/tmp/cudf_prefetch_build/src/io/streaming")
CUDA_HOME = Path(os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", "/usr/local/cuda")))
# In conda/rapidsai environments CUDA stubs may be under targets/x86_64-linux/include
_CONDA_TARGETS_INC = CONDA / "targets" / "x86_64-linux" / "include"

# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------

def _find_include(candidates, sentinel):
    for d in candidates:
        if d and (Path(d) / sentinel).exists():
            return str(d)
    raise FileNotFoundError(
        f"Cannot find include dir containing {sentinel}. "
        f"Tried: {[str(d) for d in candidates if d]}"
    )

# cudf public headers (cudf/io/datasource.hpp, cudf/io/text/byte_range_info.hpp)
cudf_include = _find_include(
    [CONDA / "include", CUDF_SRC / "cpp" / "include"],
    "cudf/io/datasource.hpp",
)

# cudf streaming headers (cudf/io/streaming/scan_manager/config.hpp)
cudf_streaming_include = _find_include(
    [CONDA / "include", CUDF_SRC / "cpp" / "include"],
    "cudf/io/streaming/scan_manager/config.hpp",
)

# cudf streaming root — needed for relative includes like exec/invocable.hpp
# These are resolved as `#include "exec/invocable.hpp"` from inside the streaming headers.
cudf_streaming_root = _find_include(
    [
        CONDA / "include" / "cudf" / "io" / "streaming",
        CUDF_SRC / "cpp" / "include" / "cudf" / "io" / "streaming",
    ],
    "exec/invocable.hpp",
)

# moodycamel vendor headers (blockingconcurrentqueue.h, concurrentqueue.h)
# Installed to include/cudf/io/streaming/vendor/ or found in the source tree.
vendor_include = _find_include(
    [
        CONDA / "include" / "cudf" / "io" / "streaming" / "vendor",
        CUDF_SRC / "cpp" / "src" / "io" / "streaming" / "vendor",
    ],
    "blockingconcurrentqueue.h",
)

# cuCascade headers
cucascade_include = _find_include(
    [CONDA / "include", CUCASCADE_SRC / "include"],
    "cucascade/memory/memory_reservation_manager.hpp",
)

# pylibcudf Cython declarations (needed for `cimport pylibcudf.io.datasource`)
def _find_pylibcudf_include():
    import sysconfig
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        CUDF_SRC / "python" / "pylibcudf",
        CONDA / "lib" / py_ver / "site-packages",
        CONDA / "lib" / "python3.14" / "site-packages",
        CONDA / "lib" / "python3.13" / "site-packages",
        CONDA / "lib" / "python3.12" / "site-packages",
        CONDA / "lib" / "python3.11" / "site-packages",
    ]
    for d in candidates:
        if d and (d / "pylibcudf" / "io" / "datasource.pxd").exists():
            return str(d)
    try:
        import importlib.util
        spec = importlib.util.find_spec("pylibcudf")
        if spec and spec.submodule_search_locations:
            parent = Path(list(spec.submodule_search_locations)[0]).parent
            if (parent / "pylibcudf" / "io" / "datasource.pxd").exists():
                return str(parent)
    except Exception:
        pass
    raise FileNotFoundError(
        "Cannot find pylibcudf include (need pylibcudf/io/datasource.pxd). "
        "Install pylibcudf or set CUDF_SRC."
    )

pylibcudf_include = _find_pylibcudf_include()

# Package source directory (contains _impl.hpp)
pkg_src = Path(__file__).parent / "sirius_cache"

def _find_cuda_include():
    candidates = [
        CUDA_HOME / "include",
        _CONDA_TARGETS_INC,
    ]
    for d in candidates:
        if (d / "vector_types.h").exists():
            return str(d)
    return None

cuda_include = _find_cuda_include()

# libcudacxx headers: cuda/std/cstdint etc — lives under include/rapids in conda rapids builds
_RAPIDS_INC = CONDA / "include" / "rapids"
rapids_include = str(_RAPIDS_INC) if (_RAPIDS_INC / "cuda" / "std" / "cstdint").exists() else None

include_dirs = list(dict.fromkeys(filter(None, [
    cudf_include,
    cudf_streaming_include,
    cudf_streaming_root,   # for relative includes: exec/invocable.hpp etc.
    vendor_include,         # for blockingconcurrentqueue.h
    cucascade_include,
    pylibcudf_include,
    str(pkg_src.parent),   # so #include "sirius_cache/_impl.hpp" resolves
    cuda_include,          # vector_types.h (CUDA toolkit)
    rapids_include,        # cuda/std/cstdint etc (libcudacxx via conda-rapids)
])))

# ---------------------------------------------------------------------------
# Library directories
# ---------------------------------------------------------------------------

def _find_lib(candidates, libname):
    for d in candidates:
        if d and (Path(d) / f"lib{libname}.so").exists():
            return str(d)
    raise FileNotFoundError(
        f"Cannot find lib{libname}.so. Tried: {[str(d) for d in candidates if d]}"
    )

lib_prefetch = _find_lib(
    [CONDA / "lib", PREFETCH_BUILD, DOCKER_LIBS],
    "cudf_prefetch_io",
)
lib_cucascade = _find_lib(
    [CONDA / "lib", DOCKER_LIBS],
    "cucascade",
)
lib_cudf = _find_lib(
    [CONDA / "lib", DOCKER_LIBS],
    "cudf",
)

library_dirs = list(dict.fromkeys([lib_prefetch, lib_cucascade, lib_cudf]))

# ---------------------------------------------------------------------------
# Extension
# ---------------------------------------------------------------------------

ext = Extension(
    name="sirius_cache.sirius_cache",
    sources=["sirius_cache/sirius_cache.pyx"],
    language="c++",
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=["cudf_prefetch_io", "cucascade", "cudf"],
    extra_compile_args=[
        "-std=c++23",
        "-O2",
        "-fvisibility=hidden",
        "-Wno-unused-parameter",
        "-Wno-deprecated-declarations",
    ],
    extra_link_args=[
        f"-Wl,-rpath,{lib_prefetch}",
        f"-Wl,-rpath,{lib_cucascade}",
        f"-Wl,-rpath,{lib_cudf}",
    ],
)

setup(
    name="sirius_cache",
    version="0.1.0.dev0",
    description="Sirius prefetch-cache IO stack for cudf-polars",
    packages=["sirius_cache"],
    ext_modules=cythonize(
        [ext],
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
        },
    ),
    python_requires=">=3.11",
    zip_safe=False,
)
