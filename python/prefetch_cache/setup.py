#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Build the prefetch_cache Cython extension.
#
# Dependencies: libcudf (headers + .so), CUDA runtime, pylibcudf (Cython .pxd).
# No cuCascade, no libcudf_streaming, no libcudf_prefetch_io needed.
#
# Docker usage:
#   pip install --no-build-isolation .
#
# Local dev:
#   pip install --no-build-isolation -e .

import os
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup

CONDA = Path(os.environ.get("CONDA_PREFIX", ""))
CUDF_SRC = Path(os.environ.get("CUDF_SRC", "/raid/mmurray/cudf"))


def _find_include(candidates, sentinel):
    for d in candidates:
        if d and (Path(d) / sentinel).exists():
            return str(d)
    raise FileNotFoundError(
        f"Cannot find include dir with {sentinel}. Tried: {[str(d) for d in candidates if d]}"
    )


def _find_lib(candidates, libname):
    for d in candidates:
        if d and (Path(d) / f"lib{libname}.so").exists():
            return str(d)
    raise FileNotFoundError(
        f"Cannot find lib{libname}.so. Tried: {[str(d) for d in candidates if d]}"
    )


def _find_pylibcudf_include():
    for d in [
        CUDF_SRC / "python" / "pylibcudf",
        CONDA / "lib" / "python3.14" / "site-packages",
        CONDA / "lib" / "python3.13" / "site-packages",
        CONDA / "lib" / "python3.12" / "site-packages",
        CONDA / "lib" / "python3.11" / "site-packages",
    ]:
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
    raise FileNotFoundError("Cannot find pylibcudf include (need pylibcudf/io/datasource.pxd)")


# cudf public headers (cudf/io/datasource.hpp, rmm/cuda_stream_view.hpp)
cudf_include = _find_include(
    [CONDA / "include", CUDF_SRC / "cpp" / "include"],
    "cudf/io/datasource.hpp",
)

# CUDA include — try multiple discovery methods.
def _find_cuda_home() -> Path | None:
    # 1. CUDA_HOME / CUDA_ROOT env vars
    for var in ("CUDA_HOME", "CUDA_ROOT"):
        v = os.environ.get(var)
        if v:
            return Path(v)
    # 2. Find via nvcc
    import subprocess
    try:
        r = subprocess.run(["which", "nvcc"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip()).parent.parent
    except Exception:
        pass
    # 3. find cuda_runtime.h
    try:
        r = subprocess.run(
            ["find", "/usr", "/opt/conda", "/usr/local", "-name", "cuda_runtime.h",
             "-not", "-path", "*/stubs/*", "-not", "-path", "*/test*"],
            capture_output=True, text=True, timeout=30,
        )
        hits = [l.strip() for l in r.stdout.splitlines() if l.strip()]
        if hits:
            return Path(hits[0]).parent.parent.parent
    except Exception:
        pass
    return None

def _cuda_candidates() -> list[Path]:
    cands = []
    cuda_home = _find_cuda_home()
    if cuda_home:
        p = cuda_home
        cands += [
            p / "include",
            p / "targets" / "x86_64-linux" / "include",
        ]
    for ver in ["", "-13.0", "-13", "-12.9", "-12.8", "-12.6", "-12.5",
                "-12.4", "-12.3", "-12.2", "-12.1", "-12.0", "-12",
                "-11.8", "-11.7", "-11", "-11.0"]:
        base = Path(f"/usr/local/cuda{ver}")
        cands += [base / "include", base / "targets" / "x86_64-linux" / "include"]
    cands += [CONDA / "include", CONDA / "targets" / "x86_64-linux" / "include"]
    return cands

cuda_include = _find_include(_cuda_candidates(), "cuda_runtime.h")

pylibcudf_include = _find_pylibcudf_include()
pkg_src = Path(__file__).parent / "prefetch_cache"

def _cccl_candidates() -> list[str]:
    """CCCL headers needed by cudf (for <cuda/std/cstdint> etc.)"""
    cands = []
    cuda_home = _find_cuda_home()
    if cuda_home:
        cands += [
            str(cuda_home / "targets" / "x86_64-linux" / "include" / "cccl"),
            str(cuda_home / "include" / "cccl"),
        ]
    # Conda-installed CCCL package puts headers under targets/.../include/cccl/
    cands += [
        str(CONDA / "targets" / "x86_64-linux" / "include" / "cccl"),
        str(CONDA / "include" / "cccl"),
        str(CONDA / "include" / "rapids"),
    ]
    return [d for d in cands if Path(d).is_dir()]

include_dirs = list(dict.fromkeys(filter(None, [
    cudf_include,
    cuda_include,
    *_cccl_candidates(),
    pylibcudf_include,
    str(pkg_src.parent),  # so #include "prefetch_cache/prefetch_datasource.hpp" resolves
])))

cudf_lib = _find_lib([
    CONDA / "lib",
    CUDF_SRC / "cpp" / "build" / "conda" / "cuda-13.0" / "release",
    CUDF_SRC / "cpp" / "build" / "conda" / "cuda-13.2" / "release",
    CUDF_SRC / "cpp" / "build" / "release",
    CUDF_SRC / "cpp" / "build",
    CUDF_SRC / "cpp" / "libcudf_streaming" / "build_gcc13" / "_deps" / "cudf-build",
], "cudf")
def _cudart_candidates() -> list[Path]:
    cands = []
    cuda_home = _find_cuda_home()
    if cuda_home:
        p = cuda_home
        cands += [p / "lib64", p / "targets" / "x86_64-linux" / "lib", p / "lib"]
    for ver in ["", "-13.0", "-13", "-12.9", "-12.8", "-12.6", "-12.5",
                "-12.4", "-12.3", "-12.2", "-12.1", "-12.0", "-12",
                "-11.8", "-11.7", "-11", "-11.0"]:
        base = Path(f"/usr/local/cuda{ver}")
        cands += [base / "lib64", base / "targets" / "x86_64-linux" / "lib"]
    cands += [CONDA / "targets" / "x86_64-linux" / "lib", CONDA / "lib"]
    return cands

cuda_lib = _find_lib(_cudart_candidates(), "cudart")

library_dirs = list(dict.fromkeys([cudf_lib, cuda_lib]))

ext = Extension(
    name="prefetch_cache.prefetch_datasource",
    sources=[
        "prefetch_cache/prefetch_datasource.pyx",
        "prefetch_cache/prefetch_datasource_impl.cpp",
    ],
    language="c++",
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=["cudf", "cudart"],
    extra_compile_args=[
        "-std=c++20",
        # Force-include our CCCL compat shim before ANY other header.
        # The Cython-generated .cpp pulls in cudf/column/column_view.hpp at
        # line ~1418 (before our .hpp is seen), so we cannot rely on including
        # the shim from prefetch_datasource.hpp alone.
        "-include", str(pkg_src / "cuda_compat.hpp"),
        "-O2",
        "-fvisibility=hidden",
        "-Wno-unused-parameter",
        "-Wno-deprecated-declarations",
    ],
    extra_link_args=[
        f"-Wl,-rpath,{cudf_lib}",
        f"-Wl,-rpath,{cuda_lib}",
    ],
)

setup(
    name="prefetch_cache",
    version="0.1.0.dev0",
    description="Pinned-RAM prefetch datasource for cudf-polars",
    packages=["prefetch_cache"],
    ext_modules=cythonize(
        [ext],
        language_level=3,
        include_path=[pylibcudf_include],
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
        },
    ),
    python_requires=">=3.11",
    zip_safe=False,
)
