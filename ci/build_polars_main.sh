#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

polars_sha=${1:?Pass the resolved Polars commit SHA}
output_dir=$(realpath -m "${2:?Pass the artifact directory}")
mkdir -p "${output_dir}"
git init polars
git -C polars remote add origin https://github.com/pola-rs/polars.git
git -C polars fetch --depth 1 origin "${polars_sha}"
git -C polars checkout --detach FETCH_HEAD
test "$(git -C polars rev-parse HEAD)" = "${polars_sha}"

python -m pip install 'maturin==1.15.0' build
export POLARS_BUILD_COMMIT="${polars_sha}"
# Build portable CPU code: the build and GPU test hosts may have different CPUs.
export RUSTFLAGS='-C target-cpu=x86-64 -C target-feature=+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b'
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-8}
(
    cd polars
    rustup show
    maturin build --locked --profile fast-release \
        --manifest-path py-polars/runtime/polars-runtime-32/Cargo.toml \
        --out "${output_dir}"
    python -m build --wheel --outdir "${output_dir}" py-polars
    git bundle create "${output_dir}/polars.bundle" HEAD
    python -c 'import pathlib, tomllib; print(tomllib.loads(pathlib.Path("py-polars/pyproject.toml").read_text())["project"]["version"])' \
        > "${output_dir}/polars-version.txt"
    rustc --version > "${output_dir}/rustc-version.txt"
)
python -m pip freeze > "${output_dir}/build-requirements.txt"
python -m pip install --no-index "${output_dir}/"*.whl
python -c 'import polars; polars.show_versions()' > "${output_dir}/polars-build.txt"
grep -Eq "^Build: +${polars_sha}$" "${output_dir}/polars-build.txt"
(
    cd "${output_dir}"
    sha256sum ./*.whl polars.bundle > SHA256SUMS
)
printf '%s\n' "${polars_sha}" > "${output_dir}/polars-sha.txt"
