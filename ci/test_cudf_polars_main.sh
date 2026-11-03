#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip
source_sha=${1:?Pass the cuDF source SHA}
suite=${2:?Pass unit, in-memory, spmd, or spmd-small}
case "${suite}" in
    unit|in-memory|spmd|spmd-small) ;;
    *) exit 1 ;;
esac
mkdir -p polars-main-results .polars-main-work
result_dir=$(realpath polars-main-results)
work_dir=$(realpath .polars-main-work)

# custom-job uploads its named artifact only if the script succeeds. The final
# workflow job reads this status and fails after the evidence has been uploaded.
printf '{"stage":"setup","suite":"%s"}\n' "${suite}" > "${result_dir}/result.json"
trap 'exit 0' ERR

gh run download "${GITHUB_RUN_ID}" --repo "${GITHUB_REPOSITORY}" \
    --name "polars-main-build-${GITHUB_RUN_ATTEMPT}" --dir "${work_dir}/polars-build"
(
    cd "${work_dir}/polars-build"
    sha256sum --check SHA256SUMS
)
git clone "${work_dir}/polars-build/polars.bundle" "${work_dir}/polars"
test "$(git -C "${work_dir}/polars" rev-parse HEAD)" = "$(cat "${work_dir}/polars-build/polars-sha.txt")"
cp "${work_dir}/polars-build/"*.txt "${result_dir}/"

# Keep RAPIDS_SHA pointing at the nightly binary artifacts. Only the Python
# sources under test may come from the repair PR.
if [[ "$(git rev-parse HEAD)" != "${source_sha}" ]]; then
    git fetch origin "${source_sha}"
    git checkout --detach FETCH_HEAD
fi
test "$(git rev-parse HEAD)" = "${source_sha}"

wheels=()
for spec in 'wheel_cpp libcudf' 'wheel_python pylibcudf' 'wheel_cpp libcudf-streaming' 'wheel_python cudf-streaming'; do
    read -r kind package <<< "${spec}"
    extra=()
    [[ "${kind}" == wheel_python ]] && extra=(--stable)
    wheelhouse=$(rapids-download-from-github "$(rapids-artifact-name "${kind}" "${package}" cudf --cuda "${RAPIDS_CUDA_VERSION}" "${extra[@]}")")
    wheels+=("${wheelhouse}"/*.whl)
done
quent=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cudf-polars-quent cudf --stable)")
wheels+=("${quent}"/cudf_polars_quent-*-cp312-abi3-*.whl)

python -m pip install build
python -m build --wheel --outdir "${work_dir}/original-wheel" python/cudf_polars
version=$(cat "${work_dir}/polars-build/polars-version.txt")
python ci/cudf_polars/upstream_wheel.py "${work_dir}/original-wheel/"*.whl "${work_dir}/candidate-wheel" "${version}"
candidate_wheels=("${work_dir}/candidate-wheel/"*.whl)
test "${#candidate_wheels[@]}" -eq 1
rapids-generate-pip-constraints py_test_cudf_polars "${PIP_CONSTRAINT}"
# Candidate constraints apply only to this isolated workflow, never a release.
sed -i -E '/^polars([<>=!~[]|$)/d; /^polars-runtime-/d' "${PIP_CONSTRAINT}"
printf '\npolars==%s\npolars-runtime-32==%s\n' "${version}" "${version}" >> "${PIP_CONSTRAINT}"
polars_dir="${work_dir}/polars"
sed -i '/^polars-cloud$/d' "${polars_dir}/py-polars/requirements-dev.txt"
sed -i 's/^pyarrow$/pyarrow<25/' "${polars_dir}/py-polars/requirements-dev.txt"
rapids-pip-retry install --prefer-binary --constraint "${PIP_CONSTRAINT}" \
    "${work_dir}/polars-build/"*.whl \
    "${candidate_wheels[0]}[test,dask,ray]" \
    "${wheels[@]}" \
    -r "${polars_dir}/py-polars/requirements-dev.txt" \
    -r "${polars_dir}/py-polars/requirements-ci.txt" \
    'polars-iceberg>=0.1.0,<1.0.0'
# Main can retain a release's version string. Always install the built bytes,
# even if an upstream test dependency already installed the same version.
rapids-pip-retry install --force-reinstall --no-deps "${work_dir}/polars-build/"*.whl
python -m pip check
python -m pip freeze > "${result_dir}/requirements.txt"
python -c 'import polars; polars.show_versions()' > "${result_dir}/polars-build.txt"
grep -Eq "^Build: +$(cat "${result_dir}/polars-sha.txt")$" "${result_dir}/polars-build.txt"
printf '%s\n' "${source_sha}" > "${result_dir}/cudf-source-sha.txt"
printf '%s\n' "${RAPIDS_SHA}" > "${result_dir}/cudf-binary-sha.txt"
cp ci/cudf_polars/gpu_test_run_list.json "${result_dir}/stable-gpu-run-list.json"

unset CUDF_POLARS_GPU_RUN_LIST
export POLARS_DIR="${polars_dir}"
export RAPIDS_ARTIFACTS_DIR="${result_dir}"
trap - ERR
set +e
case "${suite}" in
    unit)
        ./ci/run_cudf_polars_pytests.sh -x --numprocesses=4 --dist=worksteal \
            --cov-config=./pyproject.toml --cov=cudf_polars --cov-fail-under=100 \
            --cov-report=term-missing --junitxml="${result_dir}/unit.xml"
        ;;
    *)
        engine=${suite}
        blocksize=default
        if [[ "${suite}" == spmd-small ]]; then
            engine=spmd
            blocksize=small
        fi
        ./ci/run_cudf_polars_polars_tests.sh --engine "${engine}" --run-slow \
            --inject-gpu-engine-blocksize "${blocksize}"
        ;;
esac
test_status=$?
printf '{"stage":"tests","suite":"%s","returncode":%d}\n' \
    "${suite}" "${test_status}" > "${result_dir}/result.json"
exit 0
