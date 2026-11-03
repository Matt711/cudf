# Testing Polars main

The normal test matrix and package metadata describe supported stable Polars
versions. `polars-main.yaml` is a separate workflow for preparing for the next
release. Its failures do not affect the required nightly or unrelated PRs.

## Nightly source build

Set repository variable `POLARS_MAIN_ENABLED=true` to dispatch the workflow after
successful nightly cudf-polars wheel tests on main. No private Polars nightly
channel is needed. The workflow resolves main once, builds the Python wrapper
and 32-bit runtime from that commit, and shares them with all four test jobs:
unit tests, upstream in-memory, upstream SPMD, and upstream small-partition SPMD.

The build uses Python 3.12 and portable CPU instructions. Source bundles, wheel
checksums, compiler versions, installed requirements, and test reports are
retained as artifacts. A completed build is cached by SHA and build script.
The upstream checkout used for tests comes from the same bundle as the wheels.
The installed runtime's reported build SHA must match before tests start.

Run a snapshot manually with a cuDF commit and date that have nightly wheels:

```bash
gh workflow run polars-main.yaml --ref main \
    --field cudf_sha=<full-cudf-sha> --field date=<YYYY-MM-DD> \
    --field polars_sha=<full-polars-sha>
```

The workflow rebuilds a cuDF-polars wheel from the tested Python source and
replaces only its Polars requirement in an isolated test copy. The stable
metadata is unchanged. The test wheel is never published. All remaining
dependencies resolve normally, followed by `pip check`. Nightly libcudf and
other binary dependencies remain pinned to the recorded cuDF artifact revision.

Each suite uploads its status before the report job gates the workflow. Setup
errors, missing reports, collection errors, timeouts, and worker crashes are
failures, not successful compatibility results. Upstream suites run without
the stable-version GPU selection and retain fallback telemetry. Inspect lost
GPU execution as well as correctness; a passing suite can still lose coverage
through CPU fallback. A passing default-SPMD test that appears in the latest
stable GPU run list must still execute on the GPU. Disappeared or renamed
upstream tests need separate review. Do not regenerate stable run lists from
partial evidence, or remove stable expectations to hide a candidate regression.

## CodeRabbit repair PRs

First pilot the observer manually. To enable automatic draft repairs, configure
`POLARS_MAIN_BOT_TOKEN` with a repository-approved GitHub App/user identity that
can create branches and draft PRs, post comments and commit statuses, and dispatch
workflows. CodeRabbit must accept commands from that identity. A repository
`GITHUB_TOKEN` alone does not establish that integration. Set
`POLARS_MAIN_REPAIR_ENABLED=true` only after testing the identity and permissions.
Do not enable CodeRabbit Autofix on these PRs: the controller bounds repair
requests itself.

On complete test failures the controller creates one draft branch named
`bot/polars-main-<run-id>` and records its source target in
`ci/cudf_polars/upstream-target.json`. An active draft is reused when a newer
snapshot fails. A separate run tests the actual repair PR head, publishes the
`cudf-polars/Polars-main` status, and requests
`@coderabbitai fix-ci commit`. CodeRabbit's commits trigger another candidate
run. At most two repair requests are made per Polars SHA. Setup and incomplete
runs never initiate code repairs. Changes to the target do not widen package
bounds.

CodeRabbit should follow `.agents/skills/upgrade-polars/SKILL.md`, preserving
supported stable behavior. Investigate CPU Polars behavior before attributing
a failure to cuDF. Preserve correctness assertions, strict xfails, and GPU
execution. Prefer a justified fallback to a wrong result. Keep larger work
separate. Missing local GPU verification must be reported, not guessed.

Mark the PR ready for review to freeze automatic retargeting and new repair
requests. Any submitted human review also freezes it. New upstream failures
continue appearing in the main observer, without changing the reviewed patch.
A CodeRabbit task already in flight may still deliver a commit; stop it in
CodeRabbit before reviewing and review any subsequent source change again.
Frozen PRs remain manually testable:

```bash
gh workflow run polars-main.yaml --ref bot/polars-main-<run-id> \
    --field repair_pr=<pr-number>
```

This tests the current PR source against its recorded Polars snapshot and
nightly binary artifacts. Missing/expired artifacts must be refreshed through
an explicit target update. Source, binary, and upstream revisions are reported
separately. PR changes that require newer libcudf binaries need ordinary PR CI;
the compatibility workflow cannot supply unbuilt binary changes.

Normal stable CI and human approval remain necessary for merge. RAPIDS draft
CI approval (`/ok to test <sha>`) is unchanged. Automatic CodeRabbit reviews skip
drafts in this repository; the controller uses the explicit coding command.
Verify that command on a draft during the pilot.

## Stable releases

After Polars publishes a stable version, use the full upgrade skill to validate
its real distributions, upstream suite, Narwhals, supported-version matrix,
coverage, and GPU selection. Widen dependency bounds in a reviewed promotion
PR. A green source snapshot does not change the published support range. Freeze
the validated range when cutting a cuDF release; a later support change needs
a new package release or reviewed release-branch backport.

## Controller checks

```bash
python -m unittest discover -s ci/cudf_polars/tests -v
```

These tests do not need a GPU or GitHub write access. They cover frozen review
targets, stale results, retry limits, incomplete evidence, and test-only wheel
metadata integrity. End-to-end source builds, GPU runners, artifact download,
and CodeRabbit delivery must also be validated in CI before enabling repairs.
