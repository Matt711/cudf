# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve nightly targets and maintain the optional CodeRabbit repair PR."""

from __future__ import annotations

import argparse
import base64
import datetime
import json
import os
import re
import subprocess
from pathlib import Path

TARGET = "ci/cudf_polars/upstream-target.json"
BRANCH_PREFIX = "bot/polars-main-"
SUITES = ("unit", "in-memory", "spmd", "spmd-small")


def api(path, method="GET", body=None):
    command = ["gh", "api", "--method", method, path]
    if body is not None:
        command += ["--input", "-"]
    result = subprocess.run(
        command,
        input=json.dumps(body) if body is not None else None,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout) if result.stdout.strip() else None


def sha(value):
    if not re.fullmatch("[0-9a-f]{40}", value):
        raise ValueError("Expected a full commit SHA")
    return value


def read_target(repo, ref):
    result = api(f"repos/{repo}/contents/{TARGET}?ref={sha(ref)}")
    return json.loads(base64.b64decode(result["content"]))


def repair_pr(repo, number):
    pr = api(f"repos/{repo}/pulls/{int(number)}")
    if (
        pr["state"] != "open"
        or pr["base"]["ref"] != "main"
        or pr["head"]["repo"]["full_name"] != repo
        or not pr["head"]["ref"].startswith(BRANCH_PREFIX)
    ):
        raise ValueError("Not an open, same-repository Polars repair PR")
    return pr


def pages(path):
    for page in range(1, 101):
        separator = "&" if "?" in path else "?"
        items = api(f"{path}{separator}per_page=100&page={page}")
        yield from items
        if len(items) < 100:
            return
    raise ValueError("Too many API pages")


def frozen(repo, pr):
    return not pr["draft"] or any(
        review["user"]["type"] != "Bot" and review["state"] != "PENDING"
        for review in pages(f"repos/{repo}/pulls/{pr['number']}/reviews")
    )


def prepare(args):
    repo = os.environ["GITHUB_REPOSITORY"]
    target = None
    pr_number = args.pr
    if args.push_sha:
        target = read_target(repo, args.push_sha)
        matches = [
            pr
            for pr in pages(f"repos/{repo}/pulls?state=open&base=main")
            if pr["head"]["ref"] == args.branch
        ]
        if len(matches) != 1:
            raise ValueError("Push does not identify one open repair PR")
        pr_number = matches[0]["number"]
    if pr_number:
        pr = repair_pr(repo, pr_number)
        source = sha(pr["head"]["sha"])
        target = read_target(repo, source)
    else:
        source = sha(args.cudf_sha)
    if target is not None:
        cudf_sha = sha(target["cudf_sha"])
        date = target["date"]
        polars_sha = sha(target["polars_sha"])
    else:
        cudf_sha = sha(args.cudf_sha)
        date = args.date
        polars_sha = (
            sha(args.polars_sha)
            if args.polars_sha
            else sha(api("repos/pola-rs/polars/commits/main")["sha"])
        )
    datetime.date.fromisoformat(date)
    values = dict(
        cudf_sha=cudf_sha,
        date=date,
        source_sha=source,
        polars_sha=polars_sha,
        pr=pr_number,
    )
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        for name, value in values.items():
            output.write(f"{name}={value}\n")
    print(json.dumps(values, indent=2))


def outcomes(directory):
    results = {}
    for path in directory.rglob("result.json"):
        result = json.loads(path.read_text())
        suite = result["suite"]
        if suite not in SUITES or suite in results:
            raise ValueError("Unknown or duplicate test suite")
        results[suite] = result
    complete = set(results) == set(SUITES) and all(
        item["stage"] == "tests" and item["returncode"] in (0, 1)
        for item in results.values()
    )
    passed = complete and all(
        item["returncode"] == 0 for item in results.values()
    )
    return results, complete, passed


def telemetry(directory, source_sha, polars_sha):
    """Require complete observations and real GPU execution for a green run."""
    baselines = {
        path.read_text()
        for path in directory.rglob("stable-gpu-run-list.json")
    }
    if len(baselines) != 1:
        return False, False, {}
    selections = json.loads(baselines.pop())["selections"]
    latest = max(
        (key for key in selections if key.endswith("|spmd|default")),
        key=lambda key: tuple(map(int, key.split("|")[0].split("."))),
    )
    expected_gpu = set(selections[latest]["nodes"])
    observed = {}
    for path in directory.rglob("cudf-polars-fallback-*.json"):
        data = json.loads(path.read_text())
        key = (
            data["engine"]
            if data["blocksize"] == "default"
            else data["engine"] + "-small"
        )
        if key in observed or key not in SUITES[1:]:
            return False, False, observed
        tests = data["tests"]
        nodes = data["collected_nodeids"]
        if (
            data["cudf_sha"] != source_sha
            or data["upstream_sha"] != polars_sha
            or data["exitstatus"] != 0
            or data["selection_mode"] != "full"
            or data["consistent"] is not True
            or not nodes
            or len(nodes) != len(tests)
            or {test["nodeid"] for test in tests} != set(nodes)
            or not all(test["complete"] is True for test in tests)
        ):
            return False, False, observed
        observed[key] = {
            "collected": len(tests),
            "gpu_executed": sum(
                test["outcome"] == "passed" and test["gpu_executed"] == "true"
                for test in tests
            ),
            "fallback": sum(test["fallback"] == "true" for test in tests),
            "lost_gpu": [
                test["nodeid"]
                for test in tests
                if key == "spmd"
                and test["nodeid"] in expected_gpu
                and test["outcome"] == "passed"
                and test["gpu_executed"] != "true"
            ],
        }
    complete = set(observed) == set(SUITES[1:])
    accelerated = complete and all(
        item["gpu_executed"] > 0 and not item["lost_gpu"]
        for item in observed.values()
    )
    return complete, accelerated, observed


def dispatch(repo, branch, pr):
    api(
        f"repos/{repo}/actions/workflows/polars-main.yaml/dispatches",
        "POST",
        {"ref": branch, "inputs": {"repair_pr": str(pr)}},
    )


def maintain_pr(repo, args, url):
    matches = [
        pr
        for pr in pages(f"repos/{repo}/pulls?state=open&base=main")
        if pr["head"]["ref"].startswith(BRANCH_PREFIX)
        and pr["head"]["repo"]["full_name"] == repo
    ]
    if len(matches) > 1:
        raise ValueError(
            "Multiple open Polars repair PRs need maintainer attention"
        )
    pr = matches[0] if matches else None
    if pr and frozen(repo, pr):
        print("Repair PR is under human review; leaving its target unchanged")
        return
    branch = (
        pr["head"]["ref"]
        if pr
        else BRANCH_PREFIX + os.environ["GITHUB_RUN_ID"]
    )
    target = dict(
        polars_sha=sha(args.polars_sha),
        cudf_sha=sha(args.cudf_sha),
        date=args.date,
        failure_run=url,
    )
    body = {
        "message": "Test Polars main compatibility snapshot",
        "content": base64.b64encode(
            (json.dumps(target, indent=2) + "\n").encode()
        ).decode(),
        "branch": branch,
    }
    if pr:
        current = api(f"repos/{repo}/contents/{TARGET}?ref={branch}")
        previous = json.loads(base64.b64decode(current["content"]))
        if previous["polars_sha"] == args.polars_sha:
            print("Snapshot already has an active repair PR")
            return
        body["sha"] = current["sha"]
    else:
        api(
            f"repos/{repo}/git/refs",
            "POST",
            {
                "ref": "refs/heads/" + branch,
                "sha": args.cudf_sha,
            },
        )
        # A previous merged repair can leave its target file on main.
        try:
            current = api(f"repos/{repo}/contents/{TARGET}?ref={branch}")
        except subprocess.CalledProcessError as error:
            if "(HTTP 404)" not in error.stderr:
                raise
        else:
            body["sha"] = current["sha"]
    api(f"repos/{repo}/contents/{TARGET}", "PUT", body)
    if not pr:
        pr = api(
            f"repos/{repo}/pulls",
            "POST",
            {
                "head": branch,
                "base": "main",
                "draft": True,
                "title": "Fix cudf-polars compatibility with Polars main",
                "body": (
                    "## Description\n\n"
                    f"The nightly Polars source snapshot failed GPU compatibility tests: {url}.\n\n"
                    f"The exact source and binary targets are recorded in `{TARGET}`. "
                    "This draft starts with that target; CodeRabbit will investigate the failing candidate check.\n\n"
                    "Follow `.agents/skills/upgrade-polars/SKILL.md` and "
                    "`ci/cudf_polars/UPSTREAM.md`. Preserve stable dependency bounds, "
                    "test coverage, and GPU execution. Compare failing cases with CPU Polars "
                    "before deciding where to fix them. Do not hide wrong results with xfails. "
                    "Keep changes small. Report missing GPU verification honestly.\n\n"
                    "Mark this PR ready for review to freeze automatic target updates. "
                    "Any submitted human review also freezes them.\n\n"
                    "## Verification\n\n"
                    "The candidate workflow tests this branch after the target commit. "
                    "Normal stable CI and maintainer approval are also required before merge.\n\n"
                    "Tools: Codex\n"
                ),
            },
        )
    dispatch(repo, branch, pr["number"])


def report(args):
    results, complete, passed = outcomes(args.results)
    observations = {}
    if passed:
        telemetry_complete, accelerated, observations = telemetry(
            args.results, args.source_sha, args.polars_sha
        )
        complete = complete and telemetry_complete
        passed = complete and accelerated
    repo = os.environ["GITHUB_REPOSITORY"]
    url = f"{os.environ['GITHUB_SERVER_URL']}/{repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    summary = (
        f"Polars `{sha(args.polars_sha)}`, cuDF source `{sha(args.source_sha)}`, "
        f"cuDF binaries `{sha(args.cudf_sha)}`.\n\n"
        + "\n".join(
            f"- {suite}: {item}" for suite, item in sorted(results.items())
        )
        + f"\n\nGPU observations: {observations}\n"
        + f"\n\nComplete: {complete}. Passed: {passed}.\n"
    )
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as output:
        output.write(summary)
    enabled = os.environ.get("POLARS_MAIN_REPAIR_ENABLED") == "true"
    if enabled and args.pr:
        pr = repair_pr(repo, args.pr)
        if pr["head"]["sha"] != args.source_sha:
            print(
                "PR moved during tests; discarding stale status and repair request"
            )
            return 1
        api(
            f"repos/{repo}/statuses/{args.source_sha}",
            "POST",
            {
                "state": "success" if passed else "failure",
                "context": "cudf-polars/Polars-main",
                "target_url": url,
                "description": "Candidate tests passed"
                if passed
                else "Candidate tests failed or incomplete",
            },
        )
        if complete and not passed and not frozen(repo, pr):
            marker = f"<!-- polars-main:{args.polars_sha} -->"
            source_marker = f"<!-- source:{args.source_sha} -->"
            comments = list(pages(f"repos/{repo}/issues/{args.pr}/comments"))
            attempts = sum(marker in comment["body"] for comment in comments)
            already_requested = any(
                source_marker in comment["body"] for comment in comments
            )
            if attempts < 2 and not already_requested:
                api(
                    f"repos/{repo}/issues/{args.pr}/comments",
                    "POST",
                    {
                        "body": (
                            "@coderabbitai fix-ci commit\n\n"
                            f"Candidate failure: {url}. Follow the upgrade skill and "
                            "`ci/cudf_polars/UPSTREAM.md`. Keep the target and stable pins unchanged.\n\n"
                            f"{marker}\n{source_marker}\n\nTools: Codex, CodeRabbit"
                        ),
                    },
                )
    elif enabled and complete and not passed:
        maintain_pr(repo, args, url)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "report"))
    parser.add_argument("--cudf-sha", default="")
    parser.add_argument("--date", default="")
    parser.add_argument("--polars-sha", default="")
    parser.add_argument("--source-sha", default="")
    parser.add_argument("--pr", type=int, default=0)
    parser.add_argument("--push-sha", default="")
    parser.add_argument("--branch", default="")
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare(args)
        return 0
    return report(args)


if __name__ == "__main__":
    raise SystemExit(main())
