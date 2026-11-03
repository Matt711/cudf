# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from ci.cudf_polars import upstream_ci as ci
from ci.cudf_polars.upstream_wheel import rewrite_wheel

CUDF_SHA = "a" * 40
POLARS_SHA = "b" * 40
SOURCE_SHA = "c" * 40


def pull_request(*, draft=True):
    return {
        "number": 123,
        "state": "open",
        "draft": draft,
        "base": {"ref": "main"},
        "head": {
            "sha": SOURCE_SHA,
            "ref": ci.BRANCH_PREFIX + "123",
            "repo": {"full_name": "NVIDIA/cudf"},
        },
    }


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = patch.dict(
            os.environ,
            {
                "GITHUB_REPOSITORY": "NVIDIA/cudf",
                "GITHUB_SERVER_URL": "https://github.com",
                "GITHUB_RUN_ID": "456",
                "GITHUB_STEP_SUMMARY": str(self.root / "summary.md"),
                "GITHUB_OUTPUT": str(self.root / "output"),
                "POLARS_MAIN_REPAIR_ENABLED": "true",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.args = argparse.Namespace(
            results=self.root,
            pr=123,
            source_sha=SOURCE_SHA,
            polars_sha=POLARS_SHA,
            cudf_sha=CUDF_SHA,
            date="2026-11-03",
            push_sha="",
            branch="",
        )
        self.pr = pull_request()
        self.reviews = []
        self.comments = []
        self.calls = []
        for suite in ci.SUITES:
            self.result(suite)
        for suite in ci.SUITES[1:]:
            self.observation(suite)
        (self.root / "stable-gpu-run-list.json").write_text(
            json.dumps(
                {"selections": {"2.1.0|spmd|default": {"nodes": ["test_gpu"]}}}
            )
        )

    def observation(self, suite, gpu="true"):
        path = self.root / suite / "cudf-polars-fallback-test.json"
        path.write_text(
            json.dumps(
                {
                    "engine": "spmd" if suite == "spmd-small" else suite,
                    "blocksize": "small"
                    if suite == "spmd-small"
                    else "default",
                    "cudf_sha": SOURCE_SHA,
                    "upstream_sha": POLARS_SHA,
                    "exitstatus": 0,
                    "selection_mode": "full",
                    "consistent": True,
                    "collected_nodeids": ["test_gpu"],
                    "tests": [
                        {
                            "nodeid": "test_gpu",
                            "outcome": "passed",
                            "complete": True,
                            "gpu_executed": gpu,
                            "fallback": "false" if gpu == "true" else "true",
                        }
                    ],
                }
            )
        )

    def result(self, suite, code=0, stage="tests"):
        path = self.root / suite / "result.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "suite": suite,
                    "stage": stage,
                    "returncode": code,
                }
            )
        )

    def api(self, path, method="GET", body=None):
        self.calls.append((path, method, body))
        if path == "repos/NVIDIA/cudf/pulls/123":
            return self.pr
        if "/reviews?" in path:
            return self.reviews
        if "/comments?" in path:
            return self.comments
        if method == "POST":
            return {}
        raise AssertionError(path)

    def test_green_requires_every_suite(self):
        with patch.object(ci, "api", self.api):
            self.assertEqual(ci.report(self.args), 0)
        statuses = [
            body for path, _, body in self.calls if "/statuses/" in path
        ]
        self.assertEqual(statuses[0]["state"], "success")
        (self.root / "unit/result.json").unlink()
        self.assertEqual(ci.outcomes(self.root)[1:], (False, False))

    def test_all_cpu_fallback_cannot_pass(self):
        self.observation("spmd", gpu="false")
        with patch.dict(os.environ, {"POLARS_MAIN_REPAIR_ENABLED": "false"}):
            self.assertEqual(ci.report(self.args), 1)

    def test_missing_gpu_observations_cannot_pass(self):
        (self.root / "spmd/cudf-polars-fallback-test.json").unlink()
        with patch.object(ci, "api", self.api):
            self.assertEqual(ci.report(self.args), 1)
        self.assertFalse(
            any(path.endswith("/comments") for path, _, _ in self.calls)
        )

    def test_partial_loss_of_known_gpu_execution_cannot_pass(self):
        path = self.root / "spmd/cudf-polars-fallback-test.json"
        data = json.loads(path.read_text())
        data["tests"][0]["gpu_executed"] = "false"
        data["collected_nodeids"].append("another_gpu_test")
        data["tests"].append(
            {
                "nodeid": "another_gpu_test",
                "outcome": "passed",
                "complete": True,
                "gpu_executed": "true",
                "fallback": "false",
            }
        )
        path.write_text(json.dumps(data))
        with patch.dict(os.environ, {"POLARS_MAIN_REPAIR_ENABLED": "false"}):
            self.assertEqual(ci.report(self.args), 1)

    def test_wrong_snapshot_observations_cannot_pass(self):
        self.args.polars_sha = "d" * 40
        with patch.dict(os.environ, {"POLARS_MAIN_REPAIR_ENABLED": "false"}):
            self.assertEqual(ci.report(self.args), 1)

    def test_incomplete_suite_does_not_request_code_changes(self):
        for code, stage in ((2, "tests"), (1, "setup")):
            with self.subTest(code=code, stage=stage):
                self.calls.clear()
                self.result("unit", code, stage)
                with patch.object(ci, "api", self.api):
                    self.assertEqual(ci.report(self.args), 1)
                self.assertFalse(
                    any("/comments" in path for path, _, _ in self.calls)
                )

    def test_test_failure_requests_direct_commit(self):
        self.result("unit", 1)
        with patch.object(ci, "api", self.api):
            self.assertEqual(ci.report(self.args), 1)
        comments = [
            body
            for path, method, body in self.calls
            if path.endswith("/comments") and method == "POST"
        ]
        self.assertEqual(len(comments), 1)
        self.assertTrue(
            comments[0]["body"].startswith("@coderabbitai fix-ci commit\n")
        )

    def test_no_writes_when_repairs_disabled(self):
        self.result("spmd", 1)
        with (
            patch.dict(os.environ, {"POLARS_MAIN_REPAIR_ENABLED": "false"}),
            patch.object(ci, "api") as api,
        ):
            self.assertEqual(ci.report(self.args), 1)
        api.assert_not_called()

    def test_stale_results_cannot_request_a_repair(self):
        self.result("unit", 1)
        self.pr["head"]["sha"] = "d" * 40
        with patch.object(ci, "api", self.api):
            self.assertEqual(ci.report(self.args), 1)
        self.assertTrue(all(method == "GET" for _, method, _ in self.calls))

    def test_retry_budget_is_per_snapshot(self):
        self.result("unit", 1)
        self.comments = [{"body": f"<!-- polars-main:{POLARS_SHA} -->"}] * 2
        with patch.object(ci, "api", self.api):
            self.assertEqual(ci.report(self.args), 1)
        self.assertFalse(
            any(path.endswith("/comments") for path, _, _ in self.calls)
        )

    def test_workflow_retry_does_not_duplicate_agent_request(self):
        self.result("unit", 1)
        self.comments = [
            {
                "body": f"<!-- polars-main:{POLARS_SHA} -->\n<!-- source:{SOURCE_SHA} -->"
            }
        ]
        with patch.object(ci, "api", self.api):
            self.assertEqual(ci.report(self.args), 1)
        self.assertFalse(
            any(path.endswith("/comments") for path, _, _ in self.calls)
        )

    def test_human_review_freezes_new_requests(self):
        self.result("spmd", 1)
        self.reviews = [{"user": {"type": "User"}, "state": "COMMENTED"}]
        with patch.object(ci, "api", self.api):
            self.assertEqual(ci.report(self.args), 1)
        self.assertFalse(
            any(path.endswith("/comments") for path, _, _ in self.calls)
        )

    def test_ready_pr_freezes_without_reading_reviews(self):
        with patch.object(ci, "api") as api:
            self.assertTrue(
                ci.frozen("NVIDIA/cudf", pull_request(draft=False))
            )
        api.assert_not_called()

    def test_bot_review_does_not_freeze_repairs(self):
        self.reviews = [{"user": {"type": "Bot"}, "state": "COMMENTED"}]
        with patch.object(ci, "api", self.api):
            self.assertFalse(ci.frozen("NVIDIA/cudf", self.pr))

    def test_reject_foreign_repair_branch(self):
        self.pr["head"]["repo"]["full_name"] = "someone/cudf"
        with patch.object(ci, "api", self.api), self.assertRaises(ValueError):
            ci.repair_pr("NVIDIA/cudf", 123)

    def test_prepare_preserves_frozen_snapshot(self):
        target = {
            "polars_sha": POLARS_SHA,
            "cudf_sha": CUDF_SHA,
            "date": "2026-11-01",
        }

        def api(path, method="GET", body=None):
            if "/contents/" in path:
                return {
                    "content": base64.b64encode(
                        json.dumps(target).encode()
                    ).decode()
                }
            return self.api(path, method, body)

        with patch.object(ci, "api", api):
            ci.prepare(self.args)
        output = (self.root / "output").read_text()
        self.assertIn(f"polars_sha={POLARS_SHA}\n", output)
        self.assertIn(f"source_sha={SOURCE_SHA}\n", output)
        self.assertIn("date=2026-11-01\n", output)

    def test_observer_does_not_retarget_a_reviewed_pr(self):
        self.reviews = [{"user": {"type": "User"}, "state": "APPROVED"}]

        def api(path, method="GET", body=None):
            if path.startswith("repos/NVIDIA/cudf/pulls?state=open"):
                return [self.pr]
            return self.api(path, method, body)

        with patch.object(ci, "api", api):
            ci.maintain_pr("NVIDIA/cudf", self.args, "https://github.com/run")
        self.assertTrue(all(method == "GET" for _, method, _ in self.calls))

    def test_new_repair_can_replace_a_previously_merged_target(self):
        calls = []

        def api(path, method="GET", body=None):
            calls.append((path, method, body))
            if path.startswith("repos/NVIDIA/cudf/pulls?"):
                return []
            if "/contents/" in path and method == "GET":
                return {"sha": "old-target-blob"}
            if path == "repos/NVIDIA/cudf/pulls" and method == "POST":
                return {"number": 124}
            return {}

        with patch.object(ci, "api", api):
            ci.maintain_pr("NVIDIA/cudf", self.args, "https://github.com/run")
        writes = [body for _, method, body in calls if method == "PUT"]
        self.assertEqual(writes[0]["sha"], "old-target-blob")
        target = json.loads(base64.b64decode(writes[0]["content"]))
        self.assertEqual(target["polars_sha"], POLARS_SHA)
        dispatched = [
            body for path, _, body in calls if path.endswith("/dispatches")
        ]
        self.assertEqual(dispatched[0]["inputs"], {"repair_pr": "124"})


class WheelTests(unittest.TestCase):
    def test_only_polars_requirement_changes_and_record_is_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "cudf_polars-1-py3-none-any.whl"
            with zipfile.ZipFile(original, "w") as wheel:
                wheel.writestr("cudf_polars/__init__.py", "payload")
                wheel.writestr(
                    "cudf_polars-1.dist-info/METADATA",
                    (
                        "Metadata-Version: 2.1\nName: cudf-polars\nVersion: 1\n"
                        "Requires-Dist: polars<2.2,>=2.0\n"
                        "Requires-Dist: pylibcudf==1.0\n"
                    ),
                )
                wheel.writestr("cudf_polars-1.dist-info/RECORD", "old")
            original_bytes = original.read_bytes()
            output = rewrite_wheel(original, root / "candidate", "2.2.0.dev3")
            self.assertEqual(original.read_bytes(), original_bytes)
            with zipfile.ZipFile(output) as wheel:
                metadata = wheel.read(
                    "cudf_polars-1.dist-info/METADATA"
                ).decode()
                self.assertIn("Requires-Dist: polars==2.2.0.dev3\n", metadata)
                self.assertIn("Requires-Dist: pylibcudf==1.0\n", metadata)
                record = wheel.read("cudf_polars-1.dist-info/RECORD").decode()
                for name, digest, size in csv.reader(io.StringIO(record)):
                    if name.endswith("/RECORD"):
                        self.assertEqual((digest, size), ("", ""))
                        continue
                    data = wheel.read(name)
                    expected = (
                        base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                        .decode()
                        .rstrip("=")
                    )
                    self.assertEqual(digest, "sha256=" + expected)
                    self.assertEqual(int(size), len(data))

    def test_reject_version_injection(self):
        with self.assertRaises(ValueError):
            rewrite_wheel(
                Path("unused"), Path("unused"), "2.2\nRequires-Dist: injected"
            )


if __name__ == "__main__":
    unittest.main()
