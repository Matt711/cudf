#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check agent skills under .agents/skills for the problems reviewers find.

Usage: python check_skill.py <skill-dir-or-.agents/skills> [...]

Problems are printed one per line. A repository path that the skill names but that does not
exist is only a warning, because a skill may name build outputs or paths in another repository.

Exit codes: 0 no problems (warnings allowed), 1 usage error, 2 problems found.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_DESCRIPTION = 1024
# Backticked text that starts like a repository path is checked for existence.
REPO_PREFIXES = (
    ".agents/",
    ".github/",
    "ci/",
    "conda/",
    "cpp/",
    "docs/",
    "java/",
    "python/",
)
BACKTICKED = re.compile(r"`([^`\s]+)`")
LINK = re.compile(r"\]\(([^)\s]+)\)")


def frontmatter(text: str) -> dict[str, str] | None:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    fields: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return fields
        key, sep, value = line.partition(":")
        if sep and not line.startswith((" ", "\t")):
            fields[key.strip()] = value.strip()
    return None


def is_placeholder(path: str) -> bool:
    return any(c in path for c in "<>*{}$")


def check(skill: Path, root: Path) -> tuple[list[str], list[str]]:
    where = skill.relative_to(root)
    doc = skill / "SKILL.md"
    if not doc.is_file():
        return [f"{where}: no SKILL.md"], []
    text = doc.read_text()
    fields = frontmatter(text)
    if fields is None:
        return [f"{where}/SKILL.md: no frontmatter between '---' lines"], []
    problems = []
    warnings = []
    name = fields.get("name", "")
    if name != skill.name:
        problems.append(
            f"{where}/SKILL.md: name '{name}' does not match the folder"
        )
    if not NAME.match(skill.name):
        problems.append(
            f"{where}: folder name is not lowercase words joined by hyphens"
        )
    description = fields.get("description", "")
    if not description:
        problems.append(f"{where}/SKILL.md: no description")
    elif len(description) > MAX_DESCRIPTION:
        problems.append(
            f"{where}/SKILL.md: description is {len(description)} characters "
            f"(at most {MAX_DESCRIPTION})"
        )
    for path in BACKTICKED.findall(text):
        path = path.rstrip(".,:;")
        if path.startswith(REPO_PREFIXES) and not is_placeholder(path):
            if not (root / path).exists():
                warnings.append(
                    f"warning: {where}/SKILL.md: `{path}` does not exist in the repository"
                )
    for target in LINK.findall(text):
        target = target.split("#", 1)[0]
        if not target or "://" in target or is_placeholder(target):
            continue
        if not (skill / target).exists():
            problems.append(
                f"{where}/SKILL.md: link target '{target}' does not exist"
            )
    return problems, warnings


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.strip(), file=sys.stderr)  # noqa: T201
        return 1
    try:
        root = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        print("run this from a cuDF checkout", file=sys.stderr)  # noqa: T201
        return 1
    problems = []
    warnings = []
    for arg in argv:
        path = Path(arg).resolve()
        if not path.is_dir():
            print(f"not a directory: {arg}", file=sys.stderr)  # noqa: T201
            return 1
        skills = (
            [path]
            if (path / "SKILL.md").exists()
            else sorted(p for p in path.iterdir() if p.is_dir())
        )
        for skill in skills:
            found, warned = check(skill, root)
            problems.extend(found)
            warnings.extend(warned)
    for line in problems + warnings:
        print(line)  # noqa: T201
    return 2 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
