# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make a test-only cuDF wheel accept an exact unreleased Polars version."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import re
import zipfile
from pathlib import Path


def rewrite_wheel(source: Path, destination: Path, version: str) -> Path:
    """Replace only the Polars requirement and regenerate wheel RECORD."""
    if not re.fullmatch(r"[0-9][a-zA-Z0-9.+!-]*", version):
        raise ValueError("Invalid candidate version")
    with zipfile.ZipFile(source) as wheel:
        files = {name: wheel.read(name) for name in wheel.namelist()}
    (metadata,) = (
        name for name in files if name.endswith(".dist-info/METADATA")
    )
    record = metadata.removesuffix("METADATA") + "RECORD"
    text, count = re.subn(
        r"^Requires-Dist: polars(?:\[[^\]]+\])?\s*(?:[<>=!~].*)?$",
        f"Requires-Dist: polars=={version}",
        files[metadata].decode(),
        flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError(f"Expected one Polars requirement, found {count}")
    files[metadata] = text.encode()
    rows = []
    for name, data in files.items():
        if name == record:
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
        rows.append((name, "sha256=" + digest.decode().rstrip("="), len(data)))
    stream = io.StringIO(newline="")
    csv.writer(stream).writerows([*rows, (record, "", "")])
    files[record] = stream.getvalue().encode()
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / source.name
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as wheel:
        for name, data in files.items():
            wheel.writestr(name, data)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("version")
    args = parser.parse_args()
    rewrite_wheel(args.wheel, args.destination, args.version)
