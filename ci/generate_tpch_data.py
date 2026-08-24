#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate TPC-H data at a given scale factor using tpchgen-cli."""

from __future__ import annotations

import argparse
import os
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Scale factor."
    )
    parser.add_argument(
        "--parts", type=int, default=4, help="Number of parts per table."
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("TPCH_DATA_DIR"),
        help="Output directory. Defaults to TPCH_DATA_DIR environment variable.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        parser.error("--output-dir is required (or set TPCH_DATA_DIR).")

    subprocess.run(
        [
            "tpchgen-cli",
            "parquet",
            "-s",
            str(args.scale),
            f"--parts={args.parts}",
            f"--output-dir={args.output_dir}",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
