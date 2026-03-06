#!/usr/bin/env python3
"""Backward-compatible wrapper — runs the DXF CLI pipeline from the new package structure."""

import sys
from pathlib import Path

# Ensure the src directory is importable.
_src = Path(__file__).resolve().parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from daru.dxf.pipeline import parse_args, run_pipeline  # noqa: E402


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
