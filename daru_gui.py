#!/usr/bin/env python3
"""Backward-compatible wrapper — launches the GUI from the new package structure."""

import sys
from pathlib import Path

# Ensure the src directory is importable.
_src = Path(__file__).resolve().parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from src.daru.gui.app import main  # noqa: E402

if __name__ == "__main__":
    main()
