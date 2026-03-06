"""Shared fixtures for daru tests."""

import sys
from pathlib import Path

# Ensure src/ is importable when running tests from repo root.
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
