"""Shared dependency guards and scalar conversions for PDF processing."""

from __future__ import annotations

from typing import Sequence


def _noop_log(_message: str) -> None:
    return


def _require_dependency(obj: object, message: str) -> None:
    if obj is None:
        raise RuntimeError(message)


def _safe_int(seq: Sequence, index: int, default: int = 0) -> int:
    try:
        return int(seq[index])
    except Exception:
        return default


def _safe_float(seq: Sequence, index: int, default: float = 0.0) -> float:
    try:
        return float(seq[index])
    except Exception:
        return default
