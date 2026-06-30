"""Shared Codex pre-translation analysis state for one translation job."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Callable, Optional

CODEX_ANALYSIS_MAX_APPROVED_CHARS = 20000


class CodexAnalysisCancelled(RuntimeError):
    """Raised when the user cancels translation at the analysis review step."""


@dataclass(frozen=True)
class CodexAnalysisReview:
    """Editable analysis payload presented before translation starts."""

    text: str
    model: str
    reasoning_effort: str
    context_label: str
    used_fallback: bool = False
    warning: str = ""


AnalysisReviewer = Callable[[CodexAnalysisReview], Optional[str]]


class CodexAnalysisSession:
    """Resolve and retain one approved analysis across all engines in a job."""

    def __init__(self, reviewer: Optional[AnalysisReviewer] = None) -> None:
        self._reviewer = reviewer
        self._lock = Lock()
        self._approved_text: Optional[str] = None
        self._cancelled = False

    @property
    def approved_text(self) -> Optional[str]:
        with self._lock:
            return self._approved_text

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def resolve(self, review: CodexAnalysisReview) -> str:
        with self._lock:
            if self._cancelled:
                raise CodexAnalysisCancelled(
                    "Перевод отменён на этапе предварительного анализа."
                )
            if self._approved_text is not None:
                return self._approved_text

        candidate = review.text if self._reviewer is None else self._reviewer(review)
        if candidate is None:
            with self._lock:
                self._cancelled = True
            raise CodexAnalysisCancelled(
                "Перевод отменён на этапе предварительного анализа."
            )

        approved = str(candidate).strip()
        if not approved:
            raise ValueError("Утверждённый предварительный анализ не может быть пустым.")
        approved = approved[:CODEX_ANALYSIS_MAX_APPROVED_CHARS]
        with self._lock:
            if self._approved_text is None:
                self._approved_text = approved
            return self._approved_text


__all__ = [
    "CODEX_ANALYSIS_MAX_APPROVED_CHARS",
    "AnalysisReviewer",
    "CodexAnalysisCancelled",
    "CodexAnalysisReview",
    "CodexAnalysisSession",
]
