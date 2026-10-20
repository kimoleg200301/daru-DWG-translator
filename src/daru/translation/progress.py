"""Format-independent stage progress, isolated to the current job context.

Counts describe the current stage, never an estimated percentage of the whole
job. A completion event is emitted only by the job runner after a successful
pipeline return; nested PDF fallback pipelines cannot finish the parent job.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterator, Literal

ProgressStage = Literal["read", "ocr", "analysis", "translate", "apply", "export", "complete"]


@dataclass(frozen=True)
class ProgressEvent:
    stage: ProgressStage
    completed: int | None = None
    total: int | None = None


_listener: ContextVar[Callable[[ProgressEvent], None] | None] = ContextVar(
    "daru_progress_listener", default=None,
)


@contextmanager
def progress_scope(listener: Callable[[ProgressEvent], None]) -> Iterator[None]:
    context = _listener.set(listener)
    try:
        yield
    finally:
        _listener.reset(context)


def report_progress(
    stage: ProgressStage, completed: int | None = None, total: int | None = None,
) -> None:
    listener = _listener.get()
    if listener is not None:
        listener(ProgressEvent(stage, completed, total))
