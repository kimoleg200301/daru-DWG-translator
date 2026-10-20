"""Cooperative cancellation shared by a job's pipeline and translation calls."""

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event
from typing import Iterator


class TranslationCancelled(RuntimeError):
    """The active job was cancelled before its next processing step."""


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise TranslationCancelled(
                "Перевод отменён. Сохранённые пакеты доступны для продолжения."
            )

    def wait(self, seconds: float) -> None:
        self._event.wait(seconds)
        self.check()


_active_token: ContextVar[CancellationToken | None] = ContextVar(
    "daru_job_token", default=None
)


@contextmanager
def cancellation_scope(token: CancellationToken) -> Iterator[None]:
    context = _active_token.set(token)
    try:
        token.check()
        yield
    finally:
        _active_token.reset(context)


def check_cancelled() -> None:
    token = _active_token.get()
    if token is not None:
        token.check()


def retry_wait(seconds: float) -> None:
    token = _active_token.get()
    if token is None:
        Event().wait(seconds)
    else:
        token.wait(seconds)
