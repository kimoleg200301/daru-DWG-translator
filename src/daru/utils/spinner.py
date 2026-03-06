"""CLI spinner and terminal colorization helpers."""

import sys
import threading
import time
from itertools import cycle
from typing import Optional


def colorize(text: str, color_code: str) -> str:
    if sys.stdout.isatty():
        return f"\033[{color_code}m{text}\033[0m"
    return text


class Spinner:
    def __init__(self, message: str = "", enabled: bool = True, interval: float = 0.1) -> None:
        self.message = message
        self.enabled = enabled
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self):
        if self.enabled:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            self._stop_event.set()
            if self._thread and self._thread.is_alive():
                self._thread.join()
            sys.stdout.write("\r" + " " * (len(self.message) + 4) + "\r")
            sys.stdout.flush()

    def _spin(self) -> None:
        spinner = cycle("|/-\\")
        while not self._stop_event.is_set():
            sys.stdout.write(f"\r{next(spinner)} {self.message}")
            sys.stdout.flush()
            time.sleep(self.interval)
