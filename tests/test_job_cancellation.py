import os
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from daru.config import SettingsManager
from daru.gui import worker as worker_module
from daru.gui.main_window import MainWindow
from daru.gui.worker import TranslateWorker
from daru.translation.analysis import CodexAnalysisReview
from daru.translation.engine import TranslationEngine
from daru.translation.job import (
    CancellationToken,
    TranslationCancelled,
    cancellation_scope,
    check_cancelled,
    retry_wait,
)


def test_cancelled_scope_restores_context():
    token = CancellationToken()
    token.cancel()
    with pytest.raises(TranslationCancelled), cancellation_scope(token):
        pytest.fail("cancelled scope entered")
    check_cancelled()


def test_cancel_after_provider_returns_discards_current_batch():
    token = CancellationToken()
    engine = TranslationEngine(provider="noop")

    def respond(texts):
        token.cancel()
        return ["Translated"]

    engine._translate_batch = respond
    with cancellation_scope(token), pytest.raises(TranslationCancelled):
        engine.translate_many(["Original"])


def test_cancel_wakes_worker_waiting_for_analysis():
    app = QApplication.instance() or QApplication([])
    worker = TranslateWorker({})
    result = []

    def review():
        try:
            worker._review_analysis(
                CodexAnalysisReview("Analysis", "model", "high", "DOCUMENT")
            )
        except TranslationCancelled:
            result.append("cancelled")

    thread = threading.Thread(target=review)
    thread.start()
    deadline = time.monotonic() + 2
    while not worker._analysis_waiting and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    worker.cancel()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert result == ["cancelled"]
    assert not worker._analysis_waiting


def test_closing_window_cancels_and_waits_for_actual_thread(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    ready = threading.Event()

    def translate(**kwargs):
        ready.set()
        retry_wait(30)
        raise AssertionError("cancellation was ignored")

    monkeypatch.setattr(worker_module, "translate_docx", translate)
    window = MainWindow(SettingsManager(tmp_path / "settings.json"))
    window.show()
    window._launch_worker({"job_type": "docx"}, "Start", "Running")
    worker = window.worker
    cancelled, errors, finished = [], [], []
    worker.cancelled_signal.connect(cancelled.append)
    worker.error_signal.connect(errors.append)
    worker.finished_signal.connect(finished.append)
    assert ready.wait(2)
    window.close()
    assert worker.wait(2000)
    app.processEvents()
    assert window.worker is None
    assert not window.isVisible()
    assert cancelled and not errors and not finished
