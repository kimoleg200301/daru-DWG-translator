"""GUI and worker tests for editable Codex pre-translation analysis."""

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog

from daru.gui import analysis_dialog
from daru.gui import main_window
from daru.gui import worker as worker_module
from daru.gui.analysis_dialog import CodexAnalysisDialog
from daru.gui.main_window import MainWindow
from daru.gui.worker import TranslateWorker
from daru.config import SettingsManager
from daru.translation.analysis import (
    CODEX_ANALYSIS_MAX_APPROVED_CHARS,
    CodexAnalysisReview,
)


def _payload(**overrides):
    payload = {
        "text": "Исходный анализ",
        "model": "gpt-5.5",
        "reasoning_effort": "high",
        "context_label": "DOCX DOCUMENT",
        "used_fallback": False,
        "warning": "",
    }
    payload.update(overrides)
    return payload


def test_analysis_dialog_allows_editing_and_acceptance():
    app = QApplication.instance() or QApplication([])
    dialog = CodexAnalysisDialog(_payload())

    dialog.editor.setPlainText("Отредактированный анализ")
    dialog.start_button.click()

    assert dialog.result() == QDialog.Accepted
    assert dialog.approved_text() == "Отредактированный анализ"
    dialog.close()
    app.processEvents()


def test_analysis_dialog_rejects_empty_text(monkeypatch):
    app = QApplication.instance() or QApplication([])
    warnings = []
    monkeypatch.setattr(
        analysis_dialog.QMessageBox,
        "warning",
        lambda _parent, _title, message: warnings.append(message),
    )
    dialog = CodexAnalysisDialog(_payload())

    dialog.editor.clear()
    dialog.start_button.click()

    assert dialog.result() == QDialog.Rejected
    assert warnings == ["Анализ не может быть пустым."]
    dialog.close()
    app.processEvents()


def test_analysis_dialog_limits_text_and_shows_fallback_warning():
    app = QApplication.instance() or QApplication([])
    dialog = CodexAnalysisDialog(
        _payload(
            used_fallback=True,
            warning="Показан исходный контекст.",
        )
    )

    dialog.editor.setPlainText("x" * (CODEX_ANALYSIS_MAX_APPROVED_CHARS + 100))

    assert len(dialog.editor.toPlainText()) == CODEX_ANALYSIS_MAX_APPROVED_CHARS
    assert not dialog.warning_label.isHidden()
    assert "Показан исходный контекст" in dialog.warning_label.text()
    dialog.cancel_button.click()
    assert dialog.result() == QDialog.Rejected
    dialog.close()
    app.processEvents()


def test_worker_waits_for_analysis_response_and_resumes():
    worker = TranslateWorker({"job_type": "docx"})
    review = CodexAnalysisReview(
        text="Анализ",
        model="gpt-5.5",
        reasoning_effort="high",
        context_label="DOCX DOCUMENT",
    )
    result = {}

    thread = threading.Thread(
        target=lambda: result.setdefault("value", worker._review_analysis(review))
    )
    thread.start()
    deadline = time.monotonic() + 2
    while not worker._analysis_waiting and time.monotonic() < deadline:
        time.sleep(0.01)

    assert worker._analysis_waiting
    worker.submit_analysis("Подтверждённый анализ")
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert result["value"] == "Подтверждённый анализ"


def test_worker_returns_none_when_analysis_is_cancelled():
    worker = TranslateWorker({"job_type": "docx"})
    review = CodexAnalysisReview(
        text="Анализ",
        model="gpt-5.5",
        reasoning_effort="high",
        context_label="DOCX DOCUMENT",
    )
    result = {}

    thread = threading.Thread(
        target=lambda: result.setdefault("value", worker._review_analysis(review))
    )
    thread.start()
    deadline = time.monotonic() + 2
    while not worker._analysis_waiting and time.monotonic() < deadline:
        time.sleep(0.01)

    worker.submit_analysis(None)
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert result["value"] is None


def test_worker_emits_cancelled_without_finishing_translation(monkeypatch):
    app = QApplication.instance() or QApplication([])
    cancelled = []
    finished = []

    def fake_translate_docx(**kwargs):
        kwargs["codex_analysis_session"].resolve(
            CodexAnalysisReview(
                text="Анализ",
                model="gpt-5.5",
                reasoning_effort="high",
                context_label="DOCX DOCUMENT",
            )
        )
        raise AssertionError("translation must not continue after cancellation")

    monkeypatch.setattr(worker_module, "translate_docx", fake_translate_docx)
    worker = TranslateWorker(
        {
            "job_type": "docx",
            "translator_name": "codex",
        }
    )
    worker.cancelled_signal.connect(cancelled.append)
    worker.finished_signal.connect(finished.append)

    worker.start()
    deadline = time.monotonic() + 2
    while not worker._analysis_waiting and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    worker.submit_analysis(None)
    assert worker.wait(2000)
    app.processEvents()

    assert len(cancelled) == 1
    assert finished == []


def test_main_window_submits_edited_analysis_and_cancel(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow(SettingsManager(path=tmp_path / "settings.json"))

    class FakeWorker:
        def __init__(self):
            self.responses = []

        def submit_analysis(self, text):
            self.responses.append(text)

    class FakeDialog:
        accepted = True

        def __init__(self, payload, parent):
            self.payload = payload
            self.parent = parent

        def exec(self):
            return QDialog.Accepted if self.accepted else QDialog.Rejected

        def approved_text(self):
            return "Изменённый анализ"

    fake_worker = FakeWorker()
    window.worker = fake_worker
    monkeypatch.setattr(main_window, "CodexAnalysisDialog", FakeDialog)

    window.handle_analysis_ready(_payload())
    assert fake_worker.responses == ["Изменённый анализ"]
    assert window.status_label.text() == "Перевод выполняется..."

    FakeDialog.accepted = False
    window.handle_analysis_ready(_payload())
    assert fake_worker.responses[-1] is None
    assert window.status_label.text() == "Перевод отменяется..."

    window.worker = None
    window.close()
    app.processEvents()
