import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from daru.config import SettingsManager
from daru.gui import worker as worker_module
from daru.gui.main_window import MainWindow
from daru.gui.worker import TranslateWorker
from daru.translation.progress import ProgressEvent, progress_scope, report_progress


def test_progress_scopes_are_isolated_between_concurrent_jobs_and_restored_on_error():
    barrier = Barrier(2)

    def job(count):
        outer, inner = [], []
        with progress_scope(outer.append):
            barrier.wait(timeout=3)
            report_progress("translate", count, 5)
            with pytest.raises(RuntimeError), progress_scope(inner.append):
                report_progress("export")
                raise RuntimeError("failed nested stage")
            report_progress("apply")
        report_progress("complete")
        assert outer == [ProgressEvent("translate", count, 5), ProgressEvent("apply")]
        assert inner == [ProgressEvent("export")]

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(job, (1, 3)))


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
def test_worker_emits_completion_only_after_successful_pipeline_return(monkeypatch, outcome):
    app = QApplication.instance() or QApplication([])
    worker = TranslateWorker({"job_type": "docx"})
    events, finished, errors, cancelled = [], [], [], []
    worker.progress_signal.connect(events.append)
    worker.finished_signal.connect(finished.append)
    worker.error_signal.connect(errors.append)
    worker.cancelled_signal.connect(cancelled.append)

    def translate(**kwargs):
        report_progress("translate", 3, 3)
        report_progress("export")
        if outcome == "failure":
            raise OSError("export failed")
        if outcome == "cancel":
            worker.cancel()
        return {"output_path": "saved.docx"}

    monkeypatch.setattr(worker_module, "translate_docx", translate)
    worker.run()
    app.processEvents()
    assert events[:3] == [ProgressEvent("read"), ProgressEvent("translate", 3, 3), ProgressEvent("export")]
    assert (events[-1].stage == "complete") is (outcome == "success")
    assert bool(finished) is (outcome == "success")
    assert bool(errors) is (outcome == "failure")
    assert bool(cancelled) is (outcome == "cancel")
    count = len(events)
    report_progress("read")
    assert len(events) == count


def test_gui_progress_does_not_overwrite_cancel_or_analysis_status(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow(SettingsManager(tmp_path / "settings.json"))
    window.worker = object()
    try:
        event = ProgressEvent("translate", 12, 20)
        window.handle_progress(event)
        assert window.status_label.text() == "Перевод текста: 12 из 20"
        window._cancel_requested = True
        window.status_label.setText("Отмена...")
        window.handle_progress(event)
        assert window.status_label.text() == "Отмена..."
        window._cancel_requested = False
        window._analysis_dialog = object()
        window.status_label.setText("Ожидание анализа...")
        window.handle_progress(event)
        assert window.status_label.text() == "Ожидание анализа..."
    finally:
        window.worker = None
        window._analysis_dialog = None
        window.close()
        app.processEvents()
