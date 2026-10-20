"""Background translation worker thread."""

from threading import Condition
from typing import Any, Dict

from PySide6.QtCore import QThread, Signal

from ..translation.job import CancellationToken, TranslationCancelled, cancellation_scope
from ..translation.progress import progress_scope, report_progress
from ..translation.analysis import (
    CodexAnalysisCancelled,
    CodexAnalysisReview,
    CodexAnalysisSession,
)


def translate_docx(**kwargs):
    from ..docx import translate_docx as run

    return run(**kwargs)


def translate_dxf(**kwargs):
    from ..dxf.pipeline import translate_dxf as run

    return run(**kwargs)


def translate_pdf(**kwargs):
    from ..pdf.pipeline import translate_pdf as run

    return run(**kwargs)


class TranslateWorker(QThread):
    log_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal(dict)
    analysis_ready_signal = Signal(dict)
    cancelled_signal = Signal(str)
    progress_signal = Signal(object)

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self._params = params
        self._cancellation = CancellationToken()
        self._analysis_condition = Condition()
        self._analysis_waiting = False
        self._analysis_response_ready = False
        self._analysis_response: str | None = None

    def cancel(self) -> None:
        self._cancellation.cancel()
        self.requestInterruption()
        with self._analysis_condition:
            self._analysis_condition.notify_all()

    def _log(self, message: str) -> None:
        self._cancellation.check()
        self.log_signal.emit(message)

    def submit_analysis(self, text: str | None) -> None:
        with self._analysis_condition:
            if not self._analysis_waiting:
                return
            self._analysis_response = text
            self._analysis_response_ready = True
            self._analysis_condition.notify_all()

    def _review_analysis(self, review: CodexAnalysisReview) -> str | None:
        payload = {
            "text": review.text,
            "model": review.model,
            "reasoning_effort": review.reasoning_effort,
            "context_label": review.context_label,
            "used_fallback": review.used_fallback,
            "warning": review.warning,
        }
        with self._analysis_condition:
            self._cancellation.check()
            self._analysis_waiting = True
            self._analysis_response_ready = False
            self._analysis_response = None
            self.analysis_ready_signal.emit(payload)
            try:
                while not self._analysis_response_ready:
                    self._cancellation.check()
                    self._analysis_condition.wait()
                self._cancellation.check()
                return self._analysis_response
            finally:
                self._analysis_waiting = False

    def run(self) -> None:
        params = dict(self._params)
        job_type = params.pop("job_type", "cad")
        if params.get("translator_name") == "codex":
            params.setdefault(
                "codex_analysis_session",
                CodexAnalysisSession(self._review_analysis),
            )
        try:
            with cancellation_scope(self._cancellation), progress_scope(self.progress_signal.emit):
                report_progress("read")
                if job_type == "pdf":
                    result = translate_pdf(log=self._log, **params)
                elif job_type == "docx":
                    result = translate_docx(log=self._log, **params)
                elif job_type == "cad":
                    result = translate_dxf(log=self._log, **params)
                else:
                    raise RuntimeError(f"Неизвестный тип задания: {job_type}")
                self._cancellation.check()
                result["job_type"] = job_type
                report_progress("complete")
                self.finished_signal.emit(result)
        except (CodexAnalysisCancelled, TranslationCancelled) as exc:
            self.cancelled_signal.emit(str(exc))
        except Exception as exc:
            if self._cancellation.cancelled:
                self.cancelled_signal.emit("Перевод отменён. Сохранённые пакеты доступны для продолжения.")
            else:
                self.error_signal.emit(str(exc))
