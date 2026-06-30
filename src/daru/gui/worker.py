"""Background translation worker thread."""

from threading import Condition
from typing import Any, Dict

from PySide6.QtCore import QThread, Signal

from ..docx import translate_docx
from ..dxf.pipeline import translate_dxf
from ..pdf.pipeline import translate_pdf
from ..translation.analysis import (
    CodexAnalysisCancelled,
    CodexAnalysisReview,
    CodexAnalysisSession,
)


class TranslateWorker(QThread):
    log_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal(dict)
    analysis_ready_signal = Signal(dict)
    cancelled_signal = Signal(str)

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self._params = params
        self._analysis_condition = Condition()
        self._analysis_waiting = False
        self._analysis_response_ready = False
        self._analysis_response: str | None = None

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
            self._analysis_waiting = True
            self._analysis_response_ready = False
            self._analysis_response = None
            self.analysis_ready_signal.emit(payload)
            while not self._analysis_response_ready:
                self._analysis_condition.wait()
            response = self._analysis_response
            self._analysis_waiting = False
            return response

    def run(self) -> None:
        params = dict(self._params)
        job_type = params.pop("job_type", "cad")
        if params.get("translator_name") == "codex":
            params.setdefault(
                "codex_analysis_session",
                CodexAnalysisSession(self._review_analysis),
            )
        try:
            if job_type == "pdf":
                result = translate_pdf(log=self.log_signal.emit, **params)
            elif job_type == "docx":
                result = translate_docx(log=self.log_signal.emit, **params)
            elif job_type == "cad":
                result = translate_dxf(log=self.log_signal.emit, **params)
            else:
                raise RuntimeError(f"Неизвестный тип задания: {job_type}")
            result["job_type"] = job_type
            self.finished_signal.emit(result)
        except CodexAnalysisCancelled as exc:
            self.cancelled_signal.emit(str(exc))
        except Exception as exc:
            self.error_signal.emit(str(exc))
