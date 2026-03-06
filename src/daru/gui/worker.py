"""Background translation worker thread."""

from typing import Any, Dict

from PySide6.QtCore import QThread, Signal

from ..dxf.pipeline import translate_dxf
from ..pdf.pipeline import translate_pdf


class TranslateWorker(QThread):
    log_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal(dict)

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self._params = params

    def run(self) -> None:
        params = dict(self._params)
        job_type = params.pop("job_type", "cad")
        try:
            if job_type == "pdf":
                result = translate_pdf(log=self.log_signal.emit, **params)
            else:
                result = translate_dxf(log=self.log_signal.emit, **params)
            result["job_type"] = job_type
            self.finished_signal.emit(result)
        except Exception as exc:
            self.error_signal.emit(str(exc))
