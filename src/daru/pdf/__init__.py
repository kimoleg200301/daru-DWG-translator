from .config import PdfProcessingConfig
from .config import (
    PDF_PROCESSING_TEXTRACT,
    PDF_PROCESSING_TEXTRACT_VISION,
    PDF_TYPE_SCANNED,
    PDF_TYPE_NATIVE,
)

__all__ = [
    "translate_pdf",
    "process_pdf",
    "PdfProcessingConfig",
    "PDF_TYPE_SCANNED",
    "PDF_TYPE_NATIVE",
    "PDF_PROCESSING_TEXTRACT",
    "PDF_PROCESSING_TEXTRACT_VISION",
]


def __getattr__(name: str):
    if name in {"translate_pdf", "process_pdf"}:
        from . import pipeline

        value = getattr(pipeline, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
