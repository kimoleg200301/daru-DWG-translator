"""PDF processing configuration — re-exports from the monolithic module for gradual splitting."""

# The full PdfProcessingConfig lives in the pipeline module for now.
# This file provides a clean import path: from daru.pdf.config import PdfProcessingConfig
from .pipeline import PdfProcessingConfig, TEXTRACT_LOG_PATH, _CACHE_DIR

__all__ = ["PdfProcessingConfig", "TEXTRACT_LOG_PATH", "_CACHE_DIR"]
