import subprocess
import sys
from pathlib import Path

import pytest

from daru.pdf.config import PdfProcessingConfig


@pytest.mark.parametrize("languages", ["en ru es", "en\tru\nes", "en+ru;es", "en/ru,es"])
def test_ocr_languages_preserve_codes_and_split_separators(languages):
    assert PdfProcessingConfig(easyocr_languages=languages).normalized().easyocr_languages == [
        "en", "ru", "es",
    ]


def test_data_modules_import_without_ocr_backends_or_filesystem_writes():
    src = str(Path(__file__).resolve().parents[1] / "src")
    script = f"""
import importlib.abc
import sys
from unittest.mock import patch
sys.path.insert(0, {src!r})
blocked = {{'numpy', 'PIL', 'cv2', 'boto3', 'fitz', 'easyocr', 'pytesseract', 'pdf2image'}}
class NoBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Unexpected backend import: ' + fullname)
sys.meta_path.insert(0, NoBackends())
with patch('pathlib.Path.mkdir', side_effect=AssertionError('Import must not write')):
    from daru.pdf.config import PdfProcessingConfig
    from daru.pdf.models import BlockRegion
    assert PdfProcessingConfig().normalized().dpi == 400
    assert BlockRegion(0, 'id', 'TEXT', (0, 0, 1, 1)).source_text == ''
assert 'daru.pdf._core' not in sys.modules
assert 'daru.pdf.pipeline' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_legacy_pdf_imports_expose_the_same_objects():
    from daru import pdf
    from daru.pdf import _core, models, pipeline

    assert _core.PdfProcessingConfig is pipeline.PdfProcessingConfig is PdfProcessingConfig
    assert _core.BlockRegion is pipeline.BlockRegion is models.BlockRegion
    assert pdf.translate_pdf is pipeline.translate_pdf
    assert pdf.process_pdf is pipeline.process_pdf
