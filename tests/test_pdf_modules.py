"""PDF stage boundaries and the retained public facade."""

import os
import subprocess
import sys
from pathlib import Path


def test_pdf_stages_import_without_loading_the_orchestrator():
    source = Path(__file__).resolve().parents[1] / "src"
    code = """
import importlib, sys
from importlib.abc import MetaPathFinder

class NoOrchestrator(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'daru.pdf._core', 'daru.pdf.pipeline'}:
            raise AssertionError('PDF stage depends on the orchestrator: ' + fullname)

sys.meta_path.insert(0, NoOrchestrator())
for name in ('geometry', 'common', 'imaging', 'rendering', 'export', 'raster',
             'textract', 'ocr', 'regions', 'block_translation'):
    importlib.import_module('daru.pdf.' + name)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], env=dict(os.environ, PYTHONPATH=str(source)),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_geometry_import_does_not_load_imaging_or_network_dependencies():
    source = Path(__file__).resolve().parents[1] / "src"
    code = """
import sys
from daru.pdf.geometry import _union_bbox
assert _union_bbox([(1, 2, 3, 4), (2, 1, 6, 8)]) == (1, 1, 6, 8)
assert not ({'numpy', 'cv2', 'boto3', 'fitz', 'PIL', 'easyocr', 'pytesseract',
             'daru.pdf._core'} & sys.modules.keys())
"""
    result = subprocess.run(
        [sys.executable, "-c", code], env=dict(os.environ, PYTHONPATH=str(source)),
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_public_facade_exposes_original_stage_objects():
    from daru.pdf import (
        block_translation,
        export,
        ocr,
        pipeline,
        raster,
        rendering,
        textract,
    )

    for module, names in (
        (raster, ("PageLoader", "pdf_to_images")),
        (rendering, ("Renderer",)),
        (export, ("Exporter", "images_to_pdf")),
        (ocr, ("RegionDetector", "run_ocr", "LayoutAnalyzer")),
        (textract, ("TextractLayoutExtractor",)),
        (block_translation, ("RegionTranslator", "BlockRegionTranslator")),
    ):
        for name in names:
            assert getattr(pipeline, name) is getattr(module, name)
