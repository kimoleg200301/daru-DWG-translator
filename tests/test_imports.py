"""Smoke tests — verify all package modules are importable."""

import importlib

import pytest

MODULES = [
    "daru",
    "daru.config",
    "daru.utils.io",
    "daru.utils.odafc",
    "daru.utils.spinner",
    "daru.translation",
    "daru.translation.analysis",
    "daru.translation.checkpoint",
    "daru.translation.engine",
    "daru.translation.legacy",
    "daru.dxf",
    "daru.dxf.entities",
    "daru.dxf.extractor",
    "daru.dxf.applier",
    "daru.dxf.pipeline",
    "daru.pdf",
    "daru.pdf.pipeline",
    "daru.pdf.native",
    "daru.pdf.vision",
    "daru.pdf.config",
    "daru.pdf.models",
    "daru.docx",
    "daru.docx.pipeline",
    "daru.docx.segments",
    "daru.gui.worker",
    "daru.gui.analysis_dialog",
    "daru.gui.app",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_import(module_name):
    mod = importlib.import_module(module_name)
    assert mod is not None
