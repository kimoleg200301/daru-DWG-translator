"""GUI state tests for native PDF mode."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from daru.config import AppSettings
from daru.gui.pdf_dialogs import PdfProcessingDialog, PdfTextractSettingsDialog


def test_native_pdf_dialog_hides_ocr_controls():
    app = QApplication.instance() or QApplication([])
    dialog = PdfProcessingDialog(AppSettings(), native_mode=True)

    assert "текстовый слой" in dialog.windowTitle().lower()
    for widget in (
        dialog.dpi_spin,
        dialog.confidence_spin,
        dialog.blur_spin,
        dialog.dilation_spin,
        dialog.lang_edit,
    ):
        assert widget.isHidden()

    dialog.close()
    app.processEvents()


def test_pdf_textract_dialog_preserves_vision_mode():
    app = QApplication.instance() or QApplication([])
    dialog = PdfTextractSettingsDialog(
        {
            "pdf_processing_mode": "textract_vision",
            "textract_region": "us-east-1",
            "textract_access_key": "",
            "textract_secret_key": "",
            "textract_session_token": "",
            "pdf_vision_backend": "codex_cli",
            "pdf_vision_model": "gpt-5.5",
            "pdf_vision_reasoning_effort": "high",
            "pdf_vision_api_key": "api-key",
            "pdf_vision_base_url": "https://api.openai.com/v1",
            "pdf_vision_project": "project",
            "pdf_vision_codex_cli_path": "C:/tools/codex.exe",
            "pdf_vision_codex_timeout_seconds": 420,
            "pdf_vision_request_mode": "single_page",
            "pdf_vision_image_quality": "original",
        }
    )

    assert dialog.mode_combo.currentData() == "textract_vision"
    assert dialog.vision_backend_combo.currentData() == "codex_cli"
    assert dialog.vision_codex_path_edit.isEnabled()
    assert dialog.vision_request_mode_combo.isEnabled()
    assert dialog.vision_image_quality_combo.isEnabled()
    assert not dialog.vision_api_key_edit.isEnabled()
    assert dialog.region_edit.isEnabled()
    values = dialog.get_values()
    assert values["pdf_processing_mode"] == "textract_vision"
    assert values["pdf_vision_backend"] == "codex_cli"
    assert values["pdf_vision_model"] == "gpt-5.5"
    assert values["pdf_vision_reasoning_effort"] == "high"
    assert values["pdf_vision_codex_cli_path"] == "C:/tools/codex.exe"
    assert values["pdf_vision_codex_timeout_seconds"] == 420
    assert values["pdf_vision_request_mode"] == "single_page"
    assert values["pdf_vision_image_quality"] == "original"

    dialog.vision_backend_combo.setCurrentIndex(
        dialog.vision_backend_combo.findData("openai_api")
    )
    app.processEvents()
    assert dialog.vision_api_key_edit.isEnabled()
    assert not dialog.vision_codex_path_edit.isEnabled()
    assert not dialog.vision_request_mode_combo.isEnabled()
    assert dialog.vision_image_quality_combo.isEnabled()

    dialog.close()
    app.processEvents()
