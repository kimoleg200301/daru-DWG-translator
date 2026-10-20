"""GUI state tests for native PDF mode."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from daru.config import AppSettings, SettingsManager
from daru.gui import pdf_dialogs
from daru.gui.main_window import MainWindow
from daru.gui.pdf_dialogs import PdfProcessingDialog, PdfTextractSettingsDialog


def _combo_texts(combo):
    return [combo.itemText(index) for index in range(combo.count())]


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


def test_pdf_processing_mode_labels_describe_actual_routing(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow(SettingsManager(path=tmp_path / "settings.json"))

    window.set_input_file(tmp_path / "drawing.pdf")

    assert window.pdf_type_label.text() == "Режим обработки PDF"
    assert _combo_texts(window.pdf_type_combo) == [
        "OCR всех страниц",
        "Текстовый слой + OCR при необходимости",
    ]
    assert [
        window.pdf_type_combo.itemData(index)
        for index in range(window.pdf_type_combo.count())
    ] == ["scanned", "native"]
    assert "каждую страницу как изображение" in window.pdf_type_combo.toolTip()
    assert "только для страниц без пригодного текстового слоя" in (
        window.pdf_type_combo.toolTip()
    )

    window.close()
    app.processEvents()


def test_pdf_codex_vision_uses_image_models_from_cli_catalog():
    app = QApplication.instance() or QApplication([])
    dialog = PdfTextractSettingsDialog(
        {
            "pdf_processing_mode": "textract_vision",
            "pdf_vision_backend": "codex_cli",
            "pdf_vision_model": "saved-vision-model",
            "pdf_vision_reasoning_effort": "high",
        }
    )
    models = [
        pdf_dialogs.CodexModelInfo(
            model="vision-model",
            display_name="Vision Model",
            supported_reasoning_efforts=("low", "max"),
            default_reasoning_effort="max",
            input_modalities=("text", "image"),
            is_default=True,
        ),
        pdf_dialogs.CodexModelInfo(
            model="text-only-model",
            display_name="Text Only",
            supported_reasoning_efforts=("low",),
            default_reasoning_effort="low",
            input_modalities=("text",),
        ),
    ]
    dialog._codex_catalog_generation = 1
    dialog._codex_catalog_loading = True

    dialog._codex_models_loaded(1, models)

    assert _combo_texts(dialog.vision_model_combo) == [
        "vision-model",
        "saved-vision-model",
    ]
    assert "text-only-model" not in _combo_texts(dialog.vision_model_combo)
    dialog.vision_model_combo.setCurrentText("vision-model")
    app.processEvents()
    assert _combo_texts(dialog.vision_reasoning_combo) == ["low", "max"]
    assert dialog.vision_reasoning_combo.currentText() == "max"

    dialog.vision_model_combo.setCurrentText("saved-vision-model")
    dialog.vision_backend_combo.setCurrentIndex(
        dialog.vision_backend_combo.findData("openai_api")
    )
    dialog.vision_backend_combo.setCurrentIndex(
        dialog.vision_backend_combo.findData("codex_cli")
    )
    assert dialog.vision_model_combo.currentText() == "saved-vision-model"

    dialog.close()
    app.processEvents()


def test_pdf_backend_switch_keeps_provider_specific_model_selection():
    app = QApplication.instance() or QApplication([])
    dialog = PdfTextractSettingsDialog(
        {
            "pdf_processing_mode": "textract_vision",
            "pdf_vision_backend": "openai_api",
            "pdf_vision_model": "gpt-4.1",
        }
    )

    dialog.vision_backend_combo.setCurrentIndex(
        dialog.vision_backend_combo.findData("codex_cli")
    )
    models = [
        pdf_dialogs.CodexModelInfo(
            model="vision-only",
            display_name="Vision Only",
            supported_reasoning_efforts=("low", "medium"),
            default_reasoning_effort="low",
            input_modalities=("text", "image"),
            is_default=True,
        )
    ]
    dialog._codex_catalog_generation = 1
    dialog._codex_catalog_loading = True
    dialog._codex_models_loaded(1, models)

    assert _combo_texts(dialog.vision_model_combo) == ["vision-only"]
    assert dialog.vision_model_combo.currentText() == "vision-only"

    dialog.vision_backend_combo.setCurrentIndex(
        dialog.vision_backend_combo.findData("openai_api")
    )
    assert dialog.vision_model_combo.currentText() == "gpt-4.1"
    dialog.vision_backend_combo.setCurrentIndex(
        dialog.vision_backend_combo.findData("codex_cli")
    )
    assert dialog.vision_model_combo.currentText() == "vision-only"

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
