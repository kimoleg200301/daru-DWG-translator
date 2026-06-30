"""GUI state tests for DOCX mode."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from daru.config import ORIGINAL_FONT_VALUE, SettingsManager
import daru.gui.main_window as main_window_module
from daru.gui.main_window import MainWindow


def test_docx_mode_configures_output_and_controls(tmp_path):
    app = QApplication.instance() or QApplication([])
    settings = SettingsManager(path=tmp_path / "settings.json")
    window = MainWindow(settings)
    source = tmp_path / "manual.docx"

    window.set_input_file(source)

    assert window.is_docx_input()
    assert not window.is_pdf_input()
    assert Path(window.output_edit.text()) == tmp_path / "manual_ru.docx"
    assert window.output_format_combo.isHidden()
    assert not window.style_font_combo.isEnabled()
    assert not window.map_checkbox.isEnabled()
    assert not window.txt_checkbox.isEnabled()

    window.close()
    app.processEvents()


def test_codex_mode_hides_api_translator_and_removes_api_keys(tmp_path):
    app = QApplication.instance() or QApplication([])
    settings = SettingsManager(path=tmp_path / "settings.json")
    settings.update(
        translator_name="deepl",
        deepl_key="deepl-secret",
        openai_key="openai-secret",
        codex_enabled=True,
        codex_cli_path="C:/tools/codex.cmd",
        codex_model="gpt-5.4-mini",
        codex_reasoning_effort="low",
        codex_analysis_model="gpt-5.5",
        codex_analysis_reasoning_effort="high",
        codex_timeout_seconds=300,
    )
    window = MainWindow(settings)

    assert window.translator_combo.isHidden()
    assert not window.codex_mode_value.isHidden()
    assert "gpt-5.4-mini" in window.codex_mode_value.text()

    params = window._base_translation_params(
        tmp_path / "source.docx",
        tmp_path / "translated.docx",
    )
    assert params["translator_name"] == "codex"
    assert params["deepl_key"] is None
    assert params["openai_key"] is None
    assert params["openai_base_url"] is None
    assert params["codex_cli_path"] == "C:/tools/codex.cmd"
    assert params["codex_analysis_model"] == "gpt-5.5"
    assert params["codex_analysis_reasoning_effort"] == "high"
    assert settings.data.translator_name == "deepl"

    window.close()
    app.processEvents()


def test_original_font_option_uses_internal_value_without_changing_default(tmp_path):
    app = QApplication.instance() or QApplication([])
    settings = SettingsManager(path=tmp_path / "settings.json")
    window = MainWindow(settings)

    original_index = window.style_font_combo.findData(ORIGINAL_FONT_VALUE)
    assert original_index >= 0
    assert window.style_font_combo.itemText(original_index) == "Оригинал"
    assert window.style_font_combo.currentData() == "DejaVuSans.ttf"

    window.style_font_combo.setCurrentIndex(original_index)
    params = window._base_translation_params(
        tmp_path / "source.dxf",
        tmp_path / "translated.dxf",
    )

    assert params["style_font"] == ORIGINAL_FONT_VALUE

    window.close()
    app.processEvents()


def test_resume_policy_dialog_can_restart_checkpoint(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    settings = SettingsManager(path=tmp_path / "settings.json")
    window = MainWindow(settings)
    source = tmp_path / "manual.docx"
    checkpoint = tmp_path / "checkpoint.json"

    monkeypatch.setattr(main_window_module, "has_valid_checkpoint", lambda *_args, **_kwargs: False)
    assert (
        window._resolve_resume_policy(
            checkpoint_path=checkpoint,
            input_path=source,
            job_type="docx",
            source_lang="en",
            target_lang="ru",
        )
        == "auto"
    )

    monkeypatch.setattr(main_window_module, "has_valid_checkpoint", lambda *_args, **_kwargs: True)

    class FakeMessageBox:
        AcceptRole = 0
        DestructiveRole = 1
        RejectRole = 2

        def __init__(self, _parent=None):
            self.buttons = {}

        def setWindowTitle(self, _title):
            pass

        def setText(self, _text):
            pass

        def addButton(self, label, _role):
            self.buttons[label] = object()
            return self.buttons[label]

        def setDefaultButton(self, _button):
            pass

        def exec(self):
            return 0

        def clickedButton(self):
            return self.buttons["Начать заново"]

    monkeypatch.setattr(main_window_module, "QMessageBox", FakeMessageBox)

    assert (
        window._resolve_resume_policy(
            checkpoint_path=checkpoint,
            input_path=source,
            job_type="docx",
            source_lang="en",
            target_lang="ru",
        )
        == "reset"
    )

    window.close()
    app.processEvents()
