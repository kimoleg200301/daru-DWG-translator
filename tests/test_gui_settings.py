"""GUI tests for model-dependent OpenAI API settings."""

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from daru.config import OPENAI_DEFAULT_BASE_URL, AppSettings
from daru.gui import settings_dialog
from daru.gui.settings_dialog import SettingsDialog


def test_openai_fields_follow_selected_model():
    app = QApplication.instance() or QApplication([])
    dialog = SettingsDialog(AppSettings())

    assert dialog.openai_model_combo.currentText() == "gpt-5.4-mini"
    assert dialog.openai_url_combo.currentText() == OPENAI_DEFAULT_BASE_URL
    assert not dialog.openai_temp_spin.isEnabled()
    assert dialog.openai_reasoning_combo.isEnabled()
    assert dialog.openai_verbosity_combo.isEnabled()
    assert dialog.openai_reasoning_combo.currentText() == "low"
    assert dialog._pdf_mode_values["pdf_vision_model"] == "gpt-5.5"
    assert dialog._pdf_mode_values["pdf_vision_reasoning_effort"] == "medium"
    assert dialog._pdf_mode_values["pdf_vision_request_mode"] == "batched"
    assert dialog._pdf_mode_values["pdf_vision_image_quality"] == "stable"

    dialog.openai_model_combo.setCurrentText("gpt-4.1")
    app.processEvents()

    assert dialog.openai_temp_spin.isEnabled()
    assert not dialog.openai_reasoning_combo.isEnabled()
    assert not dialog.openai_verbosity_combo.isEnabled()

    dialog.openai_model_combo.setCurrentText("gpt-5.5")
    app.processEvents()

    assert not dialog.openai_temp_spin.isEnabled()
    assert dialog.openai_reasoning_combo.isEnabled()
    assert dialog.openai_verbosity_combo.isEnabled()
    assert dialog.openai_reasoning_combo.currentText() == "medium"
    assert dialog.openai_reasoning_combo.findText("xhigh") >= 0

    dialog.openai_url_combo.setCurrentText("https://api.openai.com/v1/responses")
    assert dialog.get_values()["openai_base_url"] == OPENAI_DEFAULT_BASE_URL
    assert dialog.get_values()["pdf_vision_model"] == "gpt-5.5"
    assert dialog.get_values()["pdf_vision_request_mode"] == "batched"
    assert dialog.get_values()["pdf_vision_image_quality"] == "stable"

    dialog.close()
    app.processEvents()


def test_codex_mode_switches_setting_groups():
    app = QApplication.instance() or QApplication([])
    dialog = SettingsDialog(
        AppSettings(
            codex_enabled=True,
            codex_cli_path="C:/tools/codex.cmd",
            codex_model="gpt-5.4-mini",
            codex_reasoning_effort="medium",
            codex_analysis_model="gpt-5.5",
            codex_analysis_reasoning_effort="high",
            codex_timeout_seconds=420,
        )
    )

    assert dialog.api_widget.isHidden()
    assert not dialog.codex_widget.isHidden()
    assert dialog.codex_reasoning_combo.currentText() == "medium"
    assert dialog.codex_analysis_model_combo.currentText() == "gpt-5.5"
    assert dialog.codex_analysis_reasoning_combo.currentText() == "high"
    assert dialog.codex_timeout_spin.value() == 420

    values = dialog.get_values()
    assert values["codex_enabled"] is True
    assert values["codex_cli_path"] == "C:/tools/codex.cmd"
    assert values["codex_model"] == "gpt-5.4-mini"
    assert values["codex_analysis_model"] == "gpt-5.5"
    assert values["codex_analysis_reasoning_effort"] == "high"

    dialog.codex_enabled_checkbox.setChecked(False)
    app.processEvents()
    assert not dialog.api_widget.isHidden()
    assert dialog.codex_widget.isHidden()

    dialog.close()
    app.processEvents()


def test_codex_check_probes_unique_translation_and_analysis_profiles(monkeypatch):
    app = QApplication.instance() or QApplication([])
    calls = []
    messages = []

    def fake_check(path, *, timeout, model, reasoning_effort):
        calls.append((path, timeout, model, reasoning_effort))
        return f"{model}/{reasoning_effort}: доступна"

    monkeypatch.setattr(settings_dialog, "check_codex_cli", fake_check)
    monkeypatch.setattr(
        settings_dialog.QMessageBox,
        "information",
        lambda _parent, _title, message: messages.append(message),
    )
    dialog = SettingsDialog(
        AppSettings(
            codex_enabled=True,
            codex_cli_path="C:/tools/codex.cmd",
            codex_model="gpt-5.4-mini",
            codex_reasoning_effort="low",
            codex_analysis_model="gpt-5.5",
            codex_analysis_reasoning_effort="high",
        )
    )

    dialog._test_codex_cli()

    assert [(call[2], call[3]) for call in calls] == [
        ("gpt-5.4-mini", "low"),
        ("gpt-5.5", "high"),
    ]
    assert "Перевод" in messages[0]
    assert "Преданализ" in messages[0]

    dialog.codex_analysis_model_combo.setCurrentText("gpt-5.4-mini")
    dialog.codex_analysis_reasoning_combo.setCurrentText("low")
    calls.clear()
    messages.clear()
    dialog._test_codex_cli()
    assert len(calls) == 1
    assert "Перевод / Преданализ" in messages[0]

    dialog.close()
    app.processEvents()


def test_codex_check_reauthenticates_once_after_expired_session(monkeypatch):
    app = QApplication.instance() or QApplication([])
    checks = []
    logins = []
    messages = []

    def fake_check(path, *, timeout, model, reasoning_effort):
        checks.append((path, timeout, model, reasoning_effort))
        if len(checks) == 1:
            raise settings_dialog.CodexCliAuthenticationError("Сессия истекла.")
        return f"{model}/{reasoning_effort}: доступна"

    def fake_login(path, *, timeout):
        logins.append((path, timeout))
        return "logged in"

    monkeypatch.setattr(settings_dialog, "check_codex_cli", fake_check)
    monkeypatch.setattr(settings_dialog, "login_codex_cli", fake_login)
    monkeypatch.setattr(
        settings_dialog.QMessageBox,
        "question",
        lambda *_args, **_kwargs: settings_dialog.QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        settings_dialog.QMessageBox,
        "information",
        lambda _parent, _title, message: messages.append(message),
    )

    dialog = SettingsDialog(
        AppSettings(
            codex_enabled=True,
            codex_cli_path="C:/tools/codex.exe",
            codex_model="gpt-5.4-mini",
            codex_reasoning_effort="low",
            codex_analysis_model="gpt-5.4-mini",
            codex_analysis_reasoning_effort="low",
            codex_timeout_seconds=30,
        )
    )

    dialog._test_codex_cli()

    assert logins == [("C:/tools/codex.exe", 120)]
    assert len(checks) == 2
    assert messages == ["Перевод / Преданализ:\ngpt-5.4-mini/low: доступна"]

    dialog.close()
    app.processEvents()


def test_codex_install_button_runs_installer_after_confirmation(monkeypatch):
    app = QApplication.instance() or QApplication([])
    calls = []
    messages = []
    questions = []

    def fake_install(*, timeout):
        calls.append(("install", timeout))
        return SimpleNamespace(
            executable="C:/Users/admin/AppData/Local/Programs/OpenAI/Codex/bin/codex.exe",
            version="codex-cli 1.0",
            message="installed",
        )

    def fake_login(executable, *, timeout):
        calls.append(("login", executable, timeout))
        return "logged in"

    monkeypatch.setattr(settings_dialog, "install_or_update_codex_cli", fake_install)
    monkeypatch.setattr(settings_dialog, "login_codex_cli", fake_login)
    monkeypatch.setattr(
        settings_dialog.QMessageBox,
        "question",
        lambda _parent, _title, message, *_args, **_kwargs: (
            questions.append(message)
            or settings_dialog.QMessageBox.StandardButton.Yes
        ),
    )
    monkeypatch.setattr(
        settings_dialog.QMessageBox,
        "information",
        lambda _parent, _title, message: messages.append(message),
    )

    dialog = SettingsDialog(AppSettings(codex_enabled=True, codex_timeout_seconds=30))
    dialog._install_codex_cli()

    executable = "C:/Users/admin/AppData/Local/Programs/OpenAI/Codex/bin/codex.exe"
    assert calls == [
        ("install", 120),
        ("login", executable, 120),
    ]
    assert dialog.codex_path_edit.text() == executable
    assert "https://chatgpt.com/codex/install.ps1" in questions[0]
    assert "codex login" in questions[0]
    assert messages == ["installed\n\nlogged in"]

    dialog.close()
    app.processEvents()
