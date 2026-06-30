"""Translation backend settings dialog."""

from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    OPENAI_BASE_URL_CHOICES,
    OPENAI_DEFAULT_BASE_URL,
    OPENAI_DEFAULT_MODEL,
    OPENAI_MODEL_CHOICES,
    AppSettings,
    get_openai_model_profile,
    normalize_openai_base_url,
    populate_combo,
)
from ..translation.codex_cli import (
    CODEX_DEFAULT_ANALYSIS_MODEL,
    CODEX_DEFAULT_ANALYSIS_REASONING_EFFORT,
    CODEX_DEFAULT_MODEL,
    CODEX_DEFAULT_REASONING_EFFORT,
    CODEX_DEFAULT_TIMEOUT_SECONDS,
    CODEX_INSTALLER_COMMAND,
    CODEX_REASONING_EFFORTS,
    CodexCliAuthenticationError,
    check_codex_cli,
    install_or_update_codex_cli,
    login_codex_cli,
)
from .pdf_dialogs import PdfTextractSettingsDialog


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Настройки перевода")
        layout = QVBoxLayout(self)
        self._pdf_mode_values: Dict[str, Any] = {
            "pdf_processing_mode": settings.pdf_processing_mode or "textract",
            "textract_region": settings.textract_region or "",
            "textract_access_key": settings.textract_access_key or "",
            "textract_secret_key": settings.textract_secret_key or "",
            "textract_session_token": settings.textract_session_token or "",
            "pdf_vision_backend": settings.pdf_vision_backend or "openai_api",
            "pdf_vision_model": settings.pdf_vision_model or "gpt-5.5",
            "pdf_vision_reasoning_effort": (
                settings.pdf_vision_reasoning_effort or "medium"
            ),
            "pdf_vision_api_key": settings.pdf_vision_api_key or settings.openai_key or "",
            "pdf_vision_base_url": (
                settings.pdf_vision_base_url
                or settings.openai_base_url
                or OPENAI_DEFAULT_BASE_URL
            ),
            "pdf_vision_project": (
                settings.pdf_vision_project or settings.openai_project or ""
            ),
            "pdf_vision_codex_cli_path": (
                settings.pdf_vision_codex_cli_path or settings.codex_cli_path or ""
            ),
            "pdf_vision_codex_timeout_seconds": (
                settings.pdf_vision_codex_timeout_seconds
                or settings.codex_timeout_seconds
                or CODEX_DEFAULT_TIMEOUT_SECONDS
            ),
            "pdf_vision_request_mode": settings.pdf_vision_request_mode or "batched",
            "pdf_vision_image_quality": (
                settings.pdf_vision_image_quality or "stable"
            ),
        }

        self.codex_enabled_checkbox = QCheckBox(
            "Использовать Codex CLI (без API-ключа Daru)"
        )
        self.codex_enabled_checkbox.setChecked(bool(settings.codex_enabled))
        layout.addWidget(self.codex_enabled_checkbox)

        self.api_widget = QWidget()
        form = QFormLayout(self.api_widget)
        self.deepl_edit = QLineEdit(settings.deepl_key)
        self.openai_key_edit = QLineEdit(settings.openai_key)
        self.openai_model_combo = QComboBox()
        populate_combo(self.openai_model_combo, OPENAI_MODEL_CHOICES, settings.openai_model)
        self.openai_url_combo = QComboBox()
        populate_combo(
            self.openai_url_combo,
            OPENAI_BASE_URL_CHOICES,
            normalize_openai_base_url(settings.openai_base_url),
        )
        self.openai_url_combo.setToolTip(
            "Базовый URL OpenAI SDK. Указывайте адрес до /v1, без /responses."
        )
        self.openai_project_edit = QLineEdit(settings.openai_project)
        self.openai_temp_spin = QDoubleSpinBox()
        self.openai_temp_spin.setRange(0.0, 2.0)
        self.openai_temp_spin.setDecimals(2)
        self.openai_temp_spin.setSingleStep(0.1)
        self.openai_temp_spin.setValue(settings.openai_temperature)
        self.openai_reasoning_combo = QComboBox()
        populate_combo(
            self.openai_reasoning_combo,
            ("none", "minimal", "low", "medium", "high", "xhigh"),
            settings.openai_reasoning_effort,
            editable=False,
        )
        self.openai_verbosity_combo = QComboBox()
        populate_combo(
            self.openai_verbosity_combo,
            ("low", "medium", "high"),
            settings.openai_verbosity,
            editable=False,
        )
        self.openai_key_edit.setEchoMode(QLineEdit.Password)
        self.deepl_edit.setEchoMode(QLineEdit.Password)

        form.addRow("DeepL API Key", self.deepl_edit)
        form.addRow("OpenAI API Key", self.openai_key_edit)
        form.addRow("OpenAI Model", self.openai_model_combo)
        form.addRow("OpenAI Base URL", self.openai_url_combo)
        form.addRow("OpenAI Project ID", self.openai_project_edit)
        form.addRow("OpenAI Temperature", self.openai_temp_spin)
        form.addRow("OpenAI Reasoning Effort", self.openai_reasoning_combo)
        form.addRow("OpenAI Verbosity", self.openai_verbosity_combo)
        layout.addWidget(self.api_widget)

        self.codex_widget = QWidget()
        codex_form = QFormLayout(self.codex_widget)
        self.codex_path_edit = QLineEdit(settings.codex_cli_path)
        self.codex_path_edit.setPlaceholderText("Автопоиск команды codex в PATH")
        self.codex_path_browse = QPushButton("...")
        self.codex_path_browse.clicked.connect(self._browse_codex_cli)
        codex_path_widget = QWidget()
        codex_path_layout = QHBoxLayout(codex_path_widget)
        codex_path_layout.setContentsMargins(0, 0, 0, 0)
        codex_path_layout.addWidget(self.codex_path_edit)
        codex_path_layout.addWidget(self.codex_path_browse)

        self.codex_model_combo = QComboBox()
        populate_combo(
            self.codex_model_combo,
            ("gpt-5.4-mini", "gpt-5.5", "gpt-5.4"),
            settings.codex_model or CODEX_DEFAULT_MODEL,
        )
        self.codex_reasoning_combo = QComboBox()
        populate_combo(
            self.codex_reasoning_combo,
            CODEX_REASONING_EFFORTS,
            settings.codex_reasoning_effort or CODEX_DEFAULT_REASONING_EFFORT,
            editable=False,
        )
        self.codex_analysis_model_combo = QComboBox()
        populate_combo(
            self.codex_analysis_model_combo,
            ("gpt-5.5", "gpt-5.4", "gpt-5.4-mini"),
            settings.codex_analysis_model or CODEX_DEFAULT_ANALYSIS_MODEL,
        )
        self.codex_analysis_reasoning_combo = QComboBox()
        populate_combo(
            self.codex_analysis_reasoning_combo,
            CODEX_REASONING_EFFORTS,
            (
                settings.codex_analysis_reasoning_effort
                or CODEX_DEFAULT_ANALYSIS_REASONING_EFFORT
            ),
            editable=False,
        )
        self.codex_timeout_spin = QSpinBox()
        self.codex_timeout_spin.setRange(10, 3600)
        self.codex_timeout_spin.setSuffix(" с")
        self.codex_timeout_spin.setValue(
            max(10, int(settings.codex_timeout_seconds or CODEX_DEFAULT_TIMEOUT_SECONDS))
        )
        self.codex_test_button = QPushButton("Проверить Codex CLI")
        self.codex_test_button.clicked.connect(self._test_codex_cli)
        self.codex_install_button = QPushButton("Установить/обновить Codex CLI")
        self.codex_install_button.clicked.connect(self._install_codex_cli)

        codex_form.addRow("Путь к Codex CLI", codex_path_widget)
        codex_form.addRow("Модель перевода", self.codex_model_combo)
        codex_form.addRow("Reasoning перевода", self.codex_reasoning_combo)
        codex_form.addRow("Модель преданализа", self.codex_analysis_model_combo)
        codex_form.addRow(
            "Reasoning преданализа",
            self.codex_analysis_reasoning_combo,
        )
        codex_form.addRow("Таймаут", self.codex_timeout_spin)
        codex_form.addRow(self.codex_install_button)
        codex_form.addRow(self.codex_test_button)
        layout.addWidget(self.codex_widget)

        self.pdf_settings_btn = QPushButton("Настройки обработки PDF")
        self.pdf_settings_btn.clicked.connect(self._open_pdf_settings)
        layout.addWidget(self.pdf_settings_btn)

        self.openai_model_combo.currentTextChanged.connect(self._update_openai_param_visibility)
        self.codex_enabled_checkbox.toggled.connect(self._update_backend_visibility)
        self._update_openai_param_visibility(self.openai_model_combo.currentText())
        self._update_backend_visibility(self.codex_enabled_checkbox.isChecked())

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self) -> Dict[str, Any]:
        return {
            "codex_enabled": self.codex_enabled_checkbox.isChecked(),
            "codex_cli_path": self.codex_path_edit.text().strip(),
            "codex_model": self.codex_model_combo.currentText().strip()
            or CODEX_DEFAULT_MODEL,
            "codex_reasoning_effort": self.codex_reasoning_combo.currentText().strip()
            or CODEX_DEFAULT_REASONING_EFFORT,
            "codex_analysis_model": self.codex_analysis_model_combo.currentText().strip()
            or CODEX_DEFAULT_ANALYSIS_MODEL,
            "codex_analysis_reasoning_effort": (
                self.codex_analysis_reasoning_combo.currentText().strip()
                or CODEX_DEFAULT_ANALYSIS_REASONING_EFFORT
            ),
            "codex_timeout_seconds": self.codex_timeout_spin.value(),
            "deepl_key": self.deepl_edit.text().strip(),
            "openai_key": self.openai_key_edit.text().strip(),
            "openai_model": self.openai_model_combo.currentText().strip() or OPENAI_DEFAULT_MODEL,
            "openai_base_url": normalize_openai_base_url(
                self.openai_url_combo.currentText() or OPENAI_DEFAULT_BASE_URL
            ),
            "openai_project": self.openai_project_edit.text().strip(),
            "openai_temperature": self.openai_temp_spin.value(),
            "openai_reasoning_effort": self.openai_reasoning_combo.currentText().strip() or "low",
            "openai_verbosity": self.openai_verbosity_combo.currentText().strip() or "low",
            **self._pdf_mode_values,
        }

    def _update_openai_param_visibility(self, model: str) -> None:
        profile = get_openai_model_profile(model)
        efforts = tuple(profile["reasoning_efforts"])
        current_effort = self.openai_reasoning_combo.currentText().strip()

        self.openai_reasoning_combo.blockSignals(True)
        self.openai_reasoning_combo.clear()
        self.openai_reasoning_combo.addItems(list(efforts))
        if efforts:
            selected = (
                current_effort
                if current_effort in efforts
                else str(profile["default_reasoning_effort"])
            )
            self.openai_reasoning_combo.setCurrentText(selected)
        self.openai_reasoning_combo.blockSignals(False)

        self.openai_temp_spin.setEnabled(bool(profile["supports_temperature"]))
        self.openai_reasoning_combo.setEnabled(bool(efforts))
        self.openai_verbosity_combo.setEnabled(bool(profile["supports_verbosity"]))

    def _update_backend_visibility(self, codex_enabled: bool) -> None:
        self.api_widget.setVisible(not codex_enabled)
        self.codex_widget.setVisible(codex_enabled)

    def _browse_codex_cli(self) -> None:
        current = self.codex_path_edit.text().strip()
        start_path = str(Path(current).parent) if current else ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите Codex CLI",
            start_path,
            "Codex CLI (codex.exe codex.cmd codex.bat codex.ps1);;Все файлы (*)",
        )
        if path:
            self.codex_path_edit.setText(path)

    def _install_codex_cli(self) -> None:
        answer = QMessageBox.question(
            self,
            "Codex CLI",
            "Будет выполнена команда:\n\n"
            f"{CODEX_INSTALLER_COMMAND}\n\n"
            "После установки будет запущен `codex login` для авторизации "
            "через браузер.\n\n"
            "Для этого нужны PowerShell и доступ в интернет. "
            "Продолжить?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self.codex_install_button.setEnabled(False)
        self.codex_test_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            timeout = max(120, self.codex_timeout_spin.value())
            result = install_or_update_codex_cli(timeout=timeout)
            self.codex_path_edit.setText(result.executable)
            login_message = login_codex_cli(
                result.executable,
                timeout=timeout,
            )
        except RuntimeError as exc:
            QMessageBox.critical(self, "Codex CLI", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.codex_install_button.setEnabled(True)
            self.codex_test_button.setEnabled(True)

        QMessageBox.information(
            self,
            "Codex CLI",
            f"{result.message}\n\n{login_message}",
        )

    def _test_codex_cli(self) -> None:
        profiles: Dict[tuple[str, str], list[str]] = {}
        selected = (
            (
                "Перевод",
                self.codex_model_combo.currentText().strip(),
                self.codex_reasoning_combo.currentText().strip(),
            ),
            (
                "Преданализ",
                self.codex_analysis_model_combo.currentText().strip(),
                self.codex_analysis_reasoning_combo.currentText().strip(),
            ),
        )
        for label, model, effort in selected:
            profiles.setdefault((model, effort), []).append(label)

        cli_path = self.codex_path_edit.text().strip()
        check_timeout = min(60, self.codex_timeout_spin.value())
        for attempt in range(2):
            results = []
            try:
                for (model, effort), labels in profiles.items():
                    status = check_codex_cli(
                        cli_path,
                        timeout=check_timeout,
                        model=model,
                        reasoning_effort=effort,
                    )
                    results.append(f"{' / '.join(labels)}:\n{status}")
            except CodexCliAuthenticationError as exc:
                if attempt > 0:
                    QMessageBox.critical(self, "Codex CLI", str(exc))
                    return
                answer = QMessageBox.question(
                    self,
                    "Codex CLI",
                    f"{exc}\n\nВыполнить повторный вход сейчас?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    login_codex_cli(
                        cli_path,
                        timeout=max(120, self.codex_timeout_spin.value()),
                    )
                except RuntimeError as login_error:
                    QMessageBox.critical(self, "Codex CLI", str(login_error))
                    return
                finally:
                    QApplication.restoreOverrideCursor()
                continue
            except RuntimeError as exc:
                QMessageBox.critical(self, "Codex CLI", str(exc))
                return
            QMessageBox.information(self, "Codex CLI", "\n\n".join(results))
            return

    def _open_pdf_settings(self) -> None:
        dialog = PdfTextractSettingsDialog(self._pdf_mode_values, self)
        if dialog.exec() == QDialog.Accepted:
            self._pdf_mode_values = dialog.get_values()
