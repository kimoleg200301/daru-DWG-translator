"""API settings dialog."""

from typing import Any, Dict, Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    OPENAI_BASE_URL_CHOICES,
    OPENAI_MODEL_CHOICES,
    OPENAI_REASONING_MODELS,
    AppSettings,
    populate_combo,
)
from .pdf_dialogs import PdfTextractSettingsDialog


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Настройки API")
        layout = QVBoxLayout(self)
        self._pdf_mode_values: Dict[str, Any] = {
            "pdf_processing_mode": settings.pdf_processing_mode or "textract",
            "textract_region": settings.textract_region or "",
            "textract_access_key": settings.textract_access_key or "",
            "textract_secret_key": settings.textract_secret_key or "",
            "textract_session_token": settings.textract_session_token or "",
        }

        form = QFormLayout()
        self.deepl_edit = QLineEdit(settings.deepl_key)
        self.openai_key_edit = QLineEdit(settings.openai_key)
        self.openai_model_combo = QComboBox()
        populate_combo(self.openai_model_combo, OPENAI_MODEL_CHOICES, settings.openai_model)
        self.openai_url_combo = QComboBox()
        populate_combo(self.openai_url_combo, OPENAI_BASE_URL_CHOICES, settings.openai_base_url, allow_empty=True)
        self.openai_project_edit = QLineEdit(settings.openai_project)
        self.openai_temp_spin = QSpinBox()
        self.openai_temp_spin.setRange(0, 100)
        self.openai_temp_spin.setValue(int(settings.openai_temperature * 100))
        self.openai_strict_mode_combo = QComboBox()
        self.openai_strict_mode_combo.addItems(["verbosity", "effort"])
        current_mode = settings.openai_strict_mode or "verbosity"
        if current_mode not in {"verbosity", "effort"}:
            current_mode = "verbosity"
        self.openai_strict_mode_combo.setCurrentText(current_mode)
        self.openai_strict_value_spin = QSpinBox()
        self.openai_strict_value_spin.setRange(0, 100)
        self.openai_strict_value_spin.setValue(int(settings.openai_strict_value * 100))

        self.openai_key_edit.setEchoMode(QLineEdit.Password)
        self.deepl_edit.setEchoMode(QLineEdit.Password)

        form.addRow("DeepL API Key", self.deepl_edit)
        form.addRow("OpenAI API Key", self.openai_key_edit)
        form.addRow("OpenAI Model", self.openai_model_combo)
        form.addRow("OpenAI Base URL", self.openai_url_combo)
        form.addRow("OpenAI Project ID", self.openai_project_edit)
        form.addRow("OpenAI Temperature (x100)", self.openai_temp_spin)
        form.addRow("OpenAI Strict Parameter", self.openai_strict_mode_combo)
        form.addRow("OpenAI Strict Value (x100)", self.openai_strict_value_spin)
        layout.addLayout(form)

        self.pdf_settings_btn = QPushButton("Настройки обработки PDF")
        self.pdf_settings_btn.clicked.connect(self._open_pdf_settings)
        layout.addWidget(self.pdf_settings_btn)

        self.openai_model_combo.currentTextChanged.connect(self._update_openai_param_visibility)
        self._update_openai_param_visibility(self.openai_model_combo.currentText())

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self) -> Dict[str, Any]:
        return {
            "deepl_key": self.deepl_edit.text().strip(),
            "openai_key": self.openai_key_edit.text().strip(),
            "openai_model": self.openai_model_combo.currentText().strip() or "gpt-4o-mini",
            "openai_base_url": self.openai_url_combo.currentText().strip(),
            "openai_project": self.openai_project_edit.text().strip(),
            "openai_temperature": self.openai_temp_spin.value() / 100.0,
            "openai_strict_mode": self.openai_strict_mode_combo.currentText().strip() or "verbosity",
            "openai_strict_value": self.openai_strict_value_spin.value() / 100.0,
            **self._pdf_mode_values,
        }

    def _update_openai_param_visibility(self, model: str) -> None:
        is_reasoning = (model or "").lower() in OPENAI_REASONING_MODELS
        self.openai_temp_spin.setEnabled(not is_reasoning)
        self.openai_strict_mode_combo.setEnabled(is_reasoning)
        self.openai_strict_value_spin.setEnabled(is_reasoning)

    def _open_pdf_settings(self) -> None:
        dialog = PdfTextractSettingsDialog(self._pdf_mode_values, self)
        if dialog.exec() == QDialog.Accepted:
            self._pdf_mode_values = dialog.get_values()
