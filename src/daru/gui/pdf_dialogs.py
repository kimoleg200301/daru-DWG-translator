"""PDF-related settings dialogs."""

from typing import Any, Dict, Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QWidget,
)

from ..config import AppSettings


class PdfTextractSettingsDialog(QDialog):
    def __init__(self, values: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Настройки обработки PDF")
        layout = QFormLayout(self)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Локальная обработка", "Textract"])
        current_mode = values.get("pdf_processing_mode", "textract").lower()
        self.mode_combo.setCurrentText("Textract" if current_mode == "textract" else "Локальная обработка")

        self.region_edit = QLineEdit(values.get("textract_region", "us-east-1"))
        self.access_key_edit = QLineEdit(values.get("textract_access_key", ""))
        self.secret_key_edit = QLineEdit(values.get("textract_secret_key", ""))
        self.secret_key_edit.setEchoMode(QLineEdit.Password)
        self.session_token_edit = QLineEdit(values.get("textract_session_token", ""))
        self.session_token_edit.setEchoMode(QLineEdit.Password)

        layout.addRow("Режим обработки PDF", self.mode_combo)
        layout.addRow("AWS Region", self.region_edit)
        layout.addRow("AWS Access Key ID", self.access_key_edit)
        layout.addRow("AWS Secret Access Key", self.secret_key_edit)
        layout.addRow("AWS Session Token", self.session_token_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.mode_combo.currentTextChanged.connect(self._update_textract_enabled)
        self._update_textract_enabled(self.mode_combo.currentText())

    def _update_textract_enabled(self, mode_text: str) -> None:
        enabled = (mode_text or "").lower().startswith("textract")
        for widget in (self.region_edit, self.access_key_edit, self.secret_key_edit, self.session_token_edit):
            widget.setEnabled(enabled)

    def get_values(self) -> Dict[str, Any]:
        mode_text = self.mode_combo.currentText()
        mode_value = "textract" if (mode_text or "").lower().startswith("textract") else "local"
        return {
            "pdf_processing_mode": mode_value,
            "textract_region": self.region_edit.text().strip(),
            "textract_access_key": self.access_key_edit.text().strip(),
            "textract_secret_key": self.secret_key_edit.text().strip(),
            "textract_session_token": self.session_token_edit.text().strip(),
        }


class PdfProcessingDialog(QDialog):
    def __init__(self, settings: AppSettings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Обработка PDF: размытие и OCR")
        layout = QFormLayout(self)

        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(150, 800)
        self.dpi_spin.setValue(int(settings.pdf_dpi or 400))

        self.confidence_spin = QSpinBox()
        self.confidence_spin.setRange(0, 100)
        self.confidence_spin.setValue(int(settings.pdf_min_confidence or 60))

        self.blur_spin = QSpinBox()
        self.blur_spin.setRange(3, 99)
        self.blur_spin.setSingleStep(2)
        self.blur_spin.setValue(int(settings.pdf_blur_kernel or 23))

        self.dilation_spin = QSpinBox()
        self.dilation_spin.setRange(1, 15)
        self.dilation_spin.setValue(int(settings.pdf_dilation_kernel or 3))

        self.lang_edit = QLineEdit(settings.pdf_ocr_languages or "eng")

        layout.addRow("DPI", self.dpi_spin)
        layout.addRow("Мин. доверие OCR", self.confidence_spin)
        layout.addRow("Размер ядра размытия (нечётный)", self.blur_spin)
        layout.addRow("Дилатация маски", self.dilation_spin)
        layout.addRow("Языки OCR (например, eng+rus)", self.lang_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self) -> Dict[str, object]:
        return {
            "dpi": self.dpi_spin.value(),
            "min_confidence": self.confidence_spin.value(),
            "blur_kernel_size": self.blur_spin.value(),
            "dilation_kernel_size": self.dilation_spin.value(),
            "ocr_languages": self.lang_edit.text().strip() or "eng",
        }
