"""PDF-related settings dialogs."""

from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QWidget,
)

from ..config import (
    OPENAI_DEFAULT_BASE_URL,
    OPENAI_MODEL_CHOICES,
    AppSettings,
    get_openai_model_profile,
    normalize_openai_base_url,
    populate_combo,
)
from ..translation.codex_cli import CODEX_REASONING_EFFORTS, check_codex_cli


class PdfTextractSettingsDialog(QDialog):
    def __init__(self, values: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Настройки обработки PDF")
        layout = QFormLayout(self)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Локальная обработка", "local")
        self.mode_combo.addItem("Textract", "textract")
        self.mode_combo.addItem("Textract + GPT Vision", "textract_vision")
        current_mode = values.get("pdf_processing_mode", "textract").lower()
        mode_index = self.mode_combo.findData(current_mode)
        self.mode_combo.setCurrentIndex(mode_index if mode_index >= 0 else self.mode_combo.findData("textract"))

        self.region_edit = QLineEdit(values.get("textract_region", "us-east-1"))
        self.access_key_edit = QLineEdit(values.get("textract_access_key", ""))
        self.secret_key_edit = QLineEdit(values.get("textract_secret_key", ""))
        self.secret_key_edit.setEchoMode(QLineEdit.Password)
        self.session_token_edit = QLineEdit(values.get("textract_session_token", ""))
        self.session_token_edit.setEchoMode(QLineEdit.Password)

        self.vision_backend_combo = QComboBox()
        self.vision_backend_combo.addItem("OpenAI API", "openai_api")
        self.vision_backend_combo.addItem("Codex CLI", "codex_cli")
        backend = str(values.get("pdf_vision_backend", "openai_api") or "openai_api")
        backend_index = self.vision_backend_combo.findData(backend)
        self.vision_backend_combo.setCurrentIndex(
            backend_index
            if backend_index >= 0
            else self.vision_backend_combo.findData("openai_api")
        )

        self.vision_model_combo = QComboBox()
        populate_combo(
            self.vision_model_combo,
            OPENAI_MODEL_CHOICES,
            str(values.get("pdf_vision_model", "gpt-5.5") or "gpt-5.5"),
        )
        self.vision_reasoning_combo = QComboBox()
        initial_vision_reasoning = str(
            values.get("pdf_vision_reasoning_effort", "medium") or "medium"
        )

        self.vision_api_key_edit = QLineEdit(
            str(values.get("pdf_vision_api_key", "") or "")
        )
        self.vision_api_key_edit.setEchoMode(QLineEdit.Password)
        self.vision_base_url_edit = QLineEdit(
            normalize_openai_base_url(
                str(values.get("pdf_vision_base_url", OPENAI_DEFAULT_BASE_URL) or "")
            )
        )
        self.vision_project_edit = QLineEdit(
            str(values.get("pdf_vision_project", "") or "")
        )

        self.vision_codex_path_edit = QLineEdit(
            str(values.get("pdf_vision_codex_cli_path", "") or "")
        )
        self.vision_codex_path_edit.setPlaceholderText("Автопоиск команды codex в PATH")
        self.vision_codex_browse = QPushButton("...")
        self.vision_codex_browse.clicked.connect(self._browse_codex_cli)
        codex_path_widget = QWidget()
        codex_path_layout = QHBoxLayout(codex_path_widget)
        codex_path_layout.setContentsMargins(0, 0, 0, 0)
        codex_path_layout.addWidget(self.vision_codex_path_edit)
        codex_path_layout.addWidget(self.vision_codex_browse)

        self.vision_codex_timeout_spin = QSpinBox()
        self.vision_codex_timeout_spin.setRange(10, 3600)
        self.vision_codex_timeout_spin.setSuffix(" с")
        self.vision_codex_timeout_spin.setValue(
            max(10, int(values.get("pdf_vision_codex_timeout_seconds", 300) or 300))
        )
        self.vision_request_mode_combo = QComboBox()
        self.vision_request_mode_combo.addItem("Пакетно (стабильнее)", "batched")
        self.vision_request_mode_combo.addItem(
            "Одна страница одним запросом", "single_page"
        )
        request_mode = str(values.get("pdf_vision_request_mode", "batched") or "batched")
        request_mode_index = self.vision_request_mode_combo.findData(request_mode)
        self.vision_request_mode_combo.setCurrentIndex(
            request_mode_index
            if request_mode_index >= 0
            else self.vision_request_mode_combo.findData("batched")
        )
        self.vision_image_quality_combo = QComboBox()
        self.vision_image_quality_combo.addItem("Стабильно (до 4 МП)", "stable")
        self.vision_image_quality_combo.addItem("Высокое качество (до 10 МП)", "high")
        self.vision_image_quality_combo.addItem("Оригинал (без уменьшения)", "original")
        image_quality = str(
            values.get("pdf_vision_image_quality", "stable") or "stable"
        )
        image_quality_index = self.vision_image_quality_combo.findData(image_quality)
        self.vision_image_quality_combo.setCurrentIndex(
            image_quality_index
            if image_quality_index >= 0
            else self.vision_image_quality_combo.findData("stable")
        )
        self.vision_codex_test = QPushButton("Проверить Codex CLI")
        self.vision_codex_test.clicked.connect(self._test_codex_cli)

        layout.addRow("Режим обработки PDF", self.mode_combo)
        layout.addRow("AWS Region", self.region_edit)
        layout.addRow("AWS Access Key ID", self.access_key_edit)
        layout.addRow("AWS Secret Access Key", self.secret_key_edit)
        layout.addRow("AWS Session Token", self.session_token_edit)
        layout.addRow("Провайдер Vision", self.vision_backend_combo)
        layout.addRow("Модель Vision", self.vision_model_combo)
        layout.addRow("Reasoning Vision", self.vision_reasoning_combo)
        layout.addRow("OpenAI API Key для Vision", self.vision_api_key_edit)
        layout.addRow("OpenAI Base URL для Vision", self.vision_base_url_edit)
        layout.addRow("OpenAI Project ID для Vision", self.vision_project_edit)
        layout.addRow("Путь к Codex CLI для Vision", codex_path_widget)
        layout.addRow("Таймаут Codex CLI для Vision", self.vision_codex_timeout_spin)
        layout.addRow("Режим Codex Vision", self.vision_request_mode_combo)
        layout.addRow("Качество изображения Vision", self.vision_image_quality_combo)
        layout.addRow(self.vision_codex_test)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.mode_combo.currentIndexChanged.connect(self._update_controls)
        self.vision_backend_combo.currentIndexChanged.connect(
            self._handle_backend_change
        )
        self.vision_model_combo.currentTextChanged.connect(
            self._update_reasoning_choices
        )
        self._update_reasoning_choices(self.vision_model_combo.currentText())
        if self.vision_reasoning_combo.findText(initial_vision_reasoning) >= 0:
            self.vision_reasoning_combo.setCurrentText(initial_vision_reasoning)
        self._update_controls()

    def _handle_backend_change(self, _value: object = None) -> None:
        self._update_reasoning_choices(self.vision_model_combo.currentText())
        self._update_controls()

    def _update_controls(self, _value: object = None) -> None:
        mode_value = str(self.mode_combo.currentData() or "")
        textract_enabled = mode_value.startswith("textract")
        for widget in (self.region_edit, self.access_key_edit, self.secret_key_edit, self.session_token_edit):
            widget.setEnabled(textract_enabled)

        vision_enabled = mode_value == "textract_vision"
        backend = str(self.vision_backend_combo.currentData() or "openai_api")
        api_enabled = vision_enabled and backend == "openai_api"
        cli_enabled = vision_enabled and backend == "codex_cli"
        self.vision_backend_combo.setEnabled(vision_enabled)
        self.vision_model_combo.setEnabled(vision_enabled)
        self.vision_reasoning_combo.setEnabled(
            vision_enabled and self.vision_reasoning_combo.count() > 0
        )
        self.vision_image_quality_combo.setEnabled(vision_enabled)
        for widget in (
            self.vision_api_key_edit,
            self.vision_base_url_edit,
            self.vision_project_edit,
        ):
            widget.setEnabled(api_enabled)
        for widget in (
            self.vision_codex_path_edit,
            self.vision_codex_browse,
            self.vision_codex_timeout_spin,
            self.vision_request_mode_combo,
            self.vision_codex_test,
        ):
            widget.setEnabled(cli_enabled)

    def _update_reasoning_choices(self, model: str) -> None:
        backend = str(self.vision_backend_combo.currentData() or "openai_api")
        if backend == "codex_cli":
            efforts = CODEX_REASONING_EFFORTS
            default_effort = "medium"
        else:
            profile = get_openai_model_profile(model)
            efforts = tuple(profile["reasoning_efforts"])
            default_effort = str(profile["default_reasoning_effort"] or "medium")
        current = self.vision_reasoning_combo.currentText().strip()
        self.vision_reasoning_combo.blockSignals(True)
        self.vision_reasoning_combo.clear()
        self.vision_reasoning_combo.addItems(list(efforts))
        selected = current if current in efforts else default_effort
        if selected in efforts:
            self.vision_reasoning_combo.setCurrentText(selected)
        self.vision_reasoning_combo.blockSignals(False)
        self._update_controls()

    def _browse_codex_cli(self) -> None:
        current = self.vision_codex_path_edit.text().strip()
        start_path = str(Path(current).parent) if current else ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите Codex CLI",
            start_path,
            "Codex CLI (codex.exe codex.cmd codex.bat codex.ps1);;Все файлы (*)",
        )
        if path:
            self.vision_codex_path_edit.setText(path)

    def _test_codex_cli(self) -> None:
        try:
            status = check_codex_cli(
                self.vision_codex_path_edit.text().strip(),
                timeout=min(60, self.vision_codex_timeout_spin.value()),
                model=self.vision_model_combo.currentText().strip(),
                reasoning_effort=(
                    self.vision_reasoning_combo.currentText().strip() or "medium"
                ),
                require_images=True,
            )
        except RuntimeError as exc:
            QMessageBox.critical(self, "Codex CLI", str(exc))
            return
        QMessageBox.information(self, "Codex CLI", status)

    def get_values(self) -> Dict[str, Any]:
        mode_value = str(self.mode_combo.currentData() or "local")
        return {
            "pdf_processing_mode": mode_value,
            "textract_region": self.region_edit.text().strip(),
            "textract_access_key": self.access_key_edit.text().strip(),
            "textract_secret_key": self.secret_key_edit.text().strip(),
            "textract_session_token": self.session_token_edit.text().strip(),
            "pdf_vision_backend": str(
                self.vision_backend_combo.currentData() or "openai_api"
            ),
            "pdf_vision_model": self.vision_model_combo.currentText().strip()
            or "gpt-5.5",
            "pdf_vision_reasoning_effort": (
                self.vision_reasoning_combo.currentText().strip() or "medium"
            ),
            "pdf_vision_api_key": self.vision_api_key_edit.text().strip(),
            "pdf_vision_base_url": normalize_openai_base_url(
                self.vision_base_url_edit.text()
            ),
            "pdf_vision_project": self.vision_project_edit.text().strip(),
            "pdf_vision_codex_cli_path": self.vision_codex_path_edit.text().strip(),
            "pdf_vision_codex_timeout_seconds": self.vision_codex_timeout_spin.value(),
            "pdf_vision_request_mode": str(
                self.vision_request_mode_combo.currentData() or "batched"
            ),
            "pdf_vision_image_quality": str(
                self.vision_image_quality_combo.currentData() or "stable"
            ),
        }


class PdfProcessingDialog(QDialog):
    def __init__(
        self,
        settings: AppSettings,
        parent: Optional[QWidget] = None,
        *,
        native_mode: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(
            "Обработка PDF: текстовый слой"
            if native_mode
            else "Обработка PDF: размытие и OCR"
        )
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

        if native_mode:
            for widget in (
                self.dpi_spin,
                self.confidence_spin,
                self.blur_spin,
                self.dilation_spin,
                self.lang_edit,
            ):
                label = layout.labelForField(widget)
                if label is not None:
                    label.setVisible(False)
                widget.setVisible(False)
            layout.addRow(
                "Режим",
                QLabel(
                    "Текст извлекается напрямую. OCR включится только для страниц "
                    "без пригодного текстового слоя."
                ),
            )

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
