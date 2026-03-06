"""Main application window."""

from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    LANGUAGE_CHOICES,
    OUTPUT_FORMAT_CHOICES,
    PDF_TYPE_CHOICES,
    STYLE_FONT_CHOICES,
    TRANSLATOR_CHOICES,
    AppSettings,
    SettingsManager,
    populate_combo,
)
from ..pdf.pipeline import PDF_TYPE_SCANNED, _CACHE_DIR
from .pdf_dialogs import PdfProcessingDialog
from .settings_dialog import SettingsDialog
from .worker import TranslateWorker


class MainWindow(QWidget):
    def __init__(self, settings_manager: SettingsManager) -> None:
        super().__init__()
        self.settings_manager = settings_manager
        self.worker: Optional[TranslateWorker] = None
        self._pdf_input_active = False
        self.setWindowTitle("Daru DWG Translator")
        self.resize(860, 640)
        self.setAcceptDrops(True)

        self.status_label = QLabel("Перетащите DWG/DXF файл или выберите его через обозреватель.")
        self.input_edit = QLineEdit()
        self.input_edit.setReadOnly(True)
        self.input_browse = QPushButton("Обзор...")
        self.input_browse.clicked.connect(self.select_input_file)

        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("Входной файл"))
        input_row.addWidget(self.input_edit)
        input_row.addWidget(self.input_browse)

        self.pdf_type_widget = QWidget()
        pdf_type_layout = QHBoxLayout(self.pdf_type_widget)
        pdf_type_layout.setContentsMargins(0, 0, 0, 0)
        self.pdf_type_label = QLabel("Тип PDF")
        self.pdf_type_combo = QComboBox()
        for label, code in PDF_TYPE_CHOICES:
            self.pdf_type_combo.addItem(label, code)
        pdf_type_layout.addWidget(self.pdf_type_label)
        pdf_type_layout.addWidget(self.pdf_type_combo)
        pdf_type_layout.addStretch()
        self.pdf_type_widget.setVisible(False)

        self.pdf_layer_widget = QWidget()
        pdf_layer_layout = QHBoxLayout(self.pdf_layer_widget)
        pdf_layer_layout.setContentsMargins(0, 0, 0, 0)
        self.pdf_layer_checkbox = QCheckBox("Сохранить JSON слой перевода")
        self.pdf_layer_checkbox.toggled.connect(self.update_aux_controls)
        self.pdf_layer_path_edit = QLineEdit()
        self.pdf_layer_browse = QPushButton("...")
        self.pdf_layer_browse.clicked.connect(lambda: self.browse_aux_file(self.pdf_layer_path_edit))
        pdf_layer_layout.addWidget(self.pdf_layer_checkbox)
        pdf_layer_layout.addWidget(self.pdf_layer_path_edit)
        pdf_layer_layout.addWidget(self.pdf_layer_browse)
        self.pdf_layer_widget.setVisible(False)

        self.output_edit = QLineEdit()
        self.output_browse = QPushButton("Сохранить как...")
        self.output_browse.clicked.connect(self.select_output_file)
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(OUTPUT_FORMAT_CHOICES)
        self.output_format_combo.currentTextChanged.connect(self.handle_output_format_change)
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Выходной файл"))
        output_row.addWidget(self.output_edit)
        output_row.addWidget(self.output_browse)
        self.output_format_label = QLabel("Формат")
        output_row.addWidget(self.output_format_label)
        output_row.addWidget(self.output_format_combo)

        self.translator_combo = QComboBox()
        populate_combo(self.translator_combo, TRANSLATOR_CHOICES, settings_manager.data.translator_name, editable=False)
        self.source_lang_combo = QComboBox()
        populate_combo(self.source_lang_combo, LANGUAGE_CHOICES, settings_manager.data.source_lang, editable=False)
        self.target_lang_combo = QComboBox()
        populate_combo(self.target_lang_combo, LANGUAGE_CHOICES, settings_manager.data.target_lang, editable=False)
        self.style_font_combo = QComboBox()
        populate_combo(self.style_font_combo, STYLE_FONT_CHOICES, settings_manager.data.style_font, allow_empty=False, editable=False)

        self.map_checkbox = QCheckBox("Сохранить CSV карту переводов")
        self.map_path_edit = QLineEdit()
        self.map_browse = QPushButton("...")
        self.map_browse.clicked.connect(lambda: self.browse_aux_file(self.map_path_edit))
        self.map_checkbox.toggled.connect(self.update_aux_controls)

        self.txt_checkbox = QCheckBox("Сохранять TXT промежуточные файлы")
        self.extracted_path_edit = QLineEdit()
        self.extracted_browse = QPushButton("...")
        self.extracted_browse.clicked.connect(lambda: self.browse_aux_file(self.extracted_path_edit))
        self.translated_path_edit = QLineEdit()
        self.translated_browse = QPushButton("...")
        self.translated_browse.clicked.connect(lambda: self.browse_aux_file(self.translated_path_edit))
        self.txt_checkbox.toggled.connect(self.update_aux_controls)

        self.start_button = QPushButton("Запустить перевод")
        self.start_button.clicked.connect(self.start_translation)
        self.settings_button = QPushButton("Настройки API")
        self.settings_button.clicked.connect(self.open_settings)
        self.clear_log_button = QPushButton("Очистить лог")
        self.clear_log_button.clicked.connect(self.clear_log)

        self.log_view = QListWidget()
        self.log_view.setAlternatingRowColors(True)
        self.log_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.log_view.setStyleSheet(
            "QListWidget { background: palette(base); color: palette(text); border: 1px solid palette(mid); }"
            "QListWidget::item { padding: 4px; background: palette(base); color: palette(text); }"
            "QListWidget::item:alternate { background: palette(alternate-base); color: palette(text); }"
        )

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.status_label)
        main_layout.addLayout(input_row)
        main_layout.addWidget(self.pdf_type_widget)
        main_layout.addWidget(self.pdf_layer_widget)
        main_layout.addLayout(output_row)

        options_layout = QFormLayout()
        options_layout.addRow("Переводчик", self.translator_combo)
        options_layout.addRow("Исходный язык", self.source_lang_combo)
        options_layout.addRow("Целевой язык", self.target_lang_combo)
        options_layout.addRow("Шрифт стиля", self.style_font_combo)
        main_layout.addLayout(options_layout)

        map_layout = QHBoxLayout()
        map_layout.addWidget(self.map_checkbox)
        map_layout.addWidget(self.map_path_edit)
        map_layout.addWidget(self.map_browse)
        main_layout.addLayout(map_layout)

        txt_layout1 = QHBoxLayout()
        txt_layout1.addWidget(self.txt_checkbox)
        txt_layout1.addStretch()
        main_layout.addLayout(txt_layout1)

        txt_layout2 = QHBoxLayout()
        txt_layout2.addWidget(QLabel("TXT исходных"))
        txt_layout2.addWidget(self.extracted_path_edit)
        txt_layout2.addWidget(self.extracted_browse)
        main_layout.addLayout(txt_layout2)

        txt_layout3 = QHBoxLayout()
        txt_layout3.addWidget(QLabel("TXT переводов"))
        txt_layout3.addWidget(self.translated_path_edit)
        txt_layout3.addWidget(self.translated_browse)
        main_layout.addLayout(txt_layout3)

        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.start_button)
        buttons_row.addWidget(self.settings_button)
        buttons_row.addWidget(self.clear_log_button)
        buttons_row.addStretch()
        main_layout.addLayout(buttons_row)

        main_layout.addWidget(QLabel("Лог"))
        main_layout.addWidget(self.log_view, stretch=1)

        self.restore_from_settings()
        self.update_aux_controls()
        self.sync_output_suffix()

    def restore_from_settings(self) -> None:
        data = self.settings_manager.data
        self.translator_combo.setCurrentText(data.translator_name)
        self.source_lang_combo.setCurrentText(data.source_lang)
        self.target_lang_combo.setCurrentText(data.target_lang)
        self.style_font_combo.setCurrentText(data.style_font or STYLE_FONT_CHOICES[0])
        if not self.style_font_combo.currentText().strip():
            self.style_font_combo.setCurrentText(STYLE_FONT_CHOICES[0])
        self.pdf_layer_checkbox.setChecked(data.save_pdf_layer)
        self.pdf_layer_path_edit.setText(data.pdf_layer_path)
        self.map_checkbox.setChecked(data.save_map)
        self.txt_checkbox.setChecked(data.save_txt)
        self.output_format_combo.setCurrentText(data.output_format or "dwg")

    def browse_aux_file(self, target: QLineEdit) -> None:
        start_dir = Path(target.text()).parent if target.text() else Path(self.settings_manager.data.last_directory)
        filename, _ = QFileDialog.getSaveFileName(self, "Выберите файл", str(start_dir))
        if filename:
            target.setText(filename)

    def select_input_file(self) -> None:
        start_dir = self.settings_manager.data.last_directory
        filename, _ = QFileDialog.getOpenFileName(self, "Выберите файл", start_dir, "CAD/PDF files (*.dxf *.dwg *.pdf)")
        if filename:
            self.set_input_file(Path(filename))

    def select_output_file(self) -> None:
        start_dir = Path(self.output_edit.text()).parent if self.output_edit.text() else self.settings_manager.data.last_directory
        if self.is_pdf_input():
            pattern = "PDF files (*.pdf)"
            caption = "Сохранить PDF"
        else:
            fmt = (self.output_format_combo.currentText() or "dwg").lower()
            pattern = "DWG files (*.dwg)" if fmt == "dwg" else "DXF files (*.dxf)"
            caption = "Сохранить DWG" if fmt == "dwg" else "Сохранить DXF"
        filename, _ = QFileDialog.getSaveFileName(self, caption, str(start_dir), pattern)
        if filename:
            self.output_edit.setText(filename)
            self.sync_output_suffix()

    def update_aux_controls(self) -> None:
        pdf_mode = self.is_pdf_input()
        map_active = self.map_checkbox.isChecked() and not pdf_mode
        txt_active = self.txt_checkbox.isChecked() and not pdf_mode
        self.map_checkbox.setEnabled(not pdf_mode)
        self.txt_checkbox.setEnabled(not pdf_mode)
        self.source_lang_combo.setEnabled(True)
        self.map_path_edit.setEnabled(map_active)
        self.map_browse.setEnabled(map_active)
        for widget in (self.extracted_path_edit, self.extracted_browse, self.translated_path_edit, self.translated_browse):
            widget.setEnabled(txt_active)
        self.pdf_layer_widget.setVisible(pdf_mode)
        self.pdf_layer_checkbox.setEnabled(pdf_mode)
        pdf_layer_active = pdf_mode and self.pdf_layer_checkbox.isChecked()
        self.pdf_layer_path_edit.setEnabled(pdf_layer_active)
        self.pdf_layer_browse.setEnabled(pdf_layer_active)
        if pdf_layer_active:
            self._ensure_pdf_layer_path()

    def _reset_auxiliary_for_pdf(self) -> None:
        for checkbox in (self.map_checkbox, self.txt_checkbox):
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)

    def is_pdf_input(self) -> bool:
        return self._pdf_input_active

    def handle_output_format_change(self, _value: str) -> None:
        self.sync_output_suffix()
        fmt = (self.output_format_combo.currentText() or "dwg").lower()
        self.settings_manager.update(output_format=fmt)

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith((".dxf", ".dwg", ".pdf")):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event) -> None:  # type: ignore[override]
        for url in event.mimeData().urls():
            if url.isLocalFile() and url.toLocalFile().lower().endswith((".dxf", ".dwg", ".pdf")):
                self.set_input_file(Path(url.toLocalFile()))
                event.acceptProposedAction()
                return
        event.ignore()

    def set_input_file(self, path: Path) -> None:
        self.input_edit.setText(str(path))
        self.status_label.setText(f"Выбран файл: {path.name}")
        suffix = path.suffix.lower()
        self._pdf_input_active = suffix == ".pdf"
        self.pdf_type_widget.setVisible(self._pdf_input_active)
        self.output_format_combo.setVisible(not self._pdf_input_active)
        self.output_format_label.setVisible(not self._pdf_input_active)
        if self._pdf_input_active:
            self._reset_auxiliary_for_pdf()
        elif suffix == ".dwg":
            self.output_format_combo.setCurrentText("dwg")
        elif suffix == ".dxf" and self.output_format_combo.currentText() not in OUTPUT_FORMAT_CHOICES:
            self.output_format_combo.setCurrentText("dwg")
        defaults = self.derive_default_paths(path)
        self.output_edit.setText(str(defaults["output"]))
        self.map_path_edit.setText(str(defaults["map"]))
        self.extracted_path_edit.setText(str(defaults["extracted"]))
        self.translated_path_edit.setText(str(defaults["translated"]))
        if self._pdf_input_active:
            self.pdf_layer_path_edit.setText(str(defaults.get("pdf_layer", path.with_suffix(".translation.json"))))
        self.settings_manager.update(last_directory=str(path.parent))
        self.update_aux_controls()
        self.sync_output_suffix()

    def derive_default_paths(self, input_path: Path) -> Dict[str, Path]:
        stem = input_path.stem
        parent = input_path.parent
        if input_path.suffix.lower() == ".pdf":
            out_suffix = ".pdf"
        else:
            out_suffix = ".dwg" if (self.output_format_combo.currentText() or "dwg").lower() == "dwg" else ".dxf"
        pdf_layer = parent / f"{stem}_translation.json"
        return {
            "output": parent / f"{stem}_ru{out_suffix}",
            "map": parent / f"{stem}_map.csv",
            "extracted": parent / f"{stem}_texts.txt",
            "translated": parent / f"{stem}_texts_ru.txt",
            "pdf_layer": pdf_layer,
        }

    def _pdf_processing_options(self) -> Dict[str, object]:
        data = self.settings_manager.data
        return {
            "dpi": data.pdf_dpi,
            "min_confidence": data.pdf_min_confidence,
            "blur_kernel_size": data.pdf_blur_kernel,
            "dilation_kernel_size": data.pdf_dilation_kernel,
            "ocr_languages": data.pdf_ocr_languages,
            "pdf_processing_mode": data.pdf_processing_mode,
            "textract_region": data.textract_region,
            "textract_access_key": data.textract_access_key,
            "textract_secret_key": data.textract_secret_key,
            "textract_session_token": data.textract_session_token,
        }

    def _base_translation_params(self, input_obj: Path, output_obj: Path) -> Dict[str, Any]:
        data = self.settings_manager.data
        return {
            "input_path": input_obj,
            "output_path": output_obj,
            "translator_name": self.translator_combo.currentText(),
            "source_lang": self.source_lang_combo.currentText().strip() or "en",
            "target_lang": self.target_lang_combo.currentText().strip() or "ru",
            "style_font": self.style_font_combo.currentText().strip() or STYLE_FONT_CHOICES[0],
            "deepl_key": data.deepl_key or None,
            "openai_key": data.openai_key or None,
            "openai_model": data.openai_model or None,
            "openai_base_url": data.openai_base_url or None,
            "openai_project": data.openai_project or None,
            "openai_temperature": data.openai_temperature,
            "openai_strict_mode": data.openai_strict_mode or None,
            "openai_strict_value": data.openai_strict_value,
        }

    def sync_output_suffix(self) -> None:
        if self.is_pdf_input():
            suffix = ".pdf"
        else:
            fmt = (self.output_format_combo.currentText() or "dwg").lower()
            suffix = ".dwg" if fmt == "dwg" else ".dxf"
        current = self.output_edit.text().strip()
        if current:
            path = Path(current)
            if path.suffix.lower() != suffix:
                path = path.with_suffix(suffix)
                self.output_edit.setText(str(path))
        if self.is_pdf_input():
            self._ensure_pdf_layer_path()

    def _ensure_pdf_layer_path(self) -> None:
        if not self.is_pdf_input():
            return
        current = self.pdf_layer_path_edit.text().strip()
        if current:
            return
        output = self.output_edit.text().strip()
        if not output:
            return
        default = Path(output).with_suffix(".translation.json")
        self.pdf_layer_path_edit.setText(str(default))

    def append_log(self, message: str) -> None:
        item = QListWidgetItem(message)
        if message.startswith("Готово. Результат"):
            item.setForeground(QColor("#1E8F40"))
        if message.startswith("Ошибка") or message.startswith("PDF: FreeText недоступен, рисуем текст прямо на странице"):
            item.setForeground(QColor("#8F1E1E"))
        if message.startswith("PDF: шрифт"):
            item.setForeground(QColor("#8F8B1E"))
        self.log_view.addItem(item)
        self.log_view.scrollToBottom()

    def clear_log(self) -> None:
        self.log_view.clear()

    def _launch_worker(self, params: Dict[str, Any], start_message: str, status_text: str) -> None:
        if self.worker is not None:
            QMessageBox.warning(self, "Выполнение", "Подождите завершения текущей операции")
            return
        self.append_log(start_message)
        self.start_button.setEnabled(False)
        self.worker = TranslateWorker(params)
        self.worker.log_signal.connect(self.append_log)
        self.worker.error_signal.connect(self.handle_error)
        self.worker.finished_signal.connect(self.handle_finished)
        self.worker.finished.connect(self.reset_worker)
        self.worker.start()
        self.status_label.setText(status_text)

    def start_translation(self) -> None:
        input_path = self.input_edit.text().strip()
        output_path = self.output_edit.text().strip()
        if not input_path:
            QMessageBox.warning(self, "Ошибка", "Выберите входной файл")
            return
        input_obj = Path(input_path)
        output_path_obj = Path(output_path) if output_path else None
        is_pdf = input_obj.suffix.lower() == ".pdf"
        if is_pdf:
            if output_path_obj is None:
                defaults = self.derive_default_paths(input_obj)
                output_path_obj = defaults["output"]
                self.output_edit.setText(str(output_path_obj))
            if output_path_obj.suffix.lower() != ".pdf":
                output_path_obj = output_path_obj.with_suffix(".pdf")
                self.output_edit.setText(str(output_path_obj))
            layer_json_path: Optional[Path] = None
            if self.pdf_layer_checkbox.isChecked():
                json_path_text = self.pdf_layer_path_edit.text().strip()
                if not json_path_text:
                    json_path_text = str((_CACHE_DIR / f"{input_obj.stem}.translation.json").resolve())
                    self.pdf_layer_path_edit.setText(json_path_text)
                layer_json_path = Path(json_path_text)
            processing_options = self._pdf_processing_options()
            dialog = PdfProcessingDialog(self.settings_manager.data, self)
            if dialog.exec() != QDialog.Accepted:
                return
            processing_dialog_values = dialog.get_values()
            processing_options.update(processing_dialog_values)
            self.settings_manager.update(
                pdf_dpi=processing_dialog_values["dpi"],
                pdf_min_confidence=processing_dialog_values["min_confidence"],
                pdf_blur_kernel=processing_dialog_values["blur_kernel_size"],
                pdf_dilation_kernel=processing_dialog_values["dilation_kernel_size"],
                pdf_ocr_languages=processing_dialog_values["ocr_languages"],
            )
        else:
            if output_path_obj is None:
                QMessageBox.warning(self, "Ошибка", "Укажите путь для сохранения файла")
                return
            format_choice = (self.output_format_combo.currentText() or "dwg").lower()
            expected_suffix = ".dwg" if format_choice == "dwg" else ".dxf"
            if output_path_obj.suffix.lower() != expected_suffix:
                output_path_obj = output_path_obj.with_suffix(expected_suffix)
                self.output_edit.setText(str(output_path_obj))
            layer_json_path = None
            processing_options = None
        params: Dict[str, Any] = self._base_translation_params(input_obj, output_path_obj)
        if is_pdf:
            params.update(
                {
                    "pdf_type": self.pdf_type_combo.currentData() or PDF_TYPE_SCANNED,
                    "layer_json_path": layer_json_path,
                    "processing_options": processing_options,
                    "job_type": "pdf",
                }
            )
        else:
            format_choice = (self.output_format_combo.currentText() or "dwg").lower()
            params.update(
                {
                    "output_format": format_choice,
                    "map_path": Path(self.map_path_edit.text()) if self.map_checkbox.isChecked() and self.map_path_edit.text() else None,
                    "save_map": self.map_checkbox.isChecked(),
                    "extracted_txt_path": Path(self.extracted_path_edit.text()) if self.txt_checkbox.isChecked() and self.extracted_path_edit.text() else None,
                    "translated_txt_path": Path(self.translated_path_edit.text()) if self.txt_checkbox.isChecked() and self.translated_path_edit.text() else None,
                    "save_txt": self.txt_checkbox.isChecked(),
                    "job_type": "cad",
                }
            )
        self._launch_worker(params, "Запуск процесса перевода...", "Перевод выполняется...")

    def handle_error(self, message: str) -> None:
        self.append_log(f"Ошибка: {message}")
        QMessageBox.critical(self, "Ошибка", message)
        self.start_button.setEnabled(True)
        self.status_label.setText("Ошибка при переводе")

    def handle_finished(self, payload: Dict[str, Any]) -> None:
        self.append_log(f"Готово. Результат: {payload['output_path']}")
        self.append_log(f"Движок перевода: {payload['backend']}")
        QMessageBox.information(self, "Готово", f"Файл сохранён: {payload['output_path']}")
        self.start_button.setEnabled(True)
        self.status_label.setText("Перевод завершён")
        job_type = payload.get("job_type", "cad")
        update_payload: Dict[str, Any] = {
            "translator_name": self.translator_combo.currentText(),
            "source_lang": self.source_lang_combo.currentText().strip() or "en",
            "target_lang": self.target_lang_combo.currentText().strip() or "ru",
            "style_font": self.style_font_combo.currentText().strip(),
            "save_map": self.map_checkbox.isChecked(),
            "save_txt": self.txt_checkbox.isChecked(),
            "save_pdf_layer": self.pdf_layer_checkbox.isChecked(),
            "pdf_layer_path": self.pdf_layer_path_edit.text().strip(),
        }
        if job_type == "pdf":
            processing_options = None
            if self.worker and hasattr(self.worker, "_params"):
                processing_options = self.worker._params.get("processing_options")
            if isinstance(processing_options, dict):
                update_payload.update(
                    {
                        "pdf_dpi": processing_options.get("dpi", self.settings_manager.data.pdf_dpi),
                        "pdf_min_confidence": processing_options.get("min_confidence", self.settings_manager.data.pdf_min_confidence),
                        "pdf_blur_kernel": processing_options.get("blur_kernel_size", self.settings_manager.data.pdf_blur_kernel),
                        "pdf_dilation_kernel": processing_options.get("dilation_kernel_size", self.settings_manager.data.pdf_dilation_kernel),
                        "pdf_ocr_languages": processing_options.get("ocr_languages", self.settings_manager.data.pdf_ocr_languages),
                    }
                )
        else:
            format_choice = (self.output_format_combo.currentText() or "dwg").lower()
            update_payload["output_format"] = format_choice
        self.settings_manager.update(**update_payload)

    def reset_worker(self) -> None:
        self.worker = None
        self.update_aux_controls()

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.settings_manager.data, self)
        if dialog.exec() == QDialog.Accepted:
            values = dialog.get_values()
            self.settings_manager.update(**values)
            self.append_log("Настройки API сохранены")
