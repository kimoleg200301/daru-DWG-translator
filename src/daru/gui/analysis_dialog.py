"""Editable Codex document analysis shown before translation."""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..translation.analysis import CODEX_ANALYSIS_MAX_APPROVED_CHARS


class CodexAnalysisDialog(QDialog):
    def __init__(
        self,
        payload: Dict[str, Any],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Анализ документа перед переводом")
        self.resize(760, 600)

        layout = QVBoxLayout(self)
        model = str(payload.get("model") or "")
        effort = str(payload.get("reasoning_effort") or "")
        self.model_label = QLabel(
            f"Модель преданализа: {model} · Reasoning: {effort}"
        )
        layout.addWidget(self.model_label)

        self.warning_label = QLabel(str(payload.get("warning") or ""))
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #9a6700;")
        self.warning_label.setVisible(bool(payload.get("used_fallback")))
        layout.addWidget(self.warning_label)

        self.editor = QTextEdit()
        self.editor.setPlainText(str(payload.get("text") or ""))
        self.editor.textChanged.connect(self._enforce_limit)
        layout.addWidget(self.editor)

        self.count_label = QLabel()
        layout.addWidget(self.count_label)
        self._update_count()

        buttons = QDialogButtonBox()
        self.start_button = buttons.addButton(
            "Начать перевод",
            QDialogButtonBox.AcceptRole,
        )
        self.cancel_button = buttons.addButton(
            "Отмена",
            QDialogButtonBox.RejectRole,
        )
        self.start_button.clicked.connect(self._accept_analysis)
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(buttons)

    def approved_text(self) -> str:
        return self.editor.toPlainText().strip()

    def _accept_analysis(self) -> None:
        if not self.approved_text():
            QMessageBox.warning(
                self,
                "Предварительный анализ",
                "Анализ не может быть пустым.",
            )
            return
        self.accept()

    def _enforce_limit(self) -> None:
        text = self.editor.toPlainText()
        if len(text) > CODEX_ANALYSIS_MAX_APPROVED_CHARS:
            self.editor.blockSignals(True)
            self.editor.setPlainText(text[:CODEX_ANALYSIS_MAX_APPROVED_CHARS])
            cursor = self.editor.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.editor.setTextCursor(cursor)
            self.editor.blockSignals(False)
        self._update_count()

    def _update_count(self) -> None:
        count = len(self.editor.toPlainText())
        self.count_label.setText(
            f"{count} / {CODEX_ANALYSIS_MAX_APPROVED_CHARS} символов"
        )


__all__ = ["CodexAnalysisDialog"]
