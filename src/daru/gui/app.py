"""GUI application entry point."""

import sys

from PySide6.QtWidgets import QApplication

from ..config import SettingsManager
from .main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    manager = SettingsManager()
    window = MainWindow(manager)
    window.show()
    sys.exit(app.exec())
