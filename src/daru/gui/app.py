"""GUI application entry point."""

import sys

from PySide6.QtWidgets import QApplication

from .. import APP_NAME, __version__
from ..config import SettingsManager
from .main_window import MainWindow


def configure_application(app: QApplication) -> None:
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName("Daru")


def main() -> None:
    app = QApplication(sys.argv)
    configure_application(app)
    manager = SettingsManager()
    window = MainWindow(manager)
    window.show()
    sys.exit(app.exec())
