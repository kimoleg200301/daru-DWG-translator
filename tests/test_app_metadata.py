"""Application metadata tests."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from daru import APP_DISPLAY_NAME, APP_NAME, __version__
from daru.config import SettingsManager
from daru.gui.app import configure_application
from daru.gui.main_window import MainWindow


def test_gui_uses_application_name_and_version(tmp_path):
    app = QApplication.instance() or QApplication([])
    configure_application(app)
    window = MainWindow(SettingsManager(path=tmp_path / "settings.json"))

    assert app.applicationName() == APP_NAME
    assert app.applicationDisplayName() == APP_NAME
    assert app.applicationVersion() == __version__
    assert window.windowTitle() == APP_DISPLAY_NAME

    window.close()
    app.processEvents()
