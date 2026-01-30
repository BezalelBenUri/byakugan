"""Application bootstrap utilities."""
from __future__ import annotations

import sys
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication

from .logging import configure_logging
from .ui.main_window import MainWindow
from .ui.theme import apply_dark_theme


def _configure_high_dpi() -> None:
    """Configure high-DPI handling before QApplication instantiation."""
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )


def main(argv: Optional[list[str]] = None) -> int:
    """Launch the Byakugan desktop application."""
    configure_logging()
    argv = list(sys.argv if argv is None else argv)

    _configure_high_dpi()
    app = QApplication(argv)
    apply_dark_theme(app)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
