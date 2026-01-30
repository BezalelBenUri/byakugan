"""Application-wide theme helpers."""
from __future__ import annotations

from PyQt6.QtGui import QColor, QPalette


def build_dark_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 32, 36))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base, QColor(24, 25, 28))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(34, 36, 40))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Text, QColor(235, 235, 235))
    palette.setColor(QPalette.ColorRole.Button, QColor(40, 43, 48))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(235, 235, 235))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(76, 110, 245))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    return palette


def apply_dark_theme(app) -> None:
    """Apply the dark palette and subtle styling tweaks."""
    palette = build_dark_palette()
    app.setPalette(palette)
    app.setStyle("Fusion")
