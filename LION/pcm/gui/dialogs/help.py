"""Repository-backed help browser for the PCM desktop GUI."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


class PCMHelpDialog(QDialog):
    """Show the same Markdown guide that lives beside the PCM GUI source."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("PCM GUI help")
        self.resize(900, 720)

        self.guide_path = Path(__file__).resolve().parents[1] / "README.md"

        layout = QVBoxLayout(self)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setMarkdown(self.guide_path.read_text(encoding="utf-8"))
        layout.addWidget(self.browser, stretch=1)

        buttons = QHBoxLayout()
        open_button = QPushButton("Open README")
        open_button.clicked.connect(self.open_readme)
        buttons.addWidget(open_button)
        buttons.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

    def open_readme(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.guide_path)))
