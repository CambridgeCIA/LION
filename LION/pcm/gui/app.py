"""Application entry point for the PCM desktop GUI."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from LION.pcm.gui.main_window import PCMGui


def main() -> None:
    """Run the PCM GUI."""
    app = QApplication(sys.argv)
    gui = PCMGui()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
