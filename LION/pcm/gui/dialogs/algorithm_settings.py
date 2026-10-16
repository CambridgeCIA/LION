"""Algorithm settings dialog for the PCM GUI."""

from __future__ import annotations

import dataclasses
import pprint

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)


def _configuration_text(config: object) -> str:
    """Return a readable representation of one algorithm configuration."""
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return pprint.pformat(dataclasses.asdict(config), sort_dicts=False, width=100)
    if hasattr(config, "__dict__"):
        return pprint.pformat(vars(config), sort_dicts=False, width=100)
    return repr(config)


class AlgorithmSettingsDialog(QDialog):
    """Subwindow showing the current reconstruction-method configuration."""

    def __init__(
        self, algorithm: str, config: object, parent: QWidget | None = None
    ) -> None:
        """Build the algorithm settings dialog."""
        super().__init__(parent)
        self.setWindowTitle(f"Algorithm settings - {algorithm}")
        self.resize(650, 500)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Current configuration for {algorithm}"))

        config_text = QPlainTextEdit()
        config_text.setReadOnly(True)
        config_text.setPlainText(_configuration_text(config))
        layout.addWidget(config_text)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
