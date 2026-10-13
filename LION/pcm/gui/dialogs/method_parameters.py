"""Editable reconstruction-method settings dialog."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from LION.pcm.gui.help_metadata import dataclass_help
from LION.pcm.gui.widgets.dataclass_editor import DataclassEditorWidget
from LION.pcm.parameter_sweep import ParameterSweep


class MethodParameterDialog(QDialog):
    """Edit one method dataclass and optional parameter sweeps."""

    def __init__(
        self,
        method_id: str,
        config: object,
        sweeps: dict[str, ParameterSweep] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Method settings - {method_id}")
        self.resize(760, 720)

        layout = QVBoxLayout(self)

        title_row = QHBoxLayout()
        title = QLabel(method_id)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        title_row.addWidget(title)
        class_help = dataclass_help(config)
        if class_help:
            help_button = QPushButton("?")
            help_button.setFixedWidth(28)
            help_button.setToolTip(
                "Show help from the configuration dataclass docstring"
            )
            help_button.clicked.connect(
                lambda: QMessageBox.information(self, f"{method_id} help", class_help)
            )
            title_row.addWidget(help_button)
        title_row.addStretch(1)
        layout.addLayout(title_row)

        self.editor = DataclassEditorWidget(config, sweeps)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.editor)
        layout.addWidget(scroll, stretch=1)

        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._config = config
        self._sweeps = sweeps or {}

    def validate_and_accept(self) -> None:
        try:
            self._config = self.editor.updated_config()
            self._sweeps = self.editor.parameter_sweeps()
        except Exception as exc:
            self.error_label.setText(str(exc))
            return
        self.accept()

    def config(self) -> object:
        return self._config

    def sweeps(self) -> dict[str, ParameterSweep]:
        return self._sweeps
