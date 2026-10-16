"""Generic dataclass editor widgets for the PCM GUI."""

from __future__ import annotations

import ast
import copy
import dataclasses
import types
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from LION.pcm.gui.help_metadata import field_help
from LION.pcm.parameter_sweep import ParameterSweep, set_nested_value


def _resolved_type(owner: object, field: dataclasses.Field) -> object:
    try:
        return get_type_hints(type(owner)).get(field.name, field.type)
    except Exception:
        return field.type


def _allows_none(annotation: object) -> bool:
    return type(None) in get_args(annotation)


def _non_none_types(annotation: object) -> tuple[object, ...]:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        return tuple(arg for arg in get_args(annotation) if arg is not type(None))
    return (annotation,)


def _scalar_type(annotation: object, current: object) -> object:
    if annotation is Any:
        return type(current) if current is not None else str

    origin = get_origin(annotation)
    if origin is Literal:
        values = get_args(annotation)
        return type(values[0]) if values else str

    candidates = _non_none_types(annotation)
    if Path in candidates:
        return Path
    if len(candidates) == 1:
        return candidates[0]
    if current is not None:
        return type(current)
    return str


def _parse_bool(text: str) -> bool:
    normalized = text.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean, got {text!r}.")


def parse_editor_value(text: str, annotation: object, current: object = None) -> object:
    """Parse text according to a dataclass field annotation."""
    stripped = text.strip()
    if _allows_none(annotation) and stripped.lower() in {"", "none"}:
        return None

    scalar_type = _scalar_type(annotation, current)

    if scalar_type is bool:
        return _parse_bool(stripped)
    if scalar_type is int:
        return int(stripped)
    if scalar_type is float:
        return float(stripped)
    if scalar_type is Path:
        return Path(stripped).expanduser()
    if scalar_type is str:
        return stripped
    if isinstance(scalar_type, type) and issubclass(scalar_type, Enum):
        return scalar_type[stripped]

    origin = get_origin(annotation)
    if origin in (list, tuple, dict, set) or scalar_type in (list, tuple, dict, set):
        return ast.literal_eval(stripped)

    if current is not None:
        try:
            return type(current)(stripped)
        except Exception:
            pass
    return stripped


def _format_value(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, Path):
        return str(value)
    return str(value)


class PathEditorWidget(QWidget):
    """Edit a filesystem path with an optional file browser."""

    def __init__(self, value: Path | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.line_edit = QLineEdit("" if value is None else str(value))
        layout.addWidget(self.line_edit, stretch=1)

        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse)
        layout.addWidget(browse_button)

    def browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select file", self.line_edit.text()
        )
        if path:
            self.line_edit.setText(path)

    def value(self) -> Path:
        return Path(self.line_edit.text()).expanduser()


class SweepValuesDialog(QDialog):
    """Edit an explicit list of sweep values for one scalar parameter."""

    def __init__(
        self,
        path: str,
        annotation: object,
        current: object,
        values: list[object],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.annotation = annotation
        self.current = current
        self.parsed_values = list(values)

        self.setWindowTitle(f"Sweep - {path}")
        self.resize(520, 180)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Values for {path} (comma-separated)"))

        self.values_edit = QLineEdit(
            ", ".join(_format_value(value) for value in values)
        )
        self.values_edit.setPlaceholderText("e.g. 0.01, 0.03, 0.05")
        layout.addWidget(self.values_edit)

        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)

        buttons_row = QHBoxLayout()
        clear_button = QPushButton("Clear sweep")
        clear_button.clicked.connect(self.clear_sweep)
        buttons_row.addWidget(clear_button)
        buttons_row.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        buttons_row.addWidget(buttons)
        layout.addLayout(buttons_row)

    def clear_sweep(self) -> None:
        self.values_edit.clear()
        self.parsed_values = []
        self.accept()

    def validate_and_accept(self) -> None:
        try:
            parts = [
                part.strip()
                for part in self.values_edit.text().split(",")
                if part.strip()
            ]
            self.parsed_values = [
                parse_editor_value(part, self.annotation, self.current)
                for part in parts
            ]
        except Exception as exc:
            self.error_label.setText(str(exc))
            return
        self.accept()


class DataclassEditorWidget(QWidget):
    """Recursively edit one dataclass instance and optional scalar sweeps."""

    def __init__(
        self,
        config: object,
        sweeps: dict[str, ParameterSweep] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not dataclasses.is_dataclass(config) or isinstance(config, type):
            raise TypeError("DataclassEditorWidget requires a dataclass instance.")

        self.original_config = copy.deepcopy(config)
        self.sweeps = copy.deepcopy(sweeps or {})
        self.bindings: dict[str, tuple[QWidget, object, object]] = {}
        self.sweep_labels: dict[str, QLabel] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._add_dataclass(layout, config, prefix="")
        layout.addStretch(1)

    def _add_dataclass(
        self, parent_layout: QVBoxLayout, config: object, prefix: str
    ) -> None:
        form = QFormLayout()

        for field in dataclasses.fields(config):
            value = getattr(config, field.name)
            path = f"{prefix}.{field.name}" if prefix else field.name
            annotation = _resolved_type(config, field)

            help_text = field_help(config, field)

            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                group = QGroupBox(field.name)
                if help_text:
                    group.setToolTip(help_text)
                group_layout = QVBoxLayout(group)
                self._add_dataclass(group_layout, value, path)
                form.addRow(group)
                continue

            editor = self._make_editor(value, annotation)
            self.bindings[path] = (editor, annotation, value)

            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(editor, stretch=1)

            if self._is_sweepable(annotation, value):
                sweep_button = QPushButton("Sweep...")
                sweep_button.clicked.connect(
                    lambda _checked=False, p=path: self._edit_sweep(p)
                )
                row_layout.addWidget(sweep_button)

                sweep_label = QLabel("")
                sweep_label.setWordWrap(True)
                row_layout.addWidget(sweep_label)
                self.sweep_labels[path] = sweep_label
                self._update_sweep_label(path)

            if help_text:
                label_widget = QWidget()
                label_layout = QHBoxLayout(label_widget)
                label_layout.setContentsMargins(0, 0, 0, 0)
                label_layout.addWidget(QLabel(field.name))
                help_button = QToolButton()
                help_button.setText("?")
                help_button.setToolTip(help_text)
                help_button.setAutoRaise(True)
                help_button.clicked.connect(
                    lambda _checked=False, title=field.name, message=help_text: QMessageBox.information(
                        self, title, message
                    )
                )
                label_layout.addWidget(help_button)
                label_layout.addStretch(1)
                form.addRow(label_widget, row_widget)
            else:
                form.addRow(field.name, row_widget)

        parent_layout.addLayout(form)

    def _make_editor(self, value: object, annotation: object) -> QWidget:
        origin = get_origin(annotation)
        if origin is Literal:
            combo = QComboBox()
            choices = list(get_args(annotation))
            combo.addItems([str(choice) for choice in choices])
            if value is not None:
                combo.setCurrentText(str(value))
            return combo

        scalar_type = _scalar_type(annotation, value)
        if scalar_type is bool and value is not None:
            checkbox = QCheckBox()
            checkbox.setChecked(bool(value))
            return checkbox

        if isinstance(scalar_type, type) and issubclass(scalar_type, Enum):
            combo = QComboBox()
            combo.addItems([member.name for member in scalar_type])
            if value is not None:
                combo.setCurrentText(value.name)
            return combo

        if scalar_type is int and value is not None and not _allows_none(annotation):
            spin = QSpinBox()
            spin.setRange(-2_147_483_648, 2_147_483_647)
            spin.setValue(int(value))
            return spin

        if scalar_type is Path and not _allows_none(annotation):
            return PathEditorWidget(value if isinstance(value, Path) else None)

        editor = QLineEdit(_format_value(value))
        return editor

    @staticmethod
    def _is_sweepable(annotation: object, value: object) -> bool:
        origin = get_origin(annotation)
        if origin is Literal:
            return True
        scalar_type = _scalar_type(annotation, value)
        return scalar_type in {int, float, bool, str}

    def _edit_sweep(self, path: str) -> None:
        _widget, annotation, current = self.bindings[path]
        current_sweep = self.sweeps.get(path)
        dialog = SweepValuesDialog(
            path,
            annotation,
            current,
            [] if current_sweep is None else current_sweep.values,
            self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        if dialog.parsed_values:
            self.sweeps[path] = ParameterSweep(path=path, values=dialog.parsed_values)
        else:
            self.sweeps.pop(path, None)
        self._update_sweep_label(path)

    def _update_sweep_label(self, path: str) -> None:
        label = self.sweep_labels[path]
        sweep = self.sweeps.get(path)
        if sweep is None:
            label.setText("")
            return
        label.setText(f"{len(sweep.values)} values")

    def _editor_value(
        self, widget: QWidget, annotation: object, current: object
    ) -> object:
        if isinstance(widget, QCheckBox):
            return widget.isChecked()

        if isinstance(widget, QComboBox):
            text = widget.currentText()
            scalar_type = _scalar_type(annotation, current)
            if isinstance(scalar_type, type) and issubclass(scalar_type, Enum):
                return scalar_type[text]
            choices = get_args(annotation)
            for choice in choices:
                if str(choice) == text:
                    return choice
            return text

        if isinstance(widget, QSpinBox):
            return widget.value()

        if isinstance(widget, PathEditorWidget):
            return widget.value()

        if isinstance(widget, QLineEdit):
            return parse_editor_value(widget.text(), annotation, current)

        raise TypeError(f"Unsupported editor widget: {type(widget).__name__}")

    def updated_config(self) -> object:
        """Return a deep-copied config containing the edited fixed values."""
        config = copy.deepcopy(self.original_config)
        for path, (widget, annotation, current) in self.bindings.items():
            value = self._editor_value(widget, annotation, current)
            set_nested_value(config, path, value)
        return config

    def parameter_sweeps(self) -> dict[str, ParameterSweep]:
        """Return the currently configured parameter sweeps."""
        return copy.deepcopy(self.sweeps)
