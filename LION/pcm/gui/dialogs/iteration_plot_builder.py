"""Interactive builder for PnP iteration/time metric comparison plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PySide6.QtCore import QEvent, QSettings, Qt
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from LION.pcm.gui.styles import BLUE_SCROLLBAR_STYLE
from LION.pcm.output_paths import with_datetime_prefix
from LION.pcm.pnp_iteration_metrics import (
    METRIC_LABELS,
    PlotSeries,
    PnPIterationMetricsResult,
    make_metric_comparison_figure,
    method_display_label,
)

_SETTINGS_ORG = "CambridgeCIA"
_SETTINGS_APP = "LION PCM"
_SETTINGS_PREFIX = "pnp_iteration_plot_builder"


def _setting_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _setting_json(value, default):
    if value in (None, ""):
        return default
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return default


def _resize_within_screen(widget: QWidget, width: int, height: int) -> None:
    screen = widget.screen() or QApplication.primaryScreen()
    if screen is None:
        widget.resize(width, height)
        return
    available = screen.availableGeometry()
    widget.resize(
        min(width, int(available.width() * 0.9)),
        min(height, int(available.height() * 0.9)),
    )


def _figure_pixmap(figure) -> QPixmap:
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba()).copy()
    height, width, _ = rgba.shape
    image = QImage(
        rgba.data,
        width,
        height,
        rgba.strides[0],
        QImage.Format.Format_RGBA8888,
    ).copy()
    return QPixmap.fromImage(image)


class IterationPlotBuilderDialog(QDialog):
    """Choose result series and export a customised metric comparison figure."""

    def __init__(
        self,
        result: PnPIterationMetricsResult,
        *,
        initial_x_axis: str = "iteration",
        initially_selected: set[tuple[str, str]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.result = result
        self.row_keys: list[tuple[str, str]] = []
        self.current_pixmap: QPixmap | None = None
        self._updating = True
        self._settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        self._restore_series_state = initially_selected is None
        self._saved_selected_series = {
            tuple(item)
            for item in _setting_json(
                self._settings.value(f"{_SETTINGS_PREFIX}/selected_series"),
                [],
            )
            if isinstance(item, list) and len(item) == 2
        }
        self._saved_legend_labels = _setting_json(
            self._settings.value(f"{_SETTINGS_PREFIX}/legend_labels"),
            {},
        )

        self.setWindowTitle("PnP plot builder")
        _resize_within_screen(self, 1220, 800)

        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        controls_scroll = QScrollArea()
        controls_scroll.setStyleSheet(BLUE_SCROLLBAR_STYLE)
        controls_scroll.setWidgetResizable(True)
        controls_widget = QWidget()
        controls = QVBoxLayout(controls_widget)
        controls_scroll.setWidget(controls_widget)
        splitter.addWidget(controls_scroll)

        self.preview_scroll = QScrollArea()
        self.preview_scroll.setStyleSheet(BLUE_SCROLLBAR_STYLE)
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.preview_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.preview_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_scroll.viewport().installEventFilter(self)
        self.preview_scroll.setToolTip(
            "Use the mouse wheel or vertical trackpad scroll to zoom. Use the scroll bars to pan."
        )
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(1, 1)
        self.preview_scroll.setWidget(self.preview_label)
        splitter.addWidget(self.preview_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([420, 780])

        controls.addWidget(QLabel("<b>Reconstructions</b>"))

        self.series_table = QTableWidget()
        self.series_table.setColumnCount(3)
        self.series_table.setHorizontalHeaderLabels(
            ["Show", "Reconstruction", "Legend label"]
        )
        self.series_table.verticalHeader().setVisible(False)
        self.series_table.setMinimumHeight(190)
        header = self.series_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        controls.addWidget(self.series_table)

        series_buttons = QHBoxLayout()
        select_all_button = QPushButton("Select all")
        select_all_button.clicked.connect(lambda: self._set_all_series_checked(True))
        series_buttons.addWidget(select_all_button)
        deselect_all_button = QPushButton("Deselect all")
        deselect_all_button.clicked.connect(lambda: self._set_all_series_checked(False))
        series_buttons.addWidget(deselect_all_button)
        series_buttons.addStretch(1)
        controls.addLayout(series_buttons)

        self._populate_series(initially_selected)

        form = QFormLayout()

        metrics_widget = QWidget()
        metrics_layout = QHBoxLayout(metrics_widget)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        self.metric_checkboxes: dict[str, QCheckBox] = {}
        saved_metrics = _setting_json(
            self._settings.value(f"{_SETTINGS_PREFIX}/metrics"),
            result.comparison_config.plot.metrics,
        )
        initial_metrics = set(saved_metrics)
        for metric in ("psnr", "ssim", "haarpsi", "mse"):
            checkbox = QCheckBox(METRIC_LABELS[metric])
            checkbox.setChecked(metric in initial_metrics)
            metrics_layout.addWidget(checkbox)
            self.metric_checkboxes[metric] = checkbox
        form.addRow("Metrics", metrics_widget)

        self.x_axis_combo = QComboBox()
        self.x_axis_combo.addItems(["iteration", "time"])
        saved_x_axis = str(
            self._settings.value(f"{_SETTINGS_PREFIX}/x_axis", initial_x_axis)
        )
        self.x_axis_combo.setCurrentText(
            saved_x_axis
            if self._restore_series_state and saved_x_axis in {"iteration", "time"}
            else initial_x_axis
        )
        form.addRow("X axis", self.x_axis_combo)

        self.time_origin_combo = QComboBox()
        self.time_origin_combo.addItem("Reconstruction start", "reconstruction-start")
        self.time_origin_combo.addItem("First recorded point", "first-point")
        saved_time_origin = str(
            self._settings.value(
                f"{_SETTINGS_PREFIX}/time_origin", "reconstruction-start"
            )
        )
        saved_time_origin_index = self.time_origin_combo.findData(saved_time_origin)
        if saved_time_origin_index >= 0:
            self.time_origin_combo.setCurrentIndex(saved_time_origin_index)
        self.time_origin_combo.setToolTip(
            "Reconstruction start keeps absolute cumulative wall-clock time. "
            "First recorded point subtracts each series' first recorded time so curves align at zero."
        )
        self.time_origin_combo.setEnabled(self.x_axis_combo.currentText() == "time")
        form.addRow("Time origin", self.time_origin_combo)

        self.title_edit = QLineEdit(
            str(
                self._settings.value(
                    f"{_SETTINGS_PREFIX}/title",
                    "PnP reconstruction comparison",
                )
            )
        )
        form.addRow("Title", self.title_edit)

        self.legend_position_combo = QComboBox()
        self.legend_position_combo.addItems(["right", "bottom", "top", "none"])
        saved_legend_position = str(
            self._settings.value(f"{_SETTINGS_PREFIX}/legend_position", "right")
        )
        if saved_legend_position in {"right", "bottom", "top", "none"}:
            self.legend_position_combo.setCurrentText(saved_legend_position)
        form.addRow("Common legend", self.legend_position_combo)

        self.legend_columns_spin = QSpinBox()
        self.legend_columns_spin.setRange(1, 12)
        self.legend_columns_spin.setValue(
            int(self._settings.value(f"{_SETTINGS_PREFIX}/legend_columns", 1))
        )
        form.addRow("Legend columns", self.legend_columns_spin)

        self.figure_width_spin = QDoubleSpinBox()
        self.figure_width_spin.setRange(4.0, 30.0)
        self.figure_width_spin.setDecimals(1)
        self.figure_width_spin.setSingleStep(0.5)
        self.figure_width_spin.setValue(
            float(self._settings.value(f"{_SETTINGS_PREFIX}/figure_width", 10.0))
        )
        form.addRow("Figure width (in)", self.figure_width_spin)

        self.figure_height_spin = QDoubleSpinBox()
        self.figure_height_spin.setRange(3.0, 30.0)
        self.figure_height_spin.setDecimals(1)
        self.figure_height_spin.setSingleStep(0.5)
        self.figure_height_spin.setValue(
            float(self._settings.value(f"{_SETTINGS_PREFIX}/figure_height", 7.0))
        )
        form.addRow("Figure height (in)", self.figure_height_spin)

        self.display_dpi_spin = QSpinBox()
        self.display_dpi_spin.setRange(50, 400)
        self.display_dpi_spin.setValue(
            int(self._settings.value(f"{_SETTINGS_PREFIX}/display_dpi", 120))
        )
        form.addRow("Display DPI", self.display_dpi_spin)

        zoom_widget = QWidget()
        zoom_layout = QHBoxLayout(zoom_widget)
        zoom_layout.setContentsMargins(0, 0, 0, 0)
        zoom_out_button = QPushButton("-")
        zoom_out_button.setToolTip("Zoom out")
        zoom_out_button.clicked.connect(lambda: self._change_zoom(-25))
        zoom_layout.addWidget(zoom_out_button)
        self.zoom_percent_spin = QSpinBox()
        self.zoom_percent_spin.setRange(10, 400)
        self.zoom_percent_spin.setSingleStep(25)
        self.zoom_percent_spin.setSuffix("%")
        self.zoom_percent_spin.setValue(
            int(self._settings.value(f"{_SETTINGS_PREFIX}/zoom_percent", 100))
        )
        zoom_layout.addWidget(self.zoom_percent_spin)
        zoom_in_button = QPushButton("+")
        zoom_in_button.setToolTip("Zoom in")
        zoom_in_button.clicked.connect(lambda: self._change_zoom(25))
        zoom_layout.addWidget(zoom_in_button)
        fit_button = QPushButton("Fit")
        fit_button.clicked.connect(self.fit_preview)
        zoom_layout.addWidget(fit_button)
        actual_button = QPushButton("100%")
        actual_button.clicked.connect(lambda: self.zoom_percent_spin.setValue(100))
        zoom_layout.addWidget(actual_button)
        zoom_layout.addStretch(1)
        form.addRow("Preview zoom", zoom_widget)

        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(50, 1200)
        self.dpi_spin.setValue(
            int(
                self._settings.value(
                    f"{_SETTINGS_PREFIX}/save_dpi",
                    max(300, result.comparison_config.plot.dpi),
                )
            )
        )
        form.addRow("Save DPI", self.dpi_spin)

        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(0.2, 8.0)
        self.line_width_spin.setDecimals(1)
        self.line_width_spin.setSingleStep(0.2)
        self.line_width_spin.setValue(
            float(self._settings.value(f"{_SETTINGS_PREFIX}/line_width", 1.8))
        )
        form.addRow("Line width", self.line_width_spin)

        appearance_widget = QWidget()
        appearance_layout = QHBoxLayout(appearance_widget)
        appearance_layout.setContentsMargins(0, 0, 0, 0)
        self.markers_checkbox = QCheckBox("Markers")
        self.markers_checkbox.setChecked(
            _setting_bool(
                self._settings.value(f"{_SETTINGS_PREFIX}/markers"),
                False,
            )
        )
        self.grid_checkbox = QCheckBox("Grid")
        self.grid_checkbox.setChecked(
            _setting_bool(
                self._settings.value(f"{_SETTINGS_PREFIX}/grid"),
                True,
            )
        )
        appearance_layout.addWidget(self.markers_checkbox)
        appearance_layout.addWidget(self.grid_checkbox)
        appearance_layout.addStretch(1)
        form.addRow("Appearance", appearance_widget)

        controls.addLayout(form)

        refresh_button = QPushButton("Refresh preview")
        refresh_button.clicked.connect(self.refresh_preview)
        controls.addWidget(refresh_button)

        controls.addWidget(QLabel("<b>Save</b>"))
        output_row = QHBoxLayout()
        default_path = (
            result.output_root
            / "plots"
            / "custom"
            / f"comparison_vs_{initial_x_axis}.png"
        )
        self.output_path_edit = QLineEdit(str(default_path))
        output_row.addWidget(self.output_path_edit, stretch=1)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.choose_output_path)
        output_row.addWidget(browse_button)
        controls.addLayout(output_row)

        self.prefix_datetime_checkbox = QCheckBox(
            "Prefix filename with current datetime"
        )
        self.prefix_datetime_checkbox.setChecked(
            _setting_bool(
                self._settings.value(f"{_SETTINGS_PREFIX}/prefix_datetime"),
                False,
            )
        )
        controls.addWidget(self.prefix_datetime_checkbox)

        save_row = QHBoxLayout()
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_plot)
        save_row.addWidget(self.save_button)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        save_row.addWidget(self.status_label, stretch=1)
        controls.addLayout(save_row)
        controls.addStretch(1)

        self._connect_dynamic_updates()
        self._updating = False
        self.refresh_preview()

    def _populate_series(self, initially_selected: set[tuple[str, str]] | None) -> None:
        rows: list[tuple[str, str, str]] = []
        for scenario in self.result.scenarios:
            for method_id in self.result.comparison_config.method_ids:
                method_label = method_display_label(
                    self.result.comparison_config, method_id
                )
                rows.append(
                    (scenario.id, method_id, f"{method_label} | {scenario.label}")
                )

        self.series_table.setRowCount(len(rows))
        for row, (scenario_id, method_id, label) in enumerate(rows):
            key = (scenario_id, method_id)
            self.row_keys.append(key)

            show_item = QTableWidgetItem()
            show_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            if initially_selected is not None:
                should_select = key in initially_selected
            elif self._saved_selected_series:
                should_select = key in self._saved_selected_series
            else:
                should_select = True
            show_item.setCheckState(
                Qt.CheckState.Checked if should_select else Qt.CheckState.Unchecked
            )
            self.series_table.setItem(row, 0, show_item)

            name_item = QTableWidgetItem(label)
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.series_table.setItem(row, 1, name_item)

            legend_key = f"{scenario_id}::{method_id}"
            legend_label = self._saved_legend_labels.get(legend_key, label)
            self.series_table.setItem(row, 2, QTableWidgetItem(legend_label))

    def _connect_dynamic_updates(self) -> None:
        self.series_table.itemChanged.connect(self._refresh_if_ready)
        for checkbox in self.metric_checkboxes.values():
            checkbox.toggled.connect(self._refresh_if_ready)
        self.x_axis_combo.currentTextChanged.connect(self._x_axis_changed)
        self.time_origin_combo.currentIndexChanged.connect(self._refresh_if_ready)
        self.zoom_percent_spin.valueChanged.connect(self._update_preview)
        self.title_edit.editingFinished.connect(self._refresh_if_ready)
        self.legend_position_combo.currentTextChanged.connect(self._refresh_if_ready)
        self.legend_columns_spin.valueChanged.connect(self._refresh_if_ready)
        self.figure_width_spin.valueChanged.connect(self._refresh_if_ready)
        self.figure_height_spin.valueChanged.connect(self._refresh_if_ready)
        self.display_dpi_spin.valueChanged.connect(self._refresh_if_ready)
        self.line_width_spin.valueChanged.connect(self._refresh_if_ready)
        self.markers_checkbox.toggled.connect(self._refresh_if_ready)
        self.grid_checkbox.toggled.connect(self._refresh_if_ready)

    def _refresh_if_ready(self, *_args) -> None:
        if not self._updating:
            self.refresh_preview()

    def _x_axis_changed(self, x_axis: str) -> None:
        self.time_origin_combo.setEnabled(x_axis == "time")
        path = Path(self.output_path_edit.text()).expanduser()
        if path.suffix:
            path = path.with_name(f"comparison_vs_{x_axis}{path.suffix}")
            self.output_path_edit.setText(str(path))
        self._refresh_if_ready()

    def _set_all_series_checked(self, checked: bool) -> None:
        self._updating = True
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row in range(self.series_table.rowCount()):
            self.series_table.item(row, 0).setCheckState(state)
        self._updating = False
        self.refresh_preview()

    def _selected_series(self) -> list[PlotSeries]:
        series: list[PlotSeries] = []
        for row, key in enumerate(self.row_keys):
            if self.series_table.item(row, 0).checkState() != Qt.CheckState.Checked:
                continue

            scenario_id, method_id = key
            label = self.series_table.item(row, 2).text().strip()
            if not label:
                label = self.series_table.item(row, 1).text()

            series.append(
                PlotSeries(
                    label=label,
                    metrics=self.result.metrics[(scenario_id, method_id)],
                    snapshots=self.result.snapshots[(scenario_id, method_id)],
                )
            )
        return series

    def _selected_metrics(self) -> list[str]:
        return [
            metric
            for metric, checkbox in self.metric_checkboxes.items()
            if checkbox.isChecked()
        ]

    def _build_figure(self, *, preview: bool):
        dpi = self.display_dpi_spin.value() if preview else self.dpi_spin.value()
        return make_metric_comparison_figure(
            self._selected_series(),
            metrics=self._selected_metrics(),
            x_axis=self.x_axis_combo.currentText(),
            title=self.title_edit.text().strip(),
            figsize=(self.figure_width_spin.value(), self.figure_height_spin.value()),
            dpi=dpi,
            line_width=self.line_width_spin.value(),
            show_markers=self.markers_checkbox.isChecked(),
            grid=self.grid_checkbox.isChecked(),
            legend_position=self.legend_position_combo.currentText(),
            legend_columns=self.legend_columns_spin.value(),
            time_origin=self.time_origin_combo.currentData(),
        )

    def refresh_preview(self) -> None:
        try:
            figure = self._build_figure(preview=True)
            self.current_pixmap = _figure_pixmap(figure)
            plt.close(figure)
            self.status_label.clear()
            self._update_preview()
        except Exception as exc:
            self.current_pixmap = None
            self.preview_label.clear()
            self.preview_label.setText(str(exc))

    def choose_output_path(self) -> None:
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save PnP comparison plot",
            self.output_path_edit.text(),
            "PNG image (*.png);;PDF document (*.pdf);;SVG image (*.svg)",
        )
        if path:
            self.output_path_edit.setText(path)

    def save_plot(self) -> None:
        try:
            output_path = Path(self.output_path_edit.text()).expanduser()
            if not output_path.suffix:
                output_path = output_path.with_suffix(".png")
                self.output_path_edit.setText(str(output_path))

            if output_path.suffix.lower() not in {".png", ".pdf", ".svg"}:
                raise ValueError("Output extension must be .png, .pdf, or .svg.")

            output_path = with_datetime_prefix(
                output_path,
                self.prefix_datetime_checkbox.isChecked(),
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure = self._build_figure(preview=False)
            figure.savefig(output_path, dpi=self.dpi_spin.value(), bbox_inches="tight")
            plt.close(figure)
            self.status_label.setText(f"Saved: {output_path}")
        except Exception as exc:
            self.status_label.setText(f"Could not save: {exc}")

    def eventFilter(self, watched, event) -> bool:
        """Use vertical mouse-wheel/trackpad movement to zoom the preview."""
        if (
            watched is self.preview_scroll.viewport()
            and event.type() == QEvent.Type.Wheel
        ):
            angle_delta = event.angleDelta().y()
            pixel_delta = event.pixelDelta().y()

            if angle_delta:
                delta_percent = 15 if angle_delta > 0 else -15
            elif pixel_delta:
                # High-resolution trackpads report pixel deltas. Keep this smooth
                # but noticeable rather than jumping a full mouse-wheel notch.
                magnitude = max(1, min(10, abs(int(pixel_delta)) // 2 or 1))
                delta_percent = magnitude if pixel_delta > 0 else -magnitude
            else:
                return super().eventFilter(watched, event)

            self._change_zoom(delta_percent)
            event.accept()
            return True

        return super().eventFilter(watched, event)

    def closeEvent(self, event) -> None:
        """Persist plot-builder customisations for the next window/session."""
        self._save_persistent_state()
        super().closeEvent(event)

    def _save_persistent_state(self) -> None:
        selected_series: list[list[str]] = []
        legend_labels: dict[str, str] = {}
        for row, (scenario_id, method_id) in enumerate(self.row_keys):
            if self.series_table.item(row, 0).checkState() == Qt.CheckState.Checked:
                selected_series.append([scenario_id, method_id])
            legend_labels[f"{scenario_id}::{method_id}"] = self.series_table.item(
                row, 2
            ).text()

        values = {
            "selected_series": json.dumps(selected_series),
            "legend_labels": json.dumps(legend_labels),
            "metrics": json.dumps(self._selected_metrics()),
            "x_axis": self.x_axis_combo.currentText(),
            "time_origin": self.time_origin_combo.currentData(),
            "title": self.title_edit.text(),
            "legend_position": self.legend_position_combo.currentText(),
            "legend_columns": self.legend_columns_spin.value(),
            "figure_width": self.figure_width_spin.value(),
            "figure_height": self.figure_height_spin.value(),
            "display_dpi": self.display_dpi_spin.value(),
            "zoom_percent": self.zoom_percent_spin.value(),
            "save_dpi": self.dpi_spin.value(),
            "line_width": self.line_width_spin.value(),
            "markers": self.markers_checkbox.isChecked(),
            "grid": self.grid_checkbox.isChecked(),
            "prefix_datetime": self.prefix_datetime_checkbox.isChecked(),
        }
        for key, value in values.items():
            self._settings.setValue(f"{_SETTINGS_PREFIX}/{key}", value)
        self._settings.sync()

    def _change_zoom(self, delta_percent: int) -> None:
        """Change preview zoom while preserving the scrollable canvas."""
        self.zoom_percent_spin.setValue(self.zoom_percent_spin.value() + delta_percent)

    def fit_preview(self) -> None:
        """Fit the current preview inside the visible scroll-area viewport."""
        if self.current_pixmap is None:
            return
        viewport = self.preview_scroll.viewport().size()
        if viewport.width() <= 1 or viewport.height() <= 1:
            return
        width_scale = max(1, viewport.width() - 8) / self.current_pixmap.width()
        height_scale = max(1, viewport.height() - 8) / self.current_pixmap.height()
        percent = int(round(100.0 * min(width_scale, height_scale)))
        self.zoom_percent_spin.setValue(max(10, min(400, percent)))

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)

    def _update_preview(self, *_args) -> None:
        if self.current_pixmap is None:
            return
        scale = self.zoom_percent_spin.value() / 100.0
        width = max(1, int(round(self.current_pixmap.width() * scale)))
        height = max(1, int(round(self.current_pixmap.height() * scale)))
        scaled = self.current_pixmap.scaled(
            width,
            height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)
        self.preview_label.resize(scaled.size())
