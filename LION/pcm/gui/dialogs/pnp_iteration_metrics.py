"""PnP iteration-analysis dialog for the PCM GUI."""

from __future__ import annotations

import copy
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from LION.experiments.pcm_presets.presets import PRESETS
from LION.pcm.dataset_single_sample import (
    is_synthetic_dataset,
    select_synthetic_sample,
    with_iteration_methods,
)
from LION.pcm.gui.backend import available_devices
from LION.pcm.gui.dialogs.image_viewer import ImageViewerDialog, ImageViewerItem
from LION.pcm.gui.dialogs.iteration_plot_builder import IterationPlotBuilderDialog
from LION.pcm.gui.dialogs.method_parameters import MethodParameterDialog
from LION.pcm.gui.styles import BLUE_SCROLLBAR_STYLE
from LION.pcm.gui.widgets.collapsible_section import CollapsibleSection
from LION.pcm.parameter_sweep import ParameterSweep, expand_parameter_sweeps
from LION.pcm.pnp_analysis_io import (
    ANALYSIS_SCHEMA,
    ANALYSIS_SCHEMA_VERSION,
    apply_config_mapping,
    config_to_mapping,
    load_analysis_definition,
    save_analysis_definition,
)
from LION.pcm.pnp_iteration_metrics import (
    PlotSeries,
    PnPIterationMetricsParameter,
    PnPIterationMetricsResult,
    PnPIterationPlotParameter,
    PnPProgressUpdate,
    available_iteration_method_ids,
    method_display_label,
    resolve_snapshots,
    run_pnp_iteration_metrics_analysis,
)
from LION.pcm.reconstruct import ScalarTVParameter

DEFAULT_PRESET = (
    "cigs_example_256x256" if "cigs_example_256x256" in PRESETS else sorted(PRESETS)[0]
)


def _iteration_preset(preset_name: str):
    """Use dataset image/scaling settings, with PnP methods exposed for this dialog."""
    return with_iteration_methods(PRESETS[preset_name])


def _default_sample_name(preset_name: str) -> str:
    """Return a stable, readable sample label for one preset."""
    if preset_name == "e27_pos1_average_map_simulated_noise":
        return "e27_pos1_average_map"
    return preset_name


@dataclass
class MethodConfigEntry:
    """One independently configured reconstruction method in the GUI."""

    key: str
    base_method_id: str
    label: str
    config: object
    sweeps: dict[str, ParameterSweep]


class IntListEditor(QWidget):
    """Edit a short list of integer values with Add/Remove controls."""

    values_changed = Signal()

    def __init__(
        self,
        values: list[int],
        *,
        minimum: int,
        maximum: int = 2_147_483_647,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        add_row = QHBoxLayout()
        self.value_spin = QSpinBox()
        self.value_spin.setRange(minimum, maximum)
        add_row.addWidget(self.value_spin)

        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_current_value)
        add_row.addWidget(add_button)

        remove_button = QPushButton("Remove selected")
        remove_button.clicked.connect(self.remove_selected)
        add_row.addWidget(remove_button)

        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear_values)
        add_row.addWidget(clear_button)
        add_row.addStretch(1)
        layout.addLayout(add_row)

        self.list_widget = QListWidget()
        self.list_widget.setMaximumHeight(90)
        layout.addWidget(self.list_widget)

        self.set_values(values)

    def values(self) -> list[int]:
        """Return values in their displayed order."""
        return [
            int(self.list_widget.item(index).text())
            for index in range(self.list_widget.count())
        ]

    def set_values(self, values: list[int]) -> None:
        """Replace all values."""
        self.list_widget.clear()
        for value in values:
            if value not in self.values():
                self.list_widget.addItem(str(value))
        self.values_changed.emit()

    def add_current_value(self) -> None:
        value = self.value_spin.value()
        if value not in self.values():
            self.list_widget.addItem(str(value))
            self.values_changed.emit()

    def remove_selected(self) -> None:
        for item in self.list_widget.selectedItems():
            self.list_widget.takeItem(self.list_widget.row(item))
        self.values_changed.emit()

    def clear_values(self) -> None:
        self.list_widget.clear()
        self.values_changed.emit()


class PnPIterationMetricsWorker(QThread):
    """Run a PnP iteration comparison without blocking Qt."""

    result_ready = Signal(object)
    progress = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        comparison_config: PnPIterationMetricsParameter,
        experiment_config,
        analysis_definition: dict,
    ) -> None:
        super().__init__()
        self.comparison_config = comparison_config
        self.experiment_config = experiment_config
        self.analysis_definition = analysis_definition

    def run(self) -> None:
        try:
            result = run_pnp_iteration_metrics_analysis(
                self.comparison_config,
                self.experiment_config,
                progress_callback=self.progress.emit,
                analysis_definition=self.analysis_definition,
            )
        except Exception:
            details = traceback.format_exc()
            print(details, file=sys.stderr)
            self.failed.emit(details)
            return
        self.result_ready.emit(result)


def _resize_within_screen(widget: QWidget, width: int, height: int) -> None:
    """Set an initial size that fits comfortably within the available screen."""
    screen = widget.screen() or QApplication.primaryScreen()
    if screen is None:
        widget.resize(width, height)
        return

    available = screen.availableGeometry()
    widget.resize(
        min(width, int(available.width() * 0.9)),
        min(height, int(available.height() * 0.9)),
    )


class PnPIterationMetricsDialog(QDialog):
    """Configure, run, and inspect PnP iteration comparisons."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.worker: PnPIterationMetricsWorker | None = None
        self.method_entries: list[MethodConfigEntry] = []
        self.next_method_instance: dict[str, int] = {}
        self.method_variant_labels: dict[str, QLabel] = {}
        self.analysis_result: PnPIterationMetricsResult | None = None
        self.result_windows: list[QDialog] = []

        self.setWindowTitle("PnP iteration analysis")
        _resize_within_screen(self, 820, 940)

        layout = QVBoxLayout(self)

        title = QLabel("PnP iteration analysis")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        self.content_splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(self.content_splitter, stretch=1)

        settings_scroll = QScrollArea()
        settings_scroll.setStyleSheet(BLUE_SCROLLBAR_STYLE)
        settings_scroll.setWidgetResizable(True)
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        settings_scroll.setWidget(settings_widget)
        self.content_splitter.addWidget(settings_scroll)

        results_scroll = QScrollArea()
        results_scroll.setStyleSheet(BLUE_SCROLLBAR_STYLE)
        results_scroll.setWidgetResizable(True)
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_scroll.setWidget(results_widget)
        self.content_splitter.addWidget(results_scroll)
        self.content_splitter.setStretchFactor(0, 3)
        self.content_splitter.setStretchFactor(1, 1)

        experiment_section = CollapsibleSection("Experiment", expanded=True)
        experiment_form = QFormLayout()

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(sorted(PRESETS))
        self.preset_combo.setCurrentText(DEFAULT_PRESET)
        self.preset_combo.currentTextChanged.connect(self.update_preset_controls)
        experiment_form.addRow("Preset", self.preset_combo)

        self.device_combo = QComboBox()
        self.device_combo.addItems(available_devices())
        experiment_form.addRow("Device", self.device_combo)

        self.sample_name_edit = QLineEdit(_default_sample_name(DEFAULT_PRESET))
        experiment_form.addRow("Sample name", self.sample_name_edit)

        self.dataset_category_label = QLabel("Dataset category")
        self.dataset_category_combo = QComboBox()
        self.dataset_category_combo.currentTextChanged.connect(
            self._sync_dataset_sample_name
        )
        experiment_form.addRow(self.dataset_category_label, self.dataset_category_combo)

        self.dataset_sample_id_label = QLabel("Dataset sample ID")
        self.dataset_sample_id_spin = QSpinBox()
        self.dataset_sample_id_spin.setRange(1, 9999)
        self.dataset_sample_id_spin.setValue(288)
        self.dataset_sample_id_spin.valueChanged.connect(self._sync_dataset_sample_name)
        experiment_form.addRow(
            self.dataset_sample_id_label, self.dataset_sample_id_spin
        )

        self.data_root_edit = QLineEdit()
        experiment_form.addRow("Data root", self.data_root_edit)

        self.sampling_spin = QDoubleSpinBox()
        self.sampling_spin.setRange(0.001, 1.0)
        self.sampling_spin.setSingleStep(0.01)
        self.sampling_spin.setDecimals(3)
        experiment_form.addRow("Sampling ratio", self.sampling_spin)

        self.coarse_j_spin = QSpinBox()
        self.coarse_j_spin.setMinimum(0)
        experiment_form.addRow("Coarse J", self.coarse_j_spin)

        self.output_root_edit = QLineEdit()
        experiment_form.addRow("Output root", self.output_root_edit)

        self.output_name_edit = QLineEdit("cia_pnp_iteration_metrics")
        experiment_form.addRow("Output name", self.output_name_edit)

        self.prefix_datetime_checkbox = QCheckBox(
            "Prefix output name with current datetime"
        )
        self.prefix_datetime_checkbox.setChecked(False)
        experiment_form.addRow("Datetime prefix", self.prefix_datetime_checkbox)

        plan_widget = QWidget()
        plan_layout = QHBoxLayout(plan_widget)
        plan_layout.setContentsMargins(0, 0, 0, 0)
        import_button = QPushButton("Import YAML...")
        import_button.clicked.connect(self.import_analysis_yaml)
        plan_layout.addWidget(import_button)
        export_button = QPushButton("Export YAML...")
        export_button.clicked.connect(self.export_analysis_yaml)
        plan_layout.addWidget(export_button)
        plan_layout.addStretch(1)
        experiment_form.addRow("Analysis setup", plan_widget)

        experiment_section.set_content_layout(experiment_form)
        settings_layout.addWidget(experiment_section)

        methods_section = CollapsibleSection("Methods", expanded=True)
        methods_layout = QVBoxLayout()

        self.available_method_ids: list[str] = []
        self.methods_layout = QVBoxLayout()
        methods_layout.addLayout(self.methods_layout)
        methods_section.set_content_layout(methods_layout)
        settings_layout.addWidget(methods_section)

        scenarios_section = CollapsibleSection("Noise scenarios", expanded=True)
        scenarios_form = QFormLayout()
        self.include_no_noise_checkbox = QCheckBox("Include no-noise acquisition")
        self.include_no_noise_checkbox.setChecked(True)
        self.include_no_noise_checkbox.toggled.connect(self.update_run_count)
        scenarios_form.addRow("No noise", self.include_no_noise_checkbox)
        self.noise_realizations_editor = IntListEditor([0], minimum=0)
        self.noise_realizations_editor.values_changed.connect(self.update_run_count)
        scenarios_form.addRow("Noise realizations", self.noise_realizations_editor)
        self.match_dataset_noise_checkbox = QCheckBox(
            "Use original dataset noise for realization 0"
        )
        self.match_dataset_noise_checkbox.setChecked(True)
        self.match_dataset_noise_checkbox.setToolTip(
            "Use the same RNG convention as run_dataset_experiment for noise realization 0."
        )
        scenarios_form.addRow("Dataset noise", self.match_dataset_noise_checkbox)
        self.match_dataset_noise_label = scenarios_form.labelForField(
            self.match_dataset_noise_checkbox
        )
        scenarios_section.set_content_layout(scenarios_form)
        settings_layout.addWidget(scenarios_section)

        plot_section = CollapsibleSection("Plots", expanded=True)
        plot_form = QFormLayout()

        # Previous default: [1, 5, 10, 25, 50, 100]
        self.snapshot_iterations_editor = IntListEditor(
            [1, 5, 10, 20, 30, 50], minimum=1
        )
        plot_form.addRow("Snapshot iterations", self.snapshot_iterations_editor)

        self.save_slide_export_checkbox = QCheckBox(
            "Save slide images and CSVs (includes GT, zero-filled, SPGL1 and TV)"
        )
        plot_form.addRow("Slide export", self.save_slide_export_checkbox)

        every_widget = QWidget()
        every_layout = QHBoxLayout(every_widget)
        every_layout.setContentsMargins(0, 0, 0, 0)
        self.snapshot_every_checkbox = QCheckBox("Every")
        self.snapshot_every_spin = QSpinBox()
        self.snapshot_every_spin.setRange(1, 100_000)
        self.snapshot_every_spin.setValue(5)
        self.snapshot_every_spin.setEnabled(False)
        self.snapshot_every_checkbox.toggled.connect(
            self.snapshot_every_spin.setEnabled
        )
        every_layout.addWidget(self.snapshot_every_checkbox)
        every_layout.addWidget(self.snapshot_every_spin)
        every_layout.addWidget(QLabel("iterations"))
        every_layout.addStretch(1)
        plot_form.addRow("Periodic snapshots", every_widget)

        best_widget = QWidget()
        best_layout = QHBoxLayout(best_widget)
        best_layout.setContentsMargins(0, 0, 0, 0)
        self.show_best_checkbox = QCheckBox("Show best")
        self.show_best_checkbox.setChecked(True)
        self.best_metric_combo = QComboBox()
        self.best_metric_combo.addItems(["psnr", "ssim", "haarpsi", "mse"])
        best_layout.addWidget(self.show_best_checkbox)
        best_layout.addWidget(self.best_metric_combo)
        best_layout.addStretch(1)
        plot_form.addRow("Best snapshot", best_widget)

        self.show_last_checkbox = QCheckBox("Show last")
        self.show_last_checkbox.setChecked(True)
        plot_form.addRow("Last snapshot", self.show_last_checkbox)

        self.show_early_stop_checkbox = QCheckBox("Show GS-PnP early-stop point")
        self.show_early_stop_checkbox.setChecked(True)
        self.show_early_stop_checkbox.setToolTip(
            "For GS-PnP, keep the first reconstruction satisfying the configured "
            "convergence threshold, even when the method is configured to continue."
        )
        plot_form.addRow("Early-stop snapshot", self.show_early_stop_checkbox)

        self.show_backtracking_stop_checkbox = QCheckBox(
            "Show GS-PnP backtracking-stop point"
        )
        self.show_backtracking_stop_checkbox.setChecked(True)
        self.show_backtracking_stop_checkbox.setToolTip(
            "Keep the last accepted GS-PnP reconstruction if backtracking cannot "
            "find another sufficient-decrease step."
        )
        plot_form.addRow(
            "Backtracking-stop snapshot", self.show_backtracking_stop_checkbox
        )

        metrics_widget = QWidget()
        metrics_layout = QGridLayout(metrics_widget)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        self.metric_checkboxes: dict[str, QCheckBox] = {}
        for index, metric in enumerate(("psnr", "ssim", "haarpsi", "mse")):
            checkbox = QCheckBox(metric.upper() if metric != "haarpsi" else "HaarPSI")
            checkbox.setChecked(True)
            metrics_layout.addWidget(checkbox, index // 2, index % 2)
            self.metric_checkboxes[metric] = checkbox
        plot_form.addRow("Metrics", metrics_widget)

        x_axes_widget = QWidget()
        x_axes_layout = QHBoxLayout(x_axes_widget)
        x_axes_layout.setContentsMargins(0, 0, 0, 0)
        self.iteration_axis_checkbox = QCheckBox("Iteration")
        self.time_axis_checkbox = QCheckBox("Reconstruction time")
        self.iteration_axis_checkbox.setChecked(True)
        self.time_axis_checkbox.setChecked(True)
        x_axes_layout.addWidget(self.iteration_axis_checkbox)
        x_axes_layout.addWidget(self.time_axis_checkbox)
        x_axes_layout.addStretch(1)
        plot_form.addRow("X axes", x_axes_widget)

        modes_widget = QWidget()
        modes_layout = QGridLayout(modes_widget)
        modes_layout.setContentsMargins(0, 0, 0, 0)
        self.individual_checkbox = QCheckBox("Individual")
        self.methods_per_scenario_checkbox = QCheckBox("Methods per scenario")
        self.scenarios_per_method_checkbox = QCheckBox("Scenarios per method")
        for checkbox in (
            self.individual_checkbox,
            self.methods_per_scenario_checkbox,
            self.scenarios_per_method_checkbox,
        ):
            checkbox.setChecked(True)
        modes_layout.addWidget(self.individual_checkbox, 0, 0)
        modes_layout.addWidget(self.methods_per_scenario_checkbox, 0, 1)
        modes_layout.addWidget(self.scenarios_per_method_checkbox, 1, 0, 1, 2)
        plot_form.addRow("Plot modes", modes_widget)

        self.auto_save_plots_checkbox = QCheckBox(
            "Auto-save standard plots after reconstruction (slower)"
        )
        self.auto_save_plots_checkbox.setChecked(False)
        self.auto_save_plots_checkbox.setToolTip(
            "When disabled, iteration metrics are still saved, but expensive standard PNG/PDF "
            "plots are skipped. Use Custom plot after the run to save only the figures needed."
        )
        plot_form.addRow("Automatic saving", self.auto_save_plots_checkbox)

        self.snapshot_position_combo = QComboBox()
        self.snapshot_position_combo.addItems(["top", "bottom", "left", "right"])
        plot_form.addRow("Snapshot position", self.snapshot_position_combo)

        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(50, 1200)
        self.dpi_spin.setValue(300)
        plot_form.addRow("DPI", self.dpi_spin)

        plot_section.set_content_layout(plot_form)
        settings_layout.addWidget(plot_section)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run analysis")
        self.run_button.clicked.connect(self.start_analysis)
        run_row.addWidget(self.run_button)

        self.run_count_label = QLabel("")
        run_row.addWidget(self.run_count_label)

        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        run_row.addWidget(self.status_label, stretch=1)

        self.error_details_button = QPushButton("Show error details...")
        self.error_details_button.setVisible(False)
        self.error_details_button.clicked.connect(self.show_error_details)
        run_row.addWidget(self.error_details_button)
        settings_layout.addLayout(run_row)

        progress_form = QFormLayout()
        self.overall_progress_bar = QProgressBar()
        self.overall_progress_bar.setRange(0, 1)
        self.overall_progress_bar.setValue(0)
        self.overall_progress_bar.setFormat("%v / %m iterations (%p%)")
        progress_form.addRow("Overall", self.overall_progress_bar)

        self.current_progress_bar = QProgressBar()
        self.current_progress_bar.setRange(0, 1)
        self.current_progress_bar.setValue(0)
        self.current_progress_bar.setFormat("%v / %m iterations (%p%)")
        progress_form.addRow("Current", self.current_progress_bar)

        self.progress_detail_label = QLabel("Not started")
        self.progress_detail_label.setWordWrap(True)
        progress_form.addRow("Progress", self.progress_detail_label)

        self.progress_time_label = QLabel("")
        progress_form.addRow("Time", self.progress_time_label)
        settings_layout.addLayout(progress_form)

        settings_layout.addStretch(1)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(
            self.result_label.textInteractionFlags()
        )
        results_layout.addWidget(self.result_label)

        self.results_section = CollapsibleSection("Results", expanded=False)
        self.results_layout = QVBoxLayout()
        self.results_section.set_content_layout(self.results_layout)
        results_layout.addWidget(self.results_section)
        results_layout.addStretch(1)

        self.content_splitter.setSizes([700, 220])
        self.update_preset_controls(DEFAULT_PRESET)

    def _sync_dataset_sample_name(self, *_args) -> None:
        """Use the dataset runner's sample name so the sampling seed stays consistent."""
        if not is_synthetic_dataset(PRESETS[self.preset_combo.currentText()]):
            return
        category = self.dataset_category_combo.currentText()
        if category:
            self.sample_name_edit.setText(
                f"artifacts/{category}/sample_{self.dataset_sample_id_spin.value():04d}"
            )
            previous_output_name = self.output_name_edit.text()
            if (
                previous_output_name == "cia_pnp_iteration_metrics"
                or previous_output_name.startswith("cia_pnp_slide_")
            ):
                self.output_name_edit.setText(
                    f"cia_pnp_slide_{category}_{self.dataset_sample_id_spin.value():04d}"
                )

    def update_preset_controls(self, preset_name: str) -> None:
        """Load defaults and available iteration methods from one preset."""
        config = _iteration_preset(preset_name)
        is_dataset = is_synthetic_dataset(config)
        self.dataset_category_combo.blockSignals(True)
        self.dataset_category_combo.clear()
        if is_dataset:
            self.dataset_category_combo.addItems(config.data.categories)
        self.dataset_category_combo.blockSignals(False)
        for widget in (
            self.dataset_category_label,
            self.dataset_category_combo,
            self.dataset_sample_id_label,
            self.dataset_sample_id_spin,
            self.match_dataset_noise_checkbox,
            self.match_dataset_noise_label,
        ):
            widget.setVisible(is_dataset)
        if is_dataset:
            self.dataset_sample_id_spin.setValue(288)
            self._sync_dataset_sample_name()
            self.save_slide_export_checkbox.setChecked(True)
        else:
            self.save_slide_export_checkbox.setChecked(False)
            self.sample_name_edit.setText(_default_sample_name(preset_name))
            self.output_name_edit.setText("cia_pnp_iteration_metrics")
        self.data_root_edit.setText(str(config.data.root))
        self.sampling_spin.setValue(config.sampling.sampling_ratios[0])
        coarse_j = (
            config.sampling.coarse_j_values[0]
            if config.sampling.coarse_j_values is not None
            else 0
        )
        self.coarse_j_spin.setMaximum(
            config.data.j_order if config.data.j_order is not None else 20
        )
        self.coarse_j_spin.setValue(coarse_j)
        self.output_root_edit.setText(str(config.runtime.root_output_dir))

        method_ids = available_iteration_method_ids(config)
        self.available_method_ids = list(method_ids)

        self.method_entries = []
        self.next_method_instance = {}
        self._clear_layout(self.methods_layout)

        if method_ids:
            default_method = (
                "pnp_admm_drunet" if "pnp_admm_drunet" in method_ids else method_ids[0]
            )
            self._append_method_configuration(default_method)
            if (
                is_dataset
                and "pnp_gs_deepinv_drunet" in method_ids
                and default_method != "pnp_gs_deepinv_drunet"
            ):
                self._append_method_configuration("pnp_gs_deepinv_drunet")
        else:
            self.methods_layout.addWidget(
                QLabel("No supported iteration methods in this preset.")
            )

        self.update_run_count()

    def _append_method_configuration(
        self,
        base_method_id: str,
        *,
        config: object | None = None,
        sweeps: dict[str, ParameterSweep] | None = None,
        label: str | None = None,
    ) -> None:
        """Add one independently configurable method instance."""
        preset = _iteration_preset(self.preset_combo.currentText())
        if base_method_id not in preset.methods:
            raise KeyError(
                f"Method {base_method_id!r} is not available in the selected preset."
            )

        instance = self.next_method_instance.get(base_method_id, 1)
        self.next_method_instance[base_method_id] = instance + 1

        method_config = copy.deepcopy(
            preset.methods[base_method_id] if config is None else config
        )
        if (
            config is None
            and (
                self.preset_combo.currentText() == "cigs_example_256x256"
                or is_synthetic_dataset(preset)
            )
            and base_method_id in ("pnp_admm_drunet", "pnp_gs_deepinv_drunet")
        ):
            method_config.iters = 200
            if base_method_id == "pnp_gs_deepinv_drunet":
                method_config.continue_after_early_stop = True

        entry = MethodConfigEntry(
            key=f"{base_method_id}__config_{instance}",
            base_method_id=base_method_id,
            label=label or f"{base_method_id} #{instance}",
            config=method_config,
            sweeps=copy.deepcopy(sweeps or {}),
        )
        self.method_entries.append(entry)
        self._rebuild_method_rows()
        self.update_run_count()

    def add_method_configuration(self) -> None:
        """Append one configuration using the preferred default method type."""
        if not self.available_method_ids:
            return
        method_id = (
            "pnp_admm_drunet"
            if "pnp_admm_drunet" in self.available_method_ids
            else self.available_method_ids[0]
        )
        self._append_method_configuration(method_id)

    def duplicate_method_configuration(self, entry_key: str) -> None:
        """Duplicate one hand-picked method configuration."""
        entry = self._method_entry(entry_key)
        self._append_method_configuration(
            entry.base_method_id,
            config=entry.config,
            sweeps=entry.sweeps,
        )

    def remove_method_configuration(self, entry_key: str) -> None:
        """Remove one independent method configuration."""
        self.method_entries = [
            entry for entry in self.method_entries if entry.key != entry_key
        ]
        self._rebuild_method_rows()
        self.update_run_count()

    def change_method_configuration(self, entry_key: str, base_method_id: str) -> None:
        """Change one card's method type and reset it to that preset's default config."""
        entry = self._method_entry(entry_key)
        if base_method_id == entry.base_method_id:
            return

        preset = _iteration_preset(self.preset_combo.currentText())
        instance = self.next_method_instance.get(base_method_id, 1)
        self.next_method_instance[base_method_id] = instance + 1

        entry.key = f"{base_method_id}__config_{instance}"
        entry.base_method_id = base_method_id
        entry.label = f"{base_method_id} #{instance}"
        entry.config = copy.deepcopy(preset.methods[base_method_id])
        if (
            self.preset_combo.currentText() == "cigs_example_256x256"
            or is_synthetic_dataset(preset)
        ) and base_method_id in (
            "pnp_admm_drunet",
            "pnp_gs_deepinv_drunet",
        ):
            entry.config.iters = 200
            if base_method_id == "pnp_gs_deepinv_drunet":
                entry.config.continue_after_early_stop = True
        entry.sweeps = {}
        self._rebuild_method_rows()
        self.update_run_count()

    def move_method_configuration(self, entry_key: str, offset: int) -> None:
        """Move one configuration card up or down."""
        index = next(
            index
            for index, entry in enumerate(self.method_entries)
            if entry.key == entry_key
        )
        new_index = index + offset
        if new_index < 0 or new_index >= len(self.method_entries):
            return
        self.method_entries[index], self.method_entries[new_index] = (
            self.method_entries[new_index],
            self.method_entries[index],
        )
        self._rebuild_method_rows()

    def _method_entry(self, entry_key: str) -> MethodConfigEntry:
        for entry in self.method_entries:
            if entry.key == entry_key:
                return entry
        raise KeyError(entry_key)

    def _rebuild_method_rows(self) -> None:
        """Refresh the ordered list of independent method configurations."""
        self._clear_layout(self.methods_layout)
        self.method_variant_labels = {}

        if not self.method_entries:
            self.methods_layout.addWidget(QLabel("No method configurations added."))

        for index, entry in enumerate(self.method_entries):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            method_combo = QComboBox()
            method_combo.addItems(self.available_method_ids)
            method_combo.setCurrentText(entry.base_method_id)
            method_combo.currentTextChanged.connect(
                lambda method_id, key=entry.key: self.change_method_configuration(
                    key, method_id
                )
            )
            row_layout.addWidget(method_combo, stretch=1)

            sweeps = list(entry.sweeps.values())
            count = len(
                expand_parameter_sweeps(
                    entry.key,
                    entry.config,
                    sweeps,
                    base_label=entry.label,
                )
            )
            variant_label = QLabel("1 run" if count == 1 else f"{count} sweep runs")
            row_layout.addWidget(variant_label)
            self.method_variant_labels[entry.key] = variant_label

            up_button = QPushButton("Up")
            up_button.setEnabled(index > 0)
            up_button.clicked.connect(
                lambda _checked=False, key=entry.key: self.move_method_configuration(
                    key, -1
                )
            )
            row_layout.addWidget(up_button)

            down_button = QPushButton("Down")
            down_button.setEnabled(index < len(self.method_entries) - 1)
            down_button.clicked.connect(
                lambda _checked=False, key=entry.key: self.move_method_configuration(
                    key, 1
                )
            )
            row_layout.addWidget(down_button)

            settings_button = QPushButton("Settings...")
            settings_button.clicked.connect(
                lambda _checked=False, key=entry.key: self.show_method_settings(key)
            )
            row_layout.addWidget(settings_button)

            duplicate_button = QPushButton("Duplicate")
            duplicate_button.clicked.connect(
                lambda _checked=False, key=entry.key: self.duplicate_method_configuration(
                    key
                )
            )
            row_layout.addWidget(duplicate_button)

            remove_button = QPushButton("Remove")
            remove_button.clicked.connect(
                lambda _checked=False, key=entry.key: self.remove_method_configuration(
                    key
                )
            )
            row_layout.addWidget(remove_button)

            self.methods_layout.addWidget(row_widget)

        add_method_button = QPushButton("+ Add configuration")
        add_method_button.clicked.connect(self.add_method_configuration)
        self.methods_layout.addWidget(add_method_button)

    def show_method_settings(self, entry_key: str) -> None:
        """Open the generic dataclass editor for one reconstruction method instance."""
        entry = self._method_entry(entry_key)
        dialog = MethodParameterDialog(
            entry.label,
            entry.config,
            entry.sweeps,
            self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        entry.config = dialog.config()
        entry.sweeps = dialog.sweeps()
        self._rebuild_method_rows()
        self.update_run_count()

    def _selected_variant_count(self) -> int:
        count = 0
        for entry in self.method_entries:
            count += len(
                expand_parameter_sweeps(
                    entry.key,
                    entry.config,
                    list(entry.sweeps.values()),
                    base_label=entry.label,
                )
            )
        return count

    def update_run_count(self) -> None:
        scenarios = int(self.include_no_noise_checkbox.isChecked()) + len(
            self.noise_realizations_editor.values()
        )
        variants = self._selected_variant_count()
        self.run_count_label.setText(f"Planned reconstructions: {variants * scenarios}")

    def analysis_definition(self) -> dict:
        """Return the re-importable GUI analysis plan as plain YAML-safe values."""
        comparison_config, _experiment_config = self.make_configs()
        return {
            "schema": ANALYSIS_SCHEMA,
            "version": ANALYSIS_SCHEMA_VERSION,
            "preset": self.preset_combo.currentText(),
            "experiment": {
                "device": self.device_combo.currentText(),
                "sample_name": self.sample_name_edit.text().strip(),
                "dataset_category": (
                    self.dataset_category_combo.currentText()
                    if is_synthetic_dataset(PRESETS[self.preset_combo.currentText()])
                    else None
                ),
                "dataset_sample_id": (
                    self.dataset_sample_id_spin.value()
                    if is_synthetic_dataset(PRESETS[self.preset_combo.currentText()])
                    else None
                ),
                "data_root": self.data_root_edit.text().strip(),
                "sampling_ratio": self.sampling_spin.value(),
                "coarse_j": self.coarse_j_spin.value(),
                "output_root": self.output_root_edit.text().strip(),
            },
            "comparison": {
                "output_name": comparison_config.output_name,
                "prefix_datetime": comparison_config.prefix_datetime,
                "include_no_noise": comparison_config.include_no_noise,
                "noise_realizations": list(comparison_config.noise_realizations),
                "match_dataset_default_noise": comparison_config.match_dataset_default_noise,
                "plot": {
                    "snapshot_iterations": list(
                        comparison_config.plot.snapshot_iterations
                    ),
                    "snapshot_every": comparison_config.plot.snapshot_every,
                    "save_slide_export": comparison_config.plot.save_slide_export,
                    "show_best": comparison_config.plot.show_best,
                    "best_metric": comparison_config.plot.best_metric,
                    "show_last": comparison_config.plot.show_last,
                    "show_early_stop": comparison_config.plot.show_early_stop,
                    "show_backtracking_stop": comparison_config.plot.show_backtracking_stop,
                    "metrics": list(comparison_config.plot.metrics),
                    "x_axes": list(comparison_config.plot.x_axes),
                    "modes": list(comparison_config.plot.modes),
                    "snapshot_position": comparison_config.plot.snapshot_position,
                    "dpi": comparison_config.plot.dpi,
                    "save_automatic_plots": comparison_config.plot.save_automatic_plots,
                },
            },
            "methods": [
                {
                    "base_method_id": entry.base_method_id,
                    "label": entry.label,
                    "config": config_to_mapping(entry.config),
                    "sweeps": [
                        {"path": sweep.path, "values": list(sweep.values)}
                        for sweep in entry.sweeps.values()
                    ],
                }
                for entry in self.method_entries
            ],
        }

    def export_analysis_yaml(self) -> None:
        """Export the current PnP analysis setup without running it."""
        try:
            definition = self.analysis_definition()
        except Exception as exc:
            self.status_label.setText(f"Cannot export settings: {exc}")
            return

        suggested = (
            Path(self.output_root_edit.text()).expanduser()
            / f"{self.output_name_edit.text().strip() or 'analysis'}.yaml"
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export PnP iteration analysis",
            str(suggested),
            "YAML files (*.yaml *.yml)",
        )
        if not path:
            return
        try:
            saved_path = save_analysis_definition(Path(path), definition)
        except Exception as exc:
            self.status_label.setText(f"Failed to export YAML: {exc}")
            return
        self.status_label.setText(f"Exported {saved_path}")

    def import_analysis_yaml(self) -> None:
        """Restore a previously exported analysis setup for further editing."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import PnP iteration analysis",
            self.output_root_edit.text(),
            "YAML files (*.yaml *.yml)",
        )
        if not path:
            return
        try:
            definition = load_analysis_definition(Path(path))
            self.apply_analysis_definition(definition)
        except Exception as exc:
            self.status_label.setText(f"Failed to import YAML: {exc}")
            return
        self.status_label.setText(f"Imported {path}")

    def apply_analysis_definition(self, definition: dict) -> None:
        """Apply one validated analysis YAML definition to the GUI controls."""
        preset_name = definition["preset"]
        if preset_name not in PRESETS:
            raise ValueError(
                f"Preset {preset_name!r} is not available in this checkout."
            )

        self.preset_combo.setCurrentText(preset_name)
        preset = _iteration_preset(preset_name)
        experiment = definition.get("experiment", {})
        comparison = definition.get("comparison", {})
        plot = comparison.get("plot", {})

        device = str(experiment.get("device", self.device_combo.currentText()))
        if self.device_combo.findText(device) < 0:
            self.device_combo.addItem(device)
        self.device_combo.setCurrentText(device)
        if is_synthetic_dataset(preset):
            category = experiment.get("dataset_category") or "CIGS"
            self.dataset_category_combo.setCurrentText(str(category))
            self.dataset_sample_id_spin.setValue(
                int(experiment.get("dataset_sample_id") or 288)
            )
        self.sample_name_edit.setText(
            str(experiment.get("sample_name", self.sample_name_edit.text()))
        )
        self.data_root_edit.setText(str(experiment.get("data_root", preset.data.root)))
        self.sampling_spin.setValue(
            float(experiment.get("sampling_ratio", preset.sampling.sampling_ratios[0]))
        )
        coarse_default = (
            preset.sampling.coarse_j_values[0] if preset.sampling.coarse_j_values else 0
        )
        self.coarse_j_spin.setValue(int(experiment.get("coarse_j", coarse_default)))
        self.output_root_edit.setText(
            str(experiment.get("output_root", preset.runtime.root_output_dir))
        )

        self.output_name_edit.setText(
            str(comparison.get("output_name", "cia_pnp_iteration_metrics"))
        )
        self.prefix_datetime_checkbox.setChecked(
            bool(comparison.get("prefix_datetime", False))
        )
        self.include_no_noise_checkbox.setChecked(
            bool(comparison.get("include_no_noise", True))
        )
        self.noise_realizations_editor.set_values(
            [int(v) for v in comparison.get("noise_realizations", [0])]
        )
        self.match_dataset_noise_checkbox.setChecked(
            bool(
                comparison.get(
                    "match_dataset_default_noise", is_synthetic_dataset(preset)
                )
            )
        )

        # Previous import fallback: [1, 5, 10, 25, 50, 100]
        self.snapshot_iterations_editor.set_values(
            [int(v) for v in plot.get("snapshot_iterations", [1, 5, 10, 20, 30, 50])]
        )
        self.save_slide_export_checkbox.setChecked(
            bool(plot.get("save_slide_export", False))
        )
        snapshot_every = int(plot.get("snapshot_every", 0))
        self.snapshot_every_checkbox.setChecked(snapshot_every > 0)
        if snapshot_every > 0:
            self.snapshot_every_spin.setValue(snapshot_every)
        self.show_best_checkbox.setChecked(bool(plot.get("show_best", True)))
        self.best_metric_combo.setCurrentText(str(plot.get("best_metric", "psnr")))
        self.show_last_checkbox.setChecked(bool(plot.get("show_last", True)))
        self.show_early_stop_checkbox.setChecked(
            bool(plot.get("show_early_stop", True))
        )
        self.show_backtracking_stop_checkbox.setChecked(
            bool(plot.get("show_backtracking_stop", True))
        )

        selected_metrics = set(plot.get("metrics", ["psnr", "ssim", "haarpsi", "mse"]))
        for metric, checkbox in self.metric_checkboxes.items():
            checkbox.setChecked(metric in selected_metrics)
        selected_axes = set(plot.get("x_axes", ["iteration", "time"]))
        self.iteration_axis_checkbox.setChecked("iteration" in selected_axes)
        self.time_axis_checkbox.setChecked("time" in selected_axes)
        selected_modes = set(
            plot.get(
                "modes", ["individual", "methods-per-scenario", "scenarios-per-method"]
            )
        )
        self.individual_checkbox.setChecked("individual" in selected_modes)
        self.methods_per_scenario_checkbox.setChecked(
            "methods-per-scenario" in selected_modes
        )
        self.scenarios_per_method_checkbox.setChecked(
            "scenarios-per-method" in selected_modes
        )
        self.snapshot_position_combo.setCurrentText(
            str(plot.get("snapshot_position", "top"))
        )
        self.dpi_spin.setValue(int(plot.get("dpi", 300)))
        self.auto_save_plots_checkbox.setChecked(
            bool(plot.get("save_automatic_plots", False))
        )

        self.method_entries = []
        self.next_method_instance = {}
        for item in definition["methods"]:
            base_method_id = str(item["base_method_id"])
            if base_method_id not in preset.methods:
                raise ValueError(
                    f"Method {base_method_id!r} from the YAML is not available in preset {preset_name!r}."
                )
            config = apply_config_mapping(
                preset.methods[base_method_id], dict(item.get("config", {}))
            )
            sweeps = {
                str(sweep["path"]): ParameterSweep(
                    path=str(sweep["path"]),
                    values=list(sweep.get("values", [])),
                )
                for sweep in item.get("sweeps", [])
            }
            self._append_method_configuration(
                base_method_id,
                config=config,
                sweeps=sweeps,
                label=str(item.get("label") or "") or None,
            )

        self._rebuild_method_rows()
        self.update_run_count()

    def make_configs(self):
        """Build package dataclasses and concrete parameter-sweep variants."""
        preset_name = self.preset_combo.currentText()
        experiment_config = _iteration_preset(preset_name)
        experiment_config.runtime.device = self.device_combo.currentText()
        experiment_config.data.root = Path(self.data_root_edit.text()).expanduser()
        experiment_config.runtime.root_output_dir = Path(
            self.output_root_edit.text()
        ).expanduser()
        experiment_config.runtime.num_trials = 1
        experiment_config.runtime.num_trials_skip = 0
        experiment_config.sampling.sampling_ratios = [self.sampling_spin.value()]
        experiment_config.sampling.coarse_j_values = [self.coarse_j_spin.value()]
        dataset_category = None
        dataset_sample_id = None
        if is_synthetic_dataset(experiment_config):
            dataset_category = self.dataset_category_combo.currentText()
            dataset_sample_id = self.dataset_sample_id_spin.value()
            select_synthetic_sample(
                experiment_config, dataset_category, dataset_sample_id
            )
        experiment_config.log.save_metrics = True
        experiment_config.log.save_plots = self.auto_save_plots_checkbox.isChecked()

        if not self.method_entries:
            raise ValueError("Add at least one method configuration.")

        concrete_methods: dict[str, object] = {}
        method_labels: dict[str, str] = {}
        for entry in self.method_entries:
            variants = expand_parameter_sweeps(
                entry.key,
                entry.config,
                list(entry.sweeps.values()),
                base_label=entry.label,
            )
            for variant in variants:
                if variant.id in concrete_methods:
                    raise ValueError(f"Duplicate method variant ID: {variant.id}")
                concrete_methods[variant.id] = variant.config
                method_labels[variant.id] = variant.label

        experiment_config.methods = concrete_methods
        if self.save_slide_export_checkbox.isChecked():
            source_methods = _iteration_preset(preset_name).methods
            if "spgl1" not in source_methods:
                raise ValueError("The selected preset has no SPGL1 reference method.")
            experiment_config.methods["spgl1"] = copy.deepcopy(source_methods["spgl1"])
            # Scalar TV uses the same sample, sampling mask and measurements.
            experiment_config.methods["scalar_tv"] = ScalarTVParameter(
                lambda_reg=0.001, iters=1024, metric_every=4
            )

        modes = []
        if self.individual_checkbox.isChecked():
            modes.append("individual")
        if self.methods_per_scenario_checkbox.isChecked():
            modes.append("methods-per-scenario")
        if self.scenarios_per_method_checkbox.isChecked():
            modes.append("scenarios-per-method")
        if not modes:
            raise ValueError("Select at least one plot mode.")

        metrics = [
            metric
            for metric, checkbox in self.metric_checkboxes.items()
            if checkbox.isChecked()
        ]
        if not metrics:
            raise ValueError("Select at least one metric.")

        x_axes = []
        if self.iteration_axis_checkbox.isChecked():
            x_axes.append("iteration")
        if self.time_axis_checkbox.isChecked():
            x_axes.append("time")
        if not x_axes:
            raise ValueError("Select at least one x-axis.")

        comparison_config = PnPIterationMetricsParameter(
            method_ids=list(concrete_methods),
            method_labels=method_labels,
            sample_name=self.sample_name_edit.text().strip(),
            output_name=self.output_name_edit.text().strip(),
            category=dataset_category,
            sample_id=dataset_sample_id,
            match_dataset_default_noise=dataset_category is not None
            and self.match_dataset_noise_checkbox.isChecked(),
            prefix_datetime=self.prefix_datetime_checkbox.isChecked(),
            include_no_noise=self.include_no_noise_checkbox.isChecked(),
            noise_realizations=self.noise_realizations_editor.values(),
            plot=PnPIterationPlotParameter(
                snapshot_iterations=self.snapshot_iterations_editor.values(),
                snapshot_every=self.snapshot_every_spin.value()
                if self.snapshot_every_checkbox.isChecked()
                else 0,
                save_slide_export=self.save_slide_export_checkbox.isChecked(),
                show_best=self.show_best_checkbox.isChecked(),
                best_metric=self.best_metric_combo.currentText(),
                show_last=self.show_last_checkbox.isChecked(),
                show_early_stop=self.show_early_stop_checkbox.isChecked(),
                show_backtracking_stop=self.show_backtracking_stop_checkbox.isChecked(),
                metrics=metrics,
                x_axes=x_axes,
                modes=modes,
                snapshot_position=self.snapshot_position_combo.currentText(),
                dpi=self.dpi_spin.value(),
                save_automatic_plots=self.auto_save_plots_checkbox.isChecked(),
            ),
        )
        return comparison_config, experiment_config

    def start_analysis(self) -> None:
        try:
            comparison_config, experiment_config = self.make_configs()
        except Exception as exc:
            self.status_label.setText(f"Invalid settings: {exc}")
            return

        self.run_button.setEnabled(False)
        self.status_label.setText("Starting...")
        self.overall_progress_bar.setRange(0, 1)
        self.overall_progress_bar.setValue(0)
        self.current_progress_bar.setRange(0, 1)
        self.current_progress_bar.setValue(0)
        self.progress_detail_label.setText("Preparing analysis...")
        self.progress_time_label.clear()
        self.result_label.clear()
        self.last_error_details = ""
        self.error_details_button.setVisible(False)
        self.analysis_result = None
        self._clear_layout(self.results_layout)

        try:
            analysis_definition = self.analysis_definition()
        except Exception as exc:
            self.run_button.setEnabled(True)
            self.status_label.setText(f"Cannot serialize analysis settings: {exc}")
            return

        self.worker = PnPIterationMetricsWorker(
            comparison_config,
            experiment_config,
            analysis_definition,
        )
        self.worker.progress.connect(self.show_progress)
        self.worker.result_ready.connect(self.show_result)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()

    @staticmethod
    def _format_duration(seconds: float | None) -> str:
        """Format a short human-readable duration for progress estimates."""
        if seconds is None:
            return "estimating..."
        seconds = max(0, int(round(seconds)))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
        if minutes:
            return f"{minutes:d}m {seconds:02d}s"
        return f"{seconds:d}s"

    def show_progress(self, update: PnPProgressUpdate) -> None:
        """Update GUI progress from the package-level analysis progress stream."""
        self.overall_progress_bar.setRange(0, max(1, update.total_iterations_all))
        self.overall_progress_bar.setValue(update.completed_iterations)

        if update.phase == "running":
            self.current_progress_bar.setRange(0, max(1, update.total_iterations))
            self.current_progress_bar.setValue(update.iteration)
            self.progress_detail_label.setText(
                f"Run {update.current_run}/{update.total_runs}: "
                f"{update.method_label} | {update.scenario_label}"
            )
            self.status_label.setText(
                f"Running {update.current_run}/{update.total_runs}: {update.method_label}"
            )
            remaining = self._format_duration(update.estimated_remaining_seconds)
            self.progress_time_label.setText(
                f"Elapsed: {self._format_duration(update.elapsed_seconds)} | "
                f"Approx. remaining: {remaining}"
            )
            return

        self.current_progress_bar.setRange(0, 1)
        self.current_progress_bar.setValue(1)
        if update.phase == "saving":
            self.progress_detail_label.setText(
                "Reconstruction finished. Saving plots..."
            )
            self.status_label.setText("Saving plots...")
            self.progress_time_label.setText(
                f"Elapsed: {self._format_duration(update.elapsed_seconds)}"
            )
        else:
            self.progress_detail_label.setText("Finished")
            self.status_label.setText("Finished")
            self.progress_time_label.setText(
                f"Total elapsed: {self._format_duration(update.elapsed_seconds)}"
            )

    def show_result(self, result: PnPIterationMetricsResult) -> None:
        """Show links to plots and interactive snapshot viewers."""
        self.analysis_result = result
        self.status_label.setText("Finished")
        timing_text = ""
        if result.phase_timings:
            timing_text = (
                f"\nRuns/data/metric logging: "
                f"{self._format_duration(result.phase_timings.get('runs'))}"
                f" | Automatic plots: "
                f"{self._format_duration(result.phase_timings.get('automatic_plots'))}"
                f" | Total: {self._format_duration(result.phase_timings.get('total'))}"
            )
        self.result_label.setText(f"Results: {result.output_root}{timing_text}")
        self._clear_layout(self.results_layout)

        result_actions = QHBoxLayout()
        output_button = QPushButton("Open output folder")
        output_button.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(result.output_root))
            )
        )
        result_actions.addWidget(output_button)

        custom_plot_button = QPushButton("Custom plot...")
        custom_plot_button.clicked.connect(self.open_custom_plot)
        result_actions.addWidget(custom_plot_button)
        result_actions.addStretch(1)
        self.results_layout.addLayout(result_actions)

        for scenario in result.scenarios:
            for method_id in result.comparison_config.method_ids:
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)

                label = method_display_label(result.comparison_config, method_id)
                metrics = result.metrics[(scenario.id, method_id)]
                snapshots = result.snapshots[(scenario.id, method_id)]
                accepted = metrics.iterations[-1] if metrics.iterations else 0
                requested = snapshots.requested_iterations
                if requested is None:
                    status = snapshots.termination_reason or ""
                else:
                    status = f"{accepted}/{requested} accepted" + (
                        f" | {snapshots.termination_reason}"
                        if snapshots.termination_reason
                        else ""
                    )
                row_layout.addWidget(
                    QLabel(
                        f"{scenario.label} | {label}"
                        + (f" | {status}" if status else "")
                    ),
                    stretch=1,
                )

                snapshots_button = QPushButton("Snapshots...")
                snapshots_button.clicked.connect(
                    lambda _checked=False, sid=scenario.id, mid=method_id: self.open_snapshots(
                        sid, mid
                    )
                )
                row_layout.addWidget(snapshots_button)

                for x_axis in ("iteration", "time"):
                    plot_button = QPushButton(
                        "Iteration plot..." if x_axis == "iteration" else "Time plot..."
                    )
                    plot_button.clicked.connect(
                        lambda _checked=False, sid=scenario.id, mid=method_id, axis=x_axis: self.open_individual_plot(
                            sid, mid, axis
                        )
                    )
                    row_layout.addWidget(plot_button)

                self.results_layout.addWidget(row)

        self.results_section.toggle_button.setChecked(True)

    def open_snapshots(self, scenario_id: str, method_id: str) -> None:
        result = self.analysis_result
        if result is None:
            return

        metrics = result.metrics[(scenario_id, method_id)]
        snapshots = result.snapshots[(scenario_id, method_id)]
        series = PlotSeries(
            label=method_display_label(result.comparison_config, method_id),
            metrics=metrics,
            snapshots=snapshots,
        )
        displays = resolve_snapshots(series, result.comparison_config.plot)
        if not displays:
            self.status_label.setText("No snapshots were selected for this run.")
            return

        items: list[ImageViewerItem] = []
        for display in displays:
            index = metrics.iterations.index(display.iteration)
            summary = (
                f"Iteration: {display.iteration} | "
                f"Time: {metrics.times[index]:.3f} s | "
                f"PSNR: {metrics.psnr[index]:.2f} dB | "
                f"SSIM: {metrics.ssim[index]:.4f} | "
                f"HaarPSI: {metrics.haarpsi[index]:.4f} | "
                f"MSE: {metrics.mse[index]:.3e}"
            )
            items.append(
                ImageViewerItem(
                    title=display.title.replace("\n", " - "),
                    image=display.image,
                    summary=summary,
                )
            )

        scenario = next(item for item in result.scenarios if item.id == scenario_id)
        title = f"{method_display_label(result.comparison_config, method_id)} | {scenario.label}"
        self._show_result_window(
            ImageViewerDialog(title, items, plot_helper=result.plot_helper, parent=self)
        )

    def open_custom_plot(self) -> None:
        """Open a plot builder with all completed reconstructions selected."""
        result = self.analysis_result
        if result is None:
            return
        self._show_result_window(
            IterationPlotBuilderDialog(
                result,
                initial_x_axis="iteration",
                initially_selected=None,
                parent=self,
            )
        )

    def open_individual_plot(
        self, scenario_id: str, method_id: str, x_axis: str
    ) -> None:
        """Open the plot builder starting from one selected reconstruction."""
        result = self.analysis_result
        if result is None:
            return
        self._show_result_window(
            IterationPlotBuilderDialog(
                result,
                initial_x_axis=x_axis,
                initially_selected={(scenario_id, method_id)},
                parent=self,
            )
        )

    def _show_result_window(self, dialog: QDialog) -> None:
        """Open a non-modal result window and keep it alive until closed."""
        dialog.setModal(False)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.result_windows.append(dialog)
        dialog.destroyed.connect(
            lambda _object=None, window=dialog: self._forget_result_window(window)
        )
        dialog.show()

    def _forget_result_window(self, dialog: QDialog) -> None:
        if dialog in self.result_windows:
            self.result_windows.remove(dialog)

    def show_error(self, message: str) -> None:
        self.status_label.setText("Failed")
        self.last_error_details = message
        summary = (
            message.strip().splitlines()[-1] if message.strip() else "Unknown error"
        )
        self.result_label.setText(summary)
        self.error_details_button.setVisible(True)

    def show_error_details(self) -> None:
        """Open the full Python traceback from the failed worker."""
        if not getattr(self, "last_error_details", ""):
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("PnP iteration analysis error details")
        dialog.resize(900, 600)

        layout = QVBoxLayout(dialog)
        details = QPlainTextEdit()
        details.setReadOnly(True)
        details.setPlainText(self.last_error_details)
        layout.addWidget(details, stretch=1)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button)

        self._show_result_window(dialog)

    def worker_finished(self) -> None:
        self.run_button.setEnabled(True)
        self.worker = None

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            child_layout = item.layout()
            if child_layout is not None:
                PnPIterationMetricsDialog._clear_layout(child_layout)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.status_label.setText("Analysis is still running.")
            event.ignore()
            return
        super().closeEvent(event)
