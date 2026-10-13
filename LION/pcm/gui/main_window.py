"""Main window for the PCM desktop GUI."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from LION.experiments.pcm_presets.presets import PRESETS
from LION.pcm.gui.backend import (
    GuiAcquisitionPreview,
    GuiRunRequest,
    GuiRunResult,
    available_devices,
    run_gui_experiment,
)
from LION.pcm.gui.dialogs.algorithm_settings import AlgorithmSettingsDialog
from LION.pcm.gui.widgets.acquisition_settings import AcquisitionSettingsWidget
from LION.pcm.gui.widgets.output_settings import OutputSettingsWidget
from LION.pcm.gui.widgets.result_view import ResultView
from LION.pcm.results import ZERO_FILLED_METHOD_ID

DEFAULT_PRESET = (
    "e27_pos1_average_map_simulated_noise"
    if "e27_pos1_average_map_simulated_noise" in PRESETS
    else sorted(PRESETS)[0]
)


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


class ReconstructionWorker(QThread):
    """Run the reconstruction without blocking the Qt GUI."""

    acquisition_ready = Signal(object)
    result_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, request: GuiRunRequest) -> None:
        """Initialize the reconstruction worker."""
        super().__init__()
        self.request = request

    def run(self) -> None:
        """Run the selected reconstruction."""
        try:
            result = run_gui_experiment(
                self.request,
                acquisition_ready_callback=self.acquisition_ready.emit,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        self.result_ready.emit(result)


class PCMGui(QMainWindow):
    """Desktop GUI for single-sample PCM acquisition and reconstruction."""

    def __init__(self) -> None:
        """Build the GUI."""
        super().__init__()
        self.worker: ReconstructionWorker | None = None
        self.help_dialog = None

        self.setWindowTitle("A3SmartML PCM")
        _resize_within_screen(self, 1200, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        title = QLabel("PCM acquisition and reconstruction")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        basic_controls = QFormLayout()
        layout.addLayout(basic_controls)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(sorted(PRESETS))
        self.preset_combo.setCurrentText(DEFAULT_PRESET)
        self.preset_combo.currentTextChanged.connect(self.update_preset_controls)
        basic_controls.addRow("Preset", self.preset_combo)

        algorithm_widget = QWidget()
        algorithm_layout = QHBoxLayout(algorithm_widget)
        algorithm_layout.setContentsMargins(0, 0, 0, 0)

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.currentTextChanged.connect(self.update_algorithm_controls)
        algorithm_layout.addWidget(self.algorithm_combo, stretch=1)

        self.algorithm_settings_button = QPushButton("Settings...")
        self.algorithm_settings_button.clicked.connect(self.show_algorithm_settings)
        algorithm_layout.addWidget(self.algorithm_settings_button)
        basic_controls.addRow("Algorithm", algorithm_widget)

        self.device_combo = QComboBox()
        self.device_combo.addItems(available_devices())
        basic_controls.addRow("Device", self.device_combo)

        self.acquisition_settings = AcquisitionSettingsWidget()
        layout.addWidget(self.acquisition_settings)

        self.output_settings = OutputSettingsWidget()
        layout.addWidget(self.output_settings)

        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run reconstruction")
        self.run_button.clicked.connect(self.start_reconstruction)
        button_layout.addWidget(self.run_button)

        self.iteration_analysis_button = QPushButton("PnP iteration analysis...")
        self.iteration_analysis_button.clicked.connect(self.show_iteration_analysis)
        button_layout.addWidget(self.iteration_analysis_button)

        self.help_button = QPushButton("Help...")
        self.help_button.clicked.connect(self.show_help)
        button_layout.addWidget(self.help_button)

        self.status_label = QLabel("Ready")
        button_layout.addWidget(self.status_label)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.result_view = ResultView()
        layout.addWidget(self.result_view, stretch=1)

        self.acquisition_settings.visualization_changed.connect(
            self.result_view.set_acquisition_visualization
        )
        self.result_view.set_acquisition_visualization(
            self.acquisition_settings.shows_bias_light,
            self.acquisition_settings.signal_to_bias_ratio,
        )

        self.update_preset_controls(DEFAULT_PRESET)

    def update_preset_controls(self, preset_name: str) -> None:
        """Update controls using values from the selected preset."""
        config = PRESETS[preset_name]

        self.algorithm_combo.blockSignals(True)
        self.algorithm_combo.clear()
        self.algorithm_combo.addItems([ZERO_FILLED_METHOD_ID, *sorted(config.methods)])
        if "pnp_admm_drunet" in config.methods:
            self.algorithm_combo.setCurrentText("pnp_admm_drunet")
        self.algorithm_combo.blockSignals(False)

        self.acquisition_settings.set_from_config(config)
        self.output_settings.set_from_config(config)
        self.update_algorithm_controls(self.algorithm_combo.currentText())

    def update_algorithm_controls(self, algorithm: str) -> None:
        """Update controls that depend on the selected reconstruction method."""
        self.algorithm_settings_button.setEnabled(
            bool(algorithm) and algorithm != ZERO_FILLED_METHOD_ID
        )

    def show_algorithm_settings(self) -> None:
        """Show the current algorithm configuration in a separate dialog."""
        preset_name = self.preset_combo.currentText()
        algorithm = self.algorithm_combo.currentText()

        if not algorithm or algorithm == ZERO_FILLED_METHOD_ID:
            return

        method_config = PRESETS[preset_name].methods[algorithm]
        dialog = AlgorithmSettingsDialog(algorithm, method_config, self)
        dialog.exec()

    def show_iteration_analysis(self) -> None:
        """Open the PnP iteration-analysis dialog."""
        from LION.pcm.gui.dialogs.pnp_iteration_metrics import PnPIterationMetricsDialog

        dialog = PnPIterationMetricsDialog(self)
        dialog.exec()

    def show_help(self) -> None:
        """Open the repository-backed PCM GUI guide without blocking the main window."""
        from LION.pcm.gui.dialogs.help import PCMHelpDialog

        if self.help_dialog is None:
            self.help_dialog = PCMHelpDialog(self)
            self.help_dialog.finished.connect(
                lambda _result: setattr(self, "help_dialog", None)
            )
        self.help_dialog.show()
        self.help_dialog.raise_()
        self.help_dialog.activateWindow()

    def make_request(self) -> GuiRunRequest:
        """Read one reconstruction request from the controls."""
        save = self.output_settings.values()
        return GuiRunRequest(
            preset=self.preset_combo.currentText(),
            algorithm=self.algorithm_combo.currentText(),
            device=self.device_combo.currentText(),
            sampling_ratio=self.acquisition_settings.sampling_ratio,
            coarse_j=self.acquisition_settings.coarse_j,
            uses_noise=self.acquisition_settings.uses_noise,
            output_dir=save.output_dir,
            save_config=save.save_config,
            save_metrics=save.save_metrics,
            save_plots=save.save_plots,
            save_recon=save.save_recon,
            save_diff=save.save_diff,
            save_masks=save.save_masks,
            save_zero_filled=save.save_zero_filled,
        )

    def start_reconstruction(self) -> None:
        """Start the selected reconstruction in a worker thread."""
        self.run_button.setEnabled(False)
        self.status_label.setText("Running...")
        self.result_view.clear_for_run()

        self.worker = ReconstructionWorker(self.make_request())
        self.worker.acquisition_ready.connect(self.show_acquisition)
        self.worker.result_ready.connect(self.show_result)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()

    def show_acquisition(self, preview: GuiAcquisitionPreview) -> None:
        """Display the prepared acquisition while reconstruction runs."""
        self.result_view.show_acquisition(preview)

    def show_result(self, result: GuiRunResult) -> None:
        """Display a completed reconstruction."""
        self.result_view.show_result(result)
        self.status_label.setText("Finished")

    def show_error(self, message: str) -> None:
        """Display a reconstruction error."""
        self.status_label.setText("Failed")
        self.result_view.show_error(message)

    def worker_finished(self) -> None:
        """Re-enable the GUI after reconstruction."""
        self.run_button.setEnabled(True)
        self.worker = None
