"""Minimal PySide6 front end for a single PCM reconstruction."""

from __future__ import annotations

import copy
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from LION.experiments.pcm_presets.presets import PRESETS
from LION.pcm.results import result_csv_path
from LION.pcm.run_pcm_single_sample_experiment import run_single_sample_experiment

DEFAULT_PRESET = (
    "e27_pos1_average_map_simulated_noise"
    if "e27_pos1_average_map_simulated_noise" in PRESETS
    else sorted(PRESETS)[0]
)


@dataclass(frozen=True, kw_only=True)
class GuiRunRequest:
    """Values selected in the GUI for one PCM run."""

    preset: str
    algorithm: str
    device: str
    sampling_ratio: float
    coarse_j: int
    uses_noise: bool
    output_dir: Path


@dataclass(frozen=True, kw_only=True)
class GuiRunResult:
    """Small result object returned to the GUI."""

    experiment_dir: Path
    comparison_image: Path
    metrics: dict[str, str]


def available_devices() -> list[str]:
    """Return compute devices available on the current machine."""
    devices = ["auto", "cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def read_last_result_row(path: Path) -> dict[str, str]:
    """Read the final row from one PCM result CSV."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"No results found in {path}.")

    return rows[-1]


def run_gui_experiment(request: GuiRunRequest) -> GuiRunResult:
    """Run one existing PCM experiment from GUI-selected values."""
    config = copy.deepcopy(PRESETS[request.preset])

    if request.algorithm not in config.methods:
        raise ValueError(
            f"Algorithm {request.algorithm!r} is not available in preset {request.preset!r}."
        )

    config.runtime.device = request.device
    config.runtime.root_output_dir = request.output_dir
    config.runtime.num_trials = 1
    config.runtime.num_trials_skip = 0

    config.sampling.sampling_ratios = [request.sampling_ratio]
    config.sampling.coarse_j_values = [request.coarse_j]

    config.methods = {request.algorithm: config.methods[request.algorithm]}

    if not request.uses_noise:
        config.noise = None

    config.log.save_plots = True
    config.log.save_recon = False
    config.log.save_diff = False
    config.log.save_masks = False
    config.log.save_zero_filled = False

    experiment_dir = run_single_sample_experiment(config)

    comparison_images = [
        path
        for path in experiment_dir.rglob("comparison.png")
        if path.parent.name == request.algorithm
    ]
    if len(comparison_images) != 1:
        raise RuntimeError(
            f"Expected one comparison image for {request.algorithm!r}, found {len(comparison_images)}."
        )

    metrics = read_last_result_row(result_csv_path(experiment_dir, request.algorithm))

    return GuiRunResult(
        experiment_dir=experiment_dir,
        comparison_image=comparison_images[0],
        metrics=metrics,
    )


class ReconstructionWorker(QThread):
    """Run the reconstruction without blocking the Qt GUI."""

    result_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, request: GuiRunRequest) -> None:
        """Initialize the reconstruction worker."""
        super().__init__()
        self.request = request

    def run(self) -> None:
        """Run the selected reconstruction."""
        try:
            result = run_gui_experiment(self.request)
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        self.result_ready.emit(result)


class PCMGui(QMainWindow):
    """Minimal PCM acquisition and reconstruction GUI."""

    def __init__(self) -> None:
        """Build the GUI."""
        super().__init__()
        self.worker: ReconstructionWorker | None = None
        self.result_pixmap: QPixmap | None = None

        self.setWindowTitle("A3SmartML PCM")
        self.resize(1200, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        title = QLabel("PCM acquisition and reconstruction")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        controls = QFormLayout()
        layout.addLayout(controls)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(sorted(PRESETS))
        self.preset_combo.setCurrentText(DEFAULT_PRESET)
        self.preset_combo.currentTextChanged.connect(self.update_preset_controls)
        controls.addRow("Preset", self.preset_combo)

        self.algorithm_combo = QComboBox()
        controls.addRow("Algorithm", self.algorithm_combo)

        self.device_combo = QComboBox()
        self.device_combo.addItems(available_devices())
        controls.addRow("Device", self.device_combo)

        self.sampling_spin = QDoubleSpinBox()
        self.sampling_spin.setRange(0.001, 1.0)
        self.sampling_spin.setSingleStep(0.01)
        self.sampling_spin.setDecimals(3)
        controls.addRow("Sampling ratio", self.sampling_spin)

        self.coarse_j_spin = QSpinBox()
        self.coarse_j_spin.setMinimum(0)
        controls.addRow("Coarse J", self.coarse_j_spin)

        self.noise_checkbox = QCheckBox("Use preset noise")
        controls.addRow("Noise", self.noise_checkbox)

        self.output_edit = QLineEdit("pcm_gui_results")
        controls.addRow("Output directory", self.output_edit)

        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run reconstruction")
        self.run_button.clicked.connect(self.start_reconstruction)
        button_layout.addWidget(self.run_button)

        self.status_label = QLabel("Ready")
        button_layout.addWidget(self.status_label)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.metrics_label = QLabel("")
        layout.addWidget(self.metrics_label)

        self.image_label = QLabel("Run a reconstruction to show the result.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        layout.addWidget(self.image_label, stretch=1)

        self.update_preset_controls(DEFAULT_PRESET)

    def update_preset_controls(self, preset_name: str) -> None:
        """Update controls using values from the selected preset."""
        config = PRESETS[preset_name]

        self.algorithm_combo.clear()
        self.algorithm_combo.addItems(sorted(config.methods))

        if "pnp_admm_drunet" in config.methods:
            self.algorithm_combo.setCurrentText("pnp_admm_drunet")

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

        self.noise_checkbox.setChecked(config.noise is not None)

    def make_request(self) -> GuiRunRequest:
        """Read one reconstruction request from the controls."""
        return GuiRunRequest(
            preset=self.preset_combo.currentText(),
            algorithm=self.algorithm_combo.currentText(),
            device=self.device_combo.currentText(),
            sampling_ratio=self.sampling_spin.value(),
            coarse_j=self.coarse_j_spin.value(),
            uses_noise=self.noise_checkbox.isChecked(),
            output_dir=Path(self.output_edit.text()).expanduser(),
        )

    def start_reconstruction(self) -> None:
        """Start the selected reconstruction in a worker thread."""
        self.run_button.setEnabled(False)
        self.status_label.setText("Running...")
        self.metrics_label.clear()

        self.worker = ReconstructionWorker(self.make_request())
        self.worker.result_ready.connect(self.show_result)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()

    def show_result(self, result: GuiRunResult) -> None:
        """Display reconstruction metrics and the existing LION comparison figure."""
        metrics = result.metrics
        self.metrics_label.setText(
            f"PSNR: {float(metrics['psnr']):.2f} dB    "
            f"SSIM: {float(metrics['ssim']):.4f}    "
            f"HAARPSI: {float(metrics['haarpsi']):.4f}    "
            f"Time: {float(metrics['duration']):.2f} s\n"
            f"Results: {result.experiment_dir}"
        )

        self.result_pixmap = QPixmap(str(result.comparison_image))
        self.update_result_image()
        self.status_label.setText("Finished")

    def show_error(self, message: str) -> None:
        """Display a reconstruction error."""
        self.status_label.setText("Failed")
        self.metrics_label.setText(message)

    def worker_finished(self) -> None:
        """Re-enable the GUI after reconstruction."""
        self.run_button.setEnabled(True)
        self.worker = None

    def resizeEvent(self, event) -> None:
        """Resize the displayed reconstruction figure with the window."""
        super().resizeEvent(event)
        self.update_result_image()

    def update_result_image(self) -> None:
        """Scale the current result figure to the available display area."""
        if self.result_pixmap is None:
            return

        scaled = self.result_pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)


def main() -> None:
    """Run the PCM GUI."""
    app = QApplication(sys.argv)
    gui = PCMGui()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
