"""Result display widgets for the PCM GUI."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import QLabel, QSplitter, QTabWidget, QVBoxLayout, QWidget

from LION.pcm.gui.backend import GuiAcquisitionPreview, GuiRunResult
from LION.pcm.gui.widgets.acquisition_view import AcquisitionView


def render_image_pixmap(
    image: np.ndarray | object,
    plot_helper,
    *,
    title: str,
    colorbar: bool = True,
) -> QPixmap:
    """Render one image using the preset's shared PCM plotting configuration."""
    if hasattr(image, "detach"):
        image_np = image.detach().squeeze().cpu().numpy()
    else:
        image_np = np.asarray(image).squeeze()

    if plot_helper.color.inverse_sign:
        image_np = -image_np

    figure = Figure(figsize=(5, 5), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axis = figure.subplots()

    plot = axis.imshow(
        image_np,
        cmap=plot_helper.get_cmap(),
        clim=plot_helper.color.clim,
    )
    axis.set_title(title, fontsize=10)
    axis.set_axis_off()
    if colorbar:
        figure.colorbar(plot, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    canvas.draw()

    image_rgba = np.asarray(canvas.buffer_rgba()).copy()
    height, width, _ = image_rgba.shape
    qimage = QImage(
        image_rgba.data,
        width,
        height,
        image_rgba.strides[0],
        QImage.Format.Format_RGBA8888,
    ).copy()
    return QPixmap.fromImage(qimage)


def render_reconstruction_pixmap(result: GuiRunResult) -> QPixmap:
    """Render only the reconstruction selected in the GUI."""
    run_result = result.run.results[result.algorithm]
    return render_image_pixmap(
        run_result.recon,
        result.run.config.plot,
        title=result.algorithm,
    )


class ResultView(QTabWidget):
    """Tabbed view with acquisition and reconstruction side by side."""

    def __init__(self) -> None:
        """Build the result tabs."""
        super().__init__()
        self.ground_truth_pixmap: QPixmap | None = None
        self.reconstruction_pixmap: QPixmap | None = None

        simulation_tab = QWidget()
        simulation_layout = QVBoxLayout(simulation_tab)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        ground_truth_widget = QWidget()
        ground_truth_layout = QVBoxLayout(ground_truth_widget)
        ground_truth_title = QLabel("Ground truth")
        ground_truth_title.setStyleSheet("font-weight: bold;")
        ground_truth_layout.addWidget(ground_truth_title)
        self.ground_truth_label = QLabel("Ground truth will appear when a run starts.")
        self.ground_truth_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ground_truth_label.setMinimumSize(240, 240)
        ground_truth_layout.addWidget(self.ground_truth_label, stretch=1)
        splitter.addWidget(ground_truth_widget)

        self.acquisition_view = AcquisitionView()
        splitter.addWidget(self.acquisition_view)

        reconstruction_widget = QWidget()
        reconstruction_layout = QVBoxLayout(reconstruction_widget)

        reconstruction_title = QLabel("Reconstruction")
        reconstruction_title.setStyleSheet("font-weight: bold;")
        reconstruction_layout.addWidget(reconstruction_title)

        self.reconstruction_label = QLabel("Run a reconstruction to show the result.")
        self.reconstruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reconstruction_label.setMinimumSize(400, 350)
        reconstruction_layout.addWidget(self.reconstruction_label, stretch=1)

        self.reconstruction_summary = QLabel("")
        self.reconstruction_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reconstruction_layout.addWidget(self.reconstruction_summary)

        splitter.addWidget(reconstruction_widget)
        splitter.setSizes([1, 1, 1])

        simulation_layout.addWidget(splitter)
        self.addTab(simulation_tab, "Simulation")

        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        self.metrics_label = QLabel("No result yet.")
        self.metrics_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.metrics_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        metrics_layout.addWidget(self.metrics_label)
        metrics_layout.addStretch()
        self.addTab(metrics_tab, "Metrics")

    def set_acquisition_visualization(
        self,
        show_bias_light: bool,
        signal_to_bias_ratio: float,
    ) -> None:
        """Update display-only acquisition illumination settings."""
        self.acquisition_view.set_visualization(
            show_bias_light,
            signal_to_bias_ratio,
        )

    def clear_for_run(self) -> None:
        """Reset both image panels while a new experiment starts."""
        self.acquisition_view.clear()
        self.ground_truth_pixmap = None
        self.ground_truth_label.clear()
        self.ground_truth_label.setText("Loading ground truth...")
        self.reconstruction_pixmap = None
        self.reconstruction_label.clear()
        self.reconstruction_label.setText("Reconstructing...")
        self.reconstruction_summary.clear()
        self.metrics_label.setText("Running...")
        self.setCurrentIndex(0)

    def show_acquisition(self, preview: GuiAcquisitionPreview) -> None:
        """Show the ground truth and start the simulated acquisition animation."""
        self.ground_truth_pixmap = render_image_pixmap(
            preview.target,
            preview.plot_helper,
            title="Ground truth",
        )
        self._update_ground_truth_image()
        self.acquisition_view.start_preview(preview)

    def show_result(self, result: GuiRunResult) -> None:
        """Display one completed reconstruction."""
        selected_result = result.run.results[result.algorithm]
        assert selected_result.metrics is not None
        assert selected_result.duration is not None

        metrics = selected_result.metrics
        results_text = (
            f"Results: {result.experiment_dir}"
            if result.saved_outputs
            else "Results not saved"
        )
        self.metrics_label.setText(
            f"Algorithm: {result.algorithm}\n"
            f"PSNR: {metrics['psnr']:.2f} dB\n"
            f"SSIM: {metrics['ssim']:.4f}\n"
            f"HAARPSI: {metrics['haarpsi']:.4f}\n"
            f"MSE: {metrics['mse']:.3e}\n"
            f"Time: {selected_result.duration:.2f} s\n\n"
            f"{results_text}"
        )

        self.reconstruction_summary.setText(
            f"PSNR: {metrics['psnr']:.2f} dB | "
            f"SSIM: {metrics['ssim']:.4f} | "
            f"Time: {selected_result.duration:.2f} s"
        )

        self.reconstruction_pixmap = render_reconstruction_pixmap(result)
        self._update_reconstruction_image()
        self.setCurrentIndex(0)

    def show_error(self, message: str) -> None:
        """Display an error in the metrics tab."""
        self.reconstruction_label.setText("Reconstruction failed.")
        self.metrics_label.setText(message)
        self.setCurrentIndex(1)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Resize the displayed reconstruction figure with the widget."""
        super().resizeEvent(event)
        self._update_ground_truth_image()
        self._update_reconstruction_image()

    def _update_ground_truth_image(self) -> None:
        """Scale the ground-truth figure to the available display area."""
        if self.ground_truth_pixmap is None:
            return

        scaled = self.ground_truth_pixmap.scaled(
            self.ground_truth_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.ground_truth_label.setPixmap(scaled)

    def _update_reconstruction_image(self) -> None:
        """Scale the current reconstruction figure to the available display area."""
        if self.reconstruction_pixmap is None:
            return

        scaled = self.reconstruction_pixmap.scaled(
            self.reconstruction_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.reconstruction_label.setPixmap(scaled)
