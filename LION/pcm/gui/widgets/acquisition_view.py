"""Animated view of the simulated PCM acquisition."""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

from LION.pcm.gui.backend import GuiAcquisitionPreview
from LION.pcm_tools.hadamard_matrix import hadamard_pattern_from_sorted_index

MAX_PREVIEW_PATTERNS = 64
FRAME_INTERVAL_MS = 100


def _to_green_pixmap(image: np.ndarray) -> QPixmap:
    """Convert a normalized illumination image to a green RGB pixmap."""
    image = np.asarray(image, dtype=np.float32)
    image = np.clip(image, 0.0, 1.0)

    rgb = np.stack(
        (
            0.12 * image,
            image,
            0.18 * image,
        ),
        axis=-1,
    )
    image_uint8 = np.ascontiguousarray(
        np.clip(255.0 * rgb, 0.0, 255.0).astype(np.uint8)
    )
    height, width, _ = image_uint8.shape
    qimage = QImage(
        image_uint8.data,
        width,
        height,
        image_uint8.strides[0],
        QImage.Format.Format_RGB888,
    ).copy()
    return QPixmap.fromImage(qimage)


class AcquisitionView(QWidget):
    """Animate representative positive and negative PCM pattern exposures."""

    def __init__(self) -> None:
        """Build the acquisition view."""
        super().__init__()

        self.preview: GuiAcquisitionPreview | None = None
        self.preview_positions = np.empty(0, dtype=np.int64)
        self.frame_index = 0
        self.current_pixmap: QPixmap | None = None
        self.show_bias_light = False
        self.signal_to_bias_ratio = 0.30
        self.preparation_started_at: float | None = None
        self.preview_started_at: float | None = None
        self.preparation_duration: float | None = None

        layout = QVBoxLayout(self)

        title = QLabel("Simulated acquisition preview")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        self.image_label = QLabel("Waiting for acquisition...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 350)
        layout.addWidget(self.image_label, stretch=1)

        self.pattern_label = QLabel("")
        self.pattern_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.pattern_label)

        self.measurement_label = QLabel("")
        self.measurement_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.measurement_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Not started")
        layout.addWidget(self.progress_bar)

        self.progress_time_label = QLabel("")
        self.progress_time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_time_label)

        self.timer = QTimer(self)
        self.timer.setInterval(FRAME_INTERVAL_MS)
        self.timer.timeout.connect(self._show_next_frame)

    def set_visualization(
        self, show_bias_light: bool, signal_to_bias_ratio: float
    ) -> None:
        """Update display-only illumination settings without changing acquisition physics."""
        self.show_bias_light = bool(show_bias_light)
        self.signal_to_bias_ratio = max(0.0, float(signal_to_bias_ratio))

        if self.preview is not None and len(self.preview_positions) > 0:
            # Re-render the current exposure immediately when a visual control changes.
            self.frame_index = max(0, self.frame_index - 1)
            self._show_next_frame()

    def clear(self) -> None:
        """Reset the acquisition panel for a new experiment."""
        self.timer.stop()
        self.preview = None
        self.preview_positions = np.empty(0, dtype=np.int64)
        self.frame_index = 0
        self.current_pixmap = None
        self.preparation_started_at = time.perf_counter()
        self.preview_started_at = None
        self.preparation_duration = None
        self.image_label.clear()
        self.image_label.setText("Preparing acquisition...")
        self.pattern_label.clear()
        self.measurement_label.clear()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Preparing simulated acquisition...")
        self.progress_time_label.setText("Acquisition preparation in progress")

    def start_preview(self, preview: GuiAcquisitionPreview) -> None:
        """Start an animation for one prepared acquisition."""
        self.preview = preview
        num_patterns = len(preview.sampled_indices)
        num_preview = min(num_patterns, MAX_PREVIEW_PATTERNS)

        if num_preview == 0:
            self.image_label.setText("No sampled patterns.")
            return

        self.preview_positions = np.linspace(
            0, num_patterns - 1, num=num_preview, dtype=np.int64
        )
        sampling_percentage = 100.0 * preview.sampling_ratio
        self.pattern_label.setText(
            f"{num_patterns} sampled patterns ({sampling_percentage:.1f}% sampling); "
            f"showing {num_preview} representative patterns"
        )
        now = time.perf_counter()
        self.preparation_duration = (
            now - self.preparation_started_at
            if self.preparation_started_at is not None
            else None
        )
        self.preview_started_at = now
        self.frame_index = 0
        self.progress_bar.setRange(0, 2 * num_preview)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m displayed preview exposures (%p%)")
        self._show_next_frame()
        self.timer.start()

    def _show_next_frame(self) -> None:
        """Show the next positive or negative exposure."""
        if self.preview is None or len(self.preview_positions) == 0:
            self.timer.stop()
            return

        exposure_index = self.frame_index // 2
        exposure_is_positive = self.frame_index % 2 == 0

        if exposure_index >= len(self.preview_positions):
            self.timer.stop()
            self.progress_bar.setValue(self.progress_bar.maximum())
            elapsed = (
                time.perf_counter() - self.preview_started_at
                if self.preview_started_at is not None
                else 0.0
            )
            prep_text = (
                f" | preparation {self.preparation_duration:.2f} s"
                if self.preparation_duration is not None
                else ""
            )
            self.progress_time_label.setText(
                f"Preview complete in {elapsed:.1f} s{prep_text}"
            )
            return

        acquisition_position = int(self.preview_positions[exposure_index])
        sampled_index = int(self.preview.sampled_indices[acquisition_position])

        pattern = hadamard_pattern_from_sorted_index(
            sorted_index=sampled_index,
            o_col=self.preview.j_order,
            o_row=self.preview.j_order,
        )
        mask = (
            pattern["positive_mask"]
            if exposure_is_positive
            else pattern["negative_mask"]
        )

        target = (
            -self.preview.target if self.preview.inverse_sign else self.preview.target
        )
        target_min = float(target.min())
        target_max = float(target.max())
        if target_max > target_min:
            target_display = (target - target_min) / (target_max - target_min)
        else:
            target_display = np.zeros_like(target)

        mask_float = mask.astype(np.float32)
        if self.show_bias_light:
            # Display-only model: uniform bias has amplitude 1 and the patterned
            # signal has the selected signal-to-bias amplitude ratio.
            ratio = self.signal_to_bias_ratio
            illumination = (1.0 + ratio * mask_float) / max(1.0 + ratio, 1.0)
        else:
            # Keep a small floor so the target remains faintly visible where the
            # illustration has no projected signal.
            illumination = 0.06 + 0.94 * mask_float

        projected = target_display * illumination
        self.current_pixmap = _to_green_pixmap(projected)
        self._update_image()

        exposure_name = "positive" if exposure_is_positive else "negative"
        sampling_percentage = 100.0 * self.preview.sampling_ratio
        self.pattern_label.setText(
            f"Preview {exposure_index + 1}/{len(self.preview_positions)} | "
            f"acquisition pattern {acquisition_position + 1}/{len(self.preview.sampled_indices)} "
            f"({sampling_percentage:.1f}% sampling; sample index {sampled_index}) | "
            f"{exposure_name} exposure"
        )
        measurement = float(np.ravel(self.preview.measurements)[acquisition_position])
        bias_text = (
            f" | bias shown, signal/bias={self.signal_to_bias_ratio:.2f}"
            if self.show_bias_light
            else " | bias hidden"
        )
        self.measurement_label.setText(
            f"Differential measurement: {measurement:.4e}{bias_text}"
        )

        completed = self.frame_index + 1
        self.progress_bar.setValue(completed)
        if self.preview_started_at is not None:
            elapsed = max(0.0, time.perf_counter() - self.preview_started_at)
            remaining = self.progress_bar.maximum() - completed
            eta = elapsed / completed * remaining if completed > 0 else 0.0
            prep_text = (
                f" | preparation {self.preparation_duration:.2f} s"
                if self.preparation_duration is not None
                else ""
            )
            self.progress_time_label.setText(
                f"Preview elapsed {elapsed:.1f} s | ETA {eta:.1f} s{prep_text}"
            )

        self.frame_index += 1

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Resize the current acquisition frame with the widget."""
        super().resizeEvent(event)
        self._update_image()

    def _update_image(self) -> None:
        """Scale the current acquisition frame to the available display area."""
        if self.current_pixmap is None:
            return

        scaled = self.current_pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
