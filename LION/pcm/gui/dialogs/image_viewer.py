"""Reusable image viewer dialog for PCM GUI results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from LION.utils.plot_helper import PlotHelper


@dataclass(frozen=True)
class ImageViewerItem:
    """One image and its descriptive text."""

    title: str
    image: np.ndarray | QPixmap
    summary: str = ""


def _array_pixmap(image: np.ndarray, *, plot_helper: PlotHelper) -> QPixmap:
    figure = Figure(figsize=(6, 6), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axis = figure.subplots()

    image = np.asarray(image)
    if plot_helper.color.inverse_sign:
        image = -image
    clim = plot_helper.color.clim
    cmap = plot_helper.get_cmap()
    if clim is None:
        plot = axis.imshow(image, cmap=cmap)
    else:
        plot = axis.imshow(image, cmap=cmap, vmin=clim[0], vmax=clim[1])
    axis.set_axis_off()
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


class ImageViewerDialog(QDialog):
    """View one or more images without crowding the parent dialog."""

    def __init__(
        self,
        title: str,
        items: list[ImageViewerItem],
        *,
        plot_helper: PlotHelper,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not items:
            raise ValueError("ImageViewerDialog requires at least one image.")

        self.items = items
        self.plot_helper = plot_helper
        self.current_pixmap: QPixmap | None = None

        self.setWindowTitle(title)
        _resize_within_screen(self, 800, 760)

        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.previous_button = QPushButton("Previous")
        self.previous_button.clicked.connect(self.previous_image)
        controls.addWidget(self.previous_button)

        self.image_combo = QComboBox()
        self.image_combo.addItems([item.title for item in items])
        self.image_combo.currentIndexChanged.connect(self.show_index)
        controls.addWidget(self.image_combo, stretch=1)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_image)
        controls.addWidget(self.next_button)
        layout.addLayout(controls)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(200, 200)
        layout.addWidget(self.image_label, stretch=1)

        self.summary_label = QLabel()
        self.summary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.summary_label)

        self.show_index(0)

    def show_index(self, index: int) -> None:
        item = self.items[index]
        if isinstance(item.image, QPixmap):
            self.current_pixmap = item.image
        else:
            self.current_pixmap = _array_pixmap(
                item.image, plot_helper=self.plot_helper
            )

        self.summary_label.setText(item.summary)
        self.previous_button.setEnabled(index > 0)
        self.next_button.setEnabled(index < len(self.items) - 1)
        self._update_image()

    def previous_image(self) -> None:
        self.image_combo.setCurrentIndex(max(0, self.image_combo.currentIndex() - 1))

    def next_image(self) -> None:
        self.image_combo.setCurrentIndex(
            min(len(self.items) - 1, self.image_combo.currentIndex() + 1)
        )

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_image()

    def _update_image(self) -> None:
        if self.current_pixmap is None:
            return
        scaled = self.current_pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
