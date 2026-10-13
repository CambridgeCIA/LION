from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from LION.config.LogConfig import ImageFormat
from LION.visualize.plotting_functions import save_figure_formats


def save_histogram(
    values: np.ndarray,
    image_path: Path,
    render_histograms: bool,
    image_formats: tuple[ImageFormat, ...] | list[ImageFormat],
    name: str,
    value_range: tuple[float, float] | None = None,
) -> None:
    if not render_histograms:
        return

    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return

    fig, axis = plt.subplots(figsize=(5, 4))
    axis.hist(finite_values.ravel(), bins=100, range=value_range)
    if value_range is not None:
        axis.set_xlim(*value_range)
    axis.set_xlabel(name)
    axis.set_ylabel("Count")
    fig.tight_layout()
    save_figure_formats(
        fig,
        image_path.with_name(f"{image_path.stem}_histogram.png"),
        image_formats,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def log10_positive(values: np.ndarray) -> np.ndarray:
    """Return log10(values), using NaN for non-positive or non-finite entries."""
    values = np.asarray(values, dtype=np.float64)
    log_values = np.full_like(values, np.nan, dtype=np.float64)
    positive = np.isfinite(values) & (values > 0)
    np.log10(values, out=log_values, where=positive)
    return log_values
