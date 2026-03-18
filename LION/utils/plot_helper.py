"""Helper functions for plotting images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes


@dataclass
class PlotHelper:
    """Helper class for plotting images.

    Attributes
    ----------
    zoom: float
        Magnification factor relative to the main axes.
    loc: str
        Inset axes location; e.g. 'upper right' or the integer codes 1-4.
    loc1: int
        Connector location on the main axes, passed to the plotting helper.
    loc2: int
        Connector location on the inset axes, passed to the plotting helper.
    roi: tuple[int, int, int, int]
        Region of interest for the inset zoom as (x, y, width, height).
    clim: tuple[float, float]
        Colour limits for the plotted images.
    cmap_max: float
        Maximum fraction of the ``afmhot`` colormap to use.
    adds_insets: bool
        Whether to draw inset zooms.
    show_rect: bool
        Whether to draw a rectangle around the ROI on the main image.
    """

    zoom: float
    loc: str
    loc1: int
    loc2: int
    roi: tuple[int, int, int, int]
    clim: tuple[float, float]
    cmap_max: float = 0.8
    adds_insets: bool = True
    show_rect: bool = True

    def __post_init__(self):
        """Set default styles for the ROI rectangle and connector lines, and create the colormap."""
        self.rect_kwargs = {"edgecolor": "red", "linewidth": 1.5, "facecolor": "none"}
        self.connector_kwargs = {"fc": "none", "ec": "red", "lw": 1.2}
        self.cmap = ListedColormap(
            matplotlib.colormaps["afmhot"](np.linspace(0.0, self.cmap_max, 256))
        )

    def add_zoom_inset(
        self,
        ax: Axes,
        img: np.ndarray,
        rect_kwargs: dict | None = None,
        connector_kwargs: dict | None = None,
    ) -> Axes:
        """
        Add a zoomed-in inset using zoomed_inset_axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Parent axes containing the main image.
        img : (H, W[, C]) ndarray
            Image to display.
        rect_kwargs : dict
            Style for the ROI rectangle. If None, defaults to a red rectangle with no fill.
        connector_kwargs : dict
            Style for the connector lines drawn by mark_inset. If None, defaults to red lines with no fill.

        Returns
        -------
        axins : matplotlib.axes.Axes
            The inset axes.
        """
        if rect_kwargs is None:
            rect_kwargs = self.rect_kwargs
        if connector_kwargs is None:
            connector_kwargs = self.connector_kwargs
        # Show main image
        ax.imshow(img, cmap=self.cmap, clim=self.clim)
        ax.set_axis_off()

        x, y, w, h = self.roi
        if self.show_rect:
            ax.add_patch(Rectangle((x, y), w, h, **rect_kwargs))

        # Create a zoomed inset
        axins: Axes = zoomed_inset_axes(ax, zoom=self.zoom, loc=self.loc, borderpad=1.0)

        # Draw the same image, and limit the view to the ROI
        axins.imshow(img, cmap=self.cmap, clim=self.clim)
        axins.set_xlim(x, x + w)
        axins.set_ylim(y + h, y)  # invert y for imshow's top-left origin
        axins.set_xticks([])
        axins.set_yticks([])

        # Optional connector lines between the two boxes
        mark_inset(ax, axins, loc1=self.loc1, loc2=self.loc2, **connector_kwargs)

        return axins

    def show_images_with_inset(
        self,
        images: list[torch.Tensor],
        fig_filepath: Path,
        titles: list[str] | None = None,
        suptitle: str | None = None,
        saves_fig: bool = False,
    ) -> None:
        """Plot images."""
        n_images = len(images)
        fig, axes = plt.subplots(1, n_images, squeeze=False, figsize=(n_images * 4, 4))

        for i in range(n_images):
            img_np = images[i].squeeze().cpu().numpy()
            ax: plt.Axes = axes[0][i]
            if self.adds_insets:
                self.add_zoom_inset(ax, img_np)
            else:
                ax.imshow(img_np, cmap=self.cmap, clim=self.clim)
            ax.axis("off")
            if titles:
                ax.set_title(titles[i], fontsize=10)
        if suptitle:
            fig.subplots_adjust(bottom=0.18)
            fig.text(0.5, 0.02, suptitle, ha="center", va="bottom", fontsize=16)
        if saves_fig:
            fig.savefig(fig_filepath, dpi=150)
        plt.close(fig)
