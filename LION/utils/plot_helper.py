"""Helper functions for plotting images."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from PIL import Image

from LION.utils.parameter import LIONParameter
from LION.visualize.plotting_functions import make_colormap


@dataclass(kw_only=True)
class ZoomParameter(LIONParameter):
    """Parameters for zoomed-in insets in plots.

    NOTE: From matplotlib `mpl_toolkits.axes_grid1.inset_locator` docs:

    loc1, loc2 : {1, 2, 3, 4}

    Corner of *bbox1* and *bbox2* to draw the line. Valid values are::

        'upper right'  : 1,
        'upper left'   : 2,
        'lower left'   : 3,
        'lower right'  : 4

    *loc2* is optional and defaults to *loc1*.

    Attributes:
        factor: Magnification factor relative to the main axes.
        loc: Inset axes location; e.g. 'upper right' or the integer codes 1-4.
        loc1: Connector location on the main axes, passed to the plotting helper.
        loc2: Connector location on the inset axes, passed to the plotting helper.
        roi: Region of interest for the inset zoom as (x, y, width, height).
        adds_insets: Whether to draw inset zooms.
        show_rect: Whether to draw a rectangle around the ROI on the main image.
    """

    factor: float
    loc: str
    loc1: int
    loc2: int
    roi: tuple[int, int, int, int]
    adds_insets: bool = True
    show_rect: bool = True

    def extract_roi(self, img: torch.Tensor) -> torch.Tensor:
        """Extract a region of interest (ROI) from an image.

        Region of interest for the inset zoom as (x, y, width, height).

        Args:
            img: The input image tensor.

        Returns:
            The extracted ROI.
        """
        x, y, width, height = self.roi
        assert img.ndim == 2, f"Expected 2D image, got shape {img.shape}"
        return img[y : y + height, x : x + width].contiguous()


@dataclass(kw_only=True)
class ColorParameter(LIONParameter):
    """Parameters for color mapping in plots.

    Attributes:
        clim: Color limits for the plotted images as (vmin, vmax).
        cmap_max: Maximum fraction of the colormap to use.
        cmap_name: Name of the matplotlib colormap to use for plotting the images.
        inverse_sign: Whether to invert the sign of the images when plotting.
            Useful for current maps where measurements are negative.
    """

    clim: tuple[float, float] | None = None
    cmap_max: float = 0.8
    cmap_name: str = "afmhot"
    inverse_sign: bool = True


@dataclass(kw_only=True)
class PlotHelper:
    """Helper class for plotting images.

    Attributes:
        zoom: Zoom parameters for the inset axes.
        color: Color parameters for the plotted images.
        dpi: Dots per inch for saved figures.
        axis_off : Whether to turn off axes for the images. Useful to recall the dimensions.
    """

    zoom: ZoomParameter | None = None
    color: ColorParameter = field(default_factory=ColorParameter)
    dpi: int = 200
    axis_off: bool = True

    def __post_init__(self):
        """Set default styles for the ROI rectangle and connector lines, and create the colormap."""
        self.rect_kwargs = {"edgecolor": "red", "linewidth": 1.5, "facecolor": "none"}
        self.connector_kwargs = {"fc": "none", "ec": "red", "lw": 1.2}

    def get_cmap(self, cmap_name: str | None = None) -> ListedColormap:
        """Get a colormap with the specified maximum fraction."""
        if cmap_name is None:
            cmap_name = self.color.cmap_name
        return make_colormap(cmap_name, self.color.cmap_max)

    def add_zoom_inset(
        self,
        ax: Axes,
        img: np.ndarray,
        rect_kwargs: dict | None = None,
        connector_kwargs: dict | None = None,
        cmap_name: str | None = None,
    ) -> Axes | None:
        """Add a zoomed-in inset using zoomed_inset_axes.

        Args:
            ax: Parent axes containing the main image.
            img: (H, W[, C]) ndarray image to display.
            rect_kwargs: Style for the ROI rectangle. If None, defaults to a red rectangle with no fill.
            connector_kwargs: Style for connector lines drawn by mark_inset. Defaults to red lines with no fill.
            cmap_name: Name of the matplotlib colormap to use. If None, defaults to the object's config.

        Returns:
            The inset axes.
        """
        if rect_kwargs is None:
            rect_kwargs = self.rect_kwargs
        if connector_kwargs is None:
            connector_kwargs = self.connector_kwargs
        if cmap_name is None:
            cmap_name = self.color.cmap_name
        cmap = self.get_cmap(cmap_name)
        # Show main image
        ax.imshow(img, cmap=cmap, clim=self.color.clim)
        if self.axis_off:
            ax.set_axis_off()

        if self.zoom is None:
            return None

        x, y, w, h = self.zoom.roi
        if self.zoom.show_rect:
            ax.add_patch(Rectangle((x, y), w, h, **rect_kwargs))

        # Create a zoomed inset
        axins: Axes = zoomed_inset_axes(
            ax, zoom=self.zoom.factor, loc=self.zoom.loc, borderpad=1.0
        )

        # Draw the same image, and limit the view to the ROI
        axins.imshow(img, cmap=cmap, clim=self.color.clim)
        axins.set_xlim(x, x + w)
        axins.set_ylim(y + h, y)  # invert y for imshow's top-left origin
        axins.set_xticks([])
        axins.set_yticks([])

        # Optional connector lines between the two boxes
        mark_inset(
            ax, axins, loc1=self.zoom.loc1, loc2=self.zoom.loc2, **connector_kwargs
        )

        return axins

    def show_images_with_inset(
        self,
        images: list[torch.Tensor],
        fig_filepath: Path,
        titles: list[str] | None = None,
        suptitle: str | None = None,
        saves_fig: bool = False,
        cmap_name: str | None = None,
    ) -> None:
        """Plot images."""
        n_images = len(images)
        fig, axes = plt.subplots(1, n_images, squeeze=False, figsize=(n_images * 4, 5))
        if cmap_name is None:
            cmap_name = self.color.cmap_name
        cmap = self.get_cmap(cmap_name)
        for i in range(n_images):
            img_np = images[i].squeeze().cpu().numpy()
            if self.color.inverse_sign:
                img_np = -img_np
            ax: plt.Axes = axes[0][i]
            if self.zoom is not None and self.zoom.adds_insets:
                self.add_zoom_inset(ax, img_np)
            else:
                ax.imshow(img_np, cmap=cmap, clim=self.color.clim)
            if self.axis_off:
                ax.axis("off")
            if titles:
                ax.set_title(titles[i], fontsize=8)
            plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
            # print(f"Shape of image {i}: {img_np.shape}")
        if suptitle:
            fig.subplots_adjust(bottom=0.18)
            fig.text(0.5, 0.01, suptitle, ha="center", va="bottom", fontsize=12)
        if saves_fig:
            fig.savefig(fig_filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def save_image(
        self,
        img: torch.Tensor,
        filepath: Path,
        roi: tuple[float, float, float, float] | Literal["preset"] | None = None,
        *,
        cmap_name: str | None = None,
        clim: tuple[float, float] | None = None,
        inverse_sign: bool | None = None,
    ) -> None:
        """Save a single image."""
        img_np = img.detach().squeeze().cpu().numpy()
        inverse_sign = self.color.inverse_sign if inverse_sign is None else inverse_sign
        if inverse_sign:
            img_np = -img_np
        cmap_name = self.color.cmap_name if cmap_name is None else cmap_name
        clim = self.color.clim if clim is None else clim
        cmap = self.get_cmap(cmap_name)
        norm = plt.Normalize(vmin=clim[0], vmax=clim[1]) if clim else None
        img_mapped = cmap(norm(img_np))[:, :, :3]  # Get RGB values from the colormap
        img_uint8 = (img_mapped * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(filepath)
        # if roi is not None and self.zoom is not None:
        #     if roi == "preset":
        #         x, y, w, h = self.zoom.roi
        #     else:
        #         x, y, w, h = roi
        #     print(f"Saving zoomed-in ROI of image shape {img_uint8.shape} with ROI {x, y, w, h} to {filepath}")
        #     zoomed_img_uint8 = img_uint8[y : y + h, x : x + w]
        #     Image.fromarray(zoomed_img_uint8).save(filepath)
        #     original_img_uint8_with_inset = cv2.rectangle(
        #         img_uint8, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2
        #     )
        #     # Add suffix "_inset" to the filename before the extension
        #     inset_filename = filepath.stem + "_inset" + filepath.suffix
        #     inset_filepath = filepath.parent / inset_filename
        #     Image.fromarray(original_img_uint8_with_inset).save(inset_filepath)
        # else:
        #     Image.fromarray(img_uint8).save(filepath)
