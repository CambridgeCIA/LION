from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np


class PlotHelper:
    def __init__(
        self,
        roi: tuple[int, int, int, int] = None,
        zoom: float = None,
        loc: str = None,
        show_rect: bool = None,
        rect_kwargs: dict = None,
        connector_kwargs: dict = None,
        cmap: str | ListedColormap = None,
        clim: tuple[float, float] = None,
        loc1: int = 2,
        loc2: int = 1,
    ):
        """
        Initialize the PlotHelper with zoom inset parameters.

        roi : (x, y, w, h)
            Region of interest in pixel coordinates.
        zoom : float
            Magnification factor relative to the main axes.
        loc : str|int
            Inset location; e.g. 'upper right' or the integer codes 1-4.
        show_rect : bool
            Draw a rectangle around the ROI on the main image.
        rect_kwargs : dict
            Style for the ROI rectangle.
        connector_kwargs : dict
            Style for the connector lines drawn by mark_inset.
        cmap : str|None
            Colormap for grayscale images.
        """
        self.roi = roi
        self.zoom = zoom
        self.loc = loc
        self.show_rect = show_rect
        if rect_kwargs is None:
            rect_kwargs = dict(edgecolor="red", linewidth=1.5, facecolor="none")
        if connector_kwargs is None:
            connector_kwargs = dict(fc="none", ec="red", lw=1.2)
        self.rect_kwargs = rect_kwargs
        self.connector_kwargs = connector_kwargs
        self.cmap = cmap
        self.clim = clim
        self.loc1 = loc1
        self.loc2 = loc2

    def add_zoom_inset(self, ax: Axes, img: np.ndarray) -> Axes:
        """
        Add a zoomed-in inset using zoomed_inset_axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Parent axes containing the main image.
        img : (H, W[, C]) ndarray
            Image to display.

        Returns
        -------
        axins : matplotlib.axes.Axes
            The inset axes.
        """
        # Show main image
        ax.imshow(img, cmap=self.cmap, clim=self.clim)
        ax.set_axis_off()

        x, y, w, h = self.roi
        if self.show_rect:
            ax.add_patch(Rectangle((x, y), w, h, **self.rect_kwargs))

        # Create a zoomed inset
        axins: Axes = zoomed_inset_axes(ax, zoom=self.zoom, loc=self.loc, borderpad=1.0)

        # Draw the same image, and limit the view to the ROI
        axins.imshow(img, cmap=self.cmap, clim=self.clim)
        axins.set_xlim(x, x + w)
        axins.set_ylim(y + h, y)  # invert y for imshow's top-left origin
        axins.set_xticks([])
        axins.set_yticks([])

        # Optional connector lines between the two boxes
        mark_inset(ax, axins, loc1=self.loc1, loc2=self.loc2, **self.connector_kwargs)

        return axins
