import matplotlib
import numpy as np
from matplotlib.colors import ListedColormap

from LION.pcm.config import PlotConfig
from LION.utils.plot_helper import PlotHelper


def build_plot_helper(config: PlotConfig) -> PlotHelper:
    """Build a ``PlotHelper`` from a structured plotting configuration.

    Parameters
    ----------
    config : PlotConfig
        Plotting configuration.

    Returns
    -------
    PlotHelper
        Configured plot helper.
    """
    cmap = ListedColormap(
        matplotlib.colormaps["afmhot"](np.linspace(0.0, config.cmap_max, 256))
    )
    return PlotHelper(
        roi=config.roi,
        zoom=config.zoom,
        loc=config.loc,
        show_rect=config.show_rect,
        cmap=cmap,
        clim=config.clim,
        loc1=config.loc1,
        loc2=config.loc2,
    )
