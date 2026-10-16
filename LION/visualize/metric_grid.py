"""Generic 2-D metric-grid plotting functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def _arrays(x, y, values) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    if not (x.shape == y.shape == values.shape):
        raise ValueError("x, y, and values must have the same shape after ravel().")
    return x, y, values


def plot_metric_scatter(
    x,
    y,
    values,
    *,
    ax=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    add_colorbar=True,
    colorbar_label="value",
    xlim=None,
    ylim=None,
    figsize=None,
    **scatter_kwargs,
):
    """Plot metric values as a colored 2-D scatter plot."""
    x, y, values = _arrays(x, y, values)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    mappable = ax.scatter(
        x, y, c=values, cmap=cmap, vmin=vmin, vmax=vmax, **scatter_kwargs
    )

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if add_colorbar:
        colorbar = fig.colorbar(mappable, ax=ax, extend="both")
        colorbar.set_label(colorbar_label)

    ax.grid(True)
    return fig, ax, mappable


def plot_metric_binned(
    x,
    y,
    values,
    *,
    ax=None,
    bins=60,
    cmap="viridis",
    vmin=None,
    vmax=None,
    add_colorbar=True,
    colorbar_label="value",
    xlim=None,
    ylim=None,
    figsize=None,
):
    """Plot bin-wise mean metric values using a 2-D histogram grid."""
    x, y, values = _arrays(x, y, values)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if xlim is None:
        xlim = (float(np.nanmin(x)), float(np.nanmax(x)))
    if ylim is None:
        ylim = (float(np.nanmin(y)), float(np.nanmax(y)))

    counts, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])
    sums, _, _ = np.histogram2d(x, y, bins=bins, range=[xlim, ylim], weights=values)
    mean = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)

    mappable = ax.pcolormesh(
        xedges, yedges, mean.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto"
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    if add_colorbar:
        colorbar = fig.colorbar(mappable, ax=ax, extend="both")
        colorbar.set_label(colorbar_label)

    ax.grid(True)
    return fig, ax, mappable


def plot_metric_triangulation(
    x,
    y,
    values,
    *,
    ax=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    levels=40,
    add_colorbar=True,
    colorbar_label="value",
    xlim=None,
    ylim=None,
    figsize=None,
):
    """Plot a filled triangulated contour map of metric values."""
    x, y, values = _arrays(x, y, values)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))

    if np.isscalar(levels):
        level_bounds = np.linspace(vmin, vmax, int(levels) + 1)
    else:
        level_bounds = np.asarray(levels, dtype=float)

    triangulation = mtri.Triangulation(x, y)
    mappable = ax.tricontourf(
        triangulation, values, levels=level_bounds, cmap=cmap, extend="both"
    )

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if add_colorbar:
        colorbar = fig.colorbar(mappable, ax=ax, extend="both")
        colorbar.set_label(colorbar_label)

    ax.grid(True)
    return fig, ax, mappable
