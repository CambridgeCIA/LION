import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def plot_metric_triangulation(
    sampling_ratio,
    in_order_ratio,
    val,
    *,
    ax=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    levels=40,
    add_colorbar=True,
    colorbar_label="val",
    xlim=(0, 1),
    ylim=(0, 1),
):
    x = np.asarray(sampling_ratio, dtype=float).ravel()
    y = np.asarray(in_order_ratio, dtype=float).ravel()
    z = np.asarray(val, dtype=float).ravel()

    if not (x.shape == y.shape == z.shape):
        raise ValueError(
            "sampling_ratio, in_order_ratio, and val must have the same shape after ravel()."
        )

    # if np.any(y > x + 1e-12):
    #     raise ValueError("Constraint violated: in_order_ratio must be <= sampling_ratio elementwise.")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    tri = mtri.Triangulation(x, y)

    # Decide colour scale limits
    if vmin is None:
        vmin = np.nanmin(z)
    if vmax is None:
        vmax = np.nanmax(z)

    # Key point: make contour levels cover [vmin, vmax]
    if np.isscalar(levels):
        # boundaries; +1 so the top boundary is exactly vmax
        level_bounds = np.linspace(vmin, vmax, int(levels) + 1)
    else:
        level_bounds = np.asarray(levels, dtype=float)

    mappable = ax.tricontourf(
        tri,
        z,
        levels=level_bounds,
        cmap=cmap,
        extend="both",  # show out-of-range values as extensions on the colourbar
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("sampling_ratio")
    ax.set_ylabel("in_order_ratio")
    ax.set_title("Metric map over 0 <= in_order_ratio <= sampling_ratio <= 1")
    ax.set_aspect("equal", adjustable="box")

    if add_colorbar:
        cbar = fig.colorbar(mappable, ax=ax, extend="both")
        cbar.set_label(colorbar_label)

    ax.grid(True)

    return fig, ax, mappable


def plot_metric_map_from_triples(
    triples,
    *,
    ax=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    levels=40,
    add_colorbar=True,
    colorbar_label="val",
):
    """
    Convenience wrapper for an (N, 3) array/list of (sampling_ratio, in_order_ratio, val).
    """
    arr = np.asarray(triples, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("triples must have shape (N, 3).")
    return plot_metric_triangulation(
        arr[:, 0],
        arr[:, 1],
        arr[:, 2],
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        levels=levels,
        add_colorbar=add_colorbar,
        colorbar_label=colorbar_label,
    )


if __name__ == "__main__":
    # ---- Toy data + demo ----
    rng = np.random.default_rng(0)
    n = 10
    sampling_ratio = rng.uniform(0.0, 1.0, size=n)
    in_order_ratio = rng.uniform(0.0, sampling_ratio)

    val = (
        0.8 * np.sin(2 * np.pi * sampling_ratio)
        + 0.6 * np.cos(3 * np.pi * in_order_ratio)
        + 0.4 * sampling_ratio * (1.0 - in_order_ratio)
        + 0.15 * rng.normal(size=n)
    )

    fig, ax, _ = plot_metric_triangulation(
        sampling_ratio,
        in_order_ratio,
        val,
        # cmap="viridis",
        cmap="magma",
        vmin=-2.0,
        vmax=2.0,
        levels=50,
        colorbar_label="toy metric",
    )
    plt.savefig("metric_map_demo_triangulation.png", dpi=300)
    plt.show()
