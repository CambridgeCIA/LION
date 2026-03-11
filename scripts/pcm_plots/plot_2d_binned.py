import numpy as np
import matplotlib.pyplot as plt


def plot_metric_binned(
    sampling_ratio,
    in_order_ratio,
    val,
    *,
    ax=None,
    bins=60,
    cmap="viridis",
    vmin=None,
    vmax=None,
    add_colorbar=True,
    colorbar_label="val",
    xlim=(0, 1),
    ylim=(0, 1),
):
    x = np.asarray(sampling_ratio, float).ravel()
    y = np.asarray(in_order_ratio, float).ravel()
    z = np.asarray(val, float).ravel()

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Bin counts and sum, then compute mean per bin
    counts, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])
    sums, _, _ = np.histogram2d(x, y, bins=bins, range=[xlim, ylim], weights=z)

    mean = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)

    # Mask outside the triangular domain y <= x, and also mask empty bins (nan)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xc, Yc = np.meshgrid(xc, yc, indexing="ij")
    mean[(Yc > Xc)] = np.nan

    m = ax.pcolormesh(
        xedges, yedges, mean.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto"
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("sampling_ratio")
    ax.set_ylabel("in_order_ratio")
    ax.set_aspect("equal", adjustable="box")

    if add_colorbar:
        cbar = fig.colorbar(m, ax=ax, extend="both")
        cbar.set_label(colorbar_label)
    plt.grid()

    return fig, ax, m


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

    fig, ax, _ = plot_metric_binned(
        sampling_ratio,
        in_order_ratio,
        val,
        # cmap="viridis",
        cmap="magma",
        vmin=-2.0,
        vmax=2.0,
        bins=30,
        colorbar_label="toy metric",
    )
    plt.savefig("metric_map_demo_binned.png", dpi=300)
    plt.show()
