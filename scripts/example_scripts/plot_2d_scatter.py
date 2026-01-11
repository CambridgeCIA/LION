import numpy as np
import matplotlib.pyplot as plt

def plot_metric_scatter(
    sampling_ratio,
    in_order_ratio,
    val,
    *,
    ax=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    s=25,
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

    sc = ax.scatter(x, y, c=z, cmap=cmap, vmin=vmin, vmax=vmax, s=s)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("sampling_ratio")
    ax.set_ylabel("in_order_ratio")
    ax.set_aspect("equal", adjustable="box")

    if add_colorbar:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(colorbar_label)
    plt.grid()

    return fig, ax, sc


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

    fig, ax, _ = plot_metric_scatter(
        sampling_ratio,
        in_order_ratio,
        val,
        # cmap="viridis",
        cmap="magma",
        vmin=-2.0,
        vmax=2.0,
        s=100,
        colorbar_label="toy metric",
    )
    plt.savefig("metric_map_demo_scatter.png", dpi=300)
    plt.show()
