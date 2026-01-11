from sympy import plot
from plot_2d_scatter import plot_metric_scatter
from plot_2d_triangulation import plot_metric_triangulation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm as std_tqdm
from functools import partial
from typing import Callable, Any

# Use tqdm with dynamic column width that adapts to the terminal width
tqdm = partial(std_tqdm, dynamic_ncols=True)


def plot_pcm_testcase(
    csv_path: Path,
    metric: str,
    vrange: tuple[float, float],
    title: str,
    filename: str,
    test_dir: Path,
    plot_func: Callable,
    **plot_func_kwargs,
):
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    x = df["sampling_percentage"].to_numpy()
    y = df["in_order_measurements_percentage"].to_numpy()
    y = np.minimum(y, x)  # ensure y <= x
    z = df[metric].to_numpy()

    # vmin = float(np.nanmin(z))
    # vmax = float(np.nanpercentile(z, 95))  # 95th percentile to avoid outliers

    fig, ax, _ = plot_func(
        x,
        y,
        z,
        cmap="viridis",
        # cmap="magma",
        # cmap="hot",
        vmin=vrange[0],
        vmax=vrange[1],
        # s=140,
        colorbar_label=metric,
        xlim=(-5, 105),
        ylim=(-5, 105),
        **plot_func_kwargs,
    )

    ax.set_xlabel("Sampling percentage")
    ax.set_ylabel("In-order measurements percentage")
    ax.set_title(f"{title}")

    ax.plot([0, 100], [0, 100], linestyle="--", linewidth=1)

    outdir = test_dir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{filename}.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_all_pcm_testcases():
    demo_output_dir = Path("pcm_demo_output")
    test_names = [
        "20260111_204145_CIGS",
        "20260111_204307_silicon",
        "20260111_204326_organic",
        "20260111_204341_perovskite",
    ]

    psnr_vrange = (20.0, 40.0)

    psnr_cases = [
        {
            "title": "Zero-filled reconstruction PSNR",
            "filename": "psnr_zero_filled",
            "csv_path": "pnp_admm/metrics.csv",
            "metric": "psnr_zero_filled",
            "vrange": psnr_vrange,
        },
        {
            "title": "PnP-ADMM reconstruction PSNR",
            "filename": "psnr_pnp_admm",
            "csv_path": "pnp_admm/metrics.csv",
            "metric": "psnr_recon",
            "vrange": psnr_vrange,
        },
        {
            "title": "SPGL1 reconstruction PSNR",
            "filename": "psnr_spgl1",
            "csv_path": "spgl1/metrics.csv",
            "metric": "psnr_recon",
            "vrange": psnr_vrange,
        },
    ]

    for test_name in tqdm(test_names, desc="Test cases"):
        test_dir = demo_output_dir / test_name


        plot_styles = {
            "scatter_round": {
                "plot_func": plot_metric_scatter,
                "kwargs": {
                    "s": 140,
                    "marker": "o",  # round markers
                },
            },
            "scatter_square": {
                "plot_func": plot_metric_scatter,
                "kwargs": {
                    "s": 140,
                    "marker": "s",  # square markers
                },
            },
            "triangulation": {
                "plot_func": plot_metric_triangulation,
                "kwargs": {"levels": 40},
            },
        }

        for case in tqdm(psnr_cases):
            for plot_style, plot_info in plot_styles.items():
                plot_pcm_testcase(
                    csv_path=test_dir / case["csv_path"],
                    metric=case["metric"],
                    vrange=psnr_vrange,
                    title=case["title"],
                    test_dir=test_dir,
                    filename=f'{case["filename"]}_{plot_style}',
                    plot_func=plot_info["plot_func"],
                    **plot_info["kwargs"],
                )


if __name__ == "__main__":
    plot_all_pcm_testcases()
