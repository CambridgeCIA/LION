import re
from sympy import plot, use
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
    J: int,
    uses_coarse_J: bool,
    csv_path: Path,
    metric: str,
    metric_name: str,
    reverse: bool,
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

    if uses_coarse_J:
        assert "coarse_J" in df.columns, "DataFrame must contain 'coarse_J' column."
        y = df["coarse_J"].to_numpy()
        ylabel = "Coarse J"
        yrange = (0, J)
        yrange_expanded = (-0.5, J + 0.5)
    else:  # use in_order_measurements_percentage
        if "in_order_measurements_percentage" not in df.columns:
            assert "coarse_J" in df.columns, "DataFrame must contain either 'in_order_measurements_percentage' or 'coarse_J' column."
            coarse_J = df["coarse_J"].to_numpy()
            num_pixels = 1 << (2 * J)
            in_order_measurements = 1 << (2 * coarse_J)
            in_order_measurements_percentage = (in_order_measurements / num_pixels) * 100
            # Take the minimum between in_order_measurements_percentage and sampling_percentage
            in_order_measurements_percentage = np.minimum(in_order_measurements_percentage, x)
            df["in_order_measurements_percentage"] = in_order_measurements_percentage
        y = df["in_order_measurements_percentage"].to_numpy()
        y = np.minimum(y, x)  # ensure y <= x
        ylabel = "In-order measurements percentage"
        yrange = (0, 100)
        yrange_expanded = (-5, 105)

    z = df[metric].to_numpy()

    if reverse:
        x = x[::-1]
        y = y[::-1]
        z = z[::-1]

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
        colorbar_label=metric_name,
        xlim=(-5, 105),
        # ylim=(-5, 105),
        ylim=yrange_expanded,
        **plot_func_kwargs,
    )

    ax: plt.Axes = ax  # type hint for IDEs

    if uses_coarse_J:
        ax.set_aspect('auto', adjustable='box')
    else:
        ax.set_aspect('equal', adjustable='box')
        ax.plot(
            [0, 100],
            # [0, 100],
            yrange,
            linestyle="--", linewidth=1)

    ax.set_xlabel("Sampling percentage")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}")

    outdir = test_dir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{filename}.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_all_pcm_testcases():
    demo_output_dir = Path("pcm_demo_output")
    test_names = [
        # "20260111_204145_CIGS",
        # "20260111_204307_silicon",
        # "20260111_204326_organic",
        # "20260111_204341_perovskite",
        # "20260112_235416_CIGS",
        # "20260113_175847_CIGS",
        # ("20260114_050139_CIGS_256x256_multilevel_spgl1_ok", 8, True),
        # ("20260114_044258_CIGS_256x256_uniform_pnp_admm_eta_1_not_good", 8, True),
        # ("20260114_051023_CIGS_256x256_multilevel_pnp_eta_1_good", 8, True)
        # ("20260114_064001_CIGS_256x256_uniform", 8, True)
        # ("20260114_093314_silicon_512x512_multilevel_pnp_all", 9, True),
        # ("20260114_140813_silicon_512x512_multilevel_spgl1_all_BEST", 9, True),
        # ("20260114_175241_CIGS_256x256_multilevel", 8, True),
        # ("20260114_183643_silicon_512x512_multilevel_pnp_eta_0.01_admm_iters_50_all_BEST", 9, True),
        # ("20260114_200650_perovskite_256x256_multilevel_pnp_and_spgl1_all", 8, True),
        # ("20260114_220039_organic_256x256_multilevel_pnp_and_spgl1_all", 8, True),
        # ("20260115_000157_silicon_256x256_multilevel_pnp_eta_0.01_and_spgl1_all", 8, True),
        # ("20260115_082520_Si_2_256_measurement_data_multilevel_noise_0", 8, True),
        ("20260115_105359_Si_2_256_512x512_multilevel_noise_0_pnp_and_admm_all", 9, True),
    ]
    eta = 0.01
    # eta = 0.1
    # eta = 1
    # eta = 100
    # admm_iters = 20
    admm_iters = 50
    # admm_iters = 100
    cg_iters = 20
    # cg_iters = 50
    drunet_sigma = 0.05
    spgl1_factor = 1e5
    # pnp_admm_csv_path = "pnp_admm/metrics.csv"
    pnp_admm_csv_path = f"pnp_admm_iters={admm_iters}_eta={eta}_cg_iters={cg_iters}_drunet_sigma={drunet_sigma}/metrics.csv"
    # pcm_demo_output/20260115_082520_Si_2_256_measurement_data_multilevel_noise_0/pnp_admm_iters=100_eta=0.01_cg_iters=50_drunet_sigma=0.05/metrics.csv
    # spgl1_csv_path = "spgl1/metrics.csv"
    spgl1_csv_path = f"spgl1_factor={spgl1_factor}/metrics.csv"
    reverse = True

    # psnr_vrange = (10.0, 40.0)
    # psnr_vrange = (30.0, 40.0)
    # psnr_vrange = (10.0, 50.0)
    # psnr_vrange = (30.0, 50.0)
    # psnr_vrange = (10.0, 30.0)
    psnr_vrange = (20.0, 30.0)
    # psnr_vrange = (25.0, 35.0)
    # psnr_vrange = (10.0, 30.0)
    # psnr_vrange = (18.0, 28.0)
    # psnr_vrange = (20.0, 28.0)
    # psnr_vrange = (15.0, 30.0)
    # psnr_vrange = (16.0, 30.0)
    # psnr_vrange = (20.0, 40.0)
    # psnr_vrange = (35.0, 45.0)
    # psnr_vrange = (20.0, 50.0)
    # psnr_vrange = (20.0, 60.0)

    psnr_cases = [
        {
            "title": "Zero-filled reconstruction PSNR",
            "filename": "psnr_zero_filled",
            "csv_path": pnp_admm_csv_path,
            # "csv_path": spgl1_csv_path,
            "metric": "psnr_zero_filled",
            "metric_name": "PSNR",
            "vrange": psnr_vrange,
        },
        {
            "title": "PnP-ADMM reconstruction PSNR",
            "filename": "psnr_pnp_admm",
            "csv_path": pnp_admm_csv_path,
            "metric": "psnr_recon",
            "metric_name": "PSNR",
            "vrange": psnr_vrange,
        },
        {
            "title": "SPGL1 reconstruction PSNR",
            "filename": "psnr_spgl1",
            "csv_path": spgl1_csv_path,
            "metric": "psnr_recon",
            "metric_name": "PSNR",
            "vrange": psnr_vrange,
        },
    ]

    for test_name, J, uses_coarse_J in tqdm(test_names, desc="Test cases"):
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
                    J=J,
                    uses_coarse_J=uses_coarse_J,
                    csv_path=test_dir / case["csv_path"],
                    metric=case["metric"],
                    metric_name=case["metric_name"],
                    reverse=reverse,
                    vrange=psnr_vrange,
                    title=case["title"],
                    test_dir=test_dir,
                    filename=f'{case["filename"]}_{plot_style}_uses_coarse_J_{uses_coarse_J}',
                    plot_func=plot_info["plot_func"],
                    **plot_info["kwargs"],
                )


if __name__ == "__main__":
    plot_all_pcm_testcases()
