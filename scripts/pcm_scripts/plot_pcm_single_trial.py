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


def plot_testcase(
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
            assert (
                "coarse_J" in df.columns
            ), "DataFrame must contain either 'in_order_measurements_percentage' or 'coarse_J' column."
            coarse_J = df["coarse_J"].to_numpy()
            num_pixels = 1 << (2 * J)
            in_order_measurements = 1 << (2 * coarse_J)
            in_order_measurements_percentage = (
                in_order_measurements / num_pixels
            ) * 100
            # Take the minimum between in_order_measurements_percentage and sampling_percentage
            in_order_measurements_percentage = np.minimum(
                in_order_measurements_percentage, x
            )
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
        ax.set_aspect("auto", adjustable="box")
    else:
        ax.set_aspect("equal", adjustable="box")
        ax.plot(
            [0, 100],
            # [0, 100],
            yrange,
            linestyle="--",
            linewidth=1,
        )

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
        # ("20260114_175241_CIGS_256x256_multilevel_pnp_eta_0.01_and_spgl1_all", 8, True, (20, 40), (0.5, 1.0)),
        # ("20260114_183643_silicon_512x512_multilevel_pnp_eta_0.01_and_spgl1_all", 9, True, (20, 30), (0.5, 0.9)),
        # ("20260115_105359_Si_2_256_512x512_multilevel_noise_0_pnp_eta_0.01_and_spgl1_all", 9, True, (20, 30), (0.2, 0.8)),
        # ("20260115_204000_Si_256_512x512_multilevel_noise_0_pnp_eta_0.01_and_spgl1_all", 9, True, (20, 30), (0.2, 0.8)),
        (
            "20260115_225211_Si_256_measurement_data_multilevel_noise_0_pnp_eta_0.01_and_spgl1_all",
            8,
            True,
            (20, 30),
            (0.2, 0.8),
        ),
    ]
    eta = 0.01
    admm_iters = 50
    cg_iters = 20
    drunet_sigma = 0.05
    # spgl1_factor = 1e5
    spgl1_factor = 1e7

    # pnp_admm_csv_path = "pnp_admm/metrics.csv"
    pnp_admm_csv_path = f"pnp_admm_iters={admm_iters}_eta={eta}_cg_iters={cg_iters}_drunet_sigma={drunet_sigma}/metrics.csv"
    # pcm_demo_output/20260115_082520_Si_2_256_measurement_data_multilevel_noise_0/pnp_admm_iters=100_eta=0.01_cg_iters=50_drunet_sigma=0.05/metrics.csv

    # spgl1_csv_path = "spgl1/metrics.csv"
    spgl1_csv_path = f"spgl1_factor={spgl1_factor}/metrics.csv"

    zero_filled_csv_path = pnp_admm_csv_path
    # zero_filled_csv_path = spgl1_csv_path
    reverse = True

    test_cases = [
        {
            "title": "Zero-filled reconstruction PSNR",
            "filename": "psnr_zero_filled",
            "metric": "psnr_zero_filled",
            "metric_name": "PSNR",
        },
        {
            "title": "PnP-ADMM reconstruction PSNR",
            "filename": "psnr_pnp_admm",
            "metric": "psnr_recon",
            "metric_name": "PSNR",
        },
        {
            "title": "SPGL1 reconstruction PSNR",
            "filename": "psnr_spgl1",
            "metric": "psnr_recon",
            "metric_name": "PSNR",
        },
        {
            "title": "Zero-filled reconstruction SSIM",
            "filename": "ssim_zero_filled",
            "metric": "ssim_zero_filled",
            "metric_name": "SSIM",
        },
        {
            "title": "PnP-ADMM reconstruction SSIM",
            "filename": "ssim_pnp_admm",
            "metric": "ssim_recon",
            "metric_name": "SSIM",
        },
        {
            "title": "SPGL1 reconstruction SSIM",
            "filename": "ssim_spgl1",
            "metric": "ssim_recon",
            "metric_name": "SSIM",
        },
    ]

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

    for test_name, J, uses_coarse_J, psnr_range, ssim_range in tqdm(
        test_names, desc="Test cases"
    ):
        test_dir = demo_output_dir / test_name
        for test_case in tqdm(test_cases):
            for plot_style, plot_info in plot_styles.items():
                metric_prefix = test_case["metric"].split("_")[0]
                if metric_prefix == "psnr":
                    test_case["vrange"] = psnr_range
                elif metric_prefix == "ssim":
                    test_case["vrange"] = ssim_range
                else:
                    raise ValueError(
                        f"Unknown metric prefix: {metric_prefix} in metric {test_case['metric']}"
                    )
                if "pnp_admm" in test_case["filename"]:
                    csv_path = pnp_admm_csv_path
                elif "spgl1" in test_case["filename"]:
                    csv_path = spgl1_csv_path
                elif "zero_filled" in test_case["filename"]:
                    csv_path = zero_filled_csv_path
                else:
                    raise ValueError(
                        f"Unknown method in filename: {test_case['filename']}"
                    )
                plot_testcase(
                    J=J,
                    uses_coarse_J=uses_coarse_J,
                    csv_path=test_dir / csv_path,
                    metric=test_case["metric"],
                    metric_name=test_case["metric_name"],
                    reverse=reverse,
                    vrange=test_case["vrange"],
                    title=test_case["title"],
                    test_dir=test_dir,
                    filename=f'{test_case["filename"]}_{plot_style}_uses_coarse_J_{uses_coarse_J}',
                    plot_func=plot_info["plot_func"],
                    **plot_info["kwargs"],
                )


if __name__ == "__main__":
    plot_all_pcm_testcases()
