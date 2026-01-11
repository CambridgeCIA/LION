from plot_2d_scatter import plot_metric_scatter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm as std_tqdm
from functools import partial

# Use tqdm with dynamic column width that adapts to the terminal width
tqdm = partial(std_tqdm, dynamic_ncols=True)


def plot_pcm_testcase_scatter(
    csv_path: Path,
    metric: str,
    vrange: tuple[float, float],
    title: str,
    filename: str,
    test_dir: Path,
):
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    x = df["sampling_percentage"].to_numpy()
    y = df["in_order_measurements_percentage"].to_numpy()
    y = np.minimum(y, x)  # ensure y <= x
    z = df[metric].to_numpy()

    # vmin = float(np.nanmin(z))
    # vmax = float(np.nanpercentile(z, 95))  # 95th percentile to avoid outliers

    fig, ax, _ = plot_metric_scatter(
        x,
        y,
        z,
        cmap="viridis",
        # cmap="magma",
        # cmap="hot",
        vmin=vrange[0],
        vmax=vrange[1],
        s=140,
        colorbar_label=metric,
        xlim=(-5, 105),
        ylim=(-5, 105),
    )

    ax.set_xlabel("Sampling percentage")
    ax.set_ylabel("In-order measurements percentage")
    ax.set_title(f"{title}")

    ax.plot([0, 100], [0, 100], linestyle="--", linewidth=1)

    outpath = test_dir / f"{filename}.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_all_pcm_testcases():
    demo_output_dir = Path("pcm_demo_output")
    test_dir = demo_output_dir / "20260111_172029_many_testcases"

    cases = [
        {
            "title": "Zero-filled reconstruction PSNR",
            "filename": "psnr_zero_filled",
            "csv_path": test_dir / "pnp_admm.csv",
            "metric": "psnr_zero_filled",
            "vrange": (20, 40),
        },
        {
            "title": "PnP-ADMM reconstruction PSNR",
            "filename": "psnr_pnp_admm",
            "csv_path": test_dir / "pnp_admm.csv",
            "metric": "psnr_recon",
            "vrange": (20, 40),
        },
        {
            "title": "SPGL1 reconstruction PSNR",
            "filename": "psnr_spgl1",
            "csv_path": test_dir / "spgl1.csv",
            "metric": "psnr_recon",
            "vrange": (20, 40),
        },
    ]

    for case in tqdm(cases):
        plot_pcm_testcase_scatter(
            csv_path=case["csv_path"],
            metric=case["metric"],
            vrange=case["vrange"],
            title=case["title"],
            filename=case["filename"],
            test_dir=test_dir,
        )


if __name__ == "__main__":
    plot_all_pcm_testcases()
