from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def boxplot_from_metrics_csvs(
    csv_paths: Sequence[str | Path],
    *,
    metric_col: str = "psnr_recon",
    sampling_rates: Sequence[int] = (20, 50, 80),
    sampling_col: str = "sampling_percentage",
    round_sampling: bool = True,
    title: str | None = None,
    out_path: str | Path | None = None,
) -> None:
    """Make a box plot of a metric across multiple metrics.csv files.

    Each CSV is treated as one trial. For each sampling rate, one value is taken
    from each CSV, and the distribution across CSVs becomes the box.

    Parameters
    ----------
    csv_paths : Sequence[str | Path]
        List of metrics.csv filepaths.
    metric_col : str
        Column to plot (e.g. "psnr_recon", "psnr_zero_filled", "ssim_recon").
    sampling_rates : Sequence[int]
        Sampling percentages to include.
    sampling_col : str
        Sampling column name.
    round_sampling : bool
        If True, rounds sampling percentages before matching.
    title : str | None
        Optional plot title.
    out_path : str | Path | None
        If provided, saves the plot to this path. Otherwise shows it.
    """
    rates = [int(r) for r in sampling_rates]
    values_by_rate = {r: [] for r in rates}

    for p in map(Path, csv_paths):
        df = pd.read_csv(p, skipinitialspace=True)
        df.columns = [str(c).strip() for c in df.columns]

        if sampling_col not in df.columns:
            raise ValueError(f"Missing '{sampling_col}' in {p}")
        if metric_col not in df.columns:
            raise ValueError(f"Missing '{metric_col}' in {p}")

        tmp = df[[sampling_col, metric_col]].copy()
        tmp[sampling_col] = pd.to_numeric(tmp[sampling_col], errors="coerce")
        tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
        tmp = tmp.dropna()

        if round_sampling:
            tmp[sampling_col] = tmp[sampling_col].round().astype(int)
        else:
            tmp[sampling_col] = tmp[sampling_col].astype(int)

        # If a CSV has repeated rows per sampling rate, take the mean for that CSV.
        per_rate = tmp.groupby(sampling_col, as_index=True)[metric_col].mean()

        for r in rates:
            if r not in per_rate.index:
                raise ValueError(f"Sampling rate {r}% not found in {p}")
            values_by_rate[r].append(float(per_rate.loc[r]))

    data = [np.asarray(values_by_rate[r], dtype=float) for r in rates]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    ax.boxplot(data, tick_labels=[f"{r}%" for r in rates], showfliers=True)
    ax.set_xlabel("sampling rate")
    ax.set_ylabel(metric_col)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if out_path is None:
        plt.show()
    else:
        out_path = Path(out_path)
        fig.savefig(out_path)
        plt.close(fig)


if __name__ == "__main__":
    experiment_name = "20260116_053534_example_CIGS_256x256_multilevel_20_trials_pnp"
    # experiment_name = "20260116_063843_example_CIGS_256x256_multilevel_20_trials_spgl1"

    method_name = "pnp_admm_iters=50_eta=0.01_cg_iters=20_drunet_sigma=0.05"
    # factor = 1
    # method_name = f"spgl1_factor={factor}"

    experiment_dir = Path("pcm_demo_output") / experiment_name
    csvs = sorted(experiment_dir.glob(f"trial_*/{method_name}/metrics.csv"))[10:]
    print(f"Found {len(csvs)} metrics CSVs for method={method_name!r}"
          f" under {experiment_dir}")
    boxplot_path = "psnr_boxplot.png"
    boxplot_from_metrics_csvs(
        csvs,
        metric_col="psnr_recon",
        sampling_rates=(20, 50, 80),
        title="PSNR",
        out_path=boxplot_path,
    )
    print(f"Box plot saved to {boxplot_path}")
