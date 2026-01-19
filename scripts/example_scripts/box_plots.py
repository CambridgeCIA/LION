from __future__ import annotations

from pathlib import Path
import random
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import spgl1


def boxplot_from_metrics_csvs(
    csv_paths: Sequence[str | Path],
    *,
    metric_col: str = "psnr_recon",
    sampling_rates: Sequence[int] = (20, 50, 80),
    sampling_col: str = "sampling_percentage",
    round_sampling: bool = True,
    figsize: tuple[float, float] = (8, 6.5),
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

    fig, ax = plt.subplots(figsize=figsize, dpi=160)

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




def boxplot_recon_and_zero_filled_from_metrics_csvs(
    csv_paths: Sequence[str | Path],
    *,
    metric_prefix: str = "psnr",
    sampling_rates: Sequence[int] = (20, 50, 80),
    sampling_col: str = "sampling_percentage",
    round_sampling: bool = True,
    figsize: tuple[float, float] = (8, 6.5),
    title: str | None = None,
    out_path: str | Path | None = None,
) -> None:
    """Box plot for both <metric>_recon and <metric>_zero_filled from many metrics.csv files."""
    rates = [int(r) for r in sampling_rates]
    recon_col = f"{metric_prefix}_recon"
    zf_col = f"{metric_prefix}_zero_filled"

    recon_by_rate = {r: [] for r in rates}
    zf_by_rate = {r: [] for r in rates}

    for p in map(Path, csv_paths):
        df = pd.read_csv(p, skipinitialspace=True)
        df.columns = [str(c).strip() for c in df.columns]

        for col in (sampling_col, recon_col, zf_col):
            if col not in df.columns:
                raise ValueError(f"Missing '{col}' in {p}")

        tmp = df[[sampling_col, recon_col, zf_col]].copy()
        tmp[sampling_col] = pd.to_numeric(tmp[sampling_col], errors="coerce")
        tmp[recon_col] = pd.to_numeric(tmp[recon_col], errors="coerce")
        tmp[zf_col] = pd.to_numeric(tmp[zf_col], errors="coerce")
        tmp = tmp.dropna()

        if round_sampling:
            tmp[sampling_col] = tmp[sampling_col].round().astype(int)
        else:
            tmp[sampling_col] = tmp[sampling_col].astype(int)

        per_rate_recon = tmp.groupby(sampling_col, as_index=True)[recon_col].mean()
        per_rate_zf = tmp.groupby(sampling_col, as_index=True)[zf_col].mean()

        for r in rates:
            if r not in per_rate_recon.index or r not in per_rate_zf.index:
                raise ValueError(f"Sampling rate {r}% not found in {p}")
            recon_by_rate[r].append(float(per_rate_recon.loc[r]))
            zf_by_rate[r].append(float(per_rate_zf.loc[r]))

    recon_data = [np.asarray(recon_by_rate[r], dtype=float) for r in rates]
    zf_data = [np.asarray(zf_by_rate[r], dtype=float) for r in rates]

    fig, ax = plt.subplots(figsize=figsize, dpi=160)

    base_pos = np.arange(len(rates), dtype=float)
    width = 0.32
    offset = 0.18

    bp_zf = ax.boxplot(
        zf_data,
        positions=base_pos - offset,
        widths=width,
        patch_artist=True,
        showfliers=True,
        manage_ticks=False,
    )
    bp_recon = ax.boxplot(
        recon_data,
        positions=base_pos + offset,
        widths=width,
        patch_artist=True,
        showfliers=True,
        manage_ticks=False,
    )

    # Simple colouring
    for b in bp_zf["boxes"]:
        b.set_facecolor("#BDBDBD")
    for b in bp_recon["boxes"]:
        b.set_facecolor("#8ECae6")

    ax.set_xticks(base_pos)
    ax.set_xticklabels([f"{r}%" for r in rates])
    ax.set_xlabel("sampling rate")
    ax.set_ylabel(metric_prefix)
    if title is not None:
        ax.set_title(title)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend([bp_zf["boxes"][0], bp_recon["boxes"][0]], ["Zero-filled", "Recon"], loc="upper right", frameon=True)

    fig.tight_layout()

    if out_path is None:
        plt.show()
    else:
        out_path = Path(out_path)
        fig.savefig(out_path)
        plt.close(fig)




def combined_boxplot_spgl1_pnp(
    spgl1_csvs: Sequence[str | Path],
    pnp_csvs: Sequence[str | Path],
    *,
    metric_prefix: str = "psnr",
    sampling_rates: Sequence[int] = (20, 50, 80),
    sampling_col: str = "sampling_percentage",
    round_sampling: bool = True,
    figsize: tuple[float, float] = (8, 6.5),
    vrange: tuple[float, float] | None = None,
    title: str | None = None,
    out_path: str | Path | None = None,
) -> None:
    """Grouped boxplot: Zero-filled, SPGL1, PnP-ADMM for each sampling rate.

    Each CSV is treated as one trial. For each sampling rate, one value is taken
    from each CSV, and the distribution across trials becomes the box.
    """
    if len(spgl1_csvs) != len(pnp_csvs):
        raise ValueError(f"SPGL1 and PnP CSV counts differ! {len(spgl1_csvs)} vs {len(pnp_csvs)}")

    if len(spgl1_csvs) == 0:
        raise ValueError("No CSVs provided!")

    rates = [int(r) for r in sampling_rates]
    zf_col = f"{metric_prefix}_zero_filled"
    recon_col = f"{metric_prefix}_recon"

    def collect(csv_paths: Sequence[str | Path], metric_col: str) -> dict[int, np.ndarray]:
        values_by_rate = {r: [] for r in rates}
        for p in map(Path, csv_paths):
            df = pd.read_csv(p, skipinitialspace=True)
            df.columns = [str(c).strip() for c in df.columns]

            for col in (sampling_col, metric_col):
                if col not in df.columns:
                    raise ValueError(f"Missing '{col}' in {p}")

            tmp = df[[sampling_col, metric_col]].copy()
            tmp[sampling_col] = pd.to_numeric(tmp[sampling_col], errors="coerce")
            tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
            tmp = tmp.dropna()

            if round_sampling:
                tmp[sampling_col] = tmp[sampling_col].round().astype(int)
            else:
                tmp[sampling_col] = tmp[sampling_col].astype(int)

            per_rate = tmp.groupby(sampling_col, as_index=True)[metric_col].mean()

            for r in rates:
                if r not in per_rate.index:
                    raise ValueError(f"Sampling rate {r}% not found in {p}")
                values_by_rate[r].append(float(per_rate.loc[r]))

        return {r: np.asarray(values_by_rate[r], dtype=float) for r in rates}

    # Collect distributions
    zf_spgl = collect(spgl1_csvs, zf_col)
    zf_pnp = collect(pnp_csvs, zf_col)

    for r in rates:
        if zf_spgl[r].shape != zf_pnp[r].shape:
            raise ValueError(f"Zero-filled trial count mismatch at {r}%")
        if not np.allclose(zf_spgl[r], zf_pnp[r], rtol=0.0, atol=1e-10):
            trial_indices_where_differ = np.where(~np.isclose(zf_spgl[r], zf_pnp[r], rtol=0.0, atol=1e-10))[0]
            values_differ = [(zf_spgl[r][i], zf_pnp[r][i]) for i in trial_indices_where_differ]
            raise ValueError(f"Zero-filled values differ between SPGL1 and PnP at {r}% for trials {trial_indices_where_differ}. Values: {values_differ}")

    zf = zf_spgl
    spgl = collect(spgl1_csvs, recon_col)
    pnp = collect(pnp_csvs, recon_col)

    zf_data = [zf[r] for r in rates]
    spgl_data = [spgl[r] for r in rates]
    pnp_data = [pnp[r] for r in rates]

    fig, ax = plt.subplots(figsize=figsize, dpi=160)

    base_pos = np.arange(len(rates), dtype=float)
    width = 0.22
    offsets = [-0.25, 0.0, 0.25]

    bp_zf = ax.boxplot(
        zf_data,
        positions=base_pos + offsets[0],
        widths=width,
        patch_artist=True,
        showfliers=True,
        manage_ticks=False,
    )
    bp_spgl = ax.boxplot(
        spgl_data,
        positions=base_pos + offsets[1],
        widths=width,
        patch_artist=True,
        showfliers=True,
        manage_ticks=False,
    )
    bp_pnp = ax.boxplot(
        pnp_data,
        positions=base_pos + offsets[2],
        widths=width,
        patch_artist=True,
        showfliers=True,
        manage_ticks=False,
    )

    # colors = {
    #     "Zero-filled": "#9E9E9E",  # darker grey
    #     "SPGL1": "#1F77B4",        # strong blue
    #     "PnP-ADMM": "#2CA02C",     # strong green
    # }
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {"Zero-filled": cycle[0], "SPGL1": cycle[1], "PnP-ADMM": cycle[2]}

    for b in bp_zf["boxes"]:
        b.set_facecolor(colors["Zero-filled"])
    for b in bp_spgl["boxes"]:
        b.set_facecolor(colors["SPGL1"])
    for b in bp_pnp["boxes"]:
        b.set_facecolor(colors["PnP-ADMM"])

    ax.set_xticks(base_pos)
    ax.set_xticklabels([f"{r}%" for r in rates])
    ax.set_xlabel("sampling rate")
    ax.set_ylabel(metric_prefix)
    if vrange is not None:
        ax.set_ylim(*vrange)
    if title is not None:
        ax.set_title(title)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(
        [bp_zf["boxes"][0], bp_spgl["boxes"][0], bp_pnp["boxes"][0]],
        ["Zero-filled", "SPGL1", "PnP-ADMM"],
        loc="lower right",
        frameon=True,
    )

    fig.tight_layout()

    if out_path is None:
        plt.show()
    else:
        out_path = Path(out_path)
        fig.savefig(out_path)
        plt.close(fig)



if __name__ == "__main__":
    randomization_scheme = "multilevel"
    # randomization_scheme = "uniform"

    # data_name = "example_CIGS_256x256"
    # pnp_experiment_name = f"20260116_053534_{data_name}_{randomization_scheme}_20_trials_pnp"
    # spgl1_experiment_name = f"20260116_063843_{data_name}_{randomization_scheme}_20_trials_spgl1"

    data_name = "example_silicon_512x512"
    pnp_experiment_name = f"20260118_055332_{data_name}_{randomization_scheme}_10_trials_pnp_and_spgl1"
    spgl1_experiment_name = f"20260118_055332_{data_name}_{randomization_scheme}_10_trials_pnp_and_spgl1"

    data_name = "Si_2_256_512x512"
    pnp_experiment_name =   f"20260116_170524_{data_name}_{randomization_scheme}_20_trials_pnp_and_spgl1"
    spgl1_experiment_name = f"20260116_170524_{data_name}_{randomization_scheme}_20_trials_pnp_and_spgl1"

    pnp_method_name = "pnp_admm_iters=50_eta=0.01_cg_iters=20_drunet_sigma=0.05"
    # factor = 1
    factor = 100000.0
    spgl1_method_name = f"spgl1_factor={factor}"

    all_output_dir = Path("pcm_demo_output")

    pnp_experiment_dir = all_output_dir / pnp_experiment_name
    pnp_csvs = sorted(pnp_experiment_dir.glob(f"trial_*/{pnp_method_name}/metrics.csv"))
    pnp_csvs = pnp_csvs[:10]  # use only first 10 trials

    spgl1_experiment_dir = all_output_dir / spgl1_experiment_name
    spgl1_csvs = sorted(spgl1_experiment_dir.glob(f"trial_*/{spgl1_method_name}/metrics.csv"))
    spgl1_csvs = spgl1_csvs[:10]  # use only first 10 trials

    output_dir = all_output_dir / "combined_boxplots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(pnp_csvs)} metrics CSVs for method={pnp_method_name!r}"
          f" under {pnp_experiment_dir}")
    print(f"Found {len(spgl1_csvs)} metrics CSVs for method={spgl1_method_name!r}"
          f" under {spgl1_experiment_dir}")
    boxplot_path = output_dir / f"{data_name}_{randomization_scheme}_boxplot.png"

    # boxplot_from_metrics_csvs(
    #     csvs,
    #     metric_col="psnr_recon",
    #     sampling_rates=(20, 50, 80),
    #     title="PSNR",
    #     out_path=boxplot_path,
    # )
    # boxplot_recon_and_zero_filled_from_metrics_csvs(
    #     csvs,
    #     metric_prefix="psnr",
    #     sampling_rates=(20, 50, 80),
    #     title="PSNR",
    #     out_path=boxplot_path,
    # )
    combined_boxplot_spgl1_pnp(
        spgl1_csvs=spgl1_csvs,
        pnp_csvs=pnp_csvs,
        metric_prefix="psnr",
        sampling_rates=(20, 50, 80),
        figsize=(8.0, 5.5),
        vrange=(27, 44),
        title="PSNR Comparison",
        out_path=boxplot_path,
    )

    print(f"Box plot saved to {boxplot_path}")
