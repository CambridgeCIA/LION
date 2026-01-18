from __future__ import annotations

from pathlib import Path
import pandas as pd
from functools import partial
from tqdm import tqdm as std_tqdm
# Use tqdm with dynamic column width that adapts to the terminal width
tqdm = partial(std_tqdm, dynamic_ncols=True)

def summarise_trials_by_sampling_and_coarse_J(
    experiment_dir: Path,
    method_name: str,
    num_trials: int,
    *,
    metrics_filename: str = "metrics.csv",
    sampling_round_ndigits: int = 0,
    out_csv: Path | None = None,
    ddof: int = 1,
) -> pd.DataFrame:
    """
    Combine per-trial metrics CSVs and compute mean/std grouped by
    (sampling_percentage, coarse_J).

    Expected file layout
    --------------------
    {experiment_dir}/trial_{t}/{method_name}/{metrics_filename}

    Parameters
    ----------
    experiment_dir:
        Root directory containing trial folders.
    method_name:
        Method subfolder name under each trial, e.g. "pnp_admm".
    num_trials:
        Number of trials to attempt to read, using trial indices 0..num_trials-1.
    metrics_filename:
        Metrics CSV filename (default "metrics.csv").
    sampling_round_ndigits:
        Rounding applied to sampling_percentage before grouping.
        Use 0 to round to nearest integer.
    out_csv:
        Optional output CSV path to save the summary.
    ddof:
        Delta degrees of freedom for the standard deviation.
        ddof=1 gives sample std (pandas default). ddof=0 gives population std.

    Returns
    -------
    summary:
        DataFrame with columns like "mse_recon_mean", "mse_recon_std", and "n".
    """
    dfs: list[pd.DataFrame] = []
    missing: list[Path] = []

    for t in range(num_trials):
        trial_dir = experiment_dir / f"trial_{t}"
        # assert trial_dir.is_dir(), f"Trial directory not found: {trial_dir}"
        method_dir = trial_dir / method_name
        # assert method_dir.is_dir(), f"Method directory not found: {method_dir}"
        csv_path = method_dir / metrics_filename
        if not csv_path.is_file():
            missing.append(csv_path)
            continue

        df = pd.read_csv(csv_path, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        dfs.append(df)

    if not dfs:
        msg = f"No metrics CSVs found for method={method_name!r} under {experiment_dir}."
        if missing:
            msg += f" Example missing path: {missing[0]}"
        raise FileNotFoundError(msg)

    all_df = pd.concat(dfs, ignore_index=True)

    all_df["sampling_percentage"] = pd.to_numeric(all_df["sampling_percentage"], errors="coerce")
    all_df["coarse_J"] = pd.to_numeric(all_df["coarse_J"], errors="coerce")

    all_df = all_df.dropna(subset=["sampling_percentage", "coarse_J"]).copy()

    all_df["sampling_percentage"] = all_df["sampling_percentage"].round(sampling_round_ndigits).astype(int)
    all_df["coarse_J"] = all_df["coarse_J"].round(0).astype(int)

    group_cols = ["sampling_percentage", "coarse_J"]
    value_cols = [c for c in all_df.columns if c not in group_cols]

    grouped = all_df.groupby(group_cols)

    mean_df = grouped[value_cols].mean(numeric_only=True).add_suffix("_mean")
    std_df = grouped[value_cols].std(numeric_only=True, ddof=ddof).add_suffix("_std")
    n_df = grouped.size().rename("n")

    summary = pd.concat([mean_df, std_df, n_df], axis=1).reset_index()
    summary = summary.sort_values(group_cols, ascending=True, kind="stable").reset_index(drop=True)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_csv, index=False)

    return summary


# Example usage
if __name__ == "__main__":
    # experiment_name = "20260116_053534_example_CIGS_256x256_multilevel_20_trials_pnp"
    # experiment_name = "20260116_063843_example_CIGS_256x256_multilevel_20_trials_spgl1"
    experiment_name = "20260116_170524_Si_2_256_512x512_multilevel_20_trials_pnp_and_spgl1"
    experiment_dir = Path("pcm_demo_output") / experiment_name
    assert experiment_dir.is_dir(), f"Experiment directory not found: {experiment_dir}"
    # factor = 1
    factor = 1e5
    methods = [
        "pnp_admm_iters=50_eta=0.01_cg_iters=20_drunet_sigma=0.05",
        f"spgl1_factor={factor}",
    ]
    num_trials = 10

    for method_name in tqdm(methods):
        summary = summarise_trials_by_sampling_and_coarse_J(
            experiment_dir=experiment_dir,
            method_name=method_name,
            num_trials=num_trials,
            metrics_filename="metrics.csv",
            sampling_round_ndigits=0,  # nearest integer
            out_csv=experiment_dir / f"summary_{method_name}.csv",
        )
