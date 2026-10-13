"""Plot PCM reconstruction metrics against sampling percentage."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from LION.pcm.grid_plotting import discover_result_csvs, load_result_dataframe
from LION.pcm.results import RESULT_METRIC_COLUMNS


def plot_pcm_metric_vs_sampling(
    run_dir: Path,
    *,
    metric: str,
    methods: list[str] | None = None,
    categories: list[str] | None = None,
    coarse_j: int | None = None,
    show_std: bool = True,
    figsize: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    dpi: int = 200,
) -> Path:
    """Plot one metric against sampling percentage, with one line per method."""
    if metric not in RESULT_METRIC_COLUMNS:
        raise ValueError(
            f"Unknown PCM result metric {metric!r}. "
            f"Choose one of: {', '.join(RESULT_METRIC_COLUMNS)}."
        )

    csvs = discover_result_csvs(run_dir, methods=methods, categories=categories)

    method_parts: dict[str, list[pd.DataFrame]] = {}
    for method_id, _, csv_path in csvs:
        dataframe = load_result_dataframe(csv_path)
        if coarse_j is not None:
            dataframe = dataframe[dataframe["coarse_j"] == coarse_j]
        if not dataframe.empty:
            method_parts.setdefault(method_id, []).append(dataframe)

    if not method_parts:
        raise ValueError("No result rows match the requested filters.")

    if coarse_j is None:
        coarse_levels = {
            int(value)
            for parts in method_parts.values()
            for dataframe in parts
            for value in dataframe["coarse_j"].unique()
        }
        if len(coarse_levels) != 1:
            raise ValueError(
                "More than one coarse_j is present. "
                "Specify --coarse-j so different coarse levels are not mixed in one line."
            )
        coarse_j = next(iter(coarse_levels))

    fig, ax = plt.subplots(figsize=figsize)

    method_ids = methods if methods is not None else list(method_parts)
    for method_id in method_ids:
        if method_id not in method_parts:
            continue

        dataframe = pd.concat(method_parts[method_id], ignore_index=True)
        dataframe = dataframe[dataframe["coarse_j"] == coarse_j]

        grouped = (
            dataframe.groupby("sampling_rate", as_index=False)[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("sampling_rate")
        )

        x = 100.0 * grouped["sampling_rate"].to_numpy(dtype=float)
        mean = grouped["mean"].to_numpy(dtype=float)
        line = ax.plot(x, mean, marker="o", label=method_id)[0]

        if show_std:
            std = grouped["std"].to_numpy(dtype=float)
            valid = np.isfinite(std)
            if valid.any():
                ax.fill_between(
                    x[valid],
                    mean[valid] - std[valid],
                    mean[valid] + std[valid],
                    alpha=0.2,
                    color=line.get_color(),
                )

    ax.set_xlabel("Sampling percentage")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} vs sampling percentage | coarse J={coarse_j}")
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(0.0, 100.0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid()
    ax.legend()
    fig.tight_layout()

    out_path = (
        Path(run_dir)
        / "plots"
        / "sampling_curve"
        / f"{metric}_vs_sampling_cJ_{coarse_j}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
