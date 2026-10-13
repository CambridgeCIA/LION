"""Plot aggregated PCM metrics over sampling/coarse-level grids."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from LION.pcm.results import RESULT_COLUMNS, RESULT_METRIC_COLUMNS, iter_result_csvs
from LION.pcm_tools.pcm_sampling import in_order_sampling_ratio
from LION.visualize.metric_grid import (
    plot_metric_binned,
    plot_metric_scatter,
    plot_metric_triangulation,
)

GridStatistic = Literal["mean", "std"]
GridStyle = Literal["triangulation", "scatter", "binned"]
GridYAxis = Literal["coarse_j", "in_order_ratio"]


def load_run_j_order(run_dir: Path) -> int:
    """Load ``data.j_order`` from the resolved configuration saved with a run."""
    with (Path(run_dir) / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return int(config["data"]["j_order"])


def load_result_dataframe(csv_path: Path) -> pd.DataFrame:
    """Load one PCM result CSV using the shared result-column contract."""
    dataframe = pd.read_csv(csv_path)
    missing = [column for column in RESULT_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing PCM result columns in {csv_path}: {missing}")
    return dataframe


def aggregate_metric_grid(
    dataframe: pd.DataFrame, *, metric: str, statistic: GridStatistic
) -> pd.DataFrame:
    """Aggregate one metric by sampling rate and coarse level."""
    if metric not in RESULT_METRIC_COLUMNS:
        raise ValueError(
            f"Unknown PCM result metric {metric!r}. "
            f"Choose one of: {', '.join(RESULT_METRIC_COLUMNS)}."
        )

    return (
        dataframe.groupby(["sampling_rate", "coarse_j"], as_index=False)[metric]
        .agg(statistic)
        .dropna(subset=[metric])
    )


def discover_result_csvs(
    run_dir: Path,
    *,
    methods: list[str] | None = None,
    categories: list[str] | None = None,
) -> list[tuple[str, str | None, Path]]:
    """Discover result CSVs, optionally filtering methods and categories."""
    results = []
    for method_id, category, csv_path in iter_result_csvs(run_dir):
        if methods is not None and method_id not in methods:
            continue
        if categories is not None and category not in categories:
            continue
        results.append((method_id, category, csv_path))
    return results


def grid_plot_path(
    run_dir: Path,
    *,
    method_id: str,
    category: str | None,
    metric: str,
    statistic: GridStatistic,
    style: GridStyle,
    y_axis: GridYAxis,
) -> Path:
    """Return the output path for one PCM metric-grid plot."""
    category_name = category if category is not None else "results"
    return (
        Path(run_dir)
        / "plots"
        / category_name
        / method_id
        / f"{metric}_{statistic}_{style}_{y_axis}.png"
    )


def plot_pcm_metric_grid(
    run_dir: Path,
    *,
    method_id: str,
    category: str | None,
    csv_paths: list[Path],
    metric: str,
    statistic: GridStatistic,
    style: GridStyle,
    y_axis: GridYAxis,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    levels: int = 40,
    bins: int = 60,
    marker_size: float = 140.0,
    figsize: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    dpi: int = 200,
) -> Path:
    """Aggregate and save one PCM metric-grid plot."""
    dataframe = pd.concat(
        [load_result_dataframe(csv_path) for csv_path in csv_paths], ignore_index=True
    )
    grid = aggregate_metric_grid(dataframe, metric=metric, statistic=statistic)

    j_order = load_run_j_order(run_dir)
    x = 100.0 * grid["sampling_rate"].to_numpy(dtype=float)

    if y_axis == "coarse_j":
        y = grid["coarse_j"].to_numpy(dtype=float)
        ylabel = "Coarse J"
        default_ylim = (-0.5, j_order + 0.5)
    else:
        y = 100.0 * np.asarray(
            [
                in_order_sampling_ratio(j_order, int(coarse_j), float(sampling_rate))
                for sampling_rate, coarse_j in zip(
                    grid["sampling_rate"], grid["coarse_j"], strict=True
                )
            ]
        )
        ylabel = "In-order measurements percentage"
        default_ylim = (-5.0, 105.0)

    plot_ylim = ylim if ylim is not None else default_ylim

    values = grid[metric].to_numpy(dtype=float)
    xlim = (-5.0, 105.0)
    colorbar_label = f"{metric} {statistic}"

    if style == "triangulation":
        fig, ax, _ = plot_metric_triangulation(
            x,
            y,
            values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            colorbar_label=colorbar_label,
            xlim=xlim,
            ylim=plot_ylim,
            figsize=figsize,
        )
    elif style == "scatter":
        fig, ax, _ = plot_metric_scatter(
            x,
            y,
            values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar_label=colorbar_label,
            xlim=xlim,
            ylim=plot_ylim,
            figsize=figsize,
            s=marker_size,
            marker="s",
        )
    elif style == "binned":
        fig, ax, _ = plot_metric_binned(
            x,
            y,
            values,
            bins=bins,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar_label=colorbar_label,
            xlim=xlim,
            ylim=plot_ylim,
            figsize=figsize,
        )
    else:
        raise ValueError(f"Unknown grid style {style!r}.")

    ax.set_xlabel("Sampling percentage")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{method_id} | {category or 'results'} | {metric} {statistic}")

    if y_axis == "in_order_ratio":
        ax.set_aspect("equal", adjustable="box")
        ax.plot([0.0, 100.0], [0.0, 100.0], linestyle="--", linewidth=1)
    else:
        ax.set_aspect("auto", adjustable="box")

    out_path = grid_plot_path(
        run_dir,
        method_id=method_id,
        category=category,
        metric=metric,
        statistic=statistic,
        style=style,
        y_axis=y_axis,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
