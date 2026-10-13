"""Box plots comparing PCM reconstruction methods."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from LION.pcm.grid_plotting import discover_result_csvs, load_result_dataframe
from LION.pcm.results import RESULT_METRIC_COLUMNS

BoxPlotStyle = Literal["box", "box_points", "box_paired"]


def _select_condition(
    dataframe: pd.DataFrame,
    *,
    sampling_rate: float,
    coarse_j: int,
) -> pd.DataFrame:
    """Select one sampling condition from a PCM result dataframe."""
    return dataframe[
        np.isclose(dataframe["sampling_rate"], sampling_rate)
        & (dataframe["coarse_j"] == coarse_j)
    ].copy()


def _common_paired_rows(
    dataframes: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Keep observations that are present for every method."""
    pair_columns = ["category", "sample_id", "trial_num", "rng_seed"]

    common = None
    for dataframe in dataframes.values():
        keys = dataframe[pair_columns].drop_duplicates()
        common = (
            keys if common is None else common.merge(keys, on=pair_columns, how="inner")
        )

    assert common is not None

    return {
        method_id: common.merge(dataframe, on=pair_columns, how="left")
        for method_id, dataframe in dataframes.items()
    }


def pcm_box_plot_path(
    run_dir: Path,
    *,
    category: str | None,
    metric: str,
    sampling_rate: float,
    coarse_j: int,
    style: BoxPlotStyle,
) -> Path:
    """Return the output path for one PCM method-comparison box plot."""
    category_name = category if category is not None else "results"
    return (
        Path(run_dir)
        / "plots"
        / category_name
        / "method_comparison"
        / f"{metric}_{style}_s_{sampling_rate:.2f}_cJ_{coarse_j}.png"
    )


def plot_pcm_method_boxplot(
    run_dir: Path,
    *,
    metric: str,
    categories: list[str | None],
    sampling_rate: float,
    coarse_j: int,
    methods: list[str] | None = None,
    style: BoxPlotStyle = "box_points",
    figsize: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    dpi: int = 200,
) -> Path:
    """Compare one PCM metric across reconstruction methods."""
    if metric not in RESULT_METRIC_COLUMNS:
        raise ValueError(
            f"Unknown PCM result metric {metric!r}. "
            f"Choose one of: {', '.join(RESULT_METRIC_COLUMNS)}."
        )

    csvs = discover_result_csvs(run_dir, methods=methods, categories=categories)

    selected_parts: dict[str, list[pd.DataFrame]] = {}
    for method_id, category, csv_path in csvs:
        dataframe = load_result_dataframe(csv_path)
        dataframe = _select_condition(
            dataframe, sampling_rate=sampling_rate, coarse_j=coarse_j
        )
        if not dataframe.empty:
            dataframe["category"] = category
            selected_parts.setdefault(method_id, []).append(dataframe)

    method_ids = methods if methods is not None else list(selected_parts)
    selected = {
        method_id: pd.concat(selected_parts[method_id], ignore_index=True)
        for method_id in method_ids
        if method_id in selected_parts
    }

    if style == "box_paired":
        selected = _common_paired_rows(selected)

    method_ids = list(selected)
    values = [
        selected[method_id][metric].to_numpy(dtype=float) for method_id in method_ids
    ]
    category = categories[0] if len(categories) == 1 else "combined"
    category_label = category if category is not None else "results"

    fig, ax = plt.subplots(figsize=figsize)

    ax.boxplot(
        values,
        tick_labels=method_ids,
        showmeans=True,
        # showfliers=False,
    )

    if style in {"box_points", "box_paired"}:
        for index, method_id in enumerate(method_ids, start=1):
            method_values = selected[method_id][metric].to_numpy(dtype=float)
            ax.scatter(
                np.full(method_values.shape, index, dtype=float),
                method_values,
                s=12,
                alpha=0.5,
            )

    if style == "box_paired":
        paired_values = np.column_stack(
            [
                selected[method_id][metric].to_numpy(dtype=float)
                for method_id in method_ids
            ]
        )
        x = np.arange(1, len(method_ids) + 1)
        for row in paired_values:
            ax.plot(x, row, linewidth=0.6, alpha=0.25)

    ax.set_ylabel(metric.upper())
    ax.set_title(
        f"{category_label} | {metric.upper()} | "
        f"sampling rate={sampling_rate:.2f}, coarse J={coarse_j}"
    )
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y")
    ax.tick_params(axis="x", labelrotation=20)

    fig.tight_layout()

    out_path = pcm_box_plot_path(
        run_dir,
        category=category,
        metric=metric,
        sampling_rate=sampling_rate,
        coarse_j=coarse_j,
        style=style,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return out_path
