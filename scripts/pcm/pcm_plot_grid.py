"""Generate mean/std grid plots from a PCM dataset experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tyro

from LION.pcm.grid_plotting import (
    GridStatistic,
    GridStyle,
    GridYAxis,
    discover_result_csvs,
    plot_pcm_metric_grid,
)
from LION.pcm.results import RESULT_METRIC_COLUMNS


@dataclass(kw_only=True)
class PCMGridPlotConfig:
    """CLI options for PCM metric-grid plotting."""

    run_dir: Path
    metric: str
    methods: list[str] | None = None
    categories: list[str] | None = None
    statistics: list[GridStatistic] = field(default_factory=lambda: ["mean", "std"])
    styles: list[GridStyle] = field(
        default_factory=lambda: ["triangulation", "scatter", "binned"]
    )
    y_axis: GridYAxis = "in_order_ratio"
    cmap: str = "viridis"
    vmin: float | None = None
    vmax: float | None = None
    mean_vmin: float | None = None
    mean_vmax: float | None = None
    std_vmin: float | None = None
    std_vmax: float | None = None
    levels: int = 40
    bins: int = 60
    marker_size: float = 140.0
    figsize: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    dpi: int = 200


def main() -> None:
    """Generate requested PCM metric-grid plots."""
    config = tyro.cli(PCMGridPlotConfig)

    if config.metric not in RESULT_METRIC_COLUMNS:
        raise ValueError(
            f"Unknown metric {config.metric!r}. "
            f"Choose one of: {', '.join(RESULT_METRIC_COLUMNS)}."
        )

    csvs = discover_result_csvs(
        config.run_dir, methods=config.methods, categories=config.categories
    )

    plot_groups = [
        (method_id, category, [csv_path]) for method_id, category, csv_path in csvs
    ]

    combined_csvs: dict[str, list[Path]] = {}
    for method_id, category, csv_path in csvs:
        if category is not None:
            combined_csvs.setdefault(method_id, []).append(csv_path)
    plot_groups.extend(
        (method_id, "combined", csv_paths)
        for method_id, csv_paths in combined_csvs.items()
        if len(csv_paths) > 1
    )

    for method_id, category, csv_paths in plot_groups:
        for statistic in config.statistics:
            if statistic == "mean":
                vmin = config.mean_vmin if config.mean_vmin is not None else config.vmin
                vmax = config.mean_vmax if config.mean_vmax is not None else config.vmax
            else:
                vmin = config.std_vmin if config.std_vmin is not None else config.vmin
                vmax = config.std_vmax if config.std_vmax is not None else config.vmax

            for style in config.styles:
                out_path = plot_pcm_metric_grid(
                    config.run_dir,
                    method_id=method_id,
                    category=category,
                    csv_paths=csv_paths,
                    metric=config.metric,
                    statistic=statistic,
                    style=style,
                    y_axis=config.y_axis,
                    cmap=config.cmap,
                    vmin=vmin,
                    vmax=vmax,
                    levels=config.levels,
                    bins=config.bins,
                    marker_size=config.marker_size,
                    figsize=config.figsize,
                    ylim=config.ylim,
                    dpi=config.dpi,
                )
                print(out_path)


if __name__ == "__main__":
    main()
