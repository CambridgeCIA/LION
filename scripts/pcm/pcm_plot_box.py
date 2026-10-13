"""Compare PCM reconstruction methods using box plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from LION.pcm.box_plotting import BoxPlotStyle, plot_pcm_method_boxplot
from LION.pcm.grid_plotting import discover_result_csvs


@dataclass(kw_only=True)
class PCMBoxPlotConfig:
    """CLI options for PCM method-comparison box plots."""

    run_dir: Path
    metric: str
    categories: list[str] | None = None
    sampling_rate: float
    coarse_j: int
    methods: list[str] | None = None
    style: BoxPlotStyle = "box_points"
    figsize: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    dpi: int = 200


def main() -> None:
    """Generate per-category and combined PCM method-comparison box plots."""
    config = tyro.cli(PCMBoxPlotConfig)

    csvs = discover_result_csvs(
        config.run_dir, methods=config.methods, categories=config.categories
    )
    categories = sorted(
        {category for _, category, _ in csvs},
        key=lambda category: "" if category is None else category,
    )

    for category in categories:
        out_path = plot_pcm_method_boxplot(
            config.run_dir,
            metric=config.metric,
            categories=[category],
            sampling_rate=config.sampling_rate,
            coarse_j=config.coarse_j,
            methods=config.methods,
            style=config.style,
            figsize=config.figsize,
            ylim=config.ylim,
            dpi=config.dpi,
        )
        print(out_path)

    if len(categories) > 1:
        out_path = plot_pcm_method_boxplot(
            config.run_dir,
            metric=config.metric,
            categories=categories,
            sampling_rate=config.sampling_rate,
            coarse_j=config.coarse_j,
            methods=config.methods,
            style=config.style,
            figsize=config.figsize,
            ylim=config.ylim,
            dpi=config.dpi,
        )
        print(out_path)


if __name__ == "__main__":
    main()
