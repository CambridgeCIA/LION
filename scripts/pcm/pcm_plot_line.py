"""Plot PCM reconstruction metrics against sampling percentage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from LION.pcm.line_plotting import plot_pcm_metric_vs_sampling


@dataclass(kw_only=True)
class PCMLinePlotConfig:
    """CLI options for PCM metric-versus-sampling line plots."""

    run_dir: Path
    metric: str
    methods: list[str] | None = None
    categories: list[str] | None = None
    coarse_j: int | None = None
    show_std: bool = True
    figsize: tuple[float, float] | None = None
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    dpi: int = 200


def main() -> None:
    """Generate a PCM metric-versus-sampling line plot."""
    config = tyro.cli(PCMLinePlotConfig)
    out_path = plot_pcm_metric_vs_sampling(
        config.run_dir,
        metric=config.metric,
        methods=config.methods,
        categories=config.categories,
        coarse_j=config.coarse_j,
        show_std=config.show_std,
        figsize=config.figsize,
        xlim=config.xlim,
        ylim=config.ylim,
        dpi=config.dpi,
    )
    print(out_path)


if __name__ == "__main__":
    main()
