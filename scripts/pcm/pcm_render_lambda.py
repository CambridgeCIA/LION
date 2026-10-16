"""Re-render a saved PCM lambda-map tensor with a chosen visualization config."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro

from LION.config.LogConfig import (
    CommonColorbarConfig,
    ResultOutputConfig,
    ScalarImageRenderSpec,
    ScalarMapRenderConfig,
)
from LION.pcm.pcm_result_writer import PCMResultRenderer


@dataclass(kw_only=True)
class PCMRenderLambdaConfig:
    """CLI options for re-rendering one saved lambda-map tensor."""

    tensor_path: Path
    out_dir: Path | None = None

    filename_stem: str | None = None
    label: str | None = None

    image_formats: list[str] = field(default_factory=lambda: ["png", "pdf"])
    render_histograms: bool = False

    colorbar: bool = True
    colorbar_orientation: str = "horizontal"

    linear: bool = True
    log10: bool = False

    cmap: str = "magma"
    vmin: float | None = None
    vmax: float | None = None

    log10_cmap: str | None = None
    log10_vmin: float | None = None
    log10_vmax: float | None = None

    cbar_tick_format: str = ".2f"
    cbar_first_tick_leq: bool = False
    cbar_last_tick_geq: bool = False


def main() -> None:
    """Load a saved lambda tensor and render it again with new settings."""
    config = tyro.cli(PCMRenderLambdaConfig)

    tensor = torch.load(config.tensor_path, map_location="cpu", weights_only=False)

    if config.out_dir is None:
        out_dir = config.tensor_path.parent / "rerendered"
    else:
        out_dir = config.out_dir

    filename_stem = config.filename_stem or config.tensor_path.stem
    label = config.label or filename_stem

    linear_spec = None
    if config.linear:
        linear_spec = ScalarImageRenderSpec(
            vmin=config.vmin,
            vmax=config.vmax,
            cmap=config.cmap,
            cbar_tick_format=config.cbar_tick_format,
            cbar_first_tick_leq=config.cbar_first_tick_leq,
            cbar_last_tick_geq=config.cbar_last_tick_geq,
        )

    log10_spec = None
    if config.log10:
        log10_spec = ScalarImageRenderSpec(
            vmin=config.log10_vmin,
            vmax=config.log10_vmax,
            cmap=config.log10_cmap or config.cmap,
            cbar_tick_format=config.cbar_tick_format,
            cbar_first_tick_leq=config.cbar_first_tick_leq,
            cbar_last_tick_geq=config.cbar_last_tick_geq,
        )

    output_config = ResultOutputConfig(
        image_formats=config.image_formats,
        render_histograms=config.render_histograms,
        colorbar=(
            CommonColorbarConfig(orientation=config.colorbar_orientation)
            if config.colorbar
            else None
        ),
    )

    renderer = PCMResultRenderer(output_config)
    renderer.render_scalar_map(
        values=tensor,
        out_dir=out_dir,
        filename_stem=filename_stem,
        label=label,
        render_config=ScalarMapRenderConfig(
            linear=linear_spec,
            log10=log10_spec,
        ),
    )

    print(out_dir)


if __name__ == "__main__":
    main()
