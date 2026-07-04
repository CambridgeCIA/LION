"""Build PaDIS paper-style figures from completed LION experiment outputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
import pathlib

_CACHE_ROOT = pathlib.Path("/tmp") / "lion_matplotlib_cache"
(_CACHE_ROOT / "mpl").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import torch

from LION.utils.paths import LION_EXPERIMENTS_PATH


IMPLEMENTED_FIGURES = (
    "figure4_generation",
    "figure5_ct_reconstruction",
    "figure8_512_ct",
    "figureA1_ct20_additional",
    "figureA2_ct8_additional",
    "figureA5_patch_size",
    "figureA6_dataset_size",
    "figureA7_position_encoding",
    "figureA8_sampling_methods",
    "figureA9_generation_langevin",
    "figureA10_extra_ct",
    "figureA11_patch_assembly_generation",
)


@dataclass(frozen=True)
class Panel:
    source: str
    title: str
    row: str
    path: pathlib.Path
    key: str
    sample_index: int = 0
    crop_from_target: bool = True
    window: str | None = None


@dataclass(frozen=True)
class FigureSpec:
    name: str
    filename: str
    panels: tuple[tuple[Panel, ...], ...]
    window: str = "soft_tissue"
    unsupported_note: str | None = None


def torch_load(path: pathlib.Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normal_to_hu(image: torch.Tensor) -> torch.Tensor:
    return 3000.0 * image - 1000.0


def display_image(image: torch.Tensor, *, window: str) -> torch.Tensor:
    image = image.detach().cpu().float()
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3:
        image = image[0] if image.shape[0] in (1, 3) else image.mean(dim=0)
    if window == "normal":
        return image.clamp(0.0, 1.0)
    if window == "soft_tissue":
        level = 40.0
        width = 400.0
    elif window == "bone":
        level = 400.0
        width = 1800.0
    else:
        raise ValueError(f"Unknown display window: {window}")
    lower = level - width / 2.0
    return ((normal_to_hu(image) - lower) / width).clamp(0.0, 1.0)


def tensor_from_payload(
    path: pathlib.Path, key: str, sample_index: int
) -> torch.Tensor:
    payload = torch_load(path)
    if not isinstance(payload, dict):
        if key != "samples":
            raise KeyError(f"{path} is not a dict payload and cannot provide {key}.")
        tensor = torch.as_tensor(payload)
    else:
        if key not in payload:
            raise KeyError(f"{path} does not contain key {key!r}.")
        tensor = torch.as_tensor(payload[key])
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if sample_index < 0 or sample_index >= tensor.shape[0]:
        raise IndexError(
            f"Sample index {sample_index} outside {path} payload with "
            f"{tensor.shape[0]} samples."
        )
    return tensor[sample_index].detach().cpu().float()


def target_bbox(
    path: pathlib.Path, sample_index: int, *, pad: int
) -> tuple[int, int, int, int] | None:
    try:
        target = tensor_from_payload(path, "targets", sample_index)
    except (FileNotFoundError, KeyError, IndexError):
        return None
    target_2d = target.squeeze().detach().cpu().float()
    if target_2d.ndim != 2:
        return None
    mask = target_2d > 0.02
    if not torch.any(mask):
        return None
    rows = torch.where(torch.any(mask, dim=1))[0]
    cols = torch.where(torch.any(mask, dim=0))[0]
    top = max(int(rows[0]) - pad, 0)
    bottom = min(int(rows[-1]) + pad + 1, target_2d.shape[0])
    left = max(int(cols[0]) - pad, 0)
    right = min(int(cols[-1]) + pad + 1, target_2d.shape[1])
    return top, bottom, left, right


def crop(image: torch.Tensor, bbox: tuple[int, int, int, int] | None) -> torch.Tensor:
    if bbox is None:
        return image
    top, bottom, left, right = bbox
    return image[top:bottom, left:right]


def recon_path(
    root: pathlib.Path,
    *,
    method: str,
    model: str,
    implementation: str,
    experiment: str,
    group: str = "main",
) -> pathlib.Path:
    path = root / method / model / implementation / "lion" / experiment
    if group != "main":
        path = path / group
    return path / "reconstructions.pt"


def generation_path(root: pathlib.Path, preset: str) -> pathlib.Path:
    return root / "lion-paper-protocol" / preset / "samples.pt"


def recon_panel(
    root: pathlib.Path,
    title: str,
    row: str,
    *,
    method: str,
    model: str,
    implementation: str = "lion_physics",
    experiment: str = "ct_20",
    group: str = "main",
    key: str = "reconstructions",
    sample_index: int = 0,
    window: str | None = None,
) -> Panel:
    return Panel(
        source="reconstruction",
        title=title,
        row=row,
        path=recon_path(
            root,
            method=method,
            model=model,
            implementation=implementation,
            experiment=experiment,
            group=group,
        ),
        key=key,
        sample_index=sample_index,
        window=window,
    )


def generation_panel(
    root: pathlib.Path, title: str, row: str, *, preset: str, sample_index: int
) -> Panel:
    return Panel(
        source="generation",
        title=title,
        row=row,
        path=generation_path(root, preset),
        key="samples",
        sample_index=sample_index,
        crop_from_target=False,
        window="normal",
    )


def figure_specs(
    recon_root: pathlib.Path,
    generation_root: pathlib.Path,
    *,
    sample_index: int = 0,
) -> tuple[FigureSpec, ...]:
    def target(
        title: str,
        row: str,
        experiment: str,
        model: str = "patch_lidc_default",
        *,
        panel_sample_index: int = sample_index,
        window: str | None = None,
    ):
        return recon_panel(
            recon_root,
            title,
            row,
            method="padis_dps",
            model=model,
            experiment=experiment,
            key="targets",
            sample_index=panel_sample_index,
            window=window,
        )

    def standard_ct_row(
        experiment: str,
        row: str,
        *,
        panel_sample_index: int,
        window: str | None = None,
    ) -> tuple[Panel, ...]:
        return (
            recon_panel(
                recon_root,
                "FBP",
                row,
                method="baseline",
                model="patch_lidc_default",
                experiment=experiment,
                sample_index=panel_sample_index,
                window=window,
            ),
            recon_panel(
                recon_root,
                "ADMM-TV",
                row,
                method="admm_tv",
                model="patch_lidc_default",
                experiment=experiment,
                sample_index=panel_sample_index,
                window=window,
            ),
            recon_panel(
                recon_root,
                "Whole image",
                row,
                method="whole_image_diffusion",
                model="whole_lidc_default",
                experiment=experiment,
                sample_index=panel_sample_index,
                window=window,
            ),
            recon_panel(
                recon_root,
                "PaDIS",
                row,
                method="padis_dps",
                model="patch_lidc_default",
                experiment=experiment,
                sample_index=panel_sample_index,
                window=window,
            ),
            target(
                "Ground truth",
                row,
                experiment,
                panel_sample_index=panel_sample_index,
                window=window,
            ),
        )

    generation_columns = tuple(
        tuple(
            generation_panel(
                generation_root,
                f"Sample {index + 1}",
                row,
                preset=preset,
                sample_index=index,
            )
            for index in range(4)
        )
        for row, preset in (
            ("Whole image", "paper-generation-whole"),
            ("Independent patches", "paper-generation-naive-patch"),
            ("PaDIS", "paper-generation"),
        )
    )

    return (
        FigureSpec(
            "figure4_generation",
            "figure4_generation.png",
            generation_columns,
            window="normal",
        ),
        FigureSpec(
            "figure5_ct_reconstruction",
            "figure5_ct_reconstruction.png",
            (
                standard_ct_row(
                    "ct_60",
                    "60 views",
                    panel_sample_index=sample_index,
                    window="soft_tissue",
                ),
                standard_ct_row(
                    "ct_60",
                    "60 views",
                    panel_sample_index=sample_index + 1,
                    window="soft_tissue",
                ),
                standard_ct_row(
                    "ct_20",
                    "20 views",
                    panel_sample_index=sample_index,
                    window="normal",
                ),
                standard_ct_row(
                    "ct_20",
                    "20 views",
                    panel_sample_index=sample_index + 1,
                    window="normal",
                ),
            ),
        ),
        FigureSpec(
            "figure8_512_ct",
            "figure8_512_ct.png",
            (
                (
                    recon_panel(
                        recon_root,
                        "FDK",
                        "ct_512_60",
                        method="baseline",
                        model="patch_lidc_512",
                        experiment="ct_512_60",
                        sample_index=sample_index,
                        window="soft_tissue",
                    ),
                    recon_panel(
                        recon_root,
                        "ADMM-TV",
                        "ct_512_60",
                        method="admm_tv",
                        model="patch_lidc_512",
                        experiment="ct_512_60",
                        sample_index=sample_index,
                        window="soft_tissue",
                    ),
                    recon_panel(
                        recon_root,
                        "PaDIS",
                        "ct_512_60",
                        method="padis_dps",
                        model="patch_lidc_512",
                        experiment="ct_512_60",
                        sample_index=sample_index,
                        window="soft_tissue",
                    ),
                    target(
                        "Ground truth",
                        "ct_512_60",
                        "ct_512_60",
                        "patch_lidc_512",
                        panel_sample_index=sample_index,
                        window="soft_tissue",
                    ),
                ),
            ),
        ),
        FigureSpec(
            "figureA1_ct20_additional",
            "figureA1_ct20_additional.png",
            tuple(
                standard_ct_row(
                    "ct_20",
                    f"Slice {offset + 1}",
                    panel_sample_index=sample_index + offset,
                    window="normal",
                )
                for offset in range(7)
            ),
            window="normal",
        ),
        FigureSpec(
            "figureA2_ct8_additional",
            "figureA2_ct8_additional.png",
            tuple(
                standard_ct_row(
                    "ct_8",
                    f"Slice {offset + 1}",
                    panel_sample_index=sample_index + offset,
                    window="normal",
                )
                for offset in range(7)
            ),
            window="normal",
        ),
        FigureSpec(
            "figureA5_patch_size",
            "figureA5_patch_size.png",
            tuple(
                tuple(
                    recon_panel(
                        recon_root,
                        title,
                        "Patch size",
                        method="padis_dps",
                        model=model,
                        experiment="ct_20",
                        group=group,
                        sample_index=sample_index + offset,
                        window="normal",
                    )
                    for title, model, group in (
                        ("P=8", "patch_lidc_p8_default", "patch_size_p8"),
                        ("P=16", "patch_lidc_p16_default", "patch_size_p16"),
                        ("P=32", "patch_lidc_p32_default", "patch_size_p32"),
                        ("P=56", "patch_lidc_default", "patch_size_p56"),
                        ("P=96", "patch_lidc_p96_default", "patch_size_p96"),
                    )
                )
                + (
                    target(
                        "Ground truth",
                        f"Slice {offset + 1}",
                        "ct_20",
                        panel_sample_index=sample_index + offset,
                        window="normal",
                    ),
                )
                for offset in range(2)
            ),
            window="normal",
        ),
        FigureSpec(
            "figureA6_dataset_size",
            "figureA6_dataset_size.png",
            (
                (
                    recon_panel(
                        recon_root,
                        "Default",
                        "PaDIS",
                        method="padis_dps",
                        model="patch_lidc_default",
                        group="dataset_size_patch_default",
                        sample_index=sample_index,
                        window="normal",
                    ),
                    recon_panel(
                        recon_root,
                        "Full LIDC",
                        "PaDIS",
                        method="padis_dps",
                        model="patch_lidc_full",
                        group="dataset_size_patch_full",
                        sample_index=sample_index,
                        window="normal",
                    ),
                    target(
                        "Ground truth",
                        "PaDIS",
                        "ct_20",
                        panel_sample_index=sample_index,
                        window="normal",
                    ),
                ),
                (
                    recon_panel(
                        recon_root,
                        "Default",
                        "Whole image",
                        method="whole_image_diffusion",
                        model="whole_lidc_default",
                        group="dataset_size_whole_default",
                        sample_index=sample_index,
                        window="normal",
                    ),
                    recon_panel(
                        recon_root,
                        "Full LIDC",
                        "Whole image",
                        method="whole_image_diffusion",
                        model="whole_lidc_full",
                        group="dataset_size_whole_full",
                        sample_index=sample_index,
                        window="normal",
                    ),
                    target(
                        "Ground truth",
                        "Whole image",
                        "ct_20",
                        panel_sample_index=sample_index,
                        window="normal",
                    ),
                ),
            ),
            window="normal",
        ),
        FigureSpec(
            "figureA7_position_encoding",
            "figureA7_position_encoding.png",
            tuple(
                (
                    recon_panel(
                        recon_root,
                        "No position, noise",
                        "Position",
                        method="padis_dps",
                        model="patch_lidc_no_pos_default",
                        group="position_no_encoding_noise_init",
                        sample_index=sample_index + offset,
                        window="normal",
                    ),
                    recon_panel(
                        recon_root,
                        "No position, FDK",
                        "Position",
                        method="padis_dps",
                        model="patch_lidc_no_pos_default",
                        group="position_no_encoding_fdk_init",
                        sample_index=sample_index + offset,
                        window="normal",
                    ),
                    recon_panel(
                        recon_root,
                        "Position, noise",
                        "Position",
                        method="padis_dps",
                        model="patch_lidc_default",
                        group="position_with_encoding_noise_init",
                        sample_index=sample_index + offset,
                        window="normal",
                    ),
                    recon_panel(
                        recon_root,
                        "Position, FDK",
                        "Position",
                        method="padis_dps",
                        model="patch_lidc_default",
                        group="position_with_encoding_fdk_init",
                        sample_index=sample_index + offset,
                        window="normal",
                    ),
                    target(
                        "Ground truth",
                        f"Slice {offset + 1}",
                        "ct_20",
                        panel_sample_index=sample_index + offset,
                        window="normal",
                    ),
                )
                for offset in range(2)
            ),
            window="normal",
        ),
        FigureSpec(
            "figureA8_sampling_methods",
            "figureA8_sampling_methods.png",
            (
                (
                    recon_panel(
                        recon_root,
                        "Langevin",
                        "PaDIS",
                        method="langevin",
                        model="patch_lidc_default",
                        sample_index=sample_index,
                        window="normal",
                    ),
                    recon_panel(
                        recon_root,
                        "PC",
                        "PaDIS",
                        method="predictor_corrector",
                        model="patch_lidc_default",
                        sample_index=sample_index,
                        window="normal",
                    ),
                    recon_panel(
                        recon_root,
                        "VE-DDNM",
                        "PaDIS",
                        method="ve_ddnm",
                        model="patch_lidc_default",
                        sample_index=sample_index,
                        window="normal",
                    ),
                    recon_panel(
                        recon_root,
                        "VE-DPS",
                        "PaDIS",
                        method="padis_dps",
                        model="patch_lidc_default",
                        sample_index=sample_index,
                        window="normal",
                    ),
                    target(
                        "Ground truth",
                        "PaDIS",
                        "ct_20",
                        panel_sample_index=sample_index,
                        window="normal",
                    ),
                ),
                (
                    recon_panel(
                        recon_root,
                        "Langevin",
                        "Whole image",
                        method="langevin",
                        model="whole_lidc_default",
                        sample_index=sample_index,
                        window="normal",
                    ),
                    recon_panel(
                        recon_root,
                        "PC",
                        "Whole image",
                        method="predictor_corrector",
                        model="whole_lidc_default",
                        sample_index=sample_index,
                        window="normal",
                    ),
                    recon_panel(
                        recon_root,
                        "VE-DDNM",
                        "Whole image",
                        method="ve_ddnm",
                        model="whole_lidc_default",
                        sample_index=sample_index,
                        window="normal",
                    ),
                    recon_panel(
                        recon_root,
                        "VE-DPS",
                        "Whole image",
                        method="whole_image_diffusion",
                        model="whole_lidc_default",
                        sample_index=sample_index,
                        window="normal",
                    ),
                    target(
                        "Ground truth",
                        "Whole image",
                        "ct_20",
                        panel_sample_index=sample_index,
                        window="normal",
                    ),
                ),
            ),
            window="normal",
        ),
        FigureSpec(
            "figureA9_generation_langevin",
            "figureA9_generation_langevin.png",
            (
                tuple(
                    generation_panel(
                        generation_root,
                        f"Sample {index + 1}",
                        "Langevin 300 NFE",
                        preset="paper-generation-langevin-300nfe",
                        sample_index=index,
                    )
                    for index in range(4)
                ),
            ),
            window="normal",
            unsupported_note=(
                "EDM and DDIM accelerated samplers from Figure A.9 are not "
                "implemented in this LION PaDIS reproduction."
            ),
        ),
        FigureSpec(
            "figureA10_extra_ct",
            "figureA10_extra_ct.png",
            tuple(
                (
                    recon_panel(
                        recon_root,
                        "FDK",
                        experiment,
                        method="baseline",
                        model="patch_lidc_default",
                        experiment=experiment,
                        sample_index=sample_index,
                        window="soft_tissue",
                    ),
                    recon_panel(
                        recon_root,
                        "ADMM-TV",
                        experiment,
                        method="admm_tv",
                        model="patch_lidc_default",
                        experiment=experiment,
                        sample_index=sample_index,
                        window="soft_tissue",
                    ),
                    recon_panel(
                        recon_root,
                        "Whole image",
                        experiment,
                        method="whole_image_diffusion",
                        model="whole_lidc_default",
                        experiment=experiment,
                        sample_index=sample_index,
                        window="soft_tissue",
                    ),
                    recon_panel(
                        recon_root,
                        "PaDIS",
                        experiment,
                        method="padis_dps",
                        model="patch_lidc_default",
                        experiment=experiment,
                        sample_index=sample_index,
                        window="soft_tissue",
                    ),
                    target(
                        "Ground truth",
                        experiment,
                        experiment,
                        panel_sample_index=sample_index,
                        window="soft_tissue",
                    ),
                )
                for experiment in ("ct_60", "ct_fanbeam_180")
            ),
            unsupported_note=(
                "The heavy deblurring row from Figure A.10 is omitted because "
                "the LIDC PaDIS matrix does not implement that inverse problem."
            ),
        ),
        FigureSpec(
            "figureA11_patch_assembly_generation",
            "figureA11_patch_assembly_generation.png",
            tuple(
                tuple(
                    generation_panel(
                        generation_root,
                        f"Sample {index + 1}",
                        row,
                        preset=preset,
                        sample_index=index,
                    )
                    for index in range(4)
                )
                for row, preset in (
                    ("Patch stitching", "paper-generation-patch-stitch"),
                    ("Patch averaging", "paper-generation-patch-average"),
                )
            ),
            window="normal",
        ),
    )


def draw_figure(
    spec: FigureSpec,
    output_folder: pathlib.Path,
    *,
    allow_missing: bool,
    crop_body: bool,
    body_bbox_padding: int,
) -> dict:
    import matplotlib.pyplot as plt

    rows = len(spec.panels)
    cols = max(len(row) for row in spec.panels)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(2.35 * cols, 2.35 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    missing = []
    rendered = 0
    for row_index, row in enumerate(spec.panels):
        bbox = None
        if crop_body:
            for panel in row:
                if panel.source == "reconstruction":
                    bbox = target_bbox(
                        panel.path, panel.sample_index, pad=body_bbox_padding
                    )
                    if bbox is not None:
                        break
        for col_index in range(cols):
            axis = axes[row_index][col_index]
            axis.set_axis_off()
            if col_index >= len(row):
                continue
            panel = row[col_index]
            axis.set_title(panel.title, fontsize=8)
            if col_index == 0:
                axis.set_ylabel(panel.row, fontsize=8)
            try:
                image = tensor_from_payload(panel.path, panel.key, panel.sample_index)
            except (FileNotFoundError, KeyError, IndexError) as exc:
                missing.append(
                    {
                        "figure": spec.name,
                        "panel": asdict(panel) | {"path": str(panel.path)},
                        "error": str(exc),
                    }
                )
                if not allow_missing:
                    plt.close(fig)
                    raise
                axis.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
                continue
            image_2d = display_image(image, window=panel.window or spec.window)
            if crop_body and panel.crop_from_target:
                image_2d = crop(image_2d, bbox)
            axis.imshow(image_2d, cmap="gray", vmin=0.0, vmax=1.0)
            rendered += 1

    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / spec.filename
    if rendered > 0:
        fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return {
        "name": spec.name,
        "path": str(output_path),
        "rendered_panels": rendered,
        "missing": missing,
        "unsupported_note": spec.unsupported_note,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reconstruction-root",
        type=pathlib.Path,
        required=True,
        help="Root produced by PaDIS_run_reconstruction_matrix.py.",
    )
    parser.add_argument(
        "--generation-root",
        type=pathlib.Path,
        default=LION_EXPERIMENTS_PATH / "PaDIS" / "reconstruction_presets",
        help="Root produced by PaDIS_experiments.py generation presets.",
    )
    parser.add_argument(
        "--output-folder",
        type=pathlib.Path,
        required=True,
        help="Folder for rendered figure PNGs and the manifest.",
    )
    parser.add_argument(
        "--figures",
        default="all",
        help="all or comma-separated figure keys: " + ", ".join(IMPLEMENTED_FIGURES),
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--body-bbox-padding", type=int, default=8)
    parser.add_argument("--no-body-crop", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser


def selected_figures(selection: str) -> tuple[str, ...]:
    if selection == "all":
        return IMPLEMENTED_FIGURES
    names = tuple(item.strip() for item in selection.split(",") if item.strip())
    unknown = sorted(set(names) - set(IMPLEMENTED_FIGURES))
    if unknown:
        raise ValueError(
            f"Unknown figure(s): {', '.join(unknown)}. "
            f"Valid figures: {', '.join(IMPLEMENTED_FIGURES)}."
        )
    return names


def main() -> None:
    args = build_arg_parser().parse_args()
    recon_root = args.reconstruction_root.expanduser().resolve()
    generation_root = args.generation_root.expanduser().resolve()
    output_folder = args.output_folder.expanduser().resolve()
    specs = {
        spec.name: spec
        for spec in figure_specs(
            recon_root,
            generation_root,
            sample_index=args.sample_index,
        )
    }
    names = selected_figures(args.figures)
    if args.list:
        for name in names:
            spec = specs[name]
            print(f"{name}: {spec.filename}")
            if spec.unsupported_note:
                print(f"  note: {spec.unsupported_note}")
        return

    results = []
    for name in names:
        results.append(
            draw_figure(
                specs[name],
                output_folder,
                allow_missing=args.allow_missing,
                crop_body=not args.no_body_crop,
                body_bbox_padding=args.body_bbox_padding,
            )
        )
    manifest = {
        "reconstruction_root": str(recon_root),
        "generation_root": str(generation_root),
        "output_folder": str(output_folder),
        "sample_index": int(args.sample_index),
        "figures": results,
    }
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_folder / "paper_figure_manifest.json", "w") as file:
        json.dump(manifest, file, indent=2)
    print(f"Saved figure manifest to {output_folder / 'paper_figure_manifest.json'}")


if __name__ == "__main__":
    main()
