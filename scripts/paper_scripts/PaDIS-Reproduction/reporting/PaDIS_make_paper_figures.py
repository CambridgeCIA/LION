"""Build PaDIS paper-style figures from completed LION experiment outputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
import pathlib
import sys

_CACHE_ROOT = pathlib.Path("/tmp") / "lion_matplotlib_cache"
(_CACHE_ROOT / "mpl").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import torch

from LION.CTtools import ct_utils as ct
from LION.utils.paths import LION_EXPERIMENTS_PATH

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "reconstruction"))
from PaDIS_identifiers import experiment_storage_names, method_storage_names


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

PUBLICATION_DPI = 300
PDF_IMAGE_DPI = 600
LATEX_TEXT_WIDTH_PT = 437.46112
LATEX_POINTS_PER_INCH = 72.27
PUBLICATION_WIDTH_IN = LATEX_TEXT_WIDTH_PT / LATEX_POINTS_PER_INCH
PANEL_HEADING_SIZE_PT = 7.5
ROW_LABEL_SIZE_PT = 6.75
SCALE_TEXT_SIZE_PT = 6.5
COLOUR_SCALE_TEXT_SIZE_PT = 6.5
COLOUR_SCALE_LABEL_X = 8.0
GRID_DIVIDER_COLOUR = "0.45"
GRID_DIVIDER_WIDTH_PT = 0.45
HU_LOWER_PERCENTILE = 0.15
HU_UPPER_PERCENTILE = 0.95
MODEL_FIELD_OF_VIEW_MM = 300.0
SCALE_BAR_MM = 50.0
TWO_EXAMPLE_OFFSETS = (0, 5)
PATCH_SIZE_EXAMPLE_OFFSETS = (0, 1, 5, 6)
ADDITIONAL_EXAMPLE_OFFSETS = (1, 2, 3, 4, 6, 7, 8)


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
    field_of_view_mm: float = MODEL_FIELD_OF_VIEW_MM
    scale_bar_mm: float = SCALE_BAR_MM


def torch_load(path: pathlib.Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def display_image(
    image: torch.Tensor,
    *,
    window: str,
    hu_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    image = image.detach().cpu().float()
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3:
        image = image[0] if image.shape[0] in (1, 3) else image.mean(dim=0)
    if window == "normal":
        return image.clamp(0.0, 1.0)
    if window in {"soft_tissue", "bone"}:
        if hu_range is None:
            raise ValueError("HU display rows require an explicit percentile range.")
        lower, upper = hu_range
    else:
        raise ValueError(f"Unknown display window: {window}")
    return ((ct.from_normal_to_HU(image) - lower) / (upper - lower)).clamp(0.0, 1.0)


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
    height, width = (int(value) for value in target_2d.shape)
    content_top = max(int(rows[0]) - pad, 0)
    content_bottom = min(int(rows[-1]) + pad + 1, height)
    content_left = max(int(cols[0]) - pad, 0)
    content_right = min(int(cols[-1]) + pad + 1, width)

    # Remove the largest equal border possible from all four sides. The result
    # is the tightest square crop centred on the original image centre that
    # still contains every foreground pixel (and any explicitly requested
    # padding). This avoids shifting anatomy to follow an asymmetric body mask.
    inset = max(
        min(
            content_top,
            height - content_bottom,
            content_left,
            width - content_right,
        ),
        0,
    )
    return inset, height - inset, inset, width - inset


def body_hu_percentile_range(
    path: pathlib.Path,
    sample_index: int,
    *,
    lower_quantile: float = HU_LOWER_PERCENTILE,
    upper_quantile: float = HU_UPPER_PERCENTILE,
) -> tuple[float, float]:
    """Return robust HU limits from non-background ground-truth body pixels."""
    target = tensor_from_payload(path, "targets", sample_index)
    target_2d = target.squeeze().detach().cpu().float()
    if target_2d.ndim != 2:
        raise ValueError(f"Expected a 2-D target image in {path}.")
    body_mask = target_2d > 0.02
    if not torch.any(body_mask):
        raise ValueError(f"No body pixels found in target image {path}.")
    body_hu = ct.from_normal_to_HU(target_2d)[body_mask]
    limits = torch.quantile(
        body_hu,
        torch.tensor((lower_quantile, upper_quantile), dtype=body_hu.dtype),
    )
    lower, upper = (float(value) for value in limits)
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError(
            f"Invalid HU percentile range ({lower}, {upper}) for target {path}."
        )
    return lower, upper


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
    canonical_path = None
    for stored_method in method_storage_names(method):
        for stored_experiment in experiment_storage_names(experiment):
            path = (
                root
                / stored_method
                / model
                / implementation
                / "lion"
                / stored_experiment
            )
            if canonical_path is None:
                canonical_path = path
            if group != "main":
                path = path / group
            direct_path = path / "reconstructions.pt"
            if direct_path.exists():
                return direct_path
            nested_root = path / stored_experiment if group == "main" else path
            nested_paths = tuple(nested_root.rglob("reconstructions.pt"))
            if len(nested_paths) == 1:
                return nested_paths[0]
            if len(nested_paths) > 1:
                raise RuntimeError(
                    f"Expected one reconstruction payload below {path}, found "
                    f"{len(nested_paths)}."
                )
    assert canonical_path is not None
    if group != "main":
        canonical_path = canonical_path / group
    return canonical_path / "reconstructions.pt"


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
                "FDK",
                row,
                method="baseline",
                model="patch_lidc_default",
                experiment=experiment,
                sample_index=panel_sample_index,
                window=window,
            ),
            recon_panel(
                recon_root,
                "CP",
                row,
                method="cp_tv",
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

    generation_rows = tuple(
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
            ("Patch stitching", "paper-generation-patch-stitch"),
            ("Patch averaging", "paper-generation-patch-average"),
            ("PaDIS", "paper-generation"),
        )
    )

    return (
        FigureSpec(
            "figure4_generation",
            "figure4_generation.png",
            generation_rows,
            window="normal",
        ),
        FigureSpec(
            "figure5_ct_reconstruction",
            "figure5_ct_reconstruction.png",
            (
                standard_ct_row(
                    "ct_60",
                    "60 views\n360° range\nSample 1",
                    panel_sample_index=sample_index + TWO_EXAMPLE_OFFSETS[0],
                    window="soft_tissue",
                ),
                standard_ct_row(
                    "ct_60",
                    "60 views\n360° range\nSample 6",
                    panel_sample_index=sample_index + TWO_EXAMPLE_OFFSETS[1],
                    window="soft_tissue",
                ),
                standard_ct_row(
                    "ct_20",
                    "20 views\n360° range\nSample 1",
                    panel_sample_index=sample_index + TWO_EXAMPLE_OFFSETS[0],
                    window="normal",
                ),
                standard_ct_row(
                    "ct_20",
                    "20 views\n360° range\nSample 6",
                    panel_sample_index=sample_index + TWO_EXAMPLE_OFFSETS[1],
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
                        "60 views\n360° range\n512 × 512",
                        method="baseline",
                        model="patch_lidc_512",
                        experiment="ct_512_60",
                        sample_index=sample_index,
                        window="soft_tissue",
                    ),
                    recon_panel(
                        recon_root,
                        "CP",
                        "60 views\n360° range\n512 × 512",
                        method="cp_tv",
                        model="patch_lidc_512",
                        experiment="ct_512_60",
                        sample_index=sample_index,
                        window="soft_tissue",
                    ),
                    recon_panel(
                        recon_root,
                        "PaDIS",
                        "60 views\n360° range\n512 × 512",
                        method="padis_dps",
                        model="patch_lidc_512",
                        experiment="ct_512_60",
                        sample_index=sample_index,
                        window="soft_tissue",
                    ),
                    target(
                        "Ground truth",
                        "60 views\n360° range\n512 × 512",
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
                    f"20 views\n360° range\nSample {offset + 1}",
                    panel_sample_index=sample_index + offset,
                    window="normal",
                )
                for offset in ADDITIONAL_EXAMPLE_OFFSETS
            ),
            window="normal",
        ),
        FigureSpec(
            "figureA2_ct8_additional",
            "figureA2_ct8_additional.png",
            tuple(
                standard_ct_row(
                    "ct_8",
                    f"8 views\n360° range\nSample {offset + 1}",
                    panel_sample_index=sample_index + offset,
                    window="normal",
                )
                for offset in ADDITIONAL_EXAMPLE_OFFSETS
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
                        f"Sample {offset + 1}",
                        patch_label,
                        method="padis_dps",
                        model=model,
                        experiment="ct_20",
                        group=group,
                        sample_index=sample_index + offset,
                        window="normal",
                    )
                    for offset in PATCH_SIZE_EXAMPLE_OFFSETS
                )
                for patch_label, model, group in (
                    ("8 × 8 patches", "patch_lidc_p8_default", "patch_size_p8"),
                    ("16 × 16 patches", "patch_lidc_p16_default", "patch_size_p16"),
                    ("32 × 32 patches", "patch_lidc_p32_default", "patch_size_p32"),
                    ("56 × 56 patches", "patch_lidc_default", "patch_size_p56"),
                    ("96 × 96 patches", "patch_lidc_p96_default", "patch_size_p96"),
                )
            )
            + (
                tuple(
                    target(
                        f"Sample {offset + 1}",
                        "Ground truth",
                        "ct_20",
                        panel_sample_index=sample_index + offset,
                        window="normal",
                    )
                    for offset in PATCH_SIZE_EXAMPLE_OFFSETS
                ),
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
                        "Training subset",
                        "PaDIS",
                        method="padis_dps",
                        model="patch_lidc_default",
                        group="dataset_size_patch_default",
                        sample_index=sample_index,
                        window="normal",
                    ),
                    recon_panel(
                        recon_root,
                        "Full LIDC–IDRI set",
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
                        "Training subset",
                        "Whole image",
                        method="whole_image_diffusion",
                        model="whole_lidc_default",
                        group="dataset_size_whole_default",
                        sample_index=sample_index,
                        window="normal",
                    ),
                    recon_panel(
                        recon_root,
                        "Full LIDC–IDRI set",
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
                        "No position\nNoise input",
                        f"Sample {offset + 1}",
                        method="padis_dps",
                        model="patch_lidc_no_pos_default",
                        group="position_no_encoding_noise_init",
                        sample_index=sample_index + offset,
                        window="normal",
                    ),
                    recon_panel(
                        recon_root,
                        "No position\nFDK input",
                        f"Sample {offset + 1}",
                        method="padis_dps",
                        model="patch_lidc_no_pos_default",
                        group="position_no_encoding_fdk_init",
                        sample_index=sample_index + offset,
                        window="normal",
                    ),
                    recon_panel(
                        recon_root,
                        "Position\nNoise input",
                        f"Sample {offset + 1}",
                        method="padis_dps",
                        model="patch_lidc_default",
                        group="position_with_encoding_noise_init",
                        sample_index=sample_index + offset,
                        window="normal",
                    ),
                    recon_panel(
                        recon_root,
                        "Position\nFDK input",
                        f"Sample {offset + 1}",
                        method="padis_dps",
                        model="patch_lidc_default",
                        group="position_with_encoding_fdk_init",
                        sample_index=sample_index + offset,
                        window="normal",
                    ),
                    target(
                        "Ground truth",
                        f"Sample {offset + 1}",
                        "ct_20",
                        panel_sample_index=sample_index + offset,
                        window="normal",
                    ),
                )
                for offset in TWO_EXAMPLE_OFFSETS
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
                        "Langevin\n300 model evaluations",
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
                        experiment_label,
                        method="baseline",
                        model="patch_lidc_default",
                        experiment=experiment,
                        sample_index=sample_index,
                        window="soft_tissue",
                    ),
                    recon_panel(
                        recon_root,
                        "CP",
                        experiment_label,
                        method="cp_tv",
                        model="patch_lidc_default",
                        experiment=experiment,
                        sample_index=sample_index,
                        window="soft_tissue",
                    ),
                    recon_panel(
                        recon_root,
                        "Whole image",
                        experiment_label,
                        method="whole_image_diffusion",
                        model="whole_lidc_default",
                        experiment=experiment,
                        sample_index=sample_index,
                        window="soft_tissue",
                    ),
                    recon_panel(
                        recon_root,
                        "PaDIS",
                        experiment_label,
                        method="padis_dps",
                        model="patch_lidc_default",
                        experiment=experiment,
                        sample_index=sample_index,
                        window="soft_tissue",
                    ),
                    target(
                        "Ground truth",
                        experiment_label,
                        experiment,
                        panel_sample_index=sample_index,
                        window="soft_tissue",
                    ),
                )
                for experiment, experiment_label in (
                    ("ct_60", "60 views\n360° range"),
                    ("ct_20_limited_angle_120", "20 views\n120° range"),
                )
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


def should_show_panel_title(
    panels: tuple[tuple[Panel, ...], ...], row_index: int, col_index: int
) -> bool:
    """Show a column heading only when its meaning changes from the row above."""
    if row_index == 0:
        return True
    previous_row = panels[row_index - 1]
    if col_index >= len(previous_row):
        return True
    return previous_row[col_index].title != panels[row_index][col_index].title


def add_scale_bar(
    axis,
    image: torch.Tensor,
    *,
    source_width_pixels: int,
    field_of_view_mm: float,
    scale_bar_mm: float,
) -> None:
    """Draw a physical scale bar using the reconstruction model's field of view."""
    import matplotlib.patheffects as path_effects

    height, width = image.shape[-2:]
    pixels_per_mm = float(source_width_pixels) / float(field_of_view_mm)
    bar_pixels = float(scale_bar_mm) * pixels_per_mm
    margin = max(0.045 * min(height, width), 3.0)
    x_start = 1.5 * margin
    x_end = x_start + bar_pixels
    y = max(2.0 * margin, 0.08 * height)
    if x_end >= width - margin:
        raise ValueError(
            f"{scale_bar_mm:g} mm scale bar does not fit in a {width}-pixel panel."
        )
    line = axis.plot(
        (x_start, x_end),
        (y, y),
        color="white",
        linewidth=2.2,
        solid_capstyle="butt",
        zorder=5,
    )[0]
    line.set_path_effects(
        [path_effects.Stroke(linewidth=4.0, foreground="black"), path_effects.Normal()]
    )
    label = axis.text(
        x_end + 0.025 * width,
        y,
        f"{scale_bar_mm:g} mm",
        color="white",
        fontsize=SCALE_TEXT_SIZE_PT,
        fontweight="semibold",
        ha="left",
        va="center",
        zorder=6,
        clip_on=True,
        bbox={
            "boxstyle": "square,pad=0.12",
            "facecolor": "black",
            "edgecolor": "none",
            "alpha": 0.78,
        },
    )
    label.set_path_effects(
        [path_effects.Stroke(linewidth=2.0, foreground="black"), path_effects.Normal()]
    )


def add_internal_grid_dividers(
    axis, *, row_index: int, col_index: int, rows: int, cols: int
) -> None:
    """Draw each internal grid boundary once without opening a panel gutter."""
    divider_kwargs = {
        "color": GRID_DIVIDER_COLOUR,
        "linewidth": GRID_DIVIDER_WIDTH_PT,
        "transform": axis.transAxes,
        "clip_on": False,
        "solid_capstyle": "butt",
        "zorder": 8,
    }
    if col_index < cols - 1:
        axis.plot((1.0, 1.0), (0.0, 1.0), **divider_kwargs)
    if row_index < rows - 1:
        axis.plot((0.0, 1.0), (0.0, 0.0), **divider_kwargs)


def assert_text_within_figure(fig) -> None:
    """Fail before export if any visible text would be cropped by the PDF page."""
    from matplotlib.axes import Axes

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_bbox = fig.bbox
    tolerance_pixels = 1.0
    clipped = []
    artists = list(fig.texts)
    for axis in fig.findobj(match=Axes):
        artists.extend(axis.texts)
        artists.extend((axis.title, axis._left_title, axis._right_title))
        if axis.axison:
            artists.extend(axis.get_xticklabels())
            artists.extend(axis.get_yticklabels())
            artists.extend((axis.xaxis.label, axis.yaxis.label))
    unique_artists = {id(artist): artist for artist in artists}.values()
    for artist in unique_artists:
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bounds = artist.get_window_extent(renderer=renderer)
        if (
            bounds.x0 < figure_bbox.x0 - tolerance_pixels
            or bounds.y0 < figure_bbox.y0 - tolerance_pixels
            or bounds.x1 > figure_bbox.x1 + tolerance_pixels
            or bounds.y1 > figure_bbox.y1 + tolerance_pixels
        ):
            clipped.append(
                f"{artist.get_text().replace(chr(10), ' / ')} "
                f"[{bounds.x0:.1f}, {bounds.y0:.1f}, "
                f"{bounds.x1:.1f}, {bounds.y1:.1f}]"
            )
    if clipped:
        raise RuntimeError(
            "Figure text extends beyond the export page: " + ", ".join(clipped)
        )


def symmetric_horizontal_tight_bbox(fig, axes):
    """Return the minimal tight page symmetric about the image-grid edges."""
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight_bbox = fig.get_tightbbox(renderer)
    image_left_px = min(row[0].get_window_extent(renderer).x0 for row in axes)
    image_right_px = max(row[-1].get_window_extent(renderer).x1 for row in axes)
    image_left = image_left_px / fig.dpi
    image_right = image_right_px / fig.dpi
    symmetric_margin = max(
        image_left - tight_bbox.x0,
        tight_bbox.x1 - image_right,
        0.0,
    )
    return Bbox.from_extents(
        image_left - symmetric_margin,
        tight_bbox.y0,
        image_right + symmetric_margin,
        tight_bbox.y1,
    )


def add_row_intensity_colourbar(
    fig,
    colour_axis,
    image_artist,
    *,
    window: str,
    hu_range: tuple[float, float] | None,
) -> None:
    """Add one calibrated greyscale bar bounded by the image-row height."""
    colour_axis.set_axis_off()
    bounded_axis = colour_axis.inset_axes((0.0, 0.06, 1.0, 0.88))
    colourbar = fig.colorbar(image_artist, cax=bounded_axis)
    if window == "normal":
        ticks = (0.0, 0.25, 0.5, 0.75, 1.0)
        labels = ("0", "0.25", "0.5", "0.75", "1")
        colourbar.set_label("NI", fontsize=COLOUR_SCALE_TEXT_SIZE_PT)
        colourbar.ax.yaxis.label.set_multialignment("center")
    elif window in {"soft_tissue", "bone"}:
        if hu_range is None:
            raise ValueError("HU colour bars require an explicit percentile range.")
        lower, upper = hu_range
        ticks = (0.0, 0.25, 0.5, 0.75, 1.0)
        hu_ticks = tuple(lower + tick * (upper - lower) for tick in ticks)
        labels = tuple(str(int(round(value))).replace("-", "−") for value in hu_ticks)
        colourbar.set_label("HU", fontsize=COLOUR_SCALE_TEXT_SIZE_PT, labelpad=3)
    else:
        raise ValueError(f"Unknown display window for intensity colour bar: {window}")
    colourbar.set_ticks(ticks, labels=labels)
    colourbar.ax.tick_params(labelsize=COLOUR_SCALE_TEXT_SIZE_PT, length=2, pad=1.5)
    colourbar.ax.yaxis.set_label_coords(COLOUR_SCALE_LABEL_X, 0.5)


def draw_figure(
    spec: FigureSpec,
    output_folder: pathlib.Path,
    *,
    allow_missing: bool,
    crop_body: bool,
    body_bbox_padding: int,
) -> dict:
    import matplotlib.pyplot as plt

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    rows = len(spec.panels)
    cols = max(len(row) for row in spec.panels)
    left_margin_inches = 0.60
    right_margin_inches = 0.68
    maximum_heading_lines = max(
        panel.title.count("\n") + 1
        for row_index, row in enumerate(spec.panels)
        for col_index, panel in enumerate(row)
        if should_show_panel_title(spec.panels, row_index, col_index)
    )
    top_margin_inches = 0.18 + 0.10 * (maximum_heading_lines - 1)
    bottom_margin_inches = 0.03
    colour_scale_width = 0.045
    panel_width = (PUBLICATION_WIDTH_IN - left_margin_inches - right_margin_inches) / (
        cols + colour_scale_width
    )
    figure_height = panel_width * rows + top_margin_inches + bottom_margin_inches
    grid_left = left_margin_inches / PUBLICATION_WIDTH_IN
    grid_right = 1.0 - right_margin_inches / PUBLICATION_WIDTH_IN
    grid_bottom = bottom_margin_inches / figure_height
    grid_top = 1.0 - top_margin_inches / figure_height
    fig = plt.figure(figsize=(PUBLICATION_WIDTH_IN, figure_height))
    grid = fig.add_gridspec(
        rows,
        cols + 1,
        width_ratios=[1.0] * cols + [colour_scale_width],
        left=grid_left,
        right=grid_right,
        bottom=grid_bottom,
        top=grid_top,
        wspace=0.0,
        hspace=0.0,
    )
    axes = [
        [fig.add_subplot(grid[row_index, col_index]) for col_index in range(cols)]
        for row_index in range(rows)
    ]
    colour_axes = [fig.add_subplot(grid[row_index, cols]) for row_index in range(rows)]
    missing = []
    rendered = 0
    row_hu_ranges = []
    for row_index, row in enumerate(spec.panels):
        bbox = None
        row_image_artist = None
        row_windows = {panel.window or spec.window for panel in row}
        if len(row_windows) != 1:
            raise ValueError(
                f"Figure {spec.name} row {row_index} mixes display windows: "
                f"{sorted(row_windows)}"
            )
        row_window = next(iter(row_windows))
        row_hu_range = None
        if row_window in {"soft_tissue", "bone"}:
            references = sorted(row, key=lambda panel: panel.key != "targets")
            for reference_panel in references:
                try:
                    row_hu_range = body_hu_percentile_range(
                        reference_panel.path, reference_panel.sample_index
                    )
                    break
                except (FileNotFoundError, KeyError, IndexError):
                    continue
            if row_hu_range is not None:
                row_hu_ranges.append(
                    {
                        "row": row[0].row.replace("\n", " "),
                        "lower_hu": row_hu_range[0],
                        "upper_hu": row_hu_range[1],
                        "lower_percentile": 15,
                        "upper_percentile": 95,
                    }
                )
            elif not allow_missing:
                raise FileNotFoundError(
                    f"No target payload is available for {spec.name} row {row_index}."
                )
        if crop_body:
            for panel in row:
                if panel.source == "reconstruction":
                    try:
                        bbox = target_bbox(
                            panel.path, panel.sample_index, pad=body_bbox_padding
                        )
                    except (FileNotFoundError, KeyError, IndexError):
                        continue
                    if bbox is not None:
                        break
        for col_index in range(cols):
            axis = axes[row_index][col_index]
            axis.set_axis_off()
            if col_index >= len(row):
                continue
            panel = row[col_index]
            if should_show_panel_title(spec.panels, row_index, col_index):
                row_heading_lines = max(
                    candidate.title.count("\n") + 1
                    for candidate_col, candidate in enumerate(row)
                    if should_show_panel_title(spec.panels, row_index, candidate_col)
                )
                panel_heading_lines = panel.title.count("\n") + 1
                centred_title_pad = (
                    2.0
                    + 0.55
                    * (row_heading_lines - panel_heading_lines)
                    * PANEL_HEADING_SIZE_PT
                )
                axis.set_title(
                    panel.title,
                    fontsize=PANEL_HEADING_SIZE_PT,
                    fontweight="semibold",
                    pad=centred_title_pad,
                    multialignment="center",
                )
            if col_index == 0:
                row_label_lines = panel.row.count("\n") + 1
                row_label_x = -0.045 - 0.065 * (row_label_lines - 1)
                axis.text(
                    row_label_x,
                    0.5,
                    panel.row,
                    transform=axis.transAxes,
                    fontsize=ROW_LABEL_SIZE_PT,
                    fontweight="semibold",
                    ha="center",
                    va="center",
                    rotation=90,
                    clip_on=False,
                )
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
            image_2d = display_image(
                image,
                window=panel.window or spec.window,
                hu_range=row_hu_range,
            )
            source_width_pixels = int(image_2d.shape[-1])
            if crop_body and panel.crop_from_target:
                image_2d = crop(image_2d, bbox)
            row_image_artist = axis.imshow(
                image_2d,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="none",
            )
            if col_index == len(row) - 1:
                add_scale_bar(
                    axis,
                    image_2d,
                    source_width_pixels=source_width_pixels,
                    field_of_view_mm=spec.field_of_view_mm,
                    scale_bar_mm=spec.scale_bar_mm,
                )
            add_internal_grid_dividers(
                axis,
                row_index=row_index,
                col_index=col_index,
                rows=rows,
                cols=len(row),
            )
            rendered += 1
        if row_image_artist is not None:
            add_row_intensity_colourbar(
                fig,
                colour_axes[row_index],
                row_image_artist,
                window=row_window,
                hu_range=row_hu_range,
            )
        else:
            colour_axes[row_index].set_axis_off()

    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / spec.filename
    pdf_path = output_path.with_suffix(".pdf")
    if rendered > 0:
        assert_text_within_figure(fig)
        export_bbox = symmetric_horizontal_tight_bbox(fig, axes)
        save_kwargs = {
            "facecolor": "white",
            "bbox_inches": export_bbox,
            "pad_inches": 0,
        }
        fig.savefig(output_path, dpi=PUBLICATION_DPI, **save_kwargs)
        fig.savefig(pdf_path, dpi=PDF_IMAGE_DPI, **save_kwargs)
    plt.close(fig)
    return {
        "name": spec.name,
        "path": str(output_path),
        "pdf_path": str(pdf_path),
        "dpi": PUBLICATION_DPI,
        "pdf_image_dpi": PDF_IMAGE_DPI,
        "pdf_images": "native-resolution raster; vector text and annotations",
        "hu_percentile_window": [15, 95],
        "row_hu_ranges": row_hu_ranges,
        "field_of_view_mm": spec.field_of_view_mm,
        "scale_bar_mm": spec.scale_bar_mm,
        "scale_bar_position": "upper left of rightmost panel",
        "intensity_colourbar_per_row": True,
        "internal_grid_dividers": {
            "colour": GRID_DIVIDER_COLOUR,
            "width_pt": GRID_DIVIDER_WIDTH_PT,
        },
        "row_label_alignment": "centre",
        "text_bounds_validated": True,
        "row_spacing": 0.0,
        "square_crop": bool(crop_body),
        "crop_alignment": "symmetric about the source-image centre",
        "page_crop": "tight vertically; minimal symmetric margins about image grid",
        "latex_text_width_pt": LATEX_TEXT_WIDTH_PT,
        "publication_ready": not missing and spec.unsupported_note is None,
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
    parser.add_argument("--body-bbox-padding", type=int, default=0)
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
        "publication_dpi": PUBLICATION_DPI,
        "scale_bar_basis": (
            "Physical scale uses the configured 300 mm LION reconstruction-model "
            "field of view, not patient-native DICOM pixel spacing."
        ),
        "representative_sample_indices": [
            int(args.sample_index + offset) for offset in TWO_EXAMPLE_OFFSETS
        ],
        "patch_size_sample_indices": [
            int(args.sample_index + offset) for offset in PATCH_SIZE_EXAMPLE_OFFSETS
        ],
        "additional_sample_indices": [
            int(args.sample_index + offset) for offset in ADDITIONAL_EXAMPLE_OFFSETS
        ],
        "figures": results,
    }
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_folder / "paper_figure_manifest.json", "w") as file:
        json.dump(manifest, file, indent=2)
    print(f"Saved figure manifest to {output_folder / 'paper_figure_manifest.json'}")


if __name__ == "__main__":
    main()
