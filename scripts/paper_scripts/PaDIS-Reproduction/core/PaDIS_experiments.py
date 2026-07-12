"""Run named PaDIS reconstruction presets with separated result namespaces."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
import pathlib
import subprocess
import sys

PADIS_SCRIPT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PADIS_SCRIPT_ROOT / "reconstruction"))

from LION.utils.paths import LION_EXPERIMENTS_PATH
from PaDIS_run_reconstruction_matrix import (
    MODEL_BY_NAME,
    TRAINING_ROOT_PRESETS,
    checkpoint_path,
    resolve_training_root,
)


@dataclass(frozen=True)
class ReconstructionPreset:
    """Named reconstruction/generation command fragment and result namespace."""

    implementation: str
    experiment: str | None
    description: str
    arguments: tuple[str, ...]
    engine: str = "reconstruction"


def _paper_ct_arguments(
    sigma_min: float, *, initial_reconstruction: str = "noise"
) -> tuple[str, ...]:
    """Handle paper ct arguments for the PaDIS workflow."""
    return (
        "--num-steps",
        "100",
        "--inner-steps",
        "10",
        "--sigma-min",
        f"{sigma_min:g}",
        "--sigma-max",
        "10",
        "--noise-schedule",
        "geometric",
        "--zeta",
        "0.3",
        "--dps-epsilon",
        "1",
        "--sampling-epsilon",
        "1",
        "--data-consistency-gradient",
        "paper_squared_residual",
        "--adjoint-data-step-schedule",
        "paper",
        "--initial-reconstruction",
        initial_reconstruction,
        "--no-clip-output",
        "--langevin-noise-scale",
        "1",
        "--data-consistency-normalization",
        "none",
        "--data-consistency-scale",
        "1",
    )


def _lion_compatible_ct_arguments(sigma_min: float) -> tuple[str, ...]:
    """Handle lion compatible ct arguments for the PaDIS workflow."""
    return _paper_ct_arguments(sigma_min)


def _whole_image_ct_arguments(sigma_min: float) -> tuple[str, ...]:
    """Handle whole image ct arguments for the PaDIS workflow."""
    return _paper_ct_arguments(sigma_min) + (
        "--prior-mode",
        "whole-image",
        "--patch-size",
        "256",
        "--pad-width",
        "0",
    )


def _paper_generation_arguments(
    *,
    generation_epsilon: float = 1.0,
    langevin_noise_scale: float = 1.0,
) -> tuple[str, ...]:
    """Handle paper generation arguments for the PaDIS workflow."""
    return (
        "--num-steps",
        "300",
        "--inner-steps",
        "1",
        "--sigma-min",
        "0.002",
        "--sigma-max",
        "10",
        "--noise-schedule",
        "geometric",
        "--generation-epsilon",
        str(generation_epsilon),
        "--langevin-noise-scale",
        str(langevin_noise_scale),
    )


def _with_sampler(
    arguments: tuple[str, ...], algorithm: str, *, ddnm: bool = False
) -> tuple[str, ...]:
    """Handle with sampler for the PaDIS workflow."""
    sampler_arguments = arguments + ("--algorithm", algorithm)
    if ddnm:
        sampler_arguments += ("--langevin-ddnm",)
    return sampler_arguments


def _with_patch_size(patch_size: int, pad_width: int) -> tuple[str, ...]:
    """Handle with patch size for the PaDIS workflow."""
    return _paper_ct_arguments(0.002) + (
        "--patch-size",
        str(patch_size),
        "--pad-width",
        str(pad_width),
    )


def _training_arguments(
    *,
    prior_mode: str = "patch",
    max_slices_per_patient: int | None = None,
    full_lidc: bool = False,
    run_name: str,
) -> tuple[str, ...]:
    """Handle training arguments for the PaDIS workflow."""
    arguments = ("--prior-mode", prior_mode, "--run-name", run_name)
    if full_lidc:
        arguments += ("--full-lidc",)
    elif max_slices_per_patient is not None:
        arguments += ("--max-slices-per-patient", str(max_slices_per_patient))
    return arguments


PRESETS = {
    "paper-generation": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment=None,
        description="PaDIS paper-style unconditional LIDC image generation.",
        arguments=_paper_generation_arguments(
            generation_epsilon=0.8,
            langevin_noise_scale=0.85,
        ),
        engine="generation",
    ),
    "paper-generation-whole": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment=None,
        description="Whole-image diffusion unconditional LIDC image generation.",
        arguments=_paper_generation_arguments(
            generation_epsilon=0.75,
            langevin_noise_scale=0.75,
        )
        + (
            "--prior-mode",
            "whole-image",
            "--patch-size",
            "256",
            "--pad-width",
            "0",
        ),
        engine="generation",
    ),
    "paper-generation-naive-patch": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment=None,
        description="Naive independent-patch unconditional LIDC image generation.",
        arguments=_paper_generation_arguments()
        + (
            "--generation-mode",
            "naive-patch",
        ),
        engine="generation",
    ),
    "paper-generation-patch-average": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment=None,
        description="Fixed-overlap patch-averaging LIDC image generation for Figure A.11.",
        arguments=_paper_generation_arguments()
        + (
            "--patch-assembly",
            "fixed_average",
            "--fixed-overlap-layout",
            "public_overlap",
            "--patch-overlap",
            "8",
            "--fixed-overlap-checkpoint-denoiser",
        ),
        engine="generation",
    ),
    "paper-generation-patch-stitch": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment=None,
        description="Fixed-overlap patch-stitching LIDC image generation for Figure A.11.",
        arguments=_paper_generation_arguments()
        + (
            "--patch-assembly",
            "fixed_stitch",
            "--fixed-overlap-layout",
            "public_tile",
            "--patch-overlap",
            "8",
            "--fixed-overlap-checkpoint-denoiser",
        ),
        engine="generation",
    ),
    "paper-generation-langevin-300nfe": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment=None,
        description=(
            "PaDIS generation with 300 Langevin NFEs for the implemented row of "
            "the acceleration figure."
        ),
        arguments=_paper_generation_arguments(),
        engine="generation",
    ),
    "paper-fan-8": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment="PaDISFanBeam8CTRecon",
        description="PaDIS paper-style 8-view fan beam with LION's LIDC geometry.",
        arguments=_paper_ct_arguments(0.003),
    ),
    "paper-fan-20": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment="PaDISFanBeam20CTRecon",
        description="PaDIS paper-style 20-view fan beam with LION's LIDC geometry.",
        arguments=_paper_ct_arguments(0.002),
    ),
    "paper-fan-20-no-pos": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment="PaDISFanBeam20CTRecon",
        description="PaDIS paper-style 20-view fan beam without positional encoding.",
        arguments=_paper_ct_arguments(0.002) + ("--no-position-channels",),
    ),
    "paper-fan-20-no-pos-fdk": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment="PaDISFanBeam20CTRecon",
        description="PaDIS paper-style 20-view fan beam without positional encoding, using FDK initialisation.",
        arguments=_paper_ct_arguments(0.002, initial_reconstruction="fdk")
        + ("--no-position-channels",),
    ),
    "paper-fan-20-langevin": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment="PaDISFanBeam20CTRecon",
        description="PaDIS 20-view fan beam with Langevin data-consistency sampler.",
        arguments=_with_sampler(_paper_ct_arguments(0.002), "langevin"),
    ),
    "paper-fan-20-pc": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment="PaDISFanBeam20CTRecon",
        description="PaDIS 20-view fan beam with predictor-corrector sampler.",
        arguments=_with_sampler(_paper_ct_arguments(0.002), "pc"),
    ),
    "paper-fan-20-ddnm": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment="PaDISFanBeam20CTRecon",
        description="PaDIS 20-view fan beam with VE-DDNM sampler.",
        arguments=_with_sampler(_paper_ct_arguments(0.002), "langevin", ddnm=True),
    ),
    "paper-fan-20-patch-8": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment="PaDISFanBeam20CTRecon",
        description="PaDIS 20-view patch-size ablation with P=8.",
        arguments=_with_patch_size(8, 8),
    ),
    "paper-fan-20-patch-16": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment="PaDISFanBeam20CTRecon",
        description="PaDIS 20-view patch-size ablation with P=16.",
        arguments=_with_patch_size(16, 16),
    ),
    "paper-fan-20-patch-32": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment="PaDISFanBeam20CTRecon",
        description="PaDIS 20-view patch-size ablation with P=32.",
        arguments=_with_patch_size(32, 32),
    ),
    "paper-fan-20-patch-56": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment="PaDISFanBeam20CTRecon",
        description="PaDIS 20-view patch-size ablation with P=56.",
        arguments=_with_patch_size(56, 24),
    ),
    "paper-fan-20-patch-96": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment="PaDISFanBeam20CTRecon",
        description="PaDIS 20-view patch-size ablation with P=96.",
        arguments=_with_patch_size(96, 32),
    ),
    "paper-fan-20-dataset-quarter": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment="PaDISFanBeam20CTRecon",
        description="PaDIS 20-view run for the quarter-default LIDC subset checkpoint.",
        arguments=_paper_ct_arguments(0.002),
    ),
    "paper-fan-20-dataset-half": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment="PaDISFanBeam20CTRecon",
        description="PaDIS 20-view run for the half-default LIDC subset checkpoint.",
        arguments=_paper_ct_arguments(0.002),
    ),
    "paper-fan-20-dataset-full": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment="PaDISFanBeam20CTRecon",
        description="PaDIS 20-view run for the full-LIDC checkpoint.",
        arguments=_paper_ct_arguments(0.002),
    ),
    "paper-fan-60": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment="PaDISFanBeam60CTRecon",
        description="PaDIS paper-style 60-view fan beam with LION's LIDC geometry.",
        arguments=_paper_ct_arguments(0.002),
    ),
    "paper-fan-60-512": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment="PaDISFanBeam60CTRecon",
        description="PaDIS paper-style 512x512 60-view fan beam with LION's LIDC geometry.",
        arguments=_paper_ct_arguments(0.002)
        + ("--patch-size", "64", "--pad-width", "64"),
    ),
    "paper-fan-180": ReconstructionPreset(
        implementation="lion-paper-protocol",
        experiment="PaDISFanBeam120LimitedCTRecon",
        description=(
            "PaDIS limited-angle fan beam with 20 views over 120 degrees "
            "using LION's LIDC geometry."
        ),
        arguments=_paper_ct_arguments(0.002),
    ),
    "paper-whole-fan-8": ReconstructionPreset(
        implementation="lion-whole-image",
        experiment="PaDISFanBeam8CTRecon",
        description="Whole-image diffusion 8-view fan beam with LION's LIDC geometry.",
        arguments=_whole_image_ct_arguments(0.003),
    ),
    "paper-whole-fan-20": ReconstructionPreset(
        implementation="lion-whole-image",
        experiment="PaDISFanBeam20CTRecon",
        description="Whole-image diffusion 20-view fan beam with LION's LIDC geometry.",
        arguments=_whole_image_ct_arguments(0.002),
    ),
    "paper-whole-fan-20-langevin": ReconstructionPreset(
        implementation="lion-whole-image",
        experiment="PaDISFanBeam20CTRecon",
        description="Whole-image diffusion 20-view fan beam with Langevin sampler.",
        arguments=_with_sampler(_whole_image_ct_arguments(0.002), "langevin"),
    ),
    "paper-whole-fan-20-pc": ReconstructionPreset(
        implementation="lion-whole-image",
        experiment="PaDISFanBeam20CTRecon",
        description="Whole-image diffusion 20-view fan beam with predictor-corrector sampler.",
        arguments=_with_sampler(_whole_image_ct_arguments(0.002), "pc"),
    ),
    "paper-whole-fan-20-ddnm": ReconstructionPreset(
        implementation="lion-whole-image",
        experiment="PaDISFanBeam20CTRecon",
        description="Whole-image diffusion 20-view fan beam with VE-DDNM sampler.",
        arguments=_with_sampler(
            _whole_image_ct_arguments(0.002), "langevin", ddnm=True
        ),
    ),
    "paper-whole-fan-20-dataset-quarter": ReconstructionPreset(
        implementation="lion-whole-image",
        experiment="PaDISFanBeam20CTRecon",
        description="Whole-image 20-view run for the quarter-default LIDC subset checkpoint.",
        arguments=_whole_image_ct_arguments(0.002),
    ),
    "paper-whole-fan-20-dataset-half": ReconstructionPreset(
        implementation="lion-whole-image",
        experiment="PaDISFanBeam20CTRecon",
        description="Whole-image 20-view run for the half-default LIDC subset checkpoint.",
        arguments=_whole_image_ct_arguments(0.002),
    ),
    "paper-whole-fan-20-dataset-full": ReconstructionPreset(
        implementation="lion-whole-image",
        experiment="PaDISFanBeam20CTRecon",
        description="Whole-image 20-view run for the full-LIDC checkpoint.",
        arguments=_whole_image_ct_arguments(0.002),
    ),
    "paper-whole-fan-60": ReconstructionPreset(
        implementation="lion-whole-image",
        experiment="PaDISFanBeam60CTRecon",
        description="Whole-image diffusion 60-view fan beam with LION's LIDC geometry.",
        arguments=_whole_image_ct_arguments(0.002),
    ),
    "paper-whole-fan-180": ReconstructionPreset(
        implementation="lion-whole-image",
        experiment="PaDISFanBeam120LimitedCTRecon",
        description=(
            "Whole-image diffusion limited-angle fan beam with 20 views over "
            "120 degrees using LION's LIDC geometry."
        ),
        arguments=_whole_image_ct_arguments(0.002),
    ),
    "train-patch-lidc-quarter": ReconstructionPreset(
        implementation="lion-training",
        experiment=None,
        description="Train PaDIS patch prior on one slice per LIDC patient.",
        arguments=_training_arguments(
            max_slices_per_patient=1, run_name="patch_lidc_quarter_default"
        ),
        engine="training-256",
    ),
    "train-patch-lidc-half": ReconstructionPreset(
        implementation="lion-training",
        experiment=None,
        description="Train PaDIS patch prior on two slices per LIDC patient.",
        arguments=_training_arguments(
            max_slices_per_patient=2, run_name="patch_lidc_half_default"
        ),
        engine="training-256",
    ),
    "train-patch-lidc-full": ReconstructionPreset(
        implementation="lion-training",
        experiment=None,
        description="Train PaDIS patch prior on all selected LIDC slices.",
        arguments=_training_arguments(full_lidc=True, run_name="patch_lidc_full"),
        engine="training-256",
    ),
    "train-whole-lidc-quarter": ReconstructionPreset(
        implementation="lion-training",
        experiment=None,
        description="Train whole-image diffusion prior on one slice per LIDC patient.",
        arguments=_training_arguments(
            prior_mode="whole-image",
            max_slices_per_patient=1,
            run_name="whole_lidc_quarter_default",
        ),
        engine="training-256",
    ),
    "train-whole-lidc-half": ReconstructionPreset(
        implementation="lion-training",
        experiment=None,
        description="Train whole-image diffusion prior on two slices per LIDC patient.",
        arguments=_training_arguments(
            prior_mode="whole-image",
            max_slices_per_patient=2,
            run_name="whole_lidc_half_default",
        ),
        engine="training-256",
    ),
    "train-whole-lidc-full": ReconstructionPreset(
        implementation="lion-training",
        experiment=None,
        description="Train whole-image diffusion prior on all selected LIDC slices.",
        arguments=_training_arguments(
            prior_mode="whole-image", full_lidc=True, run_name="whole_lidc_full"
        ),
        engine="training-256",
    ),
    "train-patch-lidc-512": ReconstructionPreset(
        implementation="lion-training",
        experiment=None,
        description="Train PaDIS 512x512 patch prior with LION's native LIDC geometry.",
        arguments=("--run-name", "patch_lidc_512"),
        engine="training-512",
    ),
    "lion-compatible-clinical": ReconstructionPreset(
        implementation="lion-compatible",
        experiment="clinicalCTRecon",
        description="PaDIS-compatible LION clinical-dose full-view fan beam.",
        arguments=_lion_compatible_ct_arguments(0.002),
    ),
    "lion-compatible-low-dose": ReconstructionPreset(
        implementation="lion-compatible",
        experiment="LowDoseCTRecon",
        description="PaDIS-compatible LION low-dose full-view fan beam.",
        arguments=_lion_compatible_ct_arguments(0.002),
    ),
    "lion-compatible-extreme-low-dose": ReconstructionPreset(
        implementation="lion-compatible",
        experiment="ExtremeLowDoseCTRecon",
        description="PaDIS-compatible LION extreme-low-dose full-view fan beam.",
        arguments=_lion_compatible_ct_arguments(0.002),
    ),
    "lion-compatible-limited-angle": ReconstructionPreset(
        implementation="lion-compatible",
        experiment="LimitedAngleCTRecon",
        description="PaDIS-compatible LION clinical-dose limited-angle fan beam.",
        arguments=_lion_compatible_ct_arguments(0.002),
    ),
    "lion-compatible-limited-angle-low-dose": ReconstructionPreset(
        implementation="lion-compatible",
        experiment="LimitedAngleLowDoseCTRecon",
        description="PaDIS-compatible LION low-dose limited-angle fan beam.",
        arguments=_lion_compatible_ct_arguments(0.002),
    ),
    "lion-compatible-limited-angle-extreme-low-dose": ReconstructionPreset(
        implementation="lion-compatible",
        experiment="LimitedAngleExtremeLowDoseCTRecon",
        description="PaDIS-compatible LION extreme-low-dose limited-angle fan beam.",
        arguments=_lion_compatible_ct_arguments(0.002),
    ),
    "lion-compatible-sparse-50": ReconstructionPreset(
        implementation="lion-compatible",
        experiment="SparseAngleCTRecon",
        description="PaDIS-compatible LION clinical-dose 50-view fan beam.",
        arguments=_lion_compatible_ct_arguments(0.002),
    ),
    "lion-compatible-sparse-low-dose-50": ReconstructionPreset(
        implementation="lion-compatible",
        experiment="SparseAngleLowDoseCTRecon",
        description="PaDIS-compatible LION low-dose 50-view fan beam.",
        arguments=_lion_compatible_ct_arguments(0.002),
    ),
    "lion-compatible-sparse-extreme-low-dose-50": ReconstructionPreset(
        implementation="lion-compatible",
        experiment="SparseAngleExtremeLowDoseCTRecon",
        description="PaDIS-compatible LION extreme-low-dose 50-view fan beam.",
        arguments=_lion_compatible_ct_arguments(0.002),
    ),
}


FIGURE_PRESET_GROUPS = {
    "paper-generation-figures": (
        "paper-generation-whole",
        "paper-generation-naive-patch",
        "paper-generation",
        "paper-generation-langevin-300nfe",
        "paper-generation-patch-stitch",
        "paper-generation-patch-average",
    ),
}


def add_run_arguments(parser: argparse.ArgumentParser, *, include_preset: bool) -> None:
    """Add shared run and checkpoint options to a subcommand parser."""
    if include_preset:
        parser.add_argument("preset", choices=tuple(PRESETS))
    else:
        parser.add_argument("group", choices=tuple(FIGURE_PRESET_GROUPS))
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None)
    if not include_preset:
        parser.add_argument(
            "--patch-checkpoint",
            type=pathlib.Path,
            default=None,
            help="Checkpoint for patch-prior generation presets in this group.",
        )
        parser.add_argument(
            "--whole-checkpoint",
            type=pathlib.Path,
            default=None,
            help="Checkpoint for whole-image generation presets in this group.",
        )
        parser.add_argument(
            "--training-root",
            type=pathlib.Path,
            default=None,
            help=(
                "Root containing final model subfolders. Used to fill missing "
                "--patch-checkpoint and --whole-checkpoint values."
            ),
        )
        parser.add_argument(
            "--training-root-preset",
            choices=TRAINING_ROOT_PRESETS,
            default=None,
            help=(
                "Infer --training-root from the final Slurm or GCP model "
                "layout. Explicit checkpoint flags still take precedence."
            ),
        )
        parser.add_argument(
            "--run-root",
            type=pathlib.Path,
            default=None,
            help="Base PaDIS experiment root for --training-root-preset.",
        )
        parser.add_argument(
            "--run-stamp",
            default=os.environ.get("PADIS_RUN_STAMP", ""),
            help="Slurm run stamp for --training-root-preset slurm.",
        )
        parser.add_argument(
            "--gcp-run-name",
            default=os.environ.get("PADIS_GCP_RUN_NAME", "PaDIS-Reproduction-GCP"),
            help="GCP run folder for --training-root-preset gcp.",
        )
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=LION_EXPERIMENTS_PATH / "PaDIS" / "reconstruction_presets",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")


def add_figure_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared figure-source and output options."""
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
        help="Root produced by the paper-generation-figures preset group.",
    )
    parser.add_argument(
        "--output-folder",
        type=pathlib.Path,
        required=True,
        help="Folder for rendered figure PNGs and the manifest.",
    )
    parser.add_argument("--figures", default="all")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--body-bbox-padding", type=int, default=8)
    parser.add_argument("--no-body-crop", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def apply_generation_checkpoint_defaults(args: argparse.Namespace) -> None:
    """Resolve missing generation checkpoints from the selected training root."""
    if args.training_root is None and args.training_root_preset is None:
        return
    training_root = resolve_training_root(args)
    if args.patch_checkpoint is None:
        args.patch_checkpoint = checkpoint_path(
            training_root, MODEL_BY_NAME["patch_lidc_default"]
        )
    if args.whole_checkpoint is None:
        args.whole_checkpoint = checkpoint_path(
            training_root, MODEL_BY_NAME["whole_lidc_default"]
        )


def build_parser() -> argparse.ArgumentParser:
    """Construct the named-experiment command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List available reconstruction presets.")

    run_parser = subparsers.add_parser("run", help="Run one named preset.")
    add_run_arguments(run_parser, include_preset=True)

    group_parser = subparsers.add_parser(
        "run-group", help="Run a named group of figure prerequisite presets."
    )
    add_run_arguments(group_parser, include_preset=False)

    figure_parser = subparsers.add_parser(
        "make-figures", help="Render paper-style figures from completed outputs."
    )
    add_figure_arguments(figure_parser)
    return parser


def figure_command_for(args: argparse.Namespace) -> list[str]:
    """Build the reporting command for a named figure selection."""
    engine = PADIS_SCRIPT_ROOT / "reporting" / "PaDIS_make_paper_figures.py"
    command = [
        sys.executable,
        "-u",
        str(engine),
        "--reconstruction-root",
        str(args.reconstruction_root),
        "--generation-root",
        str(args.generation_root),
        "--output-folder",
        str(args.output_folder),
        "--figures",
        args.figures,
        "--sample-index",
        str(args.sample_index),
        "--body-bbox-padding",
        str(args.body_bbox_padding),
    ]
    if args.no_body_crop:
        command.append("--no-body-crop")
    if args.allow_missing:
        command.append("--allow-missing")
    return command


def command_for(args, passthrough: list[str]) -> tuple[list[str], pathlib.Path]:
    """Build a preset command and return its isolated output directory."""
    preset = PRESETS[args.preset]
    if preset.engine == "reconstruction":
        engine = PADIS_SCRIPT_ROOT / "reconstruction" / "PaDIS_LIDC_reconstruction.py"
    elif preset.engine == "generation":
        engine = PADIS_SCRIPT_ROOT / "reconstruction" / "PaDIS_LIDC_generation.py"
    elif preset.engine == "training-256":
        engine = PADIS_SCRIPT_ROOT / "training" / "PaDIS_LIDC_256.py"
    elif preset.engine == "training-512":
        engine = PADIS_SCRIPT_ROOT / "training" / "PaDIS_LIDC_512.py"
    else:
        raise ValueError(f"Unknown preset engine: {preset.engine}")
    output_folder = args.output_root / preset.implementation / args.preset
    command = [sys.executable, "-u", str(engine)]
    if preset.engine.startswith("training"):
        command.extend(
            (
                "--save-folder",
                str(output_folder),
                "--seed",
                str(args.seed),
                "--device",
                args.device,
            )
        )
    else:
        command.extend(
            (
                "--output-folder",
                str(output_folder),
                "--seed",
                str(args.seed),
                "--device",
                args.device,
            )
        )
    if preset.engine == "reconstruction":
        command.extend(
            (
                "--experiment",
                str(preset.experiment),
                "--split",
                args.split,
                "--start-index",
                str(args.start_index),
                "--max-samples",
                str(args.max_samples),
            )
        )
    elif preset.engine == "generation":
        command.extend(("--num-samples", str(args.max_samples)))
    command.extend(preset.arguments)
    if args.checkpoint is not None and not preset.engine.startswith("training"):
        command.extend(("--checkpoint", str(args.checkpoint)))
    command.extend(passthrough)
    return command, output_folder


def main() -> None:
    """List or execute a named PaDIS reconstruction/generation preset."""
    parser = build_parser()
    args, passthrough = parser.parse_known_args()
    if passthrough[:1] == ["--"]:
        passthrough = passthrough[1:]
    if args.command == "list":
        if passthrough:
            parser.error(f"Unexpected arguments: {' '.join(passthrough)}")
        for name, preset in PRESETS.items():
            print(f"{name:28} [{preset.implementation}] {preset.description}")
        if FIGURE_PRESET_GROUPS:
            print("\nPreset groups:")
            for name, presets in FIGURE_PRESET_GROUPS.items():
                print(f"{name:28} {', '.join(presets)}")
        return

    if args.command == "run-group":
        apply_generation_checkpoint_defaults(args)
        for preset_name in FIGURE_PRESET_GROUPS[args.group]:
            child_args = argparse.Namespace(**vars(args))
            child_args.preset = preset_name
            if PRESETS[preset_name].engine == "generation":
                if "whole" in preset_name and child_args.whole_checkpoint is not None:
                    child_args.checkpoint = child_args.whole_checkpoint
                elif child_args.patch_checkpoint is not None:
                    child_args.checkpoint = child_args.patch_checkpoint
            command, output_folder = command_for(child_args, passthrough)
            print("Running:", " ".join(command))
            if child_args.dry_run:
                continue

            output_folder.mkdir(parents=True, exist_ok=True)
            manifest = {
                "preset": preset_name,
                "preset_group": args.group,
                "preset_config": asdict(PRESETS[preset_name]),
                "command": command,
                "seed": child_args.seed,
                "passthrough_overrides": passthrough,
            }
            with open(output_folder / "launcher_manifest.json", "w") as file:
                json.dump(manifest, file, indent=2)
            env = os.environ.copy()
            env.setdefault("PYTHONHASHSEED", str(child_args.seed))
            with open(output_folder / "run.log", "a", buffering=1) as log_file:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="")
                    log_file.write(line)
                return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
        return

    if args.command == "make-figures":
        if passthrough:
            parser.error(f"Unexpected arguments: {' '.join(passthrough)}")
        command = figure_command_for(args)
        print("Running:", " ".join(command))
        if args.dry_run:
            return
        subprocess.run(command, check=True)
        return

    command, output_folder = command_for(args, passthrough)
    print("Running:", " ".join(command))
    if args.dry_run:
        return

    output_folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "preset": args.preset,
        "preset_config": asdict(PRESETS[args.preset]),
        "command": command,
        "seed": args.seed,
        "passthrough_overrides": passthrough,
    }
    with open(output_folder / "launcher_manifest.json", "w") as file:
        json.dump(manifest, file, indent=2)
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", str(args.seed))
    with open(output_folder / "run.log", "a", buffering=1) as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


if __name__ == "__main__":
    main()
