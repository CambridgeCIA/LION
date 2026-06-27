"""Run or enumerate the PaDIS reconstruction matrix for trained LION priors."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import pathlib
import shlex
import subprocess
import sys


IMPLEMENTATIONS = ("paper", "public_repo")
GEOMETRIES = ("lion", "padis", "padis_parallel", "padis_fanbeam")
EXPERIMENTS = ("ct_8", "ct_20", "ct_60", "ct_fanbeam_180", "ct_512_60")
UNSUPPORTED_PADIS_GEOMETRY_MESSAGE = (
    "PaDIS geometry is intentionally not implemented for LIDC-IDRI. The "
    "processed LIDC slices used by these scripts are 512x512 HU arrays without "
    "the per-scan pixel spacing/orientation needed to resample them into the "
    "PaDIS public-repo coordinate system. The public PaDIS CT operators use a "
    "40-unit image support and 80-unit detector span, while the LION LIDC CT "
    "setup uses a 300 mm field of view with detector size 900, DSO 575 mm, and "
    "DSD 1050 mm. Use --geometries lion."
)


@dataclass(frozen=True)
class ModelTask:
    name: str
    checkpoint_name: str
    prior_mode: str
    no_position_channels: bool
    default_experiments: tuple[str, ...]


@dataclass(frozen=True)
class ReconstructionJob:
    model: ModelTask
    implementation: str
    geometry: str
    experiment: str


MODEL_TASKS = (
    ModelTask(
        "patch_lidc_default",
        "padis_lidc_256.pt",
        "auto",
        False,
        ("ct_20", "ct_8", "ct_60", "ct_fanbeam_180"),
    ),
    ModelTask(
        "patch_lidc_full",
        "padis_lidc_256.pt",
        "auto",
        False,
        ("ct_20",),
    ),
    ModelTask(
        "patch_lidc_p8_default",
        "padis_lidc_256.pt",
        "auto",
        False,
        ("ct_20",),
    ),
    ModelTask(
        "patch_lidc_p16_default",
        "padis_lidc_256.pt",
        "auto",
        False,
        ("ct_20",),
    ),
    ModelTask(
        "patch_lidc_p32_default",
        "padis_lidc_256.pt",
        "auto",
        False,
        ("ct_20",),
    ),
    ModelTask(
        "patch_lidc_p96_default",
        "padis_lidc_256.pt",
        "auto",
        False,
        ("ct_20",),
    ),
    ModelTask(
        "patch_lidc_no_pos_default",
        "padis_lidc_256.pt",
        "auto",
        True,
        ("ct_20",),
    ),
    ModelTask(
        "whole_lidc_default",
        "whole_image_lidc_256.pt",
        "whole-image",
        False,
        ("ct_20", "ct_8", "ct_60", "ct_fanbeam_180"),
    ),
    ModelTask(
        "whole_lidc_full",
        "whole_image_lidc_256.pt",
        "whole-image",
        False,
        ("ct_20",),
    ),
    ModelTask(
        "patch_lidc_512",
        "padis_lidc_512.pt",
        "auto",
        False,
        ("ct_512_60",),
    ),
)

MODEL_BY_NAME = {task.name: task for task in MODEL_TASKS}


def parse_csv(value: str, *, valid: tuple[str, ...], label: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError(f"{label} selection cannot be empty.")
    if items == ("all",):
        return valid
    unknown = sorted(set(items) - set(valid))
    if unknown:
        raise ValueError(
            f"Unknown {label} value(s): {', '.join(unknown)}. "
            f"Valid values: {', '.join(valid)}."
        )
    return items


def selected_model_tasks(selection: str) -> tuple[ModelTask, ...]:
    names = parse_csv(
        selection,
        valid=tuple(MODEL_BY_NAME.keys()),
        label="model",
    )
    return tuple(MODEL_BY_NAME[name] for name in names)


def selected_experiments(selection: str, model: ModelTask) -> tuple[str, ...]:
    if selection == "paper_matrix":
        return model.default_experiments
    return parse_csv(selection, valid=EXPERIMENTS, label="experiment")


def build_jobs(args: argparse.Namespace) -> list[ReconstructionJob]:
    models = selected_model_tasks(args.models)
    implementations = parse_csv(
        args.implementations,
        valid=IMPLEMENTATIONS,
        label="implementation",
    )
    geometries = parse_csv(args.geometries, valid=GEOMETRIES, label="geometry")
    if any(geometry != "lion" for geometry in geometries):
        raise ValueError(UNSUPPORTED_PADIS_GEOMETRY_MESSAGE)

    jobs: list[ReconstructionJob] = []
    for model in models:
        for experiment in selected_experiments(args.experiments, model):
            for implementation in implementations:
                for geometry in geometries:
                    jobs.append(
                        ReconstructionJob(
                            model=model,
                            implementation=implementation,
                            geometry=geometry,
                            experiment=experiment,
                        )
                    )
    return jobs


def checkpoint_path(training_root: pathlib.Path, model: ModelTask) -> pathlib.Path:
    return training_root / model.name / model.checkpoint_name


def command_for_job(args: argparse.Namespace, job: ReconstructionJob) -> list[str]:
    checkpoint = checkpoint_path(args.training_root, job.model)
    output_folder = (
        args.output_root / job.model.name / job.implementation / job.geometry
    )
    cmd = [
        sys.executable,
        "-u",
        "scripts/paper_scripts/PaDIS/PaDIS_LIDC_reconstruction.py",
        "--checkpoint",
        str(checkpoint),
        "--output-folder",
        str(output_folder),
        "--experiment",
        job.experiment,
        "--implementation",
        job.implementation,
        "--geometry",
        job.geometry,
        "--split",
        args.split,
        "--algorithm",
        args.algorithm,
        "--max-samples",
        str(args.max_samples),
        "--start-index",
        str(args.start_index),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if job.model.prior_mode != "auto":
        cmd.extend(["--prior-mode", job.model.prior_mode])
    if job.model.no_position_channels:
        cmd.append("--no-position-channels")
    if args.data_folder is not None:
        cmd.extend(["--data-folder", str(args.data_folder)])
    if args.public_padis_image_dir is not None:
        cmd.extend(["--public-padis-image-dir", str(args.public_padis_image_dir)])
    if args.save_previews:
        cmd.append("--save-previews")
    if args.prog_bar:
        cmd.append("--prog-bar")
    if args.trace_interval is not None:
        cmd.extend(["--trace-interval", str(args.trace_interval)])
    if args.trace_images:
        cmd.append("--trace-images")
    for extra_arg in args.reconstruction_arg:
        cmd.append(extra_arg)
    return cmd


def job_json(args: argparse.Namespace, job: ReconstructionJob) -> dict:
    return {
        "model": job.model.name,
        "checkpoint": str(checkpoint_path(args.training_root, job.model)),
        "implementation": job.implementation,
        "geometry": job.geometry,
        "experiment": job.experiment,
        "command": command_for_job(args, job),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model task names, or all.",
    )
    parser.add_argument(
        "--experiments",
        default="paper_matrix",
        help=(
            "paper_matrix for the paper-relevant experiment set per model, "
            "or a comma-separated list from: " + ", ".join(EXPERIMENTS)
        ),
    )
    parser.add_argument(
        "--implementations",
        default="paper,public_repo",
        help="Comma-separated list from: " + ", ".join(IMPLEMENTATIONS),
    )
    parser.add_argument(
        "--geometries",
        default="lion",
        help="Comma-separated list from: " + ", ".join(GEOMETRIES),
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--algorithm",
        choices=("dps_langevin", "langevin", "pc"),
        default="dps_langevin",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=25,
        help="Number of test/validation slices per reconstruction job. Default 25 matches the paper CT evaluation budget.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-folder", type=pathlib.Path, default=None)
    parser.add_argument("--public-padis-image-dir", type=pathlib.Path, default=None)
    parser.add_argument("--save-previews", action="store_true")
    parser.add_argument("--prog-bar", action="store_true")
    parser.add_argument("--trace-interval", type=int, default=None)
    parser.add_argument("--trace-images", action="store_true")
    parser.add_argument(
        "--reconstruction-arg",
        action="append",
        default=[],
        help=(
            "Append one raw argument to PaDIS_LIDC_reconstruction.py. Repeat "
            "for flags and values, for example --reconstruction-arg "
            "--stop-after-outer-steps --reconstruction-arg 1."
        ),
    )
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-missing-checkpoint",
        action="store_true",
        help="Skip a task with a missing checkpoint instead of failing.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.training_root = args.training_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    jobs = build_jobs(args)
    if args.count:
        print(len(jobs))
        return
    if args.list:
        print(json.dumps([job_json(args, job) for job in jobs], indent=2))
        return
    if args.task_index is not None:
        if args.task_index < 0 or args.task_index >= len(jobs):
            raise ValueError(
                f"--task-index {args.task_index} outside matrix size {len(jobs)}."
            )
        jobs = [jobs[args.task_index]]

    for job in jobs:
        checkpoint = checkpoint_path(args.training_root, job.model)
        if not checkpoint.is_file():
            message = f"Missing checkpoint for {job.model.name}: {checkpoint}"
            if args.allow_missing_checkpoint:
                print(message)
                continue
            raise FileNotFoundError(message)
        cmd = command_for_job(args, job)
        print("Executing reconstruction job:")
        print(" ".join(shlex.quote(part) for part in cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
