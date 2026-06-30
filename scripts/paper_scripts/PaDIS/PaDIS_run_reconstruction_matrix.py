"""Run or enumerate the PaDIS reconstruction matrix for trained LION priors."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import pathlib
import shlex
import subprocess
import sys


IMPLEMENTATIONS = ("paper", "public_repo", "lion_physics", "lion_quality")
GEOMETRIES = ("lion", "padis", "padis_parallel", "padis_fanbeam")
EXPERIMENTS = ("ct_8", "ct_20", "ct_60", "ct_fanbeam_180", "ct_512_60")
METHODS = (
    "baseline",
    "admm_tv",
    "pnp_admm",
    "whole_image_diffusion",
    "langevin",
    "predictor_corrector",
    "ve_ddnm",
    "patch_average",
    "patch_stitch",
    "padis_dps",
)
NO_PADIS_PRIOR_METHODS = {"baseline", "admm_tv", "pnp_admm"}
PUBLIC_REPO_IMPLEMENTATION_METHODS = {
    "padis_dps",
    "langevin",
    "predictor_corrector",
    "ve_ddnm",
    "patch_average",
    "patch_stitch",
}
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
class MethodTask:
    name: str
    default_model: str
    implementation: str
    algorithm: str
    default_experiments: tuple[str, ...]
    model_by_experiment: dict[str, str] | None = None
    requires_pnp: bool = False


@dataclass(frozen=True)
class ReconstructionJob:
    model: ModelTask
    method: MethodTask
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

_MAIN_CT_EXPERIMENTS = ("ct_20", "ct_8")
_EXTRA_CT_EXPERIMENTS = ("ct_60", "ct_fanbeam_180")
_NATIVE_512_EXPERIMENTS = ("ct_512_60",)
_PATCH_CT_EXPERIMENTS = (
    *_MAIN_CT_EXPERIMENTS,
    *_EXTRA_CT_EXPERIMENTS,
    *_NATIVE_512_EXPERIMENTS,
)

METHOD_TASKS = (
    MethodTask(
        "baseline",
        "patch_lidc_default",
        "paper",
        "dps_langevin",
        _PATCH_CT_EXPERIMENTS,
        {"ct_512_60": "patch_lidc_512"},
    ),
    MethodTask(
        "admm_tv",
        "patch_lidc_default",
        "paper",
        "dps_langevin",
        _PATCH_CT_EXPERIMENTS,
        {"ct_512_60": "patch_lidc_512"},
    ),
    MethodTask(
        "pnp_admm",
        "patch_lidc_default",
        "paper",
        "dps_langevin",
        _MAIN_CT_EXPERIMENTS,
        requires_pnp=True,
    ),
    MethodTask(
        "whole_image_diffusion",
        "whole_lidc_default",
        "paper",
        "dps_langevin",
        (*_MAIN_CT_EXPERIMENTS, *_EXTRA_CT_EXPERIMENTS),
    ),
    MethodTask(
        "langevin",
        "patch_lidc_default",
        "lion_physics",
        "langevin",
        ("ct_20",),
    ),
    MethodTask(
        "predictor_corrector",
        "patch_lidc_default",
        "lion_physics",
        "pc",
        ("ct_20",),
    ),
    MethodTask(
        "ve_ddnm",
        "patch_lidc_default",
        "lion_physics",
        "langevin",
        ("ct_20",),
    ),
    MethodTask(
        "patch_average",
        "patch_lidc_default",
        "lion_physics",
        "dps_langevin",
        ("ct_20",),
    ),
    MethodTask(
        "patch_stitch",
        "patch_lidc_default",
        "lion_physics",
        "dps_langevin",
        ("ct_20",),
    ),
    MethodTask(
        "padis_dps",
        "patch_lidc_default",
        "lion_physics",
        "dps_langevin",
        _PATCH_CT_EXPERIMENTS,
        {"ct_512_60": "patch_lidc_512"},
    ),
)
METHOD_BY_NAME = {task.name: task for task in METHOD_TASKS}


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


def selected_method_tasks(selection: str) -> tuple[MethodTask, ...]:
    names = parse_csv(selection, valid=METHODS, label="method")
    return tuple(METHOD_BY_NAME[name] for name in names)


def selected_experiments(
    selection: str, model: ModelTask, method: MethodTask | None = None
) -> tuple[str, ...]:
    if selection == "paper_matrix":
        if method is not None:
            return method.default_experiments
        return model.default_experiments
    return parse_csv(selection, valid=EXPERIMENTS, label="experiment")


def validate_paper_experiments(
    experiments: tuple[str, ...],
    allowed: tuple[str, ...],
    *,
    label: str,
    allow_off_paper: bool,
) -> None:
    if allow_off_paper:
        return
    off_paper = tuple(
        experiment for experiment in experiments if experiment not in allowed
    )
    if off_paper:
        raise ValueError(
            f"{label} is not part of the paper reconstruction matrix for "
            f"{', '.join(off_paper)}. Paper experiments for this task are: "
            f"{', '.join(allowed)}. Use --allow-off-paper-experiments for "
            "diagnostic or ablation runs outside the paper protocol."
        )


def default_model_for_method_experiment(
    method: MethodTask, experiment: str
) -> ModelTask:
    model_name = method.default_model
    if method.model_by_experiment is not None:
        model_name = method.model_by_experiment.get(experiment, model_name)
    return MODEL_BY_NAME[model_name]


def validate_method_implementation(method: MethodTask, implementation: str) -> None:
    if (
        implementation == "public_repo"
        and method.name not in PUBLIC_REPO_IMPLEMENTATION_METHODS
    ):
        supported = ", ".join(sorted(PUBLIC_REPO_IMPLEMENTATION_METHODS))
        raise ValueError(
            f"--implementations public_repo is only supported for methods with "
            f"a public PaDIS inverse-sampler analogue: {supported}. "
            f"Method {method.name!r} has no runnable public-repo equivalent."
        )


def build_jobs(args: argparse.Namespace) -> list[ReconstructionJob]:
    methods = selected_method_tasks(args.methods)
    geometries = parse_csv(args.geometries, valid=GEOMETRIES, label="geometry")
    if any(geometry != "lion" for geometry in geometries):
        raise ValueError(UNSUPPORTED_PADIS_GEOMETRY_MESSAGE)

    jobs: list[ReconstructionJob] = []
    for method in methods:
        implementations = (
            (method.implementation,)
            if args.implementations == "method_default"
            else parse_csv(
                args.implementations,
                valid=IMPLEMENTATIONS,
                label="implementation",
            )
        )
        for implementation in implementations:
            validate_method_implementation(method, implementation)
        if args.models == "method_default":
            experiments = selected_experiments(
                args.experiments, MODEL_BY_NAME[method.default_model], method
            )
            if args.experiments != "paper_matrix":
                validate_paper_experiments(
                    experiments,
                    method.default_experiments,
                    label=f"method {method.name!r}",
                    allow_off_paper=args.allow_off_paper_experiments,
                )
            experiment_model_pairs = tuple(
                (
                    experiment,
                    default_model_for_method_experiment(method, experiment),
                )
                for experiment in experiments
            )
        else:
            experiment_model_pairs = []
            for model in selected_model_tasks(args.models):
                experiments = selected_experiments(args.experiments, model)
                if args.experiments != "paper_matrix":
                    validate_paper_experiments(
                        experiments,
                        model.default_experiments,
                        label=f"model {model.name!r}",
                        allow_off_paper=args.allow_off_paper_experiments,
                    )
                experiment_model_pairs.extend(
                    (experiment, model) for experiment in experiments
                )
            experiment_model_pairs = tuple(experiment_model_pairs)
        for experiment, model in experiment_model_pairs:
            for geometry in geometries:
                for implementation in implementations:
                    jobs.append(
                        ReconstructionJob(
                            model=model,
                            method=method,
                            implementation=implementation,
                            geometry=geometry,
                            experiment=experiment,
                        )
                    )
    return jobs


def checkpoint_path(training_root: pathlib.Path, model: ModelTask) -> pathlib.Path:
    return training_root / model.name / model.checkpoint_name


def paper_sampler_views(experiment: str) -> int:
    if experiment == "ct_8":
        return 8
    return 20


def expected_sigma_min(experiment: str) -> float:
    return 0.003 if paper_sampler_views(experiment) == 8 else 0.002


def expected_sampler_settings(job: ReconstructionJob) -> dict:
    sigma_min = expected_sigma_min(job.experiment)
    settings = {
        "num_steps": 100,
        "inner_steps": 10,
        "sigma_min": sigma_min,
        "sigma_max": 10.0,
        "noise_schedule": "geometric",
        "zeta": 0.3,
        "sampling_epsilon": 1.0,
        "noise_initialization": "padded",
        "prior_mode": (
            "whole_image" if job.model.prior_mode == "whole-image" else "patch"
        ),
    }
    if job.implementation == "paper":
        settings.update(
            {
                "initial_reconstruction": "noise",
                "clip_initial": False,
                "clip_output": False,
                "dps_epsilon": 1.0,
                "data_consistency_gradient": "paper_squared_residual",
                "adjoint_data_step_schedule": "paper",
            }
        )
    elif job.implementation == "public_repo":
        settings.update(
            {
                "initial_reconstruction": "fdk",
                "clip_initial": True,
                "clip_output": True,
                "dps_epsilon": 0.5,
                "data_consistency_gradient": "norm",
                "adjoint_data_step_schedule": "public_repo",
                "data_consistency_scale": 0.0405,
                "adjoint_data_consistency_scale": 0.1022,
            }
        )
    elif job.implementation == "lion_quality":
        settings.update(
            {
                "initial_reconstruction": "fdk",
                "clip_initial": True,
                "clip_output": True,
                "dps_epsilon": 1.0,
                "data_consistency_gradient": "paper_squared_residual",
                "adjoint_data_step_schedule": "paper",
                "initial_fdk_filter_type": "hann",
                "initial_fdk_frequency_scaling": 0.9,
                "initial_fdk_padded": False,
                "data_consistency_normalization": "operator_norm",
            }
        )
    elif job.implementation == "lion_physics":
        settings.update(
            {
                "initial_reconstruction": "fdk",
                "clip_initial": True,
                "clip_output": True,
                "zeta": 3.0,
                "dps_epsilon": 1.0,
                "data_consistency_gradient": "least_squares",
                "adjoint_data_step_schedule": "paper",
                "initial_fdk_filter_type": "hann",
                "initial_fdk_frequency_scaling": 0.9,
                "initial_fdk_padded": False,
                "data_consistency_normalization": "operator_lipschitz",
                "data_consistency_scale": 1.0,
                "adjoint_data_consistency_scale": None,
                "pc_snr": 0.08,
            }
        )

    if job.method.name == "whole_image_diffusion":
        settings["prior_mode"] = "whole_image"
    if job.method.name == "predictor_corrector":
        settings["pc_snr"] = 0.08 if job.implementation == "lion_physics" else 0.16
        if job.implementation == "lion_physics":
            settings["zeta"] = 4.25
        settings["pc_corrector_step_rule"] = "paper_linear"
        settings["pc_corrector_denoise_sigma"] = (
            "current" if job.implementation == "public_repo" else "next"
        )
        settings["pc_reuse_predictor_layout"] = job.implementation == "public_repo"
    if job.method.name == "langevin" and job.implementation == "lion_physics":
        settings["zeta"] = 4.0
        settings["sampling_epsilon"] = 0.5
    if job.method.name == "ve_ddnm":
        if job.implementation == "paper":
            settings["num_steps"] = 1000
            settings["inner_steps"] = 1
            settings["ve_ddnm_nfe_layout"] = "paper_1000x1"
        elif job.implementation in ("lion_physics", "lion_quality"):
            settings["num_steps"] = 1000
            settings["inner_steps"] = 1
            settings["ve_ddnm_nfe_layout"] = "paper_1000x1"
            settings["initial_reconstruction"] = "noise"
            settings["clip_initial"] = False
            settings["clip_output"] = False
            settings["sampling_epsilon"] = 0.1
            settings["initial_fdk_filter_type"] = None
            settings["initial_fdk_frequency_scaling"] = 1.0
            settings["initial_fdk_padded"] = True
            settings["ddnm_corrected_clip"] = True
        else:
            settings["ve_ddnm_nfe_layout"] = "public_inner"
        settings["langevin_ddnm"] = True
        settings["ddnm_pseudoinverse_clip"] = True
        settings["ddnm_projected_pseudoinverse_clip"] = True
    if job.method.name == "patch_average":
        settings["patch_assembly"] = "fixed_average"
        if job.implementation == "lion_physics":
            settings["dps_epsilon"] = 0.5
        settings["fixed_overlap_layout"] = (
            "public_overlap"
            if job.implementation in ("public_repo", "lion_physics")
            else "lion_clipped"
        )
        settings["fixed_overlap_checkpoint_denoiser"] = True
    elif job.method.name == "patch_stitch":
        settings["patch_assembly"] = "fixed_stitch"
        if job.implementation == "lion_physics":
            settings["dps_epsilon"] = 0.5
        settings["fixed_overlap_layout"] = (
            "public_tile"
            if job.implementation in ("public_repo", "lion_physics")
            else "lion_clipped"
        )
        settings["fixed_overlap_checkpoint_denoiser"] = True
    return settings


def pnp_checkpoint_for_args(args: argparse.Namespace) -> pathlib.Path:
    if args.pnp_checkpoint is not None:
        return args.pnp_checkpoint
    return args.pnp_root / "pnp_lidc_drunet.pt"


def expected_method_settings(args: argparse.Namespace, job: ReconstructionJob) -> dict:
    if job.method.name == "baseline":
        return {"baseline": "fdk"}
    if job.method.name == "admm_tv":
        return {
            "tv_lambda": float(args.tv_lambda),
            "tv_iterations": int(args.tv_iterations),
            "tv_lipschitz": None,
            "tv_non_negativity": False,
        }
    if job.method.name == "pnp_admm":
        return {
            "pnp_checkpoint": str(pnp_checkpoint_for_args(args)),
            "pnp_iterations": int(args.pnp_iterations),
            "pnp_eta": float(args.pnp_eta),
            "pnp_cg_iterations": int(args.pnp_cg_iterations),
            "pnp_cg_tolerance": float(args.pnp_cg_tolerance),
            "pnp_noise_level": (
                None if args.pnp_noise_level is None else float(args.pnp_noise_level)
            ),
        }
    return {}


def command_for_job(args: argparse.Namespace, job: ReconstructionJob) -> list[str]:
    checkpoint = checkpoint_path(args.training_root, job.model)
    output_folder = (
        args.output_root
        / job.method.name
        / job.model.name
        / job.implementation
        / job.geometry
    )
    cmd = [
        sys.executable,
        "-u",
        "scripts/paper_scripts/PaDIS/PaDIS_LIDC_reconstruction.py",
        "--output-folder",
        str(output_folder),
        "--experiment",
        job.experiment,
        "--implementation",
        job.implementation,
        "--geometry",
        job.geometry,
        "--method",
        job.method.name,
        "--split",
        args.split,
        "--algorithm",
        job.method.algorithm,
        "--max-samples",
        str(args.max_samples),
        "--start-index",
        str(args.start_index),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if job.method.name not in NO_PADIS_PRIOR_METHODS:
        cmd.extend(["--checkpoint", str(checkpoint)])
    if job.model.prior_mode != "auto":
        cmd.extend(["--prior-mode", job.model.prior_mode])
    if job.model.no_position_channels:
        cmd.append("--no-position-channels")
    if job.method.requires_pnp:
        cmd.extend(["--pnp-checkpoint", str(pnp_checkpoint_for_args(args))])
    if job.method.name == "admm_tv":
        cmd.extend(["--tv-lambda", str(args.tv_lambda)])
        cmd.extend(["--tv-iterations", str(args.tv_iterations)])
    if job.method.name == "pnp_admm":
        cmd.extend(["--pnp-iterations", str(args.pnp_iterations)])
        cmd.extend(["--pnp-eta", str(args.pnp_eta)])
        cmd.extend(["--pnp-cg-iterations", str(args.pnp_cg_iterations)])
        cmd.extend(["--pnp-cg-tolerance", str(args.pnp_cg_tolerance)])
        if args.pnp_noise_level is not None:
            cmd.extend(["--pnp-noise-level", str(args.pnp_noise_level)])
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
    prior_mode = "patch" if job.model.prior_mode == "auto" else job.model.prior_mode
    if prior_mode == "whole-image":
        prior_mode = "whole_image"
    return {
        "model": job.model.name,
        "checkpoint": (
            ""
            if job.method.name in NO_PADIS_PRIOR_METHODS
            else str(checkpoint_path(args.training_root, job.model))
        ),
        "method": job.method.name,
        "algorithm": job.method.algorithm,
        "prior_mode": prior_mode,
        "expected_sampler": expected_sampler_settings(job),
        "expected_method_settings": expected_method_settings(args, job),
        "implementation": job.implementation,
        "geometry": job.geometry,
        "experiment": job.experiment,
        "command": command_for_job(args, job),
    }


def input_check_failures(
    args: argparse.Namespace, jobs: list[ReconstructionJob]
) -> list[str]:
    failures = []
    seen_model_checkpoints: set[pathlib.Path] = set()
    seen_pnp_checkpoints: set[pathlib.Path] = set()
    for job in jobs:
        checkpoint = checkpoint_path(args.training_root, job.model)
        if (
            job.method.name not in NO_PADIS_PRIOR_METHODS
            and checkpoint not in seen_model_checkpoints
        ):
            seen_model_checkpoints.add(checkpoint)
            if not checkpoint.is_file():
                failures.append(
                    f"Missing checkpoint for {job.model.name}: {checkpoint}"
                )
        if job.method.requires_pnp:
            pnp_checkpoint = pnp_checkpoint_for_args(args)
            if pnp_checkpoint not in seen_pnp_checkpoints:
                seen_pnp_checkpoints.add(pnp_checkpoint)
                if not pnp_checkpoint.is_file():
                    failures.append(
                        f"Missing PnP denoiser checkpoint: {pnp_checkpoint}"
                    )
    return failures


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument(
        "--models",
        default="method_default",
        help=(
            "method_default to use each method's paper checkpoint family, "
            "or comma-separated model task names/all for model ablations."
        ),
    )
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="Comma-separated method names, or all. Valid values: "
        + ", ".join(METHODS),
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
        "--allow-off-paper-experiments",
        action="store_true",
        help=(
            "Allow explicit --experiments selections outside the selected "
            "method/model's paper matrix. Use only for diagnostics or ablations."
        ),
    )
    parser.add_argument(
        "--implementations",
        default="method_default",
        help=(
            "method_default to use each method's paper or public-fallback "
            "implementation, or a comma-separated list from: "
            + ", ".join(IMPLEMENTATIONS)
        ),
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
        help=(
            "Legacy option retained for compatibility; method tasks now choose "
            "their own algorithm unless extra reconstruction args override it."
        ),
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
    parser.add_argument(
        "--pnp-root",
        type=pathlib.Path,
        default=None,
        help=(
            "Folder containing pnp_lidc_drunet.pt. Defaults to "
            "<training-root>/pnp_lidc_drunet."
        ),
    )
    parser.add_argument("--pnp-checkpoint", type=pathlib.Path, default=None)
    parser.add_argument("--pnp-iterations", type=int, default=10)
    parser.add_argument("--pnp-eta", type=float, default=1e-4)
    parser.add_argument("--pnp-cg-iterations", type=int, default=100)
    parser.add_argument("--pnp-cg-tolerance", type=float, default=1e-7)
    parser.add_argument("--pnp-noise-level", type=float, default=None)
    parser.add_argument("--tv-lambda", type=float, default=0.001)
    parser.add_argument("--tv-iterations", type=int, default=500)
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
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help=(
            "Check that all checkpoints required by the selected jobs exist, "
            "then exit without running reconstructions."
        ),
    )
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
    if args.pnp_root is None:
        args.pnp_root = args.training_root / "pnp_lidc_drunet"
    else:
        args.pnp_root = args.pnp_root.expanduser().resolve()
    if args.pnp_checkpoint is not None:
        args.pnp_checkpoint = args.pnp_checkpoint.expanduser().resolve()
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

    if args.check_inputs:
        failures = input_check_failures(args, jobs)
        if failures:
            raise FileNotFoundError(
                "Reconstruction input check failed:\n  " + "\n  ".join(failures)
            )
        print(f"Input check passed for {len(jobs)} reconstruction job(s).")
        return

    for job in jobs:
        cmd = command_for_job(args, job)
        print("Executing reconstruction job:")
        print(" ".join(shlex.quote(part) for part in cmd))
        if args.dry_run:
            continue
        failures = input_check_failures(args, [job])
        if failures:
            message = failures[0]
            if args.allow_missing_checkpoint:
                print(message)
                continue
            raise FileNotFoundError(message)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
