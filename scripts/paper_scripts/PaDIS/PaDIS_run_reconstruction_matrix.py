"""Run or enumerate the PaDIS reconstruction matrix for trained LION priors."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import pathlib
import shlex
import subprocess
import sys

try:
    from scripts.paper_scripts.PaDIS.PaDIS_hparam_defaults import (
        DEFAULT_RECONSTRUCTION_HPARAM_DEFAULTS_JSON,
        HparamDefaults,
        apply_reconstruction_args_to_settings,
    )
except ModuleNotFoundError:
    from PaDIS_hparam_defaults import (  # type: ignore[no-redef]
        DEFAULT_RECONSTRUCTION_HPARAM_DEFAULTS_JSON,
        HparamDefaults,
        apply_reconstruction_args_to_settings,
    )


TRAINING_ROOT_PRESETS = ("slurm", "gcp")
IMPLEMENTATIONS = ("paper", "public_repo", "lion_physics", "lion_quality")
GEOMETRIES = ("lion", "padis", "padis_parallel", "padis_fanbeam")
EXPERIMENTS = ("ct_8", "ct_20", "ct_60", "ct_fanbeam_180", "ct_512_60")
ABLATIONS = ("schedule_init", "patch_size", "dataset_size", "position_encoding")
CHECKPOINT_POLICIES = ("model_default", "min_val", "min_intense_val")
JOB_ORDERS = ("default", "gcp_spot")
HPARAM_DEFAULT_MODES = ("none", "auto", "json")
DEFAULT_EXPENSIVE_JOB_MAX_SAMPLES = 4
EXPENSIVE_SAMPLE_CAP_METHODS = {"patch_average", "patch_stitch"}
EXPENSIVE_SAMPLE_CAP_EXPERIMENTS = {"ct_512_60"}
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
    matrix_group: str = "main"
    extra_reconstruction_args: tuple[str, ...] = ()
    sampler_overrides: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class AblationTask:
    name: str
    method: str
    model: str
    experiment: str
    ablation_type: str
    implementations: tuple[str, ...] | None = None
    reconstruction_args: tuple[str, ...] = ()
    sampler_overrides: tuple[tuple[str, object], ...] = ()


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
        "whole_image_lidc_256_min_val.pt",
        "whole-image",
        False,
        ("ct_20", "ct_8", "ct_60", "ct_fanbeam_180"),
    ),
    ModelTask(
        "whole_lidc_full",
        "whole_image_lidc_256_min_val.pt",
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
_WHOLE_IMAGE_CT_EXPERIMENTS = (*_MAIN_CT_EXPERIMENTS, *_EXTRA_CT_EXPERIMENTS)
_PATCH_512_MODEL = {"ct_512_60": "patch_lidc_512"}
_WHOLE_IMAGE_SAMPLING_METHODS = ("langevin", "predictor_corrector", "ve_ddnm")
_WHOLE_IMAGE_SAMPLING_EXPERIMENTS = ("ct_20",)
_METHOD_DISPLAY_NAMES = {
    "baseline": "Baseline FDK",
    "admm_tv": "ADMM-TV",
    "pnp_admm": "PnP-ADMM",
    "whole_image_diffusion": "Whole image - VE-DPS",
    "langevin": "Langevin",
    "predictor_corrector": "Predictor-corrector",
    "ve_ddnm": "VE-DDNM",
    "patch_average": "Patch averaging",
    "patch_stitch": "Patch stitching",
    "padis_dps": "Patch - VE-DPS",
}

CORE_IMPLEMENTATIONS_BY_METHOD = {
    "baseline": ("lion_physics",),
    "admm_tv": ("lion_physics",),
    "pnp_admm": ("lion_physics",),
    "whole_image_diffusion": ("lion_physics", "paper"),
    "langevin": ("lion_physics", "public_repo", "paper"),
    "predictor_corrector": ("lion_physics", "public_repo", "paper"),
    "ve_ddnm": ("lion_physics", "public_repo", "paper"),
    "patch_average": ("lion_physics", "public_repo"),
    "patch_stitch": ("lion_physics", "public_repo"),
    "padis_dps": ("lion_physics", "public_repo", "paper"),
}

DEFAULT_PAPER_MATRIX_RESTRICTED_EXPERIMENTS = frozenset(
    {"ct_60", "ct_fanbeam_180", "ct_512_60"}
)
DEFAULT_PAPER_MATRIX_RESTRICTED_MAIN_JOBS = frozenset(
    {
        ("baseline", "lion_physics"),
        ("admm_tv", "lion_physics"),
        ("whole_image_diffusion", "lion_physics"),
        ("padis_dps", "lion_physics"),
        ("padis_dps", "public_repo"),
    }
)
DEFAULT_PAPER_MATRIX_EXTRA_CT_VE_DDNM_EXPERIMENTS = frozenset(
    {"ct_60", "ct_fanbeam_180"}
)

METHOD_TASKS = (
    MethodTask(
        "baseline",
        "patch_lidc_default",
        "lion_physics",
        "dps_langevin",
        _PATCH_CT_EXPERIMENTS,
        _PATCH_512_MODEL,
    ),
    MethodTask(
        "admm_tv",
        "patch_lidc_default",
        "lion_physics",
        "dps_langevin",
        _PATCH_CT_EXPERIMENTS,
        _PATCH_512_MODEL,
    ),
    MethodTask(
        "pnp_admm",
        "patch_lidc_default",
        "lion_physics",
        "dps_langevin",
        _PATCH_CT_EXPERIMENTS,
        _PATCH_512_MODEL,
        requires_pnp=True,
    ),
    MethodTask(
        "whole_image_diffusion",
        "whole_lidc_default",
        "lion_physics",
        "dps_langevin",
        _WHOLE_IMAGE_CT_EXPERIMENTS,
    ),
    MethodTask(
        "langevin",
        "patch_lidc_default",
        "lion_physics",
        "langevin",
        _PATCH_CT_EXPERIMENTS,
        _PATCH_512_MODEL,
    ),
    MethodTask(
        "predictor_corrector",
        "patch_lidc_default",
        "lion_physics",
        "pc",
        _PATCH_CT_EXPERIMENTS,
        _PATCH_512_MODEL,
    ),
    MethodTask(
        "ve_ddnm",
        "patch_lidc_default",
        "lion_physics",
        "langevin",
        _PATCH_CT_EXPERIMENTS,
        _PATCH_512_MODEL,
    ),
    MethodTask(
        "patch_average",
        "patch_lidc_default",
        "lion_physics",
        "dps_langevin",
        _PATCH_CT_EXPERIMENTS,
        _PATCH_512_MODEL,
    ),
    MethodTask(
        "patch_stitch",
        "patch_lidc_default",
        "lion_physics",
        "dps_langevin",
        _PATCH_CT_EXPERIMENTS,
        _PATCH_512_MODEL,
    ),
    MethodTask(
        "padis_dps",
        "patch_lidc_default",
        "lion_physics",
        "dps_langevin",
        _PATCH_CT_EXPERIMENTS,
        _PATCH_512_MODEL,
    ),
)
METHOD_BY_NAME = {task.name: task for task in METHOD_TASKS}

_NOISE_INIT_ARGS = (
    "--initial-reconstruction",
    "noise",
    "--no-clip-initial",
    "--no-clip-output",
)
_FDK_INIT_ARGS = (
    "--initial-reconstruction",
    "fdk",
    "--clip-initial",
    "--clip-output",
    "--initial-fdk-filter-type",
    "hann",
    "--initial-fdk-frequency-scaling",
    "0.3",
    "--no-initial-fdk-padded",
)
_NOISE_INIT_OVERRIDES = (
    ("initial_reconstruction", "noise"),
    ("clip_initial", False),
    ("clip_output", False),
    ("initial_fdk_filter_type", None),
    ("initial_fdk_frequency_scaling", 1.0),
    ("initial_fdk_padded", True),
)
_FDK_INIT_OVERRIDES = (
    ("initial_reconstruction", "fdk"),
    ("clip_initial", True),
    ("clip_output", True),
    ("initial_fdk_filter_type", "hann"),
    ("initial_fdk_frequency_scaling", 0.3),
    ("initial_fdk_padded", False),
)
_SCHEDULE_INIT_IMPLEMENTATIONS = ("lion_physics", "public_repo")
_PATCH_ABLATION_IMPLEMENTATIONS = ("lion_physics", "public_repo")
_DATASET_ABLATION_IMPLEMENTATIONS = ("lion_physics", "public_repo")
_WHOLE_DATASET_ABLATION_IMPLEMENTATIONS = ("lion_physics", "paper")
_POSITION_ABLATION_IMPLEMENTATIONS = ("lion_physics", "public_repo")

SCHEDULE_INIT_ABLATION_TASKS = tuple(
    AblationTask(
        f"schedule_{schedule}_{init}_init",
        "padis_dps",
        "patch_lidc_512" if experiment == "ct_512_60" else "patch_lidc_default",
        experiment,
        "schedule_init",
        _SCHEDULE_INIT_IMPLEMENTATIONS,
        (
            ("--noise-schedule", schedule)
            + (_NOISE_INIT_ARGS if init == "noise" else _FDK_INIT_ARGS)
        ),
        (
            (("noise_schedule", schedule),)
            + (_NOISE_INIT_OVERRIDES if init == "noise" else _FDK_INIT_OVERRIDES)
        ),
    )
    for experiment in EXPERIMENTS
    for schedule in ("geometric", "edm")
    for init in ("fdk", "noise")
)

TRAINED_ABLATION_TASKS = (
    *SCHEDULE_INIT_ABLATION_TASKS,
    AblationTask(
        "patch_size_p8",
        "padis_dps",
        "patch_lidc_p8_default",
        "ct_20",
        "patch_size",
        _PATCH_ABLATION_IMPLEMENTATIONS,
    ),
    AblationTask(
        "patch_size_p16",
        "padis_dps",
        "patch_lidc_p16_default",
        "ct_20",
        "patch_size",
        _PATCH_ABLATION_IMPLEMENTATIONS,
    ),
    AblationTask(
        "patch_size_p32",
        "padis_dps",
        "patch_lidc_p32_default",
        "ct_20",
        "patch_size",
        _PATCH_ABLATION_IMPLEMENTATIONS,
    ),
    AblationTask(
        "patch_size_p56",
        "padis_dps",
        "patch_lidc_default",
        "ct_20",
        "patch_size",
        _PATCH_ABLATION_IMPLEMENTATIONS,
    ),
    AblationTask(
        "patch_size_p96",
        "padis_dps",
        "patch_lidc_p96_default",
        "ct_20",
        "patch_size",
        _PATCH_ABLATION_IMPLEMENTATIONS,
    ),
    AblationTask(
        "dataset_size_patch_default",
        "padis_dps",
        "patch_lidc_default",
        "ct_20",
        "dataset_size",
        _DATASET_ABLATION_IMPLEMENTATIONS,
    ),
    AblationTask(
        "dataset_size_patch_full",
        "padis_dps",
        "patch_lidc_full",
        "ct_20",
        "dataset_size",
        _DATASET_ABLATION_IMPLEMENTATIONS,
    ),
    AblationTask(
        "dataset_size_whole_default",
        "whole_image_diffusion",
        "whole_lidc_default",
        "ct_20",
        "dataset_size",
        _WHOLE_DATASET_ABLATION_IMPLEMENTATIONS,
    ),
    AblationTask(
        "dataset_size_whole_full",
        "whole_image_diffusion",
        "whole_lidc_full",
        "ct_20",
        "dataset_size",
        _WHOLE_DATASET_ABLATION_IMPLEMENTATIONS,
    ),
    AblationTask(
        "position_no_encoding_noise_init",
        "padis_dps",
        "patch_lidc_no_pos_default",
        "ct_20",
        "position_encoding",
        _POSITION_ABLATION_IMPLEMENTATIONS,
        _NOISE_INIT_ARGS,
        _NOISE_INIT_OVERRIDES,
    ),
    AblationTask(
        "position_no_encoding_fdk_init",
        "padis_dps",
        "patch_lidc_no_pos_default",
        "ct_20",
        "position_encoding",
        _POSITION_ABLATION_IMPLEMENTATIONS,
        _FDK_INIT_ARGS,
        _FDK_INIT_OVERRIDES,
    ),
    AblationTask(
        "position_with_encoding_noise_init",
        "padis_dps",
        "patch_lidc_default",
        "ct_20",
        "position_encoding",
        _POSITION_ABLATION_IMPLEMENTATIONS,
        _NOISE_INIT_ARGS,
        _NOISE_INIT_OVERRIDES,
    ),
    AblationTask(
        "position_with_encoding_fdk_init",
        "padis_dps",
        "patch_lidc_default",
        "ct_20",
        "position_encoding",
        _POSITION_ABLATION_IMPLEMENTATIONS,
        _FDK_INIT_ARGS,
        _FDK_INIT_OVERRIDES,
    ),
)


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


def selected_ablation_types(selection: str) -> tuple[str, ...]:
    selection = selection.strip()
    if selection in ("", "none"):
        return ()
    return parse_csv(selection, valid=ABLATIONS, label="ablation")


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


def selected_implementations_for_method(
    method: MethodTask, selection: str
) -> tuple[str, ...]:
    if selection == "method_default":
        return CORE_IMPLEMENTATIONS_BY_METHOD[method.name]
    return parse_csv(selection, valid=IMPLEMENTATIONS, label="implementation")


def include_job_in_default_paper_matrix(
    args: argparse.Namespace, job: ReconstructionJob
) -> bool:
    if args.models != "method_default" or args.experiments != "paper_matrix":
        return True
    if job.experiment not in DEFAULT_PAPER_MATRIX_RESTRICTED_EXPERIMENTS:
        return True
    if job.matrix_group != "main":
        return False
    if (
        job.method.name == "ve_ddnm"
        and job.experiment in DEFAULT_PAPER_MATRIX_EXTRA_CT_VE_DDNM_EXPERIMENTS
    ):
        return job.implementation in CORE_IMPLEMENTATIONS_BY_METHOD["ve_ddnm"]
    return (
        job.method.name,
        job.implementation,
    ) in DEFAULT_PAPER_MATRIX_RESTRICTED_MAIN_JOBS


def job_identity(job: ReconstructionJob) -> tuple:
    return (
        job.method.name,
        job.model.name,
        job.implementation,
        job.geometry,
        job.experiment,
        job.matrix_group,
        job.extra_reconstruction_args,
    )


def job_prior_mode(job: ReconstructionJob) -> str:
    if job.model.prior_mode == "whole-image":
        return "whole_image"
    if job.method.name == "whole_image_diffusion":
        return "whole_image"
    return "patch"


def job_display_label(job: ReconstructionJob) -> str:
    method_label = _METHOD_DISPLAY_NAMES.get(job.method.name, job.method.name)
    if job.method.name == "pnp_admm" and job.matrix_group == "pnp_noise_conditioned":
        return "PnP-ADMM (noise-conditioned)"
    if job.method.name in _WHOLE_IMAGE_SAMPLING_METHODS:
        prefix = "Whole image" if job_prior_mode(job) == "whole_image" else "Patch"
        return f"{prefix} - {method_label}"
    return method_label


def append_whole_image_sampling_jobs(
    args: argparse.Namespace,
    jobs: list[ReconstructionJob],
    methods: tuple[MethodTask, ...],
    geometries: tuple[str, ...],
) -> None:
    if args.models != "method_default" or args.experiments != "paper_matrix":
        return
    if args.implementations == "method_default":
        implementations = IMPLEMENTATIONS
    else:
        implementations = parse_csv(
            args.implementations,
            valid=IMPLEMENTATIONS,
            label="implementation",
        )
    if "lion_physics" not in implementations:
        return

    selected_method_names = {method.name for method in methods}
    seen = {job_identity(job) for job in jobs}
    model = MODEL_BY_NAME["whole_lidc_default"]
    for method_name in _WHOLE_IMAGE_SAMPLING_METHODS:
        if method_name not in selected_method_names:
            continue
        method = METHOD_BY_NAME[method_name]
        for experiment in _WHOLE_IMAGE_SAMPLING_EXPERIMENTS:
            for geometry in geometries:
                job = ReconstructionJob(
                    model=model,
                    method=method,
                    implementation="lion_physics",
                    geometry=geometry,
                    experiment=experiment,
                )
                identity = job_identity(job)
                if identity not in seen:
                    jobs.append(job)
                    seen.add(identity)


def append_trained_ablation_jobs(
    args: argparse.Namespace,
    jobs: list[ReconstructionJob],
    methods: tuple[MethodTask, ...],
    geometries: tuple[str, ...],
) -> None:
    ablation_types = selected_ablation_types(args.ablations)
    if not ablation_types or args.models != "method_default":
        return

    if args.experiments == "paper_matrix":
        selected_experiment_names = set(EXPERIMENTS)
    else:
        selected_experiment_names = set(
            parse_csv(args.experiments, valid=EXPERIMENTS, label="experiment")
        )
    selected_method_names = {method.name for method in methods}
    selected_ablation_type_names = set(ablation_types)
    seen = {job_identity(job) for job in jobs}

    for task in TRAINED_ABLATION_TASKS:
        if task.ablation_type not in selected_ablation_type_names:
            continue
        if task.method not in selected_method_names:
            continue
        if task.experiment not in selected_experiment_names:
            continue
        method = METHOD_BY_NAME[task.method]
        model = MODEL_BY_NAME[task.model]
        implementations = selected_implementations_for_method(
            method, args.implementations
        )
        if task.implementations is not None:
            implementations = tuple(
                implementation
                for implementation in implementations
                if implementation in task.implementations
            )
        for implementation in implementations:
            validate_method_implementation(method, implementation)
            for geometry in geometries:
                job = ReconstructionJob(
                    model=model,
                    method=method,
                    implementation=implementation,
                    geometry=geometry,
                    experiment=task.experiment,
                    matrix_group=task.name,
                    extra_reconstruction_args=task.reconstruction_args,
                    sampler_overrides=task.sampler_overrides,
                )
                if not include_job_in_default_paper_matrix(args, job):
                    continue
                identity = job_identity(job)
                if identity not in seen:
                    jobs.append(job)
                    seen.add(identity)


def append_pnp_noise_conditioned_jobs(
    args: argparse.Namespace,
    jobs: list[ReconstructionJob],
    methods: tuple[MethodTask, ...],
    geometries: tuple[str, ...],
) -> None:
    if args.models != "method_default" or args.experiments != "paper_matrix":
        return
    if "pnp_admm" not in {method.name for method in methods}:
        return
    if "lion_physics" not in selected_implementations_for_method(
        METHOD_BY_NAME["pnp_admm"], args.implementations
    ):
        return

    seen = {job_identity(job) for job in jobs}
    model = MODEL_BY_NAME["patch_lidc_default"]
    method = METHOD_BY_NAME["pnp_admm"]
    for experiment in _MAIN_CT_EXPERIMENTS:
        for geometry in geometries:
            job = ReconstructionJob(
                model=model,
                method=method,
                implementation="lion_physics",
                geometry=geometry,
                experiment=experiment,
                matrix_group="pnp_noise_conditioned",
            )
            identity = job_identity(job)
            if identity not in seen:
                jobs.append(job)
                seen.add(identity)


def build_jobs(args: argparse.Namespace) -> list[ReconstructionJob]:
    methods = selected_method_tasks(args.methods)
    geometries = parse_csv(args.geometries, valid=GEOMETRIES, label="geometry")
    if any(geometry != "lion" for geometry in geometries):
        raise ValueError(UNSUPPORTED_PADIS_GEOMETRY_MESSAGE)

    jobs: list[ReconstructionJob] = []
    for method in methods:
        implementations = selected_implementations_for_method(
            method, args.implementations
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
                    job = ReconstructionJob(
                        model=model,
                        method=method,
                        implementation=implementation,
                        geometry=geometry,
                        experiment=experiment,
                    )
                    if include_job_in_default_paper_matrix(args, job):
                        jobs.append(job)
    append_whole_image_sampling_jobs(args, jobs, methods, geometries)
    append_trained_ablation_jobs(args, jobs, methods, geometries)
    append_pnp_noise_conditioned_jobs(args, jobs, methods, geometries)
    return jobs


def checkpoint_name_for_policy(model: ModelTask, policy: str) -> str:
    if policy == "model_default":
        return model.checkpoint_name
    if policy not in CHECKPOINT_POLICIES:
        raise ValueError(
            f"Unknown checkpoint policy {policy!r}. "
            f"Valid values: {', '.join(CHECKPOINT_POLICIES)}."
        )
    if model.name.startswith("whole_lidc_"):
        prefix = "whole_image_lidc_256"
    elif model.name == "patch_lidc_512":
        prefix = "padis_lidc_512"
    else:
        prefix = "padis_lidc_256"
    return f"{prefix}_{policy}.pt"


def checkpoint_path(
    training_root: pathlib.Path,
    model: ModelTask,
    checkpoint_policy: str = "model_default",
) -> pathlib.Path:
    return (
        training_root
        / model.name
        / checkpoint_name_for_policy(model, checkpoint_policy)
    )


def ordered_jobs(
    args: argparse.Namespace, jobs: list[ReconstructionJob]
) -> list[ReconstructionJob]:
    if args.job_order == "default":
        return jobs
    if args.job_order != "gcp_spot":
        raise ValueError(
            f"Unknown job order {args.job_order!r}. "
            f"Valid values: {', '.join(JOB_ORDERS)}."
        )

    def sort_key(item: tuple[int, ReconstructionJob]) -> tuple[int, int, int, int]:
        index, job = item
        fixed_overlap_rank = {
            "patch_average": 0,
            "patch_stitch": 1,
        }
        if (
            job.method.name == "ve_ddnm"
            and job.experiment in DEFAULT_PAPER_MATRIX_EXTRA_CT_VE_DDNM_EXPERIMENTS
        ):
            return (1, 0, 0, index)
        if job.matrix_group == "pnp_noise_conditioned":
            return (2, 0, 0, index)
        if job.method.name in fixed_overlap_rank:
            return (3, fixed_overlap_rank[job.method.name], 0, index)
        if job.experiment == "ct_512_60":
            return (4, 0, 0, index)
        return (0, 0, 0, index)

    return [job for _, job in sorted(enumerate(jobs), key=sort_key)]


def default_run_root() -> pathlib.Path:
    if os.environ.get("PADIS_RUN_ROOT"):
        return pathlib.Path(os.environ["PADIS_RUN_ROOT"]).expanduser().resolve()
    if os.environ.get("LION_EXPERIMENTS_PATH"):
        return (
            pathlib.Path(os.environ["LION_EXPERIMENTS_PATH"]).expanduser().resolve()
            / "PaDIS"
        )
    if os.environ.get("LION_DATA_PATH"):
        return (
            pathlib.Path(os.environ["LION_DATA_PATH"]).expanduser().resolve()
            / "experiments"
            / "PaDIS"
        )
    lion_root = pathlib.Path(__file__).resolve().parents[3]
    return (lion_root.parent / "Data" / "experiments" / "PaDIS").resolve()


def resolve_run_root(run_root: pathlib.Path | None) -> pathlib.Path:
    if run_root is not None:
        return run_root.expanduser().resolve()
    return default_run_root()


def default_hparam_run_root(args: argparse.Namespace) -> pathlib.Path:
    if os.environ.get("PADIS_HPARAM_RUN_ROOT"):
        return pathlib.Path(os.environ["PADIS_HPARAM_RUN_ROOT"]).expanduser().resolve()
    if os.environ.get("PADIS_HPARAM_TUNING_RUN_ROOT"):
        return (
            pathlib.Path(os.environ["PADIS_HPARAM_TUNING_RUN_ROOT"])
            .expanduser()
            .resolve()
        )
    return (resolve_run_root(args.run_root) / "hparam_tuning" / "runs").resolve()


def resolve_hparam_run_root(args: argparse.Namespace) -> pathlib.Path | None:
    if args.hparam_defaults in {"none", "json"}:
        return None
    if args.hparam_run_root is not None:
        return args.hparam_run_root.expanduser().resolve()
    return default_hparam_run_root(args)


def resolve_hparam_defaults_json(args: argparse.Namespace) -> pathlib.Path:
    if args.hparam_defaults_json is not None:
        return args.hparam_defaults_json.expanduser().resolve()
    return DEFAULT_RECONSTRUCTION_HPARAM_DEFAULTS_JSON


def hparam_defaults_for_args(args: argparse.Namespace) -> HparamDefaults:
    cached = getattr(args, "_hparam_defaults_cache", None)
    if cached is not None:
        return cached
    if args.hparam_defaults == "none":
        cached = HparamDefaults(())
    elif args.hparam_defaults == "auto":
        cached = HparamDefaults.from_run_root(
            resolve_hparam_run_root(args),
            args.hparam_run_glob,
        )
    elif args.hparam_defaults == "json":
        cached = HparamDefaults.from_json(resolve_hparam_defaults_json(args))
    else:
        raise ValueError(
            f"Unknown hparam default mode {args.hparam_defaults!r}. "
            f"Valid values: {', '.join(HPARAM_DEFAULT_MODES)}."
        )
    setattr(args, "_hparam_defaults_cache", cached)
    return cached


def hparam_selection_for_job(args: argparse.Namespace, job: ReconstructionJob):
    if args.hparam_defaults == "none":
        return None
    return hparam_defaults_for_args(args).select(
        method=job.method.name,
        implementation=job.implementation,
        prior=job_prior_mode(job),
        model=job.model.name,
        experiment=job.experiment,
    )


def hparam_args_for_job(
    args: argparse.Namespace, job: ReconstructionJob
) -> tuple[str, ...]:
    selection = hparam_selection_for_job(args, job)
    if selection is None:
        return ()
    return selection.args


def latest_slurm_training_root(run_root: pathlib.Path) -> pathlib.Path | None:
    final_real_runs = run_root / "final_real_runs"
    if not final_real_runs.is_dir():
        return None
    candidates = tuple(
        path for path in final_real_runs.glob("a100_training_*") if path.is_dir()
    )
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def slurm_training_root(run_root: pathlib.Path, run_stamp: str | None) -> pathlib.Path:
    if run_stamp:
        folder_name = (
            run_stamp
            if run_stamp.startswith("a100_training_")
            else f"a100_training_{run_stamp}"
        )
        return (run_root / "final_real_runs" / folder_name).resolve()
    latest = latest_slurm_training_root(run_root)
    if latest is not None:
        return latest
    raise ValueError(
        "Could not infer the final Slurm model root. Set --training-root, "
        "--run-stamp, PADIS_RUN_STAMP, or create an a100_training_* folder under "
        f"{run_root / 'final_real_runs'}."
    )


def gcp_training_root(run_root: pathlib.Path, gcp_run_name: str) -> pathlib.Path:
    if not gcp_run_name:
        raise ValueError("--gcp-run-name cannot be empty for the gcp preset.")
    return (run_root / "final_real_runs" / gcp_run_name).resolve()


def resolve_training_root(args: argparse.Namespace) -> pathlib.Path:
    if args.training_root is not None:
        return args.training_root.expanduser().resolve()
    if args.training_root_preset is None:
        raise ValueError(
            "Set --training-root explicitly or choose --training-root-preset "
            "slurm|gcp."
        )
    run_root = resolve_run_root(args.run_root)
    if args.training_root_preset == "slurm":
        return slurm_training_root(run_root, args.run_stamp)
    if args.training_root_preset == "gcp":
        return gcp_training_root(run_root, args.gcp_run_name)
    raise ValueError(
        f"Unknown --training-root-preset {args.training_root_preset!r}. "
        f"Valid values: {', '.join(TRAINING_ROOT_PRESETS)}."
    )


def paper_sampler_views(experiment: str) -> int:
    if experiment == "ct_8":
        return 8
    return 20


def expected_sigma_min(experiment: str) -> float:
    return 0.003 if paper_sampler_views(experiment) == 8 else 0.002


def job_reconstruction_args(
    args: argparse.Namespace, job: ReconstructionJob
) -> tuple[str, ...]:
    return (
        *hparam_args_for_job(args, job),
        *job.extra_reconstruction_args,
        *tuple(args.reconstruction_arg),
    )


def expected_sampler_settings(args: argparse.Namespace, job: ReconstructionJob) -> dict:
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
        "prior_mode": job_prior_mode(job),
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
        if job.method.name == "padis_dps":
            settings["zeta"] = 0.0075
            settings["dps_epsilon"] = 0.5
        elif job.method.name == "langevin":
            settings["zeta"] = 0.03
            settings["sampling_epsilon"] = 0.5
        elif job.method.name == "predictor_corrector":
            settings["zeta"] = 0.03
            settings["pc_snr"] = 0.08
        elif job.method.name == "ve_ddnm":
            settings["sampling_epsilon"] = 0.1
    elif job.implementation == "public_repo":
        settings.update(
            {
                "initial_reconstruction": "fdk",
                "clip_initial": True,
                "clip_output": True,
                "initial_fdk_filter_type": "hann",
                "initial_fdk_frequency_scaling": 0.3,
                "initial_fdk_padded": False,
                "dps_epsilon": 0.5,
                "data_consistency_gradient": "norm",
                "adjoint_data_step_schedule": "public_repo",
                "data_consistency_scale": 0.0405,
                "adjoint_data_consistency_scale": 0.1022,
            }
        )
        if job.method.name == "padis_dps":
            settings["zeta"] = 0.2
        elif job.method.name == "langevin":
            settings["zeta"] = 0.2
            settings["sampling_epsilon"] = 0.5
        elif job.method.name == "predictor_corrector":
            settings["zeta"] = 0.5
        elif job.method.name == "ve_ddnm":
            settings["sampling_epsilon"] = 0.2
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
                "initial_fdk_frequency_scaling": 0.3,
                "initial_fdk_padded": False,
                "data_consistency_normalization": "operator_lipschitz",
                "data_consistency_scale": 1.0,
                "adjoint_data_consistency_scale": None,
                "pc_snr": 0.01,
            }
        )

    if job.method.name == "whole_image_diffusion":
        settings["prior_mode"] = "whole_image"
        if job.implementation == "lion_physics":
            settings["zeta"] = 4.0
            settings["dps_epsilon"] = 0.5
    if job.method.name == "padis_dps" and job.implementation == "lion_physics":
        settings["zeta"] = 4.25
        settings["dps_epsilon"] = 0.5
        settings["initial_reconstruction"] = "noise"
        settings["clip_initial"] = False
        settings["clip_output"] = False
        settings["initial_fdk_filter_type"] = None
        settings["initial_fdk_frequency_scaling"] = 1.0
        settings["initial_fdk_padded"] = True
    if job.experiment == "ct_512_60" and job.method.name in {
        "padis_dps",
        "langevin",
        "predictor_corrector",
        "ve_ddnm",
        "patch_average",
        "patch_stitch",
    }:
        settings["patch_batch_size"] = 1
        if job.method.name not in {"patch_average", "patch_stitch"}:
            settings["patch_checkpoint_denoiser"] = True
            settings["fixed_overlap_checkpoint_denoiser"] = False
    if job.method.name == "predictor_corrector":
        if job.implementation == "lion_physics":
            settings["pc_snr"] = 0.01
        elif job.implementation == "paper":
            settings["pc_snr"] = 0.08
        else:
            settings["pc_snr"] = 0.16
        if job.implementation == "lion_physics":
            settings["zeta"] = 4.25
        elif job.implementation == "paper":
            settings["zeta"] = 0.03
        elif job.implementation == "public_repo":
            settings["zeta"] = 0.5
        settings["pc_corrector_step_rule"] = "paper_linear"
        settings["pc_corrector_denoise_sigma"] = (
            "current" if job.implementation == "public_repo" else "next"
        )
        settings["pc_reuse_predictor_layout"] = job.implementation == "public_repo"
    if job.method.name == "langevin":
        if job.implementation == "lion_physics":
            settings["zeta"] = 4.0
            settings["sampling_epsilon"] = 0.5
        elif job.implementation == "paper":
            settings["zeta"] = 0.03
            settings["sampling_epsilon"] = 0.5
        elif job.implementation == "public_repo":
            settings["zeta"] = 0.2
            settings["sampling_epsilon"] = 0.5
    if job.method.name == "ve_ddnm":
        if job.implementation == "paper":
            settings["num_steps"] = 1000
            settings["inner_steps"] = 1
            settings["ve_ddnm_nfe_layout"] = "paper_1000x1"
            settings["sampling_epsilon"] = 0.1
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
            settings["sampling_epsilon"] = 0.2
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
        settings["patch_batch_size"] = 1
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
        settings["patch_batch_size"] = 1
    for key, value in job.sampler_overrides:
        settings[key] = value
    settings, _ = apply_reconstruction_args_to_settings(
        reconstruction_args=job_reconstruction_args(args, job),
        sampler_settings=settings,
    )
    return settings


def pnp_checkpoint_for_args(args: argparse.Namespace) -> pathlib.Path:
    if args.pnp_checkpoint is not None:
        return args.pnp_checkpoint
    return args.pnp_root / "pnp_lidc_drunet_min_val.pt"


def pnp_noise_conditioned_checkpoint_for_args(
    args: argparse.Namespace,
) -> pathlib.Path:
    if args.pnp_noise_conditioned_checkpoint is not None:
        return args.pnp_noise_conditioned_checkpoint
    return args.pnp_noise_conditioned_root / "pnp_lidc_drunet_noise_cond_min_val.pt"


def pnp_checkpoint_for_job(
    args: argparse.Namespace, job: ReconstructionJob
) -> pathlib.Path:
    if job.matrix_group == "pnp_noise_conditioned":
        return pnp_noise_conditioned_checkpoint_for_args(args)
    return pnp_checkpoint_for_args(args)


def pnp_noise_level_for_job(
    args: argparse.Namespace, job: ReconstructionJob
) -> float | None:
    if job.matrix_group == "pnp_noise_conditioned":
        return args.pnp_noise_conditioned_noise_level
    return args.pnp_noise_level


def expected_method_settings(args: argparse.Namespace, job: ReconstructionJob) -> dict:
    if job.method.name == "baseline":
        settings = {"baseline": "fdk"}
        _, settings = apply_reconstruction_args_to_settings(
            reconstruction_args=job_reconstruction_args(args, job),
            method_settings=settings,
        )
        return settings
    if job.method.name == "admm_tv":
        settings = {
            "tv_lambda": float(args.tv_lambda),
            "tv_iterations": int(args.tv_iterations),
            "tv_lipschitz": None,
            "tv_non_negativity": False,
        }
        _, settings = apply_reconstruction_args_to_settings(
            reconstruction_args=job_reconstruction_args(args, job),
            method_settings=settings,
        )
        return settings
    if job.method.name == "pnp_admm":
        pnp_noise_level = pnp_noise_level_for_job(args, job)
        settings = {
            "pnp_checkpoint": str(pnp_checkpoint_for_job(args, job)),
            "pnp_iterations": int(args.pnp_iterations),
            "pnp_eta": float(args.pnp_eta),
            "pnp_cg_iterations": int(args.pnp_cg_iterations),
            "pnp_cg_tolerance": float(args.pnp_cg_tolerance),
            "pnp_noise_level": (
                None if pnp_noise_level is None else float(pnp_noise_level)
            ),
            "pnp_clip": True,
        }
        _, settings = apply_reconstruction_args_to_settings(
            reconstruction_args=job_reconstruction_args(args, job),
            method_settings=settings,
        )
        return settings
    _, settings = apply_reconstruction_args_to_settings(
        reconstruction_args=job_reconstruction_args(args, job),
        method_settings={},
    )
    return settings


def command_for_job(args: argparse.Namespace, job: ReconstructionJob) -> list[str]:
    checkpoint = checkpoint_path(args.training_root, job.model, args.checkpoint_policy)
    output_folder = (
        args.output_root
        / job.method.name
        / job.model.name
        / job.implementation
        / job.geometry
        / job.experiment
    )
    if job.matrix_group != "main":
        output_folder = output_folder / job.matrix_group
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
        "--matrix-group",
        job.matrix_group,
        "--method",
        job.method.name,
        "--split",
        args.split,
        "--algorithm",
        job.method.algorithm,
        "--max-samples",
        str(max_samples_for_job(args, job)),
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
        cmd.extend(["--pnp-checkpoint", str(pnp_checkpoint_for_job(args, job))])
    if job.method.name == "admm_tv":
        cmd.extend(["--tv-lambda", str(args.tv_lambda)])
        cmd.extend(["--tv-iterations", str(args.tv_iterations)])
    if job.method.name == "pnp_admm":
        pnp_noise_level = pnp_noise_level_for_job(args, job)
        cmd.extend(["--pnp-iterations", str(args.pnp_iterations)])
        cmd.extend(["--pnp-eta", str(args.pnp_eta)])
        cmd.extend(["--pnp-cg-iterations", str(args.pnp_cg_iterations)])
        cmd.extend(["--pnp-cg-tolerance", str(args.pnp_cg_tolerance)])
        if pnp_noise_level is not None:
            cmd.extend(["--pnp-noise-level", str(pnp_noise_level)])
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
    for extra_arg in job_reconstruction_args(args, job):
        cmd.append(extra_arg)
    return cmd


def max_samples_for_job(args: argparse.Namespace, job: ReconstructionJob) -> int:
    max_samples = int(args.max_samples)
    expensive_cap = getattr(args, "expensive_job_max_samples", None)
    if expensive_cap is not None and (
        job.method.name in EXPENSIVE_SAMPLE_CAP_METHODS
        or job.experiment in EXPENSIVE_SAMPLE_CAP_EXPERIMENTS
    ):
        return min(max_samples, int(expensive_cap))
    return max_samples


def job_json(args: argparse.Namespace, job: ReconstructionJob) -> dict:
    prior_mode = job_prior_mode(job)
    hparam_selection = hparam_selection_for_job(args, job)
    return {
        "model": job.model.name,
        "checkpoint": (
            ""
            if job.method.name in NO_PADIS_PRIOR_METHODS
            else str(
                checkpoint_path(args.training_root, job.model, args.checkpoint_policy)
            )
        ),
        "method": job.method.name,
        "display_label": job_display_label(job),
        "algorithm": job.method.algorithm,
        "prior_mode": prior_mode,
        "expected_sampler": expected_sampler_settings(args, job),
        "expected_method_settings": expected_method_settings(args, job),
        "hparam_default": (
            None if hparam_selection is None else hparam_selection.to_json()
        ),
        "implementation": job.implementation,
        "geometry": job.geometry,
        "experiment": job.experiment,
        "matrix_group": job.matrix_group,
        "extra_reconstruction_args": list(job.extra_reconstruction_args),
        "sampler_overrides": dict(job.sampler_overrides),
        "command": command_for_job(args, job),
    }


def input_check_failures(
    args: argparse.Namespace, jobs: list[ReconstructionJob]
) -> list[str]:
    failures = []
    seen_model_checkpoints: set[pathlib.Path] = set()
    seen_pnp_checkpoints: set[pathlib.Path] = set()
    for job in jobs:
        checkpoint = checkpoint_path(
            args.training_root, job.model, args.checkpoint_policy
        )
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
            pnp_checkpoint = pnp_checkpoint_for_job(args, job)
            if pnp_checkpoint not in seen_pnp_checkpoints:
                seen_pnp_checkpoints.add(pnp_checkpoint)
                if not pnp_checkpoint.is_file():
                    failures.append(
                        f"Missing PnP denoiser checkpoint: {pnp_checkpoint}"
                    )
    return failures


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root",
        type=pathlib.Path,
        default=None,
        help=(
            "Root containing trained model subfolders. If omitted, choose "
            "--training-root-preset slurm or gcp."
        ),
    )
    parser.add_argument(
        "--training-root-preset",
        choices=TRAINING_ROOT_PRESETS,
        default=None,
        help=(
            "Infer --training-root from the final Slurm or GCP model layout. "
            "Explicit --training-root takes precedence."
        ),
    )
    parser.add_argument(
        "--run-root",
        type=pathlib.Path,
        default=None,
        help=(
            "Base PaDIS experiment root for --training-root-preset. Defaults "
            "to PADIS_RUN_ROOT, LION_EXPERIMENTS_PATH/PaDIS, "
            "LION_DATA_PATH/experiments/PaDIS, or ../Data/experiments/PaDIS."
        ),
    )
    parser.add_argument(
        "--run-stamp",
        default=os.environ.get("PADIS_RUN_STAMP", ""),
        help=(
            "Slurm run stamp used by --training-root-preset slurm. The resolved "
            "folder is final_real_runs/a100_training_<stamp>. If omitted, the "
            "latest existing a100_training_* folder is used."
        ),
    )
    parser.add_argument(
        "--gcp-run-name",
        default=os.environ.get("PADIS_GCP_RUN_NAME", "PaDIS-Reproduction-GCP"),
        help=(
            "GCP run folder used by --training-root-preset gcp. The resolved "
            "folder is final_real_runs/<name>."
        ),
    )
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument(
        "--checkpoint-policy",
        choices=CHECKPOINT_POLICIES,
        default="model_default",
        help=(
            "Which diffusion checkpoint family to use for matrix jobs. "
            "model_default preserves the historical per-model names; "
            "min_intense_val selects the validation-intensive checkpoint family."
        ),
    )
    parser.add_argument(
        "--hparam-defaults",
        choices=HPARAM_DEFAULT_MODES,
        default="none",
        help=(
            "none uses the reconstruction implementation presets. json loads "
            "repo-local tuned defaults from --hparam-defaults-json. auto loads "
            "completed fixed-validation tuning runs directly and appends the "
            "best candidate args per method/implementation/experiment."
        ),
    )
    parser.add_argument(
        "--hparam-defaults-json",
        type=pathlib.Path,
        default=None,
        help=(
            "JSON file containing selected tuned reconstruction defaults. "
            "Defaults to scripts/paper_scripts/PaDIS/config/"
            "reconstruction_hparam_defaults.json when --hparam-defaults json "
            "is used."
        ),
    )
    parser.add_argument(
        "--hparam-run-root",
        type=pathlib.Path,
        default=None,
        help=(
            "Folder containing tuning run subfolders with runs.jsonl. Defaults "
            "to PADIS_HPARAM_RUN_ROOT, PADIS_HPARAM_TUNING_RUN_ROOT, or "
            "<run-root>/hparam_tuning/runs when --hparam-defaults auto is used."
        ),
    )
    parser.add_argument(
        "--hparam-run-glob",
        default="fixedval_*",
        help=(
            "Comma-separated glob(s) of tuning run folder names to use for "
            "automatic defaults. Default fixedval_* uses corrected fixed-validation runs."
        ),
    )
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
        "--ablations",
        default="none",
        help=(
            "none, all, or a comma-separated list of trained ablation groups to "
            "append when --models method_default is used. Valid values: "
            + ", ".join(ABLATIONS)
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
    parser.add_argument(
        "--expensive-job-max-samples",
        type=int,
        default=DEFAULT_EXPENSIVE_JOB_MAX_SAMPLES,
        help=(
            "Maximum samples for expensive tail jobs: patch_average, "
            "patch_stitch, and ct_512_60. Use -1 to disable this cap."
        ),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--pnp-root",
        type=pathlib.Path,
        default=None,
        help=(
            "Folder containing pnp_lidc_drunet_min_val.pt. Defaults to "
            "<training-root>/pnp_lidc_drunet."
        ),
    )
    parser.add_argument("--pnp-checkpoint", type=pathlib.Path, default=None)
    parser.add_argument(
        "--pnp-noise-conditioned-root",
        type=pathlib.Path,
        default=None,
        help=(
            "Folder containing pnp_lidc_drunet_noise_cond_min_val.pt. Defaults "
            "to <training-root>/pnp_lidc_drunet_noise_cond."
        ),
    )
    parser.add_argument(
        "--pnp-noise-conditioned-checkpoint",
        type=pathlib.Path,
        default=None,
    )
    parser.add_argument("--pnp-iterations", type=int, default=60)
    parser.add_argument("--pnp-eta", type=float, default=3e-5)
    parser.add_argument("--pnp-cg-iterations", type=int, default=50)
    parser.add_argument("--pnp-cg-tolerance", type=float, default=1e-7)
    parser.add_argument("--pnp-noise-level", type=float, default=None)
    parser.add_argument("--pnp-noise-conditioned-noise-level", type=float, default=0.03)
    parser.add_argument("--tv-lambda", type=float, default=0.001)
    parser.add_argument("--tv-iterations", type=int, default=1000)
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
    parser.add_argument(
        "--job-order",
        choices=JOB_ORDERS,
        default="default",
        help=(
            "default preserves matrix construction order. gcp_spot runs regular "
            "jobs first, then late-added VE-DDNM extra-CT jobs, then "
            "noise-conditioned PnP jobs, then patch_average/patch_stitch jobs, "
            "then ct_512_60 jobs."
        ),
    )
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
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if args.expensive_job_max_samples == -1:
        args.expensive_job_max_samples = None
    elif args.expensive_job_max_samples <= 0:
        raise ValueError("--expensive-job-max-samples must be positive or -1.")
    args.training_root = resolve_training_root(args)
    args.output_root = args.output_root.expanduser().resolve()
    if args.pnp_root is None:
        args.pnp_root = args.training_root / "pnp_lidc_drunet"
    else:
        args.pnp_root = args.pnp_root.expanduser().resolve()
    if args.pnp_checkpoint is not None:
        args.pnp_checkpoint = args.pnp_checkpoint.expanduser().resolve()
    if args.pnp_noise_conditioned_root is None:
        args.pnp_noise_conditioned_root = (
            args.training_root / "pnp_lidc_drunet_noise_cond"
        )
    else:
        args.pnp_noise_conditioned_root = (
            args.pnp_noise_conditioned_root.expanduser().resolve()
        )
    if args.pnp_noise_conditioned_checkpoint is not None:
        args.pnp_noise_conditioned_checkpoint = (
            args.pnp_noise_conditioned_checkpoint.expanduser().resolve()
        )
    jobs = ordered_jobs(args, build_jobs(args))
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
