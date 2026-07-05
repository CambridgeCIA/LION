"""Tune PaDIS reconstruction hyperparameters on the LIDC validation split.

This script wraps PaDIS_run_reconstruction_matrix.py so tuning runs use the
same job definitions as the final inference array. It stages the external model
checkpoints into the training-root layout expected by the matrix launcher,
runs candidate reconstruction settings on the validation split, and aggregates
the resulting metrics.json files.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
import pathlib
import shlex
import subprocess
import sys
import time
from typing import Iterable

import PaDIS_run_reconstruction_matrix as matrix


DEFAULT_EXTERNAL_MODEL_ROOT = pathlib.Path(
    "/home/thomas/DiS/Project/Data/experiments/PaDIS/external_models"
)
DEFAULT_TUNING_ROOT = pathlib.Path(
    "/home/thomas/DiS/Project/Data/experiments/PaDIS/hparam_tuning"
)
DEFAULT_STAGED_TRAINING_ROOT = DEFAULT_TUNING_ROOT / "external_training_root"
DEFAULT_OUTPUT_ROOT = DEFAULT_TUNING_ROOT / "runs"


EXTERNAL_MODEL_LINKS = {
    "patch_lidc_default/padis_lidc_256.pt": "padis_lidc_default.pt",
    "patch_lidc_512/padis_lidc_512.pt": "padis_lidc_512.pt",
    "whole_lidc_default/whole_image_lidc_256_min_val.pt": "whole_lidc_default.pt",
    "pnp_lidc_drunet/pnp_lidc_drunet.pt": "pnp_lidc_drunet_min_val.pt",
}


@dataclass(frozen=True)
class Candidate:
    name: str
    method: str
    implementation: str
    prior: str | None
    args: tuple[str, ...]
    notes: str = ""

    @property
    def group(self) -> str:
        prior = self.prior if self.prior is not None else "any"
        return f"{self.method}__{self.implementation}__{prior}"


@dataclass(frozen=True)
class RunRecord:
    candidate: Candidate
    job: matrix.ReconstructionJob
    command: tuple[str, ...]
    metrics_path: pathlib.Path
    log_path: pathlib.Path
    status: str
    elapsed_seconds: float | None
    summary: dict
    error: str | None = None


def safe_name(value: str) -> str:
    return (
        value.replace("-", "_")
        .replace(".", "p")
        .replace("/", "_")
        .replace("=", "_")
        .replace("+", "plus")
    )


def flag_value_args(flag: str, values: Iterable[object]) -> tuple[tuple[str, ...], ...]:
    return tuple((flag, str(value)) for value in values)


def zeta_candidates(
    *,
    method: str,
    implementation: str,
    prior: str | None,
    values: Iterable[float],
    prefix: str = "zeta",
) -> list[Candidate]:
    return [
        Candidate(
            name=f"{prefix}_{safe_name(f'{value:g}')}",
            method=method,
            implementation=implementation,
            prior=prior,
            args=("--zeta", f"{value:g}"),
        )
        for value in values
    ]


def current_default_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    for method, implementations in matrix.CORE_IMPLEMENTATIONS_BY_METHOD.items():
        for implementation in implementations:
            priors: tuple[str | None, ...] = (None,)
            if method in {"langevin", "predictor_corrector", "ve_ddnm"}:
                priors = ("patch", "whole_image")
            for prior in priors:
                candidates.append(
                    Candidate(
                        name="current_defaults",
                        method=method,
                        implementation=implementation,
                        prior=prior,
                        args=(),
                        notes="Matrix/reconstruction defaults with no tuner override.",
                    )
                )
    return candidates


def pilot_candidates() -> list[Candidate]:
    candidates = current_default_candidates()

    candidates.extend(
        Candidate(
            name=f"hann_{safe_name(f'{value:g}')}",
            method="baseline",
            implementation="lion_physics",
            prior=None,
            args=("--initial-fdk-frequency-scaling", f"{value:g}"),
            notes="Tune the LION FDK/Hann baseline frequency scaling.",
        )
        for value in (0.2, 0.3, 0.5, 0.7, 0.9, 1.0)
    )

    candidates.extend(
        Candidate(
            name=f"tv_lam_{safe_name(f'{lam:g}')}",
            method="admm_tv",
            implementation="lion_physics",
            prior=None,
            args=("--tv-lambda", f"{lam:g}", "--tv-iterations", "500"),
        )
        for lam in (3e-4, 1e-3, 3e-3, 1e-2)
    )

    candidates.extend(
        Candidate(
            name=(f"eta_{safe_name(f'{eta:g}')}__iters_{iterations}"),
            method="pnp_admm",
            implementation="lion_physics",
            prior=None,
            args=(
                "--pnp-eta",
                f"{eta:g}",
                "--pnp-iterations",
                str(iterations),
            ),
        )
        for eta in (3e-6, 1e-5, 3e-5)
        for iterations in (10, 20, 40)
    )

    candidates.extend(
        zeta_candidates(
            method="padis_dps",
            implementation="lion_physics",
            prior=None,
            values=(2.0, 3.0, 4.0, 5.0, 6.0),
        )
    )
    candidates.extend(
        Candidate(
            name=f"zeta_{safe_name(f'{zeta:g}')}__eps_{safe_name(f'{eps:g}')}",
            method="padis_dps",
            implementation="lion_physics",
            prior=None,
            args=("--zeta", f"{zeta:g}", "--dps-epsilon", f"{eps:g}"),
        )
        for zeta in (3.0, 4.0, 5.0)
        for eps in (0.5, 0.75, 1.0)
    )
    candidates.extend(
        zeta_candidates(
            method="padis_dps",
            implementation="public_repo",
            prior=None,
            values=(0.1, 0.2, 0.3, 0.4),
        )
    )
    candidates.extend(
        zeta_candidates(
            method="padis_dps",
            implementation="paper",
            prior=None,
            values=(0.005, 0.01, 0.03, 0.1, 0.3),
        )
    )

    for prior in ("patch", "whole_image"):
        candidates.extend(
            Candidate(
                name=f"zeta_{safe_name(f'{zeta:g}')}__eps_{safe_name(f'{eps:g}')}",
                method="langevin",
                implementation="lion_physics",
                prior=prior,
                args=(
                    "--zeta",
                    f"{zeta:g}",
                    "--sampling-epsilon",
                    f"{eps:g}",
                ),
            )
            for zeta in (3.0, 4.0, 5.0)
            for eps in (0.25, 0.5, 0.75)
        )
        candidates.extend(
            Candidate(
                name=f"zeta_{safe_name(f'{zeta:g}')}__snr_{safe_name(f'{snr:g}')}",
                method="predictor_corrector",
                implementation="lion_physics",
                prior=prior,
                args=("--zeta", f"{zeta:g}", "--pc-snr", f"{snr:g}"),
            )
            for zeta in (3.5, 4.25, 5.0)
            for snr in (0.04, 0.08, 0.12)
        )
        candidates.extend(
            Candidate(
                name=f"eps_{safe_name(f'{eps:g}')}",
                method="ve_ddnm",
                implementation="lion_physics",
                prior=prior,
                args=("--sampling-epsilon", f"{eps:g}"),
            )
            for eps in (0.05, 0.1, 0.2)
        )

    for method in ("langevin", "predictor_corrector", "ve_ddnm"):
        candidates.extend(
            zeta_candidates(
                method=method,
                implementation="public_repo",
                prior="patch",
                values=(0.1, 0.2, 0.3, 0.5),
            )
        )
        candidates.extend(
            zeta_candidates(
                method=method,
                implementation="paper",
                prior="patch",
                values=(0.005, 0.01, 0.03, 0.1, 0.3),
            )
        )

    for method in ("patch_average", "patch_stitch"):
        candidates.extend(
            Candidate(
                name=f"zeta_{safe_name(f'{zeta:g}')}__eps_{safe_name(f'{eps:g}')}",
                method=method,
                implementation="lion_physics",
                prior=None,
                args=("--zeta", f"{zeta:g}", "--dps-epsilon", f"{eps:g}"),
            )
            for zeta in (3.0, 4.0, 5.0)
            for eps in (0.3, 0.5, 0.75)
        )
        candidates.extend(
            zeta_candidates(
                method=method,
                implementation="public_repo",
                prior=None,
                values=(0.1, 0.2, 0.3, 0.5),
            )
        )

    candidates.extend(
        Candidate(
            name=f"zeta_{safe_name(f'{zeta:g}')}__eps_{safe_name(f'{eps:g}')}",
            method="whole_image_diffusion",
            implementation="lion_physics",
            prior=None,
            args=("--zeta", f"{zeta:g}", "--dps-epsilon", f"{eps:g}"),
        )
        for zeta in (3.0, 4.0, 5.0)
        for eps in (0.5, 0.75, 1.0)
    )
    candidates.extend(
        zeta_candidates(
            method="whole_image_diffusion",
            implementation="paper",
            prior=None,
            values=(0.005, 0.01, 0.03, 0.1, 0.3),
        )
    )

    return unique_candidates(candidates)


def broad_candidates() -> list[Candidate]:
    candidates = pilot_candidates()
    candidates.extend(
        zeta_candidates(
            method="padis_dps",
            implementation="lion_physics",
            prior=None,
            values=(1.0, 1.5, 2.5, 3.5, 4.5, 5.5, 7.0),
        )
    )
    candidates.extend(
        Candidate(
            name=(
                f"zeta_{safe_name(f'{zeta:g}')}__eps_{safe_name(f'{eps:g}')}"
                f"__dc_{schedule}"
            ),
            method="padis_dps",
            implementation="lion_physics",
            prior=None,
            args=(
                "--zeta",
                f"{zeta:g}",
                "--dps-epsilon",
                f"{eps:g}",
                "--data-consistency-scale-schedule",
                schedule,
                "--data-consistency-scale-floor",
                "0.05",
            ),
            notes="Check whether sigma-dependent LION data weighting improves cross-experiment robustness.",
        )
        for zeta in (4.0, 5.0)
        for eps in (0.5, 1.0)
        for schedule in ("edm", "inverse_sigma")
    )
    candidates.extend(
        Candidate(
            name=f"tv_lam_{safe_name(f'{lam:g}')}__iters_{iterations}",
            method="admm_tv",
            implementation="lion_physics",
            prior=None,
            args=(
                "--tv-lambda",
                f"{lam:g}",
                "--tv-iterations",
                str(iterations),
            ),
        )
        for lam in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
        for iterations in (250, 500, 1000)
    )
    return unique_candidates(candidates)


def focused_candidates() -> list[Candidate]:
    candidates = current_default_candidates()
    candidates.extend(
        Candidate(
            name=f"hann_{safe_name(f'{value:g}')}",
            method="baseline",
            implementation="lion_physics",
            prior=None,
            args=("--initial-fdk-frequency-scaling", f"{value:g}"),
        )
        for value in (0.15, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.5)
    )
    candidates.extend(
        Candidate(
            name=f"tv_lam_{safe_name(f'{lam:g}')}",
            method="admm_tv",
            implementation="lion_physics",
            prior=None,
            args=("--tv-lambda", f"{lam:g}", "--tv-iterations", "500"),
        )
        for lam in (0.002, 0.003, 0.005, 0.0075, 0.01, 0.015)
    )
    candidates.extend(
        Candidate(
            name=f"zeta_{safe_name(f'{zeta:g}')}__eps_{safe_name(f'{eps:g}')}",
            method="padis_dps",
            implementation="lion_physics",
            prior=None,
            args=("--zeta", f"{zeta:g}", "--dps-epsilon", f"{eps:g}"),
        )
        for zeta in (3.5, 4.0, 4.5, 4.55, 4.6, 4.7, 4.8, 4.9, 5.0)
        for eps in (0.5, 0.75, 1.0)
    )
    candidates.extend(
        zeta_candidates(
            method="padis_dps",
            implementation="public_repo",
            prior=None,
            values=(0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3),
        )
    )
    candidates.extend(
        zeta_candidates(
            method="padis_dps",
            implementation="paper",
            prior=None,
            values=(0.005, 0.0075, 0.01, 0.015, 0.02),
        )
    )
    candidates.extend(
        Candidate(
            name=f"zeta_{safe_name(f'{zeta:g}')}__eps_{safe_name(f'{eps:g}')}",
            method="whole_image_diffusion",
            implementation="lion_physics",
            prior=None,
            args=("--zeta", f"{zeta:g}", "--dps-epsilon", f"{eps:g}"),
        )
        for zeta in (3.5, 4.0, 4.5)
        for eps in (0.5, 0.75, 1.0)
    )
    candidates.extend(
        Candidate(
            name=f"eta_{safe_name(f'{eta:g}')}__iters_{iterations}",
            method="pnp_admm",
            implementation="lion_physics",
            prior=None,
            args=(
                "--pnp-eta",
                f"{eta:g}",
                "--pnp-iterations",
                str(iterations),
            ),
        )
        for eta in (5e-6, 1e-5, 2e-5)
        for iterations in (20, 40, 60)
    )
    return unique_candidates(candidates)


def unique_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    seen: set[tuple] = set()
    unique: list[Candidate] = []
    for candidate in candidates:
        key = (
            candidate.method,
            candidate.implementation,
            candidate.prior,
            candidate.name,
            candidate.args,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def candidate_set(name: str) -> list[Candidate]:
    if name == "smoke":
        return current_default_candidates()
    if name == "pilot":
        return pilot_candidates()
    if name == "broad":
        return broad_candidates()
    if name == "focused":
        return focused_candidates()
    raise ValueError(f"Unknown candidate set {name!r}.")


def ensure_staged_training_root(
    *,
    external_model_root: pathlib.Path,
    training_root: pathlib.Path,
) -> None:
    external_model_root = external_model_root.expanduser().resolve()
    training_root = training_root.expanduser().resolve()
    for relative_link, source_name in EXTERNAL_MODEL_LINKS.items():
        source = external_model_root / source_name
        if not source.is_file():
            raise FileNotFoundError(f"Missing external model checkpoint: {source}")
        link = training_root / relative_link
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            existing = link.resolve()
            if existing != source.resolve():
                raise FileExistsError(
                    f"{link} already exists and points to {existing}, expected {source}"
                )
            continue
        link.symlink_to(source)


def parse_csv_selection(value: str) -> set[str] | None:
    value = value.strip()
    if value in ("", "all"):
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def job_matches_candidate(job: matrix.ReconstructionJob, candidate: Candidate) -> bool:
    if job.method.name != candidate.method:
        return False
    if job.implementation != candidate.implementation:
        return False
    if candidate.prior is not None and matrix.job_prior_mode(job) != candidate.prior:
        return False
    return True


def candidate_matches_filters(candidate: Candidate, args: argparse.Namespace) -> bool:
    methods = parse_csv_selection(args.only_methods)
    implementations = parse_csv_selection(args.only_implementations)
    groups = parse_csv_selection(args.only_groups)
    names = parse_csv_selection(args.only_candidates)
    if methods is not None and candidate.method not in methods:
        return False
    if implementations is not None and candidate.implementation not in implementations:
        return False
    if groups is not None and candidate.group not in groups:
        return False
    if names is not None and candidate.name not in names:
        return False
    return True


def build_matrix_args(args: argparse.Namespace) -> argparse.Namespace:
    parser = matrix.build_arg_parser()
    raw = [
        "--training-root",
        str(args.training_root),
        "--output-root",
        str(args.output_root),
        "--models",
        args.models,
        "--methods",
        args.methods,
        "--experiments",
        args.experiments,
        "--ablations",
        args.ablations,
        "--implementations",
        args.implementations,
        "--geometries",
        "lion",
        "--split",
        "validation",
        "--max-samples",
        str(args.max_samples),
        "--start-index",
        str(args.start_index),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--pnp-root",
        str(args.training_root / "pnp_lidc_drunet"),
        "--tv-lambda",
        str(args.tv_lambda),
        "--tv-iterations",
        str(args.tv_iterations),
        "--pnp-iterations",
        str(args.pnp_iterations),
        "--pnp-eta",
        str(args.pnp_eta),
        "--pnp-cg-iterations",
        str(args.pnp_cg_iterations),
        "--pnp-cg-tolerance",
        str(args.pnp_cg_tolerance),
    ]
    if args.allow_off_paper_experiments:
        raw.append("--allow-off-paper-experiments")
    if args.data_folder is not None:
        raw.extend(["--data-folder", str(args.data_folder)])
    if args.save_previews:
        raw.append("--save-previews")
    if args.prog_bar:
        raw.append("--prog-bar")
    if args.trace_interval is not None:
        raw.extend(["--trace-interval", str(args.trace_interval)])
    if args.trace_images:
        raw.append("--trace-images")
    parsed = parser.parse_args(raw)
    parsed.training_root = matrix.resolve_training_root(parsed)
    parsed.output_root = parsed.output_root.expanduser().resolve()
    parsed.pnp_root = parsed.pnp_root.expanduser().resolve()
    return parsed


def candidate_output_root(
    args: argparse.Namespace, candidate: Candidate
) -> pathlib.Path:
    return (
        args.output_root
        / args.run_name
        / args.candidate_set
        / candidate.group
        / candidate.name
    )


def command_metrics_path(command: list[str]) -> pathlib.Path:
    def value_after(flag: str) -> str:
        try:
            return command[command.index(flag) + 1]
        except ValueError as exc:
            raise ValueError(f"Command is missing {flag}") from exc

    output_folder = pathlib.Path(value_after("--output-folder"))
    experiment = value_after("--experiment")
    split = value_after("--split")
    method = value_after("--method")
    algorithm = value_after("--algorithm")
    return output_folder / experiment / split / method / algorithm / "metrics.json"


def command_for_candidate(
    matrix_args: argparse.Namespace,
    job: matrix.ReconstructionJob,
    candidate: Candidate,
    args: argparse.Namespace,
) -> list[str]:
    run_matrix_args = argparse.Namespace(**vars(matrix_args))
    run_matrix_args.output_root = candidate_output_root(args, candidate)
    command = matrix.command_for_job(run_matrix_args, job)
    command.extend(candidate.args)
    if args.stop_after_outer_steps is not None:
        command.extend(["--stop-after-outer-steps", str(args.stop_after_outer_steps)])
    for extra_arg in args.reconstruction_arg:
        command.append(extra_arg)
    return command


def finite_values(values: Iterable[float | None]) -> list[float]:
    return [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]


def mean_or_none(values: Iterable[float | None]) -> float | None:
    finite = finite_values(values)
    if not finite:
        return None
    return sum(finite) / len(finite)


def min_or_none(values: Iterable[float | None]) -> float | None:
    finite = finite_values(values)
    if not finite:
        return None
    return min(finite)


def max_or_none(values: Iterable[float | None]) -> float | None:
    finite = finite_values(values)
    if not finite:
        return None
    return max(finite)


def summarize_metrics(metrics_path: pathlib.Path) -> dict:
    with open(metrics_path) as file:
        payload = json.load(file)
    metrics = payload.get("metrics", [])
    sampler = payload.get("sampler", {})
    method_settings = payload.get("method_settings", {})
    psnr_values = [item.get("psnr") for item in metrics]
    ssim_values = [item.get("ssim") for item in metrics]
    mae_values = [item.get("mae") for item in metrics]
    fdk_psnr_values = [item.get("fdk_psnr") for item in metrics]
    fdk_ssim_values = [item.get("fdk_ssim") for item in metrics]
    body_psnr_values = [item.get("body_psnr") for item in metrics]
    margins = [
        item.get("psnr") - item.get("fdk_psnr")
        for item in metrics
        if item.get("psnr") is not None and item.get("fdk_psnr") is not None
    ]
    primary_values = [
        value
        for item in metrics
        for value in (item.get("psnr"), item.get("ssim"), item.get("mae"))
    ]
    nonfinite_primary_metric_count = sum(
        1
        for value in primary_values
        if value is None or not math.isfinite(float(value))
    )
    return {
        "sample_count": len(metrics),
        "nonfinite_primary_metric_count": nonfinite_primary_metric_count,
        "all_finite_primary_metrics": nonfinite_primary_metric_count == 0,
        "mean_psnr": mean_or_none(psnr_values),
        "min_psnr": min_or_none(psnr_values),
        "mean_ssim": mean_or_none(ssim_values),
        "min_ssim": min_or_none(ssim_values),
        "mean_mae": mean_or_none(mae_values),
        "max_mae": max_or_none(mae_values),
        "mean_fdk_psnr": mean_or_none(fdk_psnr_values),
        "mean_fdk_ssim": mean_or_none(fdk_ssim_values),
        "mean_body_psnr": mean_or_none(body_psnr_values),
        "mean_psnr_margin_vs_fdk": mean_or_none(margins),
        "min_psnr_margin_vs_fdk": min_or_none(margins),
        "sampler": sampler,
        "method_settings": method_settings,
        "paper_experiment": payload.get("paper_experiment", {}),
    }


def run_command(
    command: list[str], log_path: pathlib.Path
) -> tuple[str, float, str | None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with open(log_path, "w", buffering=1) as log_file:
        log_file.write(" ".join(shlex.quote(part) for part in command) + "\n\n")
        completed = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.monotonic() - started
    if completed.returncode == 0:
        return "completed", elapsed, None
    return "failed", elapsed, f"Command exited with code {completed.returncode}"


def record_for_run(
    *,
    candidate: Candidate,
    job: matrix.ReconstructionJob,
    command: list[str],
    metrics_path: pathlib.Path,
    log_path: pathlib.Path,
    status: str,
    elapsed_seconds: float | None,
    summary: dict,
    error: str | None = None,
) -> dict:
    return {
        "candidate_group": candidate.group,
        "candidate": candidate.name,
        "candidate_args": list(candidate.args),
        "candidate_notes": candidate.notes,
        "method": job.method.name,
        "implementation": job.implementation,
        "prior": matrix.job_prior_mode(job),
        "model": job.model.name,
        "experiment": job.experiment,
        "matrix_group": job.matrix_group,
        "status": status,
        "elapsed_seconds": elapsed_seconds,
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
        "command": command,
        "summary": summary,
        "error": error,
    }


def write_jsonl(path: pathlib.Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")


def aggregate_records(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        if record["status"] not in ("completed", "nonfinite"):
            continue
        key = (record["candidate_group"], record["candidate"])
        groups.setdefault(key, []).append(record)

    rows = []
    for (candidate_group, candidate), group_records in sorted(groups.items()):
        summaries = [record["summary"] for record in group_records]
        psnr_values = [summary.get("mean_psnr") for summary in summaries]
        ssim_values = [summary.get("mean_ssim") for summary in summaries]
        mae_values = [summary.get("mean_mae") for summary in summaries]
        margin_values = [
            summary.get("mean_psnr_margin_vs_fdk") for summary in summaries
        ]
        elapsed_values = [record.get("elapsed_seconds") for record in group_records]
        rows.append(
            {
                "candidate_group": candidate_group,
                "candidate": candidate,
                "completed_jobs": len(group_records),
                "nonfinite_jobs": sum(
                    1
                    for summary in summaries
                    if not summary.get("all_finite_primary_metrics", False)
                ),
                "mean_psnr": mean_or_none(psnr_values),
                "min_job_psnr": min_or_none(psnr_values),
                "mean_ssim": mean_or_none(ssim_values),
                "min_job_ssim": min_or_none(ssim_values),
                "mean_mae": mean_or_none(mae_values),
                "mean_psnr_margin_vs_fdk": mean_or_none(margin_values),
                "min_job_psnr_margin_vs_fdk": min_or_none(margin_values),
                "total_elapsed_seconds": sum(finite_values(elapsed_values)),
                "experiments": ",".join(
                    sorted({record["experiment"] for record in group_records})
                ),
                "models": ",".join(
                    sorted({record["model"] for record in group_records})
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            row["candidate_group"],
            -float(row["mean_psnr"] if row["mean_psnr"] is not None else -1e9),
            -float(row["mean_ssim"] if row["mean_ssim"] is not None else -1e9),
        )
    )
    return rows


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_group",
        "candidate",
        "completed_jobs",
        "nonfinite_jobs",
        "mean_psnr",
        "min_job_psnr",
        "mean_ssim",
        "min_job_ssim",
        "mean_mae",
        "mean_psnr_margin_vs_fdk",
        "min_job_psnr_margin_vs_fdk",
        "total_elapsed_seconds",
        "experiments",
        "models",
    ]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_runs(
    args: argparse.Namespace,
) -> tuple[
    argparse.Namespace, list[tuple[Candidate, matrix.ReconstructionJob, list[str]]]
]:
    matrix_args = build_matrix_args(args)
    jobs = matrix.build_jobs(matrix_args)
    failures = matrix.input_check_failures(matrix_args, jobs)
    if failures:
        raise FileNotFoundError(
            "Reconstruction input check failed:\n  " + "\n  ".join(failures)
        )

    selected_candidates = [
        candidate
        for candidate in candidate_set(args.candidate_set)
        if candidate_matches_filters(candidate, args)
    ]
    runs: list[tuple[Candidate, matrix.ReconstructionJob, list[str]]] = []
    for candidate in selected_candidates:
        for job in jobs:
            if not job_matches_candidate(job, candidate):
                continue
            command = command_for_candidate(matrix_args, job, candidate, args)
            runs.append((candidate, job, command))
    if args.limit is not None:
        runs = runs[: args.limit]
    return matrix_args, runs


def command_line(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_tuning(args: argparse.Namespace) -> None:
    ensure_staged_training_root(
        external_model_root=args.external_model_root,
        training_root=args.training_root,
    )
    _, runs = build_runs(args)
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_root / args.run_name / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as file:
        json.dump(
            {
                "candidate_set": args.candidate_set,
                "run_name": args.run_name,
                "training_root": str(args.training_root),
                "external_model_root": str(args.external_model_root),
                "split": "validation",
                "max_samples": args.max_samples,
                "start_index": args.start_index,
                "seed": args.seed,
                "methods": args.methods,
                "implementations": args.implementations,
                "experiments": args.experiments,
                "ablations": args.ablations,
                "run_count": len(runs),
            },
            file,
            indent=2,
        )

    records: list[dict] = []
    for run_index, (candidate, job, command) in enumerate(runs):
        metrics_path = command_metrics_path(command)
        log_path = metrics_path.parent / "tuning_command.log"
        print(
            f"[{run_index + 1}/{len(runs)}] "
            f"{candidate.group}/{candidate.name} "
            f"{job.experiment} {job.model.name}"
        )
        print(command_line(command))
        if args.dry_run:
            records.append(
                record_for_run(
                    candidate=candidate,
                    job=job,
                    command=command,
                    metrics_path=metrics_path,
                    log_path=log_path,
                    status="dry_run",
                    elapsed_seconds=None,
                    summary={},
                )
            )
            continue
        if metrics_path.is_file() and not args.rerun_existing:
            summary = summarize_metrics(metrics_path)
            records.append(
                record_for_run(
                    candidate=candidate,
                    job=job,
                    command=command,
                    metrics_path=metrics_path,
                    log_path=log_path,
                    status="completed",
                    elapsed_seconds=None,
                    summary=summary,
                )
            )
            print(f"  using existing metrics: {metrics_path}")
            continue

        status, elapsed, error = run_command(command, log_path)
        summary = {}
        if status == "completed" and metrics_path.is_file():
            summary = summarize_metrics(metrics_path)
            print(
                "  completed "
                f"PSNR={summary.get('mean_psnr')} "
                f"SSIM={summary.get('mean_ssim')} "
                f"MAE={summary.get('mean_mae')} "
                f"elapsed={elapsed:.1f}s"
            )
            if not summary.get("all_finite_primary_metrics", False):
                status = "nonfinite"
                error = (
                    "Non-finite primary metrics for "
                    f"{candidate.group}/{candidate.name} {job.experiment}."
                )
        elif status == "completed":
            status = "failed"
            error = (
                f"Command completed but metrics file was not written: {metrics_path}"
            )
            print(f"  {error}")
        else:
            print(f"  failed: {error}; see {log_path}")
        records.append(
            record_for_run(
                candidate=candidate,
                job=job,
                command=command,
                metrics_path=metrics_path,
                log_path=log_path,
                status=status,
                elapsed_seconds=elapsed,
                summary=summary,
                error=error,
            )
        )
        write_outputs(args, records)
        if status == "failed" and args.stop_on_failure:
            raise RuntimeError(error)
        if status == "nonfinite" and args.stop_on_nonfinite:
            raise RuntimeError(error)

    write_outputs(args, records)
    print(f"Wrote tuning records under {args.output_root / args.run_name}")


def write_outputs(args: argparse.Namespace, records: list[dict]) -> None:
    run_root = args.output_root / args.run_name
    write_jsonl(run_root / "runs.jsonl", records)
    summary_rows = aggregate_records(records)
    with open(run_root / "summary.json", "w") as file:
        json.dump(summary_rows, file, indent=2)
    write_csv(run_root / "summary.csv", summary_rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-model-root",
        type=pathlib.Path,
        default=DEFAULT_EXTERNAL_MODEL_ROOT,
        help="Folder containing padis_lidc_default.pt and related external checkpoints.",
    )
    parser.add_argument(
        "--training-root",
        type=pathlib.Path,
        default=DEFAULT_STAGED_TRAINING_ROOT,
        help="Staged matrix training-root layout. External checkpoints are symlinked here.",
    )
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root for tuning runs and summaries.",
    )
    parser.add_argument(
        "--run-name",
        default=time.strftime("validation_tuning_%Y%m%d_%H%M%S"),
    )
    parser.add_argument(
        "--candidate-set",
        choices=("smoke", "pilot", "broad", "focused"),
        default="pilot",
    )
    parser.add_argument("--models", default="method_default")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--experiments", default="paper_matrix")
    parser.add_argument("--ablations", default="none")
    parser.add_argument("--implementations", default="method_default")
    parser.add_argument("--allow-off-paper-experiments", action="store_true")
    parser.add_argument("--only-methods", default="all")
    parser.add_argument("--only-implementations", default="all")
    parser.add_argument(
        "--only-groups",
        default="all",
        help=(
            "Comma-separated candidate groups such as "
            "padis_dps__lion_physics__any or langevin__lion_physics__patch."
        ),
    )
    parser.add_argument("--only-candidates", default="all")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-folder", type=pathlib.Path, default=None)
    parser.add_argument("--tv-lambda", type=float, default=0.005)
    parser.add_argument("--tv-iterations", type=int, default=500)
    parser.add_argument("--pnp-iterations", type=int, default=60)
    parser.add_argument("--pnp-eta", type=float, default=2e-5)
    parser.add_argument("--pnp-cg-iterations", type=int, default=100)
    parser.add_argument("--pnp-cg-tolerance", type=float, default=1e-7)
    parser.add_argument("--save-previews", action="store_true")
    parser.add_argument("--prog-bar", action="store_true")
    parser.add_argument("--trace-interval", type=int, default=None)
    parser.add_argument("--trace-images", action="store_true")
    parser.add_argument("--stop-after-outer-steps", type=int, default=None)
    parser.add_argument(
        "--reconstruction-arg",
        action="append",
        default=[],
        help="Append one raw argument to each reconstruction command.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--stop-on-nonfinite", action="store_true")
    parser.add_argument(
        "--list-candidates",
        action="store_true",
        help="Print candidate definitions after filtering, then exit.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="Print concrete candidate/job commands after filtering, then exit.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.external_model_root = args.external_model_root.expanduser().resolve()
    args.training_root = args.training_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when set.")
    if args.stop_after_outer_steps is not None and args.stop_after_outer_steps <= 0:
        raise ValueError("--stop-after-outer-steps must be positive when set.")

    ensure_staged_training_root(
        external_model_root=args.external_model_root,
        training_root=args.training_root,
    )

    candidates = [
        candidate
        for candidate in candidate_set(args.candidate_set)
        if candidate_matches_filters(candidate, args)
    ]
    if args.list_candidates:
        print(
            json.dumps(
                [
                    {
                        "group": candidate.group,
                        "name": candidate.name,
                        "method": candidate.method,
                        "implementation": candidate.implementation,
                        "prior": candidate.prior,
                        "args": list(candidate.args),
                        "notes": candidate.notes,
                    }
                    for candidate in candidates
                ],
                indent=2,
            )
        )
        return

    _, runs = build_runs(args)
    if args.list_runs:
        print(
            json.dumps(
                [
                    {
                        "candidate_group": candidate.group,
                        "candidate": candidate.name,
                        "method": job.method.name,
                        "implementation": job.implementation,
                        "prior": matrix.job_prior_mode(job),
                        "model": job.model.name,
                        "experiment": job.experiment,
                        "matrix_group": job.matrix_group,
                        "command": command,
                    }
                    for candidate, job, command in runs
                ],
                indent=2,
            )
        )
        return

    run_tuning(args)


if __name__ == "__main__":
    main()
