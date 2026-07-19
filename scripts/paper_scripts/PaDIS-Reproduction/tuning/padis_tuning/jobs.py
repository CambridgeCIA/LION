"""Candidate filtering and matrix job construction for PaDIS tuning."""

from __future__ import annotations

import argparse
import pathlib

import PaDIS_run_reconstruction_matrix as matrix

from padis_tuning.candidates import Candidate, candidate_set
from padis_tuning.config import EXTERNAL_MODEL_LINKS


def ensure_staged_training_root(
    *,
    external_model_root: pathlib.Path,
    training_root: pathlib.Path,
) -> None:
    """Ensure staged training root."""
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
    """Parse csv selection."""
    value = value.strip()
    if value in ("", "all"):
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def job_matches_candidate(job: matrix.ReconstructionJob, candidate: Candidate) -> bool:
    """Return the job matches candidate."""
    if job.method.name != candidate.method:
        return False
    if job.implementation != candidate.implementation:
        return False
    if candidate.prior is not None and matrix.job_prior_mode(job) != candidate.prior:
        return False
    return True


def candidate_matches_filters(candidate: Candidate, args: argparse.Namespace) -> bool:
    """Handle candidate matches filters for the PaDIS workflow."""
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
    """Build matrix args."""
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
        "--checkpoint-policy",
        args.checkpoint_policy,
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
    """Handle candidate output root for the PaDIS workflow."""
    return (
        args.output_root
        / args.run_name
        / args.candidate_set
        / candidate.group
        / candidate.name
    )


def command_metrics_path(command: list[str]) -> pathlib.Path:
    """Build the command for metrics path."""

    def value_after(flag: str) -> str:
        """Handle value after for the PaDIS workflow."""
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
    """Build the command for for candidate."""
    run_matrix_args = argparse.Namespace(**vars(matrix_args))
    run_matrix_args.output_root = candidate_output_root(args, candidate)
    command = matrix.command_for_job(run_matrix_args, job)
    command.extend(candidate.args)
    if args.stop_after_outer_steps is not None:
        command.extend(["--stop-after-outer-steps", str(args.stop_after_outer_steps)])
    for extra_arg in args.reconstruction_arg:
        command.append(extra_arg)
    return command


def build_runs(
    args: argparse.Namespace,
) -> tuple[
    argparse.Namespace, list[tuple[Candidate, matrix.ReconstructionJob, list[str]]]
]:
    """Resolve matrix jobs and candidate filters into executable tuning runs."""
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
