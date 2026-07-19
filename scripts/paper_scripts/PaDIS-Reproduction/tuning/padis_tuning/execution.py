"""Command execution and run orchestration for PaDIS tuning."""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import time

import PaDIS_run_reconstruction_matrix as matrix

from padis_tuning.candidates import Candidate
from padis_tuning.jobs import (
    build_runs,
    command_metrics_path,
    ensure_staged_training_root,
)
from padis_tuning.metrics import (
    command_extra_reconstruction_args,
    summarize_metrics,
    write_outputs,
)


def run_command(
    command: list[str], log_path: pathlib.Path
) -> tuple[str, float, str | None]:
    """Run command."""
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
    """Build a record for for run."""
    candidate_args = list(candidate.args)
    for extra_arg in command_extra_reconstruction_args(command):
        if extra_arg not in candidate_args:
            candidate_args.append(extra_arg)
    return {
        "candidate_group": candidate.group,
        "candidate": candidate.name,
        "candidate_args": candidate_args,
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


def command_line(command: list[str]) -> str:
    """Build the command for line."""
    return " ".join(shlex.quote(part) for part in command)


def run_tuning(args: argparse.Namespace) -> None:
    """Run tuning."""
    if not args.use_existing_training_root:
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
                "checkpoint_policy": args.checkpoint_policy,
                "use_existing_training_root": args.use_existing_training_root,
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
