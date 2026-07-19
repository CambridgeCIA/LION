"""Tune PaDIS reconstruction hyperparameters on the LIDC validation split.

This script wraps PaDIS_run_reconstruction_matrix.py so tuning runs use the
same job definitions as the final inference array. It stages the external model
checkpoints into the training-root layout expected by the matrix launcher,
runs candidate reconstruction settings on the validation split, and aggregates
the resulting metrics.json files.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "reconstruction"))

import PaDIS_run_reconstruction_matrix as matrix


from padis_tuning import (  # noqa: F401
    DEFAULT_EXTERNAL_MODEL_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STAGED_TRAINING_ROOT,
    DEFAULT_TUNING_ROOT,
    EXTERNAL_MODEL_LINKS,
    Candidate,
    RunRecord,
    aggregate_records,
    broad_candidates,
    build_arg_parser,
    build_matrix_args,
    build_runs,
    candidate_matches_filters,
    candidate_output_root,
    candidate_set,
    command_extra_reconstruction_args,
    command_for_candidate,
    command_line,
    command_metrics_path,
    consensus_24h_candidates,
    consensus_24h_no_defaults_candidates,
    current_default_candidates,
    ensure_staged_training_root,
    finite_values,
    flag_value_args,
    focused_candidates,
    job_matches_candidate,
    lion_physics_candidate,
    lion_physics_full_candidates,
    lion_physics_pc_public_gap_candidates,
    max_or_none,
    mean_or_none,
    min_or_none,
    padis_dps_lion_full_candidates,
    paper_full_candidates,
    parse_csv_selection,
    pilot_candidates,
    public_paper_sampler_candidates,
    public_repo_full_candidates,
    record_for_run,
    reproduction_candidates,
    run_command,
    run_tuning,
    safe_name,
    sampler_candidate,
    sampler_full_candidates,
    summarize_metrics,
    unique_candidates,
    write_csv,
    write_jsonl,
    write_outputs,
    zeta_candidates,
)


def main() -> None:
    """Execute selected fixed-validation tuning candidates."""
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

    if not args.use_existing_training_root:
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
