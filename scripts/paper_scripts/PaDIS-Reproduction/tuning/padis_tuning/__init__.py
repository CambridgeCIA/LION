"""Internal components for the PaDIS reconstruction tuner."""

from padis_tuning.candidates import *  # noqa: F401,F403
from padis_tuning.candidates import __all__ as _candidate_exports
from padis_tuning.cli import build_arg_parser
from padis_tuning.config import (
    DEFAULT_EXTERNAL_MODEL_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STAGED_TRAINING_ROOT,
    DEFAULT_TUNING_ROOT,
    EXTERNAL_MODEL_LINKS,
)
from padis_tuning.execution import command_line, record_for_run, run_command, run_tuning
from padis_tuning.jobs import (
    build_matrix_args,
    build_runs,
    candidate_matches_filters,
    candidate_output_root,
    command_for_candidate,
    command_metrics_path,
    ensure_staged_training_root,
    job_matches_candidate,
    parse_csv_selection,
)
from padis_tuning.metrics import (
    aggregate_records,
    command_extra_reconstruction_args,
    finite_values,
    max_or_none,
    mean_or_none,
    min_or_none,
    summarize_metrics,
    write_csv,
    write_jsonl,
    write_outputs,
)

__all__ = [
    *_candidate_exports,
    "DEFAULT_EXTERNAL_MODEL_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_STAGED_TRAINING_ROOT",
    "DEFAULT_TUNING_ROOT",
    "EXTERNAL_MODEL_LINKS",
    "aggregate_records",
    "build_arg_parser",
    "build_matrix_args",
    "build_runs",
    "candidate_matches_filters",
    "candidate_output_root",
    "command_extra_reconstruction_args",
    "command_for_candidate",
    "command_line",
    "command_metrics_path",
    "ensure_staged_training_root",
    "finite_values",
    "job_matches_candidate",
    "max_or_none",
    "mean_or_none",
    "min_or_none",
    "parse_csv_selection",
    "record_for_run",
    "run_command",
    "run_tuning",
    "summarize_metrics",
    "write_csv",
    "write_jsonl",
    "write_outputs",
]
