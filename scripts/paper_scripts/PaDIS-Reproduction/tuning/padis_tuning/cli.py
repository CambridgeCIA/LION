"""Tune PaDIS reconstruction hyperparameters on the LIDC validation split.

This script wraps PaDIS_run_reconstruction_matrix.py so tuning runs use the
same job definitions as the final inference array. It stages the external model
checkpoints into the training-root layout expected by the matrix launcher,
runs candidate reconstruction settings on the validation split, and aggregates
the resulting metrics.json files.
"""

from __future__ import annotations

import argparse
import pathlib
import time

import PaDIS_run_reconstruction_matrix as matrix

from padis_tuning.config import (
    DEFAULT_EXTERNAL_MODEL_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STAGED_TRAINING_ROOT,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the reconstruction-tuning command-line parser."""
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
        "--use-existing-training-root",
        action="store_true",
        help=(
            "Use checkpoints already arranged as a reconstruction-matrix training "
            "root instead of staging the historical external checkpoints."
        ),
    )
    parser.add_argument(
        "--checkpoint-policy",
        choices=matrix.CHECKPOINT_POLICIES,
        default="model_default",
        help="Diffusion checkpoint family passed to the reconstruction matrix.",
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
        choices=(
            "smoke",
            "pilot",
            "broad",
            "focused",
            "padis_dps_lion_full",
            "lion_physics_full",
            "public_paper_sampler",
            "public_repo_full",
            "paper_full",
            "consensus_24h",
            "consensus_24h_no_defaults",
            "sampler_full",
            "reproduction",
        ),
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
    parser.add_argument("--tv-lambda", type=float, default=0.002)
    parser.add_argument("--tv-iterations", type=int, default=1000)
    parser.add_argument("--pnp-iterations", type=int, default=60)
    parser.add_argument("--pnp-eta", type=float, default=2e-5)
    parser.add_argument("--pnp-cg-iterations", type=int, default=50)
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
