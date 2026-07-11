#!/usr/bin/env bash
# Run or submit the complete PaDIS training and reconstruction pipeline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BACKEND="${PADIS_PIPELINE_BACKEND:-}"
DRY_RUN="${PADIS_PIPELINE_DRY_RUN:-0}"

usage() {
        cat <<'EOF'
Usage: PaDIS_run_pipeline.sh --backend gcp|slurm [--dry-run]

Selects the existing resumable backend pipeline while retaining all PADIS_*
environment-variable overrides.

  gcp    Run training, validation-intensive continuation, and reconstruction
         synchronously on the current GCP machine.
  slurm  Submit checks, cache preparation, pilot and real training, both PnP
         trainings, reconstruction, and verification as dependent Slurm jobs.

Examples:
  bash scripts/paper_scripts/PaDIS-Reproduction/pipeline/PaDIS_run_pipeline.sh --backend gcp
  PADIS_RUN_STAMP=paper-final bash scripts/paper_scripts/PaDIS-Reproduction/pipeline/PaDIS_run_pipeline.sh --backend slurm
EOF
}

while [ "$#" -gt 0 ]; do
        case "$1" in
                --backend)
                        [ "$#" -ge 2 ] || { echo "--backend requires gcp or slurm" >&2; exit 2; }
                        BACKEND="$2"
                        shift 2
                        ;;
                --dry-run)
                        DRY_RUN=1
                        shift
                        ;;
                -h|--help)
                        usage
                        exit 0
                        ;;
                *)
                        echo "Unknown argument: $1" >&2
                        usage >&2
                        exit 2
                        ;;
        esac
done

case "$BACKEND" in
        gcp)
                runner="$SCRIPT_DIR/../platforms/gcp/run_PaDIS_GCP_spot_training.sh"
                export PADIS_GCP_RECONSTRUCTION_PHASE="${PADIS_GCP_RECONSTRUCTION_PHASE:-1}"
                export PADIS_GCP_INCLUDE_PNP="${PADIS_GCP_INCLUDE_PNP:-1}"
                ;;
        slurm)
                runner="$SCRIPT_DIR/../platforms/slurm/submit_PaDIS_A100_pipeline.sh"
                export PADIS_SUBMIT_PNP_TRAINING="${PADIS_SUBMIT_PNP_TRAINING:-1}"
                export PADIS_SUBMIT_RECONSTRUCTION="${PADIS_SUBMIT_RECONSTRUCTION:-1}"
                export PADIS_RECON_VERIFY="${PADIS_RECON_VERIFY:-1}"
                ;;
        "")
                echo "Select a backend with --backend gcp|slurm or PADIS_PIPELINE_BACKEND." >&2
                exit 2
                ;;
        *)
                echo "Unsupported backend: $BACKEND (expected gcp or slurm)" >&2
                exit 2
                ;;
esac

[ -f "$runner" ] || { echo "Pipeline backend runner not found: $runner" >&2; exit 1; }

echo "PaDIS pipeline backend: $BACKEND"
echo "Backend runner: $runner"
if [ "$DRY_RUN" = "1" ]; then
        echo "Dry run: backend runner was not executed."
        exit 0
fi

if [ "$BACKEND" = "gcp" ]; then
        bash "$runner"
        export PADIS_TRAIN_ROOT="${PADIS_TRAIN_ROOT:-${PADIS_RUN_ROOT:-${LION_EXPERIMENTS_PATH:-/mnt/data/Datasets/experiments}/PaDIS}/final_real_runs/${PADIS_GCP_RUN_NAME:-PaDIS-Reproduction-GCP}}"
        export PADIS_RECON_ROOT="${PADIS_RECON_ROOT:-${PADIS_RUN_ROOT:-${LION_EXPERIMENTS_PATH:-/mnt/data/Datasets/experiments}/PaDIS}/final_real_runs/${PADIS_GCP_RUN_NAME:-PaDIS-Reproduction-GCP}_reconstruction}"
        export PADIS_TIMING_MODE="${PADIS_TIMING_MODE:-gcp}"
        export PADIS_TIMING_LOG_ROOT="${PADIS_TIMING_LOG_ROOT:-$PADIS_TRAIN_ROOT/.gcp_spot/logs}"
        finalise_env="${LION_MAMBA_ENV:-${LION_CONDA_ENV:-lion}}"
        if ! conda run -n "$finalise_env" bash "$SCRIPT_DIR/PaDIS_finalise_pipeline.sh"; then
                [ "$finalise_env" != "lion-dev" ] || exit 1
                conda run -n lion-dev bash "$SCRIPT_DIR/PaDIS_finalise_pipeline.sh"
        fi
else
        exec bash "$runner"
fi
