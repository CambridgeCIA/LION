#!/bin/bash
#
# Submit the DRUNet denoiser training job required by the paper PnP-ADMM row.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/paper_scripts/PaDIS/slurm/padis_a100_common.sh
. "$SCRIPT_DIR/padis_a100_common.sh"

LION_ROOT="$(padis_lion_root)"
cd "$LION_ROOT"

account="${PADIS_SLURM_ACCOUNT:-MPHIL-DIS-SL2-GPU}"
time_limit="${PADIS_PNP_TIME:-24:00:00}"
run_root="$(padis_default_run_root)"
run_stamp="${PADIS_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
PADIS_TRAIN_ROOT="${PADIS_TRAIN_ROOT:-$run_root/final_real_runs/a100_training_$run_stamp}"
PADIS_PNP_OUTPUT_ROOT="${PADIS_PNP_OUTPUT_ROOT:-$PADIS_TRAIN_ROOT}"
PADIS_PNP_RUN_NAME="${PADIS_PNP_RUN_NAME:-pnp_lidc_drunet}"
PADIS_PNP_FINAL_NAME="${PADIS_PNP_FINAL_NAME:-pnp_lidc_drunet.pt}"

mkdir -p "$run_root/debug_runs/slurm_logs" "$PADIS_TRAIN_ROOT"

export PADIS_RUN_STAMP="$run_stamp"
export PADIS_SLURM_DIR="$SCRIPT_DIR"
export LION_ROOT PADIS_TRAIN_ROOT PADIS_PNP_OUTPUT_ROOT PADIS_PNP_RUN_NAME
export PADIS_PNP_FINAL_NAME

pnp_job="$(
        sbatch \
                --parsable \
                -A "$account" \
                --time "$time_limit" \
                --export=ALL \
                --output "$run_root/debug_runs/slurm_logs/%x-%j.out" \
                "$SCRIPT_DIR/slurm_PaDIS_A100_pnp_training.sh"
)"

cat <<EOF
Submitted PaDIS PnP denoiser training job: $pnp_job

Run root: $run_root
Run stamp: $run_stamp
Training root: $PADIS_TRAIN_ROOT
Expected checkpoint: $PADIS_PNP_OUTPUT_ROOT/$PADIS_PNP_RUN_NAME/$PADIS_PNP_FINAL_NAME
Slurm logs: $run_root/debug_runs/slurm_logs

Monitor:
  squeue -j $pnp_job

Cancel:
  scancel $pnp_job
EOF
