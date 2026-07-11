#!/bin/bash
#
# Submit all training jobs needed by the PaDIS reconstruction matrix:
# the PaDIS diffusion-prior training array and the DRUNet denoiser used by
# the PnP-ADMM comparison row.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/paper_scripts/PaDIS-Reproduction/platforms/slurm/padis_a100_common.sh
. "$SCRIPT_DIR/padis_a100_common.sh"

LION_ROOT="$(padis_lion_root)"
cd "$LION_ROOT"

task_count="$(padis_training_task_count)"
last_task=$((task_count - 1))

account="${PADIS_SLURM_ACCOUNT:-MPHIL-DIS-SL2-GPU}"
patch_time="${PADIS_PATCH_REAL_TIME:-12:30:00}"
whole_time="${PADIS_WHOLE_REAL_TIME:-${PADIS_REAL_TIME:-24:30:00}}"
pnp_time="${PADIS_PNP_TIME:-24:00:00}"
real_limit="${PADIS_REAL_ARRAY_LIMIT:-10}"
patch_array="${PADIS_PATCH_REAL_ARRAY:-0-6,9%${real_limit}}"
whole_array="${PADIS_WHOLE_REAL_ARRAY:-7-8%${real_limit}}"
run_root="$(padis_default_run_root)"
run_stamp="${PADIS_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

export PADIS_RUN_STAMP="$run_stamp"
export PADIS_SLURM_DIR="$SCRIPT_DIR"
export LION_ROOT
export PADIS_TRAIN_ROOT="${PADIS_TRAIN_ROOT:-$run_root/final_real_runs/a100_training_$run_stamp}"
export PADIS_PNP_OUTPUT_ROOT="${PADIS_PNP_OUTPUT_ROOT:-$PADIS_TRAIN_ROOT}"
export PADIS_PNP_RUN_NAME="${PADIS_PNP_RUN_NAME:-pnp_lidc_drunet}"
export PADIS_PNP_FINAL_NAME="${PADIS_PNP_FINAL_NAME:-pnp_lidc_drunet.pt}"

mkdir -p "$run_root/debug_runs/slurm_logs" "$run_root/final_real_runs" "$PADIS_TRAIN_ROOT"

padis_configure_real_training_defaults "$whole_time" "$run_stamp"

patch_job="$(
        sbatch \
                --parsable \
                -A "$account" \
                --time "$patch_time" \
                --array "$patch_array" \
                --export=ALL \
                --output "$run_root/debug_runs/slurm_logs/%x-%A_%a.out" \
                "$SCRIPT_DIR/slurm_PaDIS_A100_training_array.sh"
)"
whole_job="$(
        sbatch \
                --parsable \
                -A "$account" \
                --time "$whole_time" \
                --array "$whole_array" \
                --export=ALL \
                --output "$run_root/debug_runs/slurm_logs/%x-%A_%a.out" \
                "$SCRIPT_DIR/slurm_PaDIS_A100_training_array.sh"
)"

pnp_job=""
if [ "${PADIS_SUBMIT_PNP_TRAINING:-1}" = "1" ]; then
        pnp_job="$(
                sbatch \
                        --parsable \
                        -A "$account" \
                        --time "$pnp_time" \
                        --export=ALL \
                        --output "$run_root/debug_runs/slurm_logs/%x-%j.out" \
                        "$SCRIPT_DIR/slurm_PaDIS_A100_pnp_training.sh"
        )"
fi

cat <<EOF
Submitted PaDIS all-training jobs.

Run root: $run_root
Run stamp: $run_stamp
PaDIS patch-prior training array (12 h + 30 min buffer): $patch_job
PaDIS whole-image training array (24 h + 30 min buffer): $whole_job
PaDIS diffusion training root: $PADIS_TRAIN_ROOT
PnP denoiser training: ${pnp_job:-skipped}
Expected PnP checkpoint: $PADIS_PNP_OUTPUT_ROOT/$PADIS_PNP_RUN_NAME/$PADIS_PNP_FINAL_NAME
Slurm logs: $run_root/debug_runs/slurm_logs

Monitor:
  squeue -j $patch_job,$whole_job${pnp_job:+,$pnp_job}

Cancel:
  scancel $patch_job $whole_job${pnp_job:+ $pnp_job}
EOF
