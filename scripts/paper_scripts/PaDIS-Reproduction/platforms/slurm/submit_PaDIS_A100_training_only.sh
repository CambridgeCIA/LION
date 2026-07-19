#!/bin/bash
#
# Submit only the real PaDIS A100 training array. Use this after the local
# check/pilot runner has passed on the development machine.

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
real_limit="${PADIS_REAL_ARRAY_LIMIT:-10}"
patch_array="${PADIS_PATCH_REAL_ARRAY:-0-6,9%${real_limit}}"
whole_array="${PADIS_WHOLE_REAL_ARRAY:-7-8%${real_limit}}"
run_root="$(padis_default_run_root)"
run_stamp="${PADIS_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$run_root/debug_runs/slurm_logs" "$run_root/final_real_runs"

export PADIS_RUN_STAMP="$run_stamp"
export PADIS_SLURM_DIR="$SCRIPT_DIR"
export LION_ROOT
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

cat <<EOF
Submitted real PaDIS training arrays.

Run root: $run_root
Run stamp: $run_stamp
Patch array: $patch_array ($patch_time allocation)
Whole-image array: $whole_array ($whole_time allocation)
Real training: $run_root/final_real_runs/a100_training_$run_stamp
Slurm logs: $run_root/debug_runs/slurm_logs

Monitor:
  squeue -j $patch_job,$whole_job

Cancel:
  scancel $patch_job $whole_job
EOF
