#!/bin/bash
#
# Submit only the real PaDIS A100 training array. Use this after the local
# check/pilot runner has passed on the development machine.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/paper_scripts/PaDIS/slurm/padis_a100_common.sh
. "$SCRIPT_DIR/padis_a100_common.sh"

LION_ROOT="$(padis_lion_root)"
cd "$LION_ROOT"

task_count="$(padis_training_task_count)"
last_task=$((task_count - 1))

account="${PADIS_SLURM_ACCOUNT:-MPHIL-DIS-SL2-GPU}"
real_time="${PADIS_REAL_TIME:-24:00:00}"
real_limit="${PADIS_REAL_ARRAY_LIMIT:-10}"
run_root="$(padis_default_run_root)"
run_stamp="${PADIS_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$run_root/debug_runs/slurm_logs" "$run_root/real_runs"

real_job="$(
        sbatch \
                --parsable \
                -A "$account" \
                --time "$real_time" \
                --array "0-${last_task}%${real_limit}" \
                --export "ALL,PADIS_RUN_STAMP=$run_stamp" \
                --output "$run_root/debug_runs/slurm_logs/%x-%A_%a.out" \
                "$SCRIPT_DIR/slurm_PaDIS_A100_training_array.sh"
)"

cat <<EOF
Submitted real PaDIS training array: $real_job

Run root: $run_root
Run stamp: $run_stamp
Real training: $run_root/real_runs/a100_training_$run_stamp
Slurm logs: $run_root/debug_runs/slurm_logs

Monitor:
  squeue -j $real_job

Cancel:
  scancel $real_job
EOF
