#!/bin/bash
#
# Submit one ten-minute GPU utilisation profile for the default PaDIS A100
# training configuration.
#
# Useful overrides:
#   PADIS_SLURM_ACCOUNT=MPHIL-DIS-SL2-GPU
#   PADIS_PROFILE_SLURM_TIME=00:25:00
#   PADIS_PROFILE_SECONDS=600
#   PADIS_PROFILE_SAMPLE_INTERVAL=1
#   PADIS_PROFILE_WARMUP_SECONDS=60
#   PADIS_RUN_ROOT=/path/to/experiments/PaDIS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/paper_scripts/PaDIS/slurm/padis_a100_common.sh
. "$SCRIPT_DIR/padis_a100_common.sh"

LION_ROOT="$(padis_lion_root)"
cd "$LION_ROOT"

account="${PADIS_SLURM_ACCOUNT:-MPHIL-DIS-SL2-GPU}"
profile_time="${PADIS_PROFILE_SLURM_TIME:-00:25:00}"
run_root="$(padis_default_run_root)"
run_stamp="${PADIS_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$run_root/debug_runs/slurm_logs" "$run_root/profile_runs"

export PADIS_RUN_STAMP="$run_stamp"
export PADIS_SLURM_DIR="$SCRIPT_DIR"
export LION_ROOT

profile_job="$(
        sbatch \
                --parsable \
                -A "$account" \
                --time "$profile_time" \
                --export=ALL \
                --output "$run_root/debug_runs/slurm_logs/%x-%j.out" \
                "$SCRIPT_DIR/slurm_PaDIS_A100_profile_default.sh"
)"

cat <<EOF
Submitted default PaDIS GPU profile: $profile_job

Run root: $run_root
Run stamp: $run_stamp
Profile output: $run_root/profile_runs/default_training_$run_stamp
Slurm log: $run_root/debug_runs/slurm_logs/PaDIS_profile-$profile_job.out

Monitor:
  squeue -j $profile_job

Results:
  $run_root/profile_runs/default_training_$run_stamp/profile_summary.json
  $run_root/profile_runs/default_training_$run_stamp/timing_metrics.jsonl
  $run_root/profile_runs/default_training_$run_stamp/nvidia_smi.csv
  $run_root/profile_runs/default_training_$run_stamp/train.log

Cancel:
  scancel $profile_job
EOF
