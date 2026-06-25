#!/bin/bash
#
# Submit the PaDIS A100 reproduction pipeline as:
#   checks -> LIDC cache preparation -> pilot training array -> real training array
#
# Useful overrides:
#   PADIS_SLURM_ACCOUNT=MPHIL-DIS-SL2-GPU
#   PADIS_CACHE_SLURM_ACCOUNT=MPHIL-DIS-SL2-CPU
#   PADIS_CHECK_TIME=00:20:00
#   PADIS_CACHE_SLURM_TIME=08:00:00
#   PADIS_CACHE_PARTITION=icelake
#   PADIS_CACHE_CPUS_PER_TASK=8
#   PADIS_CACHE_MEM=128G
#   PADIS_CACHE_PREP_VARIANTS=256-default,256-full,512-default
#   PADIS_PILOT_TIME=00:15:00
#   PADIS_REAL_TIME=24:00:00
#   PADIS_PILOT_ARRAY_LIMIT=10
#   PADIS_REAL_ARRAY_LIMIT=10
#   PADIS_RUN_ROOT=/path/to/experiments/PaDIS
#   PADIS_WANDB_PROJECT=PaDIS-Reproduction

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/paper_scripts/PaDIS/slurm/padis_a100_common.sh
. "$SCRIPT_DIR/padis_a100_common.sh"

LION_ROOT="$(padis_lion_root)"
cd "$LION_ROOT"

task_count="$(padis_training_task_count)"
last_task=$((task_count - 1))

account="${PADIS_SLURM_ACCOUNT:-MPHIL-DIS-SL2-GPU}"
cache_account="${PADIS_CACHE_SLURM_ACCOUNT:-MPHIL-DIS-SL2-CPU}"
check_time="${PADIS_CHECK_TIME:-00:20:00}"
cache_time="${PADIS_CACHE_SLURM_TIME:-08:00:00}"
cache_partition="${PADIS_CACHE_PARTITION:-icelake}"
cache_cpus="${PADIS_CACHE_CPUS_PER_TASK:-8}"
cache_mem="${PADIS_CACHE_MEM:-128G}"
cache_variants="${PADIS_CACHE_PREP_VARIANTS:-256-default,256-full,512-default}"
pilot_time="${PADIS_PILOT_TIME:-00:15:00}"
real_time="${PADIS_REAL_TIME:-24:00:00}"
pilot_limit="${PADIS_PILOT_ARRAY_LIMIT:-10}"
real_limit="${PADIS_REAL_ARRAY_LIMIT:-10}"
run_root="$(padis_default_run_root)"
run_stamp="${PADIS_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
data_root="${LION_DATA_PATH:-/home/tjh200/rds/hpc-work/Datasets}"
cache_root="${PADIS_CACHE_ROOT:-$data_root/processed/LIDC-IDRI-cache}"

mkdir -p "$run_root/debug_runs/slurm_logs" "$run_root/pilot_runs" "$run_root/real_runs" "$cache_root"

export PADIS_RUN_STAMP="$run_stamp"
export PADIS_SLURM_DIR="$SCRIPT_DIR"
export LION_ROOT
export PADIS_CACHE_ROOT="$cache_root"
export PADIS_CACHE_PREP_VARIANTS="$cache_variants"

checks_job="$(
        sbatch \
                --parsable \
                -A "$account" \
                --time "$check_time" \
                --export=ALL \
                --output "$run_root/debug_runs/slurm_logs/%x-%j.out" \
                "$SCRIPT_DIR/slurm_PaDIS_A100_checks.sh"
)"
echo "Submitted checks job: $checks_job"

cache_job="$(
        sbatch \
                --parsable \
                -A "$cache_account" \
                -p "$cache_partition" \
                --cpus-per-task "$cache_cpus" \
                --mem "$cache_mem" \
                --time "$cache_time" \
                --dependency "afterok:$checks_job" \
                --export=ALL \
                --output "$run_root/debug_runs/slurm_logs/%x-%j.out" \
                "$SCRIPT_DIR/slurm_PaDIS_A100_prepare_full_cache.sh"
)"
echo "Submitted cache preparation job: $cache_job after $checks_job"

pilot_job="$(
        sbatch \
                --parsable \
                -A "$account" \
                --time "$pilot_time" \
                --array "0-${last_task}%${pilot_limit}" \
                --dependency "afterok:$cache_job" \
                --export=ALL \
                --output "$run_root/debug_runs/slurm_logs/%x-%A_%a.out" \
                "$SCRIPT_DIR/slurm_PaDIS_A100_pilot_array.sh"
)"
echo "Submitted pilot array: $pilot_job after $cache_job"

real_job="$(
        sbatch \
                --parsable \
                -A "$account" \
                --time "$real_time" \
                --array "0-${last_task}%${real_limit}" \
                --dependency "afterok:$pilot_job" \
                --export=ALL \
                --output "$run_root/debug_runs/slurm_logs/%x-%A_%a.out" \
                "$SCRIPT_DIR/slurm_PaDIS_A100_training_array.sh"
)"
echo "Submitted real training array: $real_job after $pilot_job"

cat <<EOF

Pipeline submitted.

Run root: $run_root
Run stamp: $run_stamp
Cache root: $cache_root
Debug/checks: $run_root/debug_runs/a100_checks_$run_stamp
Pilots: $run_root/pilot_runs/a100_pilots_$run_stamp
Real training: $run_root/real_runs/a100_training_$run_stamp

Checks: $checks_job
Cache prep: $cache_job
Pilots: $pilot_job
Real training: $real_job

Monitor:
  squeue -j $checks_job,$cache_job,$pilot_job,$real_job

Cancel all:
  scancel $checks_job $cache_job $pilot_job $real_job
EOF
