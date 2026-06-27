#!/bin/bash
#
# Submit the PaDIS reconstruction matrix for models trained by
# submit_PaDIS_A100_training_only.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/paper_scripts/PaDIS/slurm/padis_a100_common.sh
. "$SCRIPT_DIR/padis_a100_common.sh"

LION_ROOT="$(padis_lion_root)"
cd "$LION_ROOT"

account="${PADIS_SLURM_ACCOUNT:-MPHIL-DIS-SL2-GPU}"
time_limit="${PADIS_RECON_TIME:-06:00:00}"
array_limit="${PADIS_RECON_ARRAY_LIMIT:-10}"
run_root="$(padis_default_run_root)"
run_stamp="${PADIS_RUN_STAMP:-}"

if [ -z "${PADIS_TRAIN_ROOT:-}" ]; then
        if [ -n "$run_stamp" ]; then
                PADIS_TRAIN_ROOT="$run_root/final_real_runs/a100_training_$run_stamp"
        else
                latest_train_root=""
                if [ -d "$run_root/final_real_runs" ]; then
                        latest_train_root="$(
                                find "$run_root/final_real_runs" \
                                        -maxdepth 1 \
                                        -type d \
                                        -name 'a100_training_*' \
                                        -printf '%T@ %p\n' \
                                        | sort -nr \
                                        | sed -n '1s/^[^ ]* //p'
                        )"
                fi
                if [ -z "$latest_train_root" ]; then
                        echo "Could not infer PADIS_TRAIN_ROOT. Set PADIS_TRAIN_ROOT or PADIS_RUN_STAMP." >&2
                        exit 1
                fi
                PADIS_TRAIN_ROOT="$latest_train_root"
                run_stamp="${PADIS_TRAIN_ROOT##*a100_training_}"
        fi
fi

run_stamp="${run_stamp:-$(date +%Y%m%d_%H%M%S)}"
PADIS_RECON_ROOT="${PADIS_RECON_ROOT:-$run_root/final_real_runs/a100_reconstruction_$run_stamp}"
PADIS_RECON_MODELS="${PADIS_RECON_MODELS:-all}"
PADIS_RECON_EXPERIMENTS="${PADIS_RECON_EXPERIMENTS:-paper_matrix}"
PADIS_RECON_IMPLEMENTATIONS="${PADIS_RECON_IMPLEMENTATIONS:-paper,public_repo}"
PADIS_RECON_GEOMETRIES="${PADIS_RECON_GEOMETRIES:-lion}"
PADIS_RECON_SPLIT="${PADIS_RECON_SPLIT:-test}"
PADIS_RECON_ALGORITHM="${PADIS_RECON_ALGORITHM:-dps_langevin}"
PADIS_RECON_MAX_SAMPLES="${PADIS_RECON_MAX_SAMPLES:-25}"
PADIS_RECON_START_INDEX="${PADIS_RECON_START_INDEX:-0}"
PADIS_RECON_SEED="${PADIS_RECON_SEED:-33}"
PADIS_RECON_DEVICE="${PADIS_RECON_DEVICE:-cuda}"

export PADIS_RUN_STAMP="$run_stamp"
export PADIS_SLURM_DIR="$SCRIPT_DIR"
export LION_ROOT PADIS_TRAIN_ROOT PADIS_RECON_ROOT
export PADIS_RECON_MODELS PADIS_RECON_EXPERIMENTS PADIS_RECON_IMPLEMENTATIONS
export PADIS_RECON_GEOMETRIES PADIS_RECON_SPLIT PADIS_RECON_ALGORITHM
export PADIS_RECON_MAX_SAMPLES PADIS_RECON_START_INDEX PADIS_RECON_SEED
export PADIS_RECON_DEVICE

mkdir -p "$run_root/debug_runs/slurm_logs" "$PADIS_RECON_ROOT"

task_count="$(
        python scripts/paper_scripts/PaDIS/PaDIS_run_reconstruction_matrix.py \
                --training-root "$PADIS_TRAIN_ROOT" \
                --output-root "$PADIS_RECON_ROOT" \
                --models "$PADIS_RECON_MODELS" \
                --experiments "$PADIS_RECON_EXPERIMENTS" \
                --implementations "$PADIS_RECON_IMPLEMENTATIONS" \
                --geometries "$PADIS_RECON_GEOMETRIES" \
                --split "$PADIS_RECON_SPLIT" \
                --algorithm "$PADIS_RECON_ALGORITHM" \
                --max-samples "$PADIS_RECON_MAX_SAMPLES" \
                --start-index "$PADIS_RECON_START_INDEX" \
                --seed "$PADIS_RECON_SEED" \
                --device "$PADIS_RECON_DEVICE" \
                --count
)"
last_task=$((task_count - 1))
array_spec="${PADIS_RECON_ARRAY:-0-${last_task}%${array_limit}}"

recon_job="$(
        sbatch \
                --parsable \
                -A "$account" \
                --time "$time_limit" \
                --array "$array_spec" \
                --export=ALL \
                --output "$run_root/debug_runs/slurm_logs/%x-%A_%a.out" \
                "$SCRIPT_DIR/slurm_PaDIS_A100_reconstruction_array.sh"
)"

cat <<EOF
Submitted PaDIS reconstruction array: $recon_job

Training root: $PADIS_TRAIN_ROOT
Reconstruction root: $PADIS_RECON_ROOT
Run stamp: $run_stamp
Array: $array_spec
Tasks: $task_count
Models: $PADIS_RECON_MODELS
Experiments: $PADIS_RECON_EXPERIMENTS
Implementations: $PADIS_RECON_IMPLEMENTATIONS
Geometries: $PADIS_RECON_GEOMETRIES
Slurm logs: $run_root/debug_runs/slurm_logs

Monitor:
  squeue -j $recon_job

Cancel:
  scancel $recon_job
EOF
