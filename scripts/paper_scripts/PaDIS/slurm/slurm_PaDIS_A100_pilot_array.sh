#!/bin/bash
#!
#! Short PaDIS A100 training pilots for every real training configuration.
#!
#SBATCH -J PaDIS_pilot
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --mail-type=NONE
#SBATCH --array=0-13%14
#SBATCH -p ampere
#SBATCH -o slurm-%x-%A_%a.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/paper_scripts/PaDIS/slurm/padis_a100_common.sh
. "$SCRIPT_DIR/padis_a100_common.sh"

padis_setup_modules
padis_activate_environment

LION_ROOT="$(padis_lion_root)"
PADIS_RUN_ROOT="$(padis_default_run_root)"
PADIS_RUN_STAMP="${PADIS_RUN_STAMP:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}}"
PADIS_PILOT_ROOT="${PADIS_PILOT_ROOT:-$PADIS_RUN_ROOT/pilot_runs/a100_pilots_$PADIS_RUN_STAMP}"
PADIS_DATA_FOLDER="${PADIS_DATA_FOLDER:-}"
PADIS_SEED="${PADIS_SEED:-33}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$PADIS_PILOT_ROOT/matplotlib}"
WANDB_DIR="${WANDB_DIR:-$PADIS_PILOT_ROOT/wandb}"
export LION_ROOT PADIS_RUN_ROOT PADIS_RUN_STAMP PADIS_PILOT_ROOT PADIS_DATA_FOLDER
export MPLCONFIGDIR WANDB_DIR PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 PYTHONHASHSEED="$PADIS_SEED"

padis_init_training_tasks
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#PADIS_TASK_NAMES[@]}" ]; then
        echo "Invalid task id $TASK_ID for ${#PADIS_TASK_NAMES[@]} tasks."
        exit 1
fi

TASK_NAME="${PADIS_TASK_NAMES[$TASK_ID]}"
TASK_ENGINE="${PADIS_TASK_ENGINES[$TASK_ID]}"
TASK_BATCH_SIZE="${PADIS_TASK_BATCH_SIZES[$TASK_ID]}"
read -r -a TASK_ARGS <<< "${PADIS_TASK_ARGUMENTS[$TASK_ID]}"

mkdir -p "$PADIS_PILOT_ROOT" "$MPLCONFIGDIR" "$WANDB_DIR"
cd "$LION_ROOT"
padis_print_job_header
echo "Pilot task: $TASK_ID $TASK_NAME ($TASK_ENGINE)"

TARGET_PATCHES="${PADIS_PILOT_TARGET_PATCHES:-8192}"
VALIDATION_INTERVAL="${PADIS_PILOT_VALIDATION_INTERVAL_PATCHES:-$TARGET_PATCHES}"
CHECKPOINT_INTERVAL="${PADIS_PILOT_CHECKPOINT_INTERVAL_PATCHES:-$TARGET_PATCHES}"
LOG_INTERVAL="${PADIS_PILOT_LOG_INTERVAL_PATCHES:-128}"
NUM_WORKERS="${PADIS_NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PADIS_PREFETCH_FACTOR:-4}"

if [ "$TASK_ENGINE" = "lidc256" ]; then
        CMD=(
                python -u scripts/paper_scripts/PaDIS/PaDIS_LIDC_256.py
                --save-folder "$PADIS_PILOT_ROOT"
                --device cuda
                --target-patches "$TARGET_PATCHES"
                --validation-interval-patches "$VALIDATION_INTERVAL"
                --checkpoint-interval-patches "$CHECKPOINT_INTERVAL"
                --log-interval-patches "$LOG_INTERVAL"
                --seed "$PADIS_SEED"
                --batch-size "$TASK_BATCH_SIZE"
                --num-workers "$NUM_WORKERS"
                --prefetch-factor "$PREFETCH_FACTOR"
                --no-wandb
                --wandb-mode disabled
        )
        if [ -n "$PADIS_DATA_FOLDER" ]; then
                CMD+=(--data-folder "$PADIS_DATA_FOLDER")
        fi
        if [ -n "${PADIS_MICROBATCH_SIZE:-}" ]; then
                CMD+=(--microbatch-size "$PADIS_MICROBATCH_SIZE")
        fi
        CACHE_DATASET="${PADIS_CACHE_DATASET:-ramdisk}"
        if [[ "$TASK_NAME" == *full ]] && [ "${PADIS_CACHE_FULL_LIDC:-0}" != "1" ]; then
                CACHE_DATASET="none"
        fi
        if [ "$CACHE_DATASET" != "none" ]; then
                CACHE_FOLDER="${PADIS_CACHE_FOLDER:-/ramdisks/$USER/lion_lidc_cache}"
                CMD+=(--cache-dataset "$CACHE_DATASET" --cache-folder "$CACHE_FOLDER")
        fi
elif [ "$TASK_ENGINE" = "lidc512" ]; then
        CMD=(
                python -u scripts/paper_scripts/PaDIS/PaDIS_LIDC_512.py
                --save-folder "$PADIS_PILOT_ROOT"
                --device cuda
                --target-patches "$TARGET_PATCHES"
                --validation-interval-patches "$VALIDATION_INTERVAL"
                --checkpoint-interval-patches "$CHECKPOINT_INTERVAL"
                --log-interval-patches "$LOG_INTERVAL"
                --seed "$PADIS_SEED"
                --batch-size "$TASK_BATCH_SIZE"
                --num-workers "${PADIS_512_NUM_WORKERS:-$NUM_WORKERS}"
                --prefetch-factor "${PADIS_512_PREFETCH_FACTOR:-$PREFETCH_FACTOR}"
                --no-wandb
                --wandb-mode disabled
        )
        if [ -n "$PADIS_DATA_FOLDER" ]; then
                CMD+=(--data-folder "$PADIS_DATA_FOLDER")
        fi
        if [ -n "${PADIS_MICROBATCH_SIZE:-}" ]; then
                CMD+=(--microbatch-size "$PADIS_MICROBATCH_SIZE")
        fi
else
        echo "Unknown task engine: $TASK_ENGINE"
        exit 1
fi

CMD+=("${TASK_ARGS[@]}")

echo "Executing pilot command:"
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

echo "Pilot task $TASK_NAME completed at $(date)."
