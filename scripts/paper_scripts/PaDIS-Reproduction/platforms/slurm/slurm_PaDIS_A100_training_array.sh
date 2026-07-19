#!/bin/bash
#!
#! Long PaDIS A100 training array for CT reproduction priors.
#!
#SBATCH -J PaDIS_train
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mail-type=NONE
#SBATCH --array=0-9%10
#SBATCH -p ampere
#SBATCH -o slurm-%x-%A_%a.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$SCRIPT_DIR/padis_a100_common.sh" ]; then
        if [ -n "${PADIS_SLURM_DIR:-}" ] && [ -f "$PADIS_SLURM_DIR/padis_a100_common.sh" ]; then
                SCRIPT_DIR="$(cd "$PADIS_SLURM_DIR" && pwd)"
        elif [ -n "${LION_ROOT:-}" ] && [ -f "$LION_ROOT/scripts/paper_scripts/PaDIS-Reproduction/platforms/slurm/padis_a100_common.sh" ]; then
                SCRIPT_DIR="$(cd "$LION_ROOT/scripts/paper_scripts/PaDIS-Reproduction/platforms/slurm" && pwd)"
        elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/padis_a100_common.sh" ]; then
                SCRIPT_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
        elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/paper_scripts/PaDIS-Reproduction/platforms/slurm/padis_a100_common.sh" ]; then
                SCRIPT_DIR="$(cd "$SLURM_SUBMIT_DIR/scripts/paper_scripts/PaDIS-Reproduction/platforms/slurm" && pwd)"
        elif [ -f "$PWD/scripts/paper_scripts/PaDIS-Reproduction/platforms/slurm/padis_a100_common.sh" ]; then
                SCRIPT_DIR="$(cd "$PWD/scripts/paper_scripts/PaDIS-Reproduction/platforms/slurm" && pwd)"
        else
                echo "Could not locate padis_a100_common.sh. Submit via a PaDIS submit wrapper or set PADIS_SLURM_DIR." >&2
                exit 1
        fi
fi
# shellcheck source=scripts/paper_scripts/PaDIS-Reproduction/platforms/slurm/padis_a100_common.sh
. "$SCRIPT_DIR/padis_a100_common.sh"

padis_setup_modules
padis_activate_environment

LION_ROOT="$(padis_lion_root)"
PADIS_RUN_ROOT="$(padis_default_run_root)"
PADIS_RUN_STAMP="${PADIS_RUN_STAMP:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}}"
PADIS_TRAIN_ROOT="${PADIS_TRAIN_ROOT:-$PADIS_RUN_ROOT/final_real_runs/a100_training_$PADIS_RUN_STAMP}"
PADIS_DATA_FOLDER="${PADIS_DATA_FOLDER:-}"
PADIS_DATA_ROOT="${LION_DATA_PATH:-/home/tjh200/rds/hpc-work/Datasets}"
PADIS_CACHE_ROOT="${PADIS_CACHE_ROOT:-$PADIS_DATA_ROOT/processed/LIDC-IDRI-cache}"
PADIS_256_CACHE_ARCHIVE_FOLDER="${PADIS_256_CACHE_ARCHIVE_FOLDER:-${PADIS_CACHE_ARCHIVE_FOLDER:-$PADIS_CACHE_ROOT/padis_256/archives}}"
PADIS_512_CACHE_ARCHIVE_FOLDER="${PADIS_512_CACHE_ARCHIVE_FOLDER:-$PADIS_CACHE_ROOT/padis_512/archives}"
PADIS_SEED="${PADIS_SEED:-33}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$PADIS_TRAIN_ROOT/matplotlib}"
WANDB_DIR="${WANDB_DIR:-$PADIS_TRAIN_ROOT/wandb}"
export LION_ROOT PADIS_RUN_ROOT PADIS_RUN_STAMP PADIS_TRAIN_ROOT PADIS_DATA_FOLDER
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

mkdir -p "$PADIS_TRAIN_ROOT" "$MPLCONFIGDIR" "$WANDB_DIR"
cd "$LION_ROOT"
padis_print_job_header
echo "Real training task: $TASK_ID $TASK_NAME ($TASK_ENGINE)"

TARGET_PATCHES="${PADIS_TARGET_PATCHES:-400000000}"
MAX_PERIODIC_CHECKPOINTS="${PADIS_MAX_PERIODIC_CHECKPOINTS:-5}"
VALIDATION_HEAVY_SECONDS="${PADIS_VALIDATION_HEAVY_SECONDS:-21600}"
NUM_WORKERS="${PADIS_NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PADIS_PREFETCH_FACTOR:-4}"
NO_WANDB_ARTIFACT="${PADIS_NO_WANDB_ARTIFACT:-0}"
PADIS_WANDB_PROJECT="${PADIS_WANDB_PROJECT:-PaDIS-Reproduction}"
PADIS_WANDB_ENTITY="${PADIS_WANDB_ENTITY:-}"
PADIS_WANDB_MODE="${PADIS_WANDB_MODE:-online}"
PADIS_NO_WANDB="${PADIS_NO_WANDB:-0}"
export PADIS_WANDB_PROJECT PADIS_WANDB_ENTITY PADIS_WANDB_MODE PADIS_NO_WANDB

if [[ "$TASK_NAME" == whole_lidc_* ]]; then
        BASE_TRAIN_SECONDS="${PADIS_WHOLE_BASE_TRAIN_SECONDS:-64800}"
        VALIDATION_INTERVAL="${PADIS_WHOLE_VALIDATION_INTERVAL_PATCHES:-${PADIS_VALIDATION_INTERVAL_PATCHES:-10000}}"
        VALIDATION_MAX_PATCHES="${PADIS_WHOLE_VALIDATION_MAX_PATCHES:-${PADIS_VALIDATION_MAX_PATCHES:-128}}"
        CHECKPOINT_INTERVAL="${PADIS_WHOLE_CHECKPOINT_INTERVAL_PATCHES:-${PADIS_CHECKPOINT_INTERVAL_PATCHES:-25000}}"
        LOG_INTERVAL="${PADIS_WHOLE_LOG_INTERVAL_PATCHES:-${PADIS_LOG_INTERVAL_PATCHES:-128}}"
else
        BASE_TRAIN_SECONDS="${PADIS_PATCH_BASE_TRAIN_SECONDS:-21600}"
        VALIDATION_INTERVAL="${PADIS_VALIDATION_INTERVAL_PATCHES:-200000}"
        VALIDATION_MAX_PATCHES="${PADIS_VALIDATION_MAX_PATCHES:-1000}"
        CHECKPOINT_INTERVAL="${PADIS_CHECKPOINT_INTERVAL_PATCHES:-1000000}"
        LOG_INTERVAL="${PADIS_LOG_INTERVAL_PATCHES:-128}"
fi
INTENSIVE_TRAIN_SECONDS="$VALIDATION_HEAVY_SECONDS"
if [ "$INTENSIVE_TRAIN_SECONDS" -le 0 ]; then
        echo "Validation-heavy duration must be positive." >&2
        exit 2
fi
MAX_TRAIN_SECONDS="$BASE_TRAIN_SECONDS"

wandb_args=()
if [ "$PADIS_NO_WANDB" = "1" ]; then
        wandb_args=(--no-wandb --wandb-mode disabled)
else
        wandb_name="${PADIS_WANDB_NAME_PREFIX:-PaDIS_A100_${PADIS_RUN_STAMP}}_${TASK_NAME}"
        wandb_args=(
                --wandb-project "$PADIS_WANDB_PROJECT"
                --wandb-name "$wandb_name"
                --wandb-mode "$PADIS_WANDB_MODE"
        )
        if [ -n "$PADIS_WANDB_ENTITY" ]; then
                wandb_args+=(--wandb-entity "$PADIS_WANDB_ENTITY")
        fi
        if [ "$NO_WANDB_ARTIFACT" = "1" ]; then
                wandb_args+=(--no-wandb-artifact)
        fi
fi

echo "Target patches: $TARGET_PATCHES"
echo "Validation interval patches: $VALIDATION_INTERVAL (max $VALIDATION_MAX_PATCHES)"
echo "Checkpoint interval patches: $CHECKPOINT_INTERVAL (retain $MAX_PERIODIC_CHECKPOINTS periodic checkpoints)"
echo "Max train seconds: ${MAX_TRAIN_SECONDS:-unset}"
echo "Validation-heavy continuation seconds: $INTENSIVE_TRAIN_SECONDS"
echo "WandB: project=${PADIS_WANDB_PROJECT:-unset} mode=${PADIS_WANDB_MODE:-unset} disabled=${PADIS_NO_WANDB}"

if [ "$TASK_ENGINE" = "lidc256" ]; then
        CMD=(
                python -u scripts/paper_scripts/PaDIS-Reproduction/training/PaDIS_LIDC_256.py
                --save-folder "$PADIS_TRAIN_ROOT"
                --device cuda
                --target-patches "$TARGET_PATCHES"
                --validation-interval-patches "$VALIDATION_INTERVAL"
                --validation-max-patches "$VALIDATION_MAX_PATCHES"
                --checkpoint-interval-patches "$CHECKPOINT_INTERVAL"
                --max-periodic-checkpoints "$MAX_PERIODIC_CHECKPOINTS"
                --log-interval-patches "$LOG_INTERVAL"
                --seed "$PADIS_SEED"
                --batch-size "$TASK_BATCH_SIZE"
                --num-workers "$NUM_WORKERS"
                --prefetch-factor "$PREFETCH_FACTOR"
        )
        CMD+=("${wandb_args[@]}")
        if [ -n "$PADIS_DATA_FOLDER" ]; then
                CMD+=(--data-folder "$PADIS_DATA_FOLDER")
        fi
        if [ -n "${PADIS_MICROBATCH_SIZE:-}" ]; then
                CMD+=(--microbatch-size "$PADIS_MICROBATCH_SIZE")
        fi
        if [ -n "$MAX_TRAIN_SECONDS" ]; then
                CMD+=(--max-train-seconds "$MAX_TRAIN_SECONDS")
        fi
        CACHE_DATASET="${PADIS_256_CACHE_DATASET:-${PADIS_CACHE_DATASET:-ramdisk}}"
        if [[ "$TASK_NAME" == *full ]] && [ "${PADIS_CACHE_FULL_LIDC:-1}" != "1" ]; then
                CACHE_DATASET="none"
        fi
        if [ "$CACHE_DATASET" != "none" ]; then
                CACHE_FOLDER="${PADIS_256_CACHE_FOLDER:-${PADIS_CACHE_FOLDER:-/ramdisks/$USER/lion_lidc_cache}}"
                CMD+=(
                        --cache-dataset "$CACHE_DATASET"
                        --cache-folder "$CACHE_FOLDER"
                        --cache-archive-folder "$PADIS_256_CACHE_ARCHIVE_FOLDER"
                )
                CACHE_SOURCE_FOLDER="${PADIS_256_CACHE_SOURCE_FOLDER:-${PADIS_CACHE_SOURCE_FOLDER:-}}"
                if [ -n "$CACHE_SOURCE_FOLDER" ]; then
                        CMD+=(--cache-source-folder "$CACHE_SOURCE_FOLDER")
                fi
                if [ "${PADIS_REQUIRE_CACHE_HIT:-1}" = "1" ]; then
                        CMD+=(--require-cache-hit)
                fi
                if [ "${PADIS_WRITE_CACHE_ARCHIVE:-0}" = "1" ]; then
                        CMD+=(--write-cache-archive)
                fi
        fi
elif [ "$TASK_ENGINE" = "lidc512" ]; then
        CMD=(
                python -u scripts/paper_scripts/PaDIS-Reproduction/training/PaDIS_LIDC_512.py
                --save-folder "$PADIS_TRAIN_ROOT"
                --device cuda
                --target-patches "$TARGET_PATCHES"
                --validation-interval-patches "$VALIDATION_INTERVAL"
                --validation-max-patches "$VALIDATION_MAX_PATCHES"
                --checkpoint-interval-patches "$CHECKPOINT_INTERVAL"
                --max-periodic-checkpoints "$MAX_PERIODIC_CHECKPOINTS"
                --log-interval-patches "$LOG_INTERVAL"
                --seed "$PADIS_SEED"
                --batch-size "$TASK_BATCH_SIZE"
                --num-workers "${PADIS_512_NUM_WORKERS:-$NUM_WORKERS}"
                --prefetch-factor "${PADIS_512_PREFETCH_FACTOR:-$PREFETCH_FACTOR}"
        )
        CMD+=("${wandb_args[@]}")
        if [ -n "$PADIS_DATA_FOLDER" ]; then
                CMD+=(--data-folder "$PADIS_DATA_FOLDER")
        fi
        if [ -n "${PADIS_MICROBATCH_SIZE:-}" ]; then
                CMD+=(--microbatch-size "$PADIS_MICROBATCH_SIZE")
        fi
        if [ -n "$MAX_TRAIN_SECONDS" ]; then
                CMD+=(--max-train-seconds "$MAX_TRAIN_SECONDS")
        fi
        CACHE_DATASET="${PADIS_512_CACHE_DATASET:-${PADIS_CACHE_DATASET:-ramdisk}}"
        if [[ " ${TASK_ARGS[*]} " == *" --full-lidc "* ]] && [ "${PADIS_CACHE_FULL_512_LIDC:-0}" != "1" ]; then
                CACHE_DATASET="none"
        fi
        if [ "$CACHE_DATASET" != "none" ]; then
                CACHE_FOLDER="${PADIS_512_CACHE_FOLDER:-/ramdisks/$USER/lion_lidc_cache_512}"
                CMD+=(
                        --cache-dataset "$CACHE_DATASET"
                        --cache-folder "$CACHE_FOLDER"
                        --cache-archive-folder "$PADIS_512_CACHE_ARCHIVE_FOLDER"
                )
                if [ -n "${PADIS_512_CACHE_SOURCE_FOLDER:-}" ]; then
                        CMD+=(--cache-source-folder "$PADIS_512_CACHE_SOURCE_FOLDER")
                fi
                if [ "${PADIS_REQUIRE_CACHE_HIT:-1}" = "1" ]; then
                        CMD+=(--require-cache-hit)
                fi
                if [ "${PADIS_WRITE_CACHE_ARCHIVE:-0}" = "1" ]; then
                        CMD+=(--write-cache-archive)
                fi
        fi
else
        echo "Unknown task engine: $TASK_ENGINE"
        exit 1
fi

CMD+=("${TASK_ARGS[@]}")

echo "Executing real training command:"
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

set_command_arg() {
        local flag="$1" value="$2" index
        for index in "${!CMD[@]}"; do
                if [ "${CMD[$index]}" = "$flag" ]; then
                        CMD[$((index + 1))]="$value"
                        return 0
                fi
        done
        CMD+=("$flag" "$value")
}

if [[ "$TASK_NAME" == whole_lidc_* ]]; then
        INTENSIVE_VALIDATION_INTERVAL="${PADIS_WHOLE_VALIDATION_HEAVY_INTERVAL_PATCHES:-2500}"
        INTENSIVE_VALIDATION_MAX="${PADIS_WHOLE_VALIDATION_HEAVY_MAX_PATCHES:-328}"
        VALIDATION_NAME="whole_image_lidc_256_min_intense_val.pt"
elif [ "$TASK_ENGINE" = "lidc512" ]; then
        INTENSIVE_VALIDATION_INTERVAL="${PADIS_PATCH_VALIDATION_HEAVY_INTERVAL_PATCHES:-20000}"
        INTENSIVE_VALIDATION_MAX="${PADIS_PATCH_VALIDATION_HEAVY_MAX_PATCHES:-4000}"
        VALIDATION_NAME="padis_lidc_512_min_intense_val.pt"
else
        INTENSIVE_VALIDATION_INTERVAL="${PADIS_PATCH_VALIDATION_HEAVY_INTERVAL_PATCHES:-20000}"
        INTENSIVE_VALIDATION_MAX="${PADIS_PATCH_VALIDATION_HEAVY_MAX_PATCHES:-4000}"
        VALIDATION_NAME="padis_lidc_256_min_intense_val.pt"
fi

set_command_arg --validation-interval-patches "$INTENSIVE_VALIDATION_INTERVAL"
set_command_arg --validation-max-patches "$INTENSIVE_VALIDATION_MAX"
set_command_arg --max-train-seconds "$INTENSIVE_TRAIN_SECONDS"
CMD+=(
        --validation-name "$VALIDATION_NAME"
        --validation-summary-key min_intense_validation_loss
        --validation-checkpoint-summary-key min_intense_validation_checkpoint
        --validation-repeat-until-max-patches
)

echo "Executing validation-heavy continuation command:"
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

echo "Real training task $TASK_NAME completed at $(date)."
