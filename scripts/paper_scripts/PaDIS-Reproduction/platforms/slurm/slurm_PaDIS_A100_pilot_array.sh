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
#SBATCH --time=00:15:00
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
PADIS_PILOT_ROOT="${PADIS_PILOT_ROOT:-$PADIS_RUN_ROOT/pilot_runs/a100_pilots_$PADIS_RUN_STAMP}"
PADIS_DATA_FOLDER="${PADIS_DATA_FOLDER:-}"
PADIS_DATA_ROOT="${LION_DATA_PATH:-/home/tjh200/rds/hpc-work/Datasets}"
PADIS_CACHE_ROOT="${PADIS_CACHE_ROOT:-$PADIS_DATA_ROOT/processed/LIDC-IDRI-cache}"
PADIS_256_CACHE_ARCHIVE_FOLDER="${PADIS_256_CACHE_ARCHIVE_FOLDER:-${PADIS_CACHE_ARCHIVE_FOLDER:-$PADIS_CACHE_ROOT/padis_256/archives}}"
PADIS_512_CACHE_ARCHIVE_FOLDER="${PADIS_512_CACHE_ARCHIVE_FOLDER:-$PADIS_CACHE_ROOT/padis_512/archives}"
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
VALIDATION_MAX_PATCHES="${PADIS_PILOT_VALIDATION_MAX_PATCHES:-${PADIS_VALIDATION_MAX_PATCHES:-1000}}"
CHECKPOINT_INTERVAL="${PADIS_PILOT_CHECKPOINT_INTERVAL_PATCHES:-$TARGET_PATCHES}"
MAX_PERIODIC_CHECKPOINTS="${PADIS_PILOT_MAX_PERIODIC_CHECKPOINTS:-${PADIS_MAX_PERIODIC_CHECKPOINTS:-5}}"
LOG_INTERVAL="${PADIS_PILOT_LOG_INTERVAL_PATCHES:-128}"
MAX_TRAIN_SECONDS="${PADIS_PILOT_MAX_TRAIN_SECONDS:-840}"
NUM_WORKERS="${PADIS_NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PADIS_PREFETCH_FACTOR:-4}"
NO_WANDB_ARTIFACT="${PADIS_NO_WANDB_ARTIFACT:-0}"
PADIS_WANDB_PROJECT="${PADIS_WANDB_PROJECT:-PaDIS-Reproduction}"
PADIS_WANDB_ENTITY="${PADIS_WANDB_ENTITY:-}"
PADIS_WANDB_MODE="${PADIS_WANDB_MODE:-online}"
PADIS_NO_WANDB="${PADIS_NO_WANDB:-0}"
export PADIS_WANDB_PROJECT PADIS_WANDB_ENTITY PADIS_WANDB_MODE PADIS_NO_WANDB

wandb_args=()
if [ "$PADIS_NO_WANDB" = "1" ] || [ "${PADIS_PILOT_NO_WANDB:-0}" = "1" ]; then
        wandb_args=(--no-wandb --wandb-mode disabled)
else
        wandb_name="${PADIS_PILOT_WANDB_NAME_PREFIX:-PaDIS_A100_pilot_${PADIS_RUN_STAMP}}_${TASK_NAME}"
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

if [ "$TASK_ENGINE" = "lidc256" ]; then
        CMD=(
                python -u scripts/paper_scripts/PaDIS-Reproduction/training/PaDIS_LIDC_256.py
                --save-folder "$PADIS_PILOT_ROOT"
                --device cuda
                --target-patches "$TARGET_PATCHES"
                --validation-interval-patches "$VALIDATION_INTERVAL"
                --validation-max-patches "$VALIDATION_MAX_PATCHES"
                --checkpoint-interval-patches "$CHECKPOINT_INTERVAL"
                --max-periodic-checkpoints "$MAX_PERIODIC_CHECKPOINTS"
                --log-interval-patches "$LOG_INTERVAL"
                --max-train-seconds "$MAX_TRAIN_SECONDS"
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
                --save-folder "$PADIS_PILOT_ROOT"
                --device cuda
                --target-patches "$TARGET_PATCHES"
                --validation-interval-patches "$VALIDATION_INTERVAL"
                --validation-max-patches "$VALIDATION_MAX_PATCHES"
                --checkpoint-interval-patches "$CHECKPOINT_INTERVAL"
                --max-periodic-checkpoints "$MAX_PERIODIC_CHECKPOINTS"
                --log-interval-patches "$LOG_INTERVAL"
                --max-train-seconds "$MAX_TRAIN_SECONDS"
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

echo "Executing pilot command:"
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

echo "Pilot task $TASK_NAME completed at $(date)."
