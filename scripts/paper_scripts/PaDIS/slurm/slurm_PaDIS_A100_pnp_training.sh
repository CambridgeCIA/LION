#!/bin/bash
#!
#! Train the DRUNet denoiser used by the PaDIS paper PnP-ADMM comparison.
#!
#SBATCH -J PaDIS_pnp
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mail-type=NONE
#SBATCH -p ampere
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$SCRIPT_DIR/padis_a100_common.sh" ]; then
        if [ -n "${PADIS_SLURM_DIR:-}" ] && [ -f "$PADIS_SLURM_DIR/padis_a100_common.sh" ]; then
                SCRIPT_DIR="$(cd "$PADIS_SLURM_DIR" && pwd)"
        elif [ -n "${LION_ROOT:-}" ] && [ -f "$LION_ROOT/scripts/paper_scripts/PaDIS/slurm/padis_a100_common.sh" ]; then
                SCRIPT_DIR="$(cd "$LION_ROOT/scripts/paper_scripts/PaDIS/slurm" && pwd)"
        else
                echo "Could not locate padis_a100_common.sh. Set PADIS_SLURM_DIR or LION_ROOT." >&2
                exit 1
        fi
fi
# shellcheck source=scripts/paper_scripts/PaDIS/slurm/padis_a100_common.sh
. "$SCRIPT_DIR/padis_a100_common.sh"

padis_setup_modules
export LION_MAMBA_ENV="${LION_MAMBA_ENV:-lion-dev}"
export LION_MAMBA_ENV_FALLBACKS="${LION_MAMBA_ENV_FALLBACKS:-padis-dev}"
padis_activate_environment

LION_ROOT="$(padis_lion_root)"
PADIS_RUN_ROOT="$(padis_default_run_root)"
PADIS_TRAIN_ROOT="${PADIS_TRAIN_ROOT:-$PADIS_RUN_ROOT/final_real_runs/a100_training_${PADIS_RUN_STAMP:-${SLURM_JOB_ID:-manual}}}"
PADIS_PNP_OUTPUT_ROOT="${PADIS_PNP_OUTPUT_ROOT:-$PADIS_TRAIN_ROOT}"
PADIS_PNP_RUN_NAME="${PADIS_PNP_RUN_NAME:-pnp_lidc_drunet}"
PADIS_PNP_BATCH_SIZE="${PADIS_PNP_BATCH_SIZE:-8}"
PADIS_PNP_EPOCHS="${PADIS_PNP_EPOCHS:-100}"
PADIS_PNP_LR="${PADIS_PNP_LR:-1e-4}"
PADIS_PNP_BETA1="${PADIS_PNP_BETA1:-0.9}"
PADIS_PNP_BETA2="${PADIS_PNP_BETA2:-0.99}"
PADIS_PNP_NOISE_MIN="${PADIS_PNP_NOISE_MIN:-0.0}"
PADIS_PNP_NOISE_MAX="${PADIS_PNP_NOISE_MAX:-0.05}"
PADIS_PNP_IMAGE_SCALING="${PADIS_PNP_IMAGE_SCALING:-0.5}"
PADIS_PNP_MAX_SLICES_PER_PATIENT="${PADIS_PNP_MAX_SLICES_PER_PATIENT:-4}"
PADIS_PNP_MAX_TRAIN_SAMPLES="${PADIS_PNP_MAX_TRAIN_SAMPLES:-}"
PADIS_PNP_MAX_VALIDATION_SAMPLES="${PADIS_PNP_MAX_VALIDATION_SAMPLES:-}"
PADIS_PNP_FULL_LIDC="${PADIS_PNP_FULL_LIDC:-0}"
PADIS_PNP_USE_NOISE_LEVEL="${PADIS_PNP_USE_NOISE_LEVEL:-0}"
PADIS_PNP_INT_CHANNELS="${PADIS_PNP_INT_CHANNELS:-64}"
PADIS_PNP_N_BLOCKS="${PADIS_PNP_N_BLOCKS:-4}"
PADIS_PNP_PATCH_SIZE="${PADIS_PNP_PATCH_SIZE:-}"
PADIS_PNP_PATCHES_PER_IMAGE="${PADIS_PNP_PATCHES_PER_IMAGE:-1}"
PADIS_PNP_VALIDATION_EVERY="${PADIS_PNP_VALIDATION_EVERY:-10}"
PADIS_PNP_CHECKPOINT_EVERY="${PADIS_PNP_CHECKPOINT_EVERY:-10}"
PADIS_PNP_SEED="${PADIS_PNP_SEED:-33}"
PADIS_PNP_DEVICE="${PADIS_PNP_DEVICE:-cuda}"
PADIS_PNP_NUM_WORKERS="${PADIS_PNP_NUM_WORKERS:-4}"
PADIS_PNP_FINAL_NAME="${PADIS_PNP_FINAL_NAME:-pnp_lidc_drunet.pt}"
PADIS_PNP_CHECKPOINT_PATTERN="${PADIS_PNP_CHECKPOINT_PATTERN:-pnp_lidc_drunet_check_*.pt}"
PADIS_PNP_VALIDATION_NAME="${PADIS_PNP_VALIDATION_NAME:-pnp_lidc_drunet_min_val.pt}"
PADIS_DATA_FOLDER="${PADIS_DATA_FOLDER:-}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$PADIS_TRAIN_ROOT/matplotlib}"
export LION_ROOT PADIS_RUN_ROOT PADIS_TRAIN_ROOT MPLCONFIGDIR PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1

mkdir -p "$PADIS_TRAIN_ROOT" "$MPLCONFIGDIR"
cd "$LION_ROOT"
padis_print_job_header

CMD=(
        python -u scripts/paper_scripts/PaDIS/PaDIS_LIDC_PnP_denoiser.py
        --output-root "$PADIS_PNP_OUTPUT_ROOT"
        --run-name "$PADIS_PNP_RUN_NAME"
        --batch-size "$PADIS_PNP_BATCH_SIZE"
        --epochs "$PADIS_PNP_EPOCHS"
        --learning-rate "$PADIS_PNP_LR"
        --beta1 "$PADIS_PNP_BETA1"
        --beta2 "$PADIS_PNP_BETA2"
        --noise-min "$PADIS_PNP_NOISE_MIN"
        --noise-max "$PADIS_PNP_NOISE_MAX"
        --image-scaling "$PADIS_PNP_IMAGE_SCALING"
        --max-slices-per-patient "$PADIS_PNP_MAX_SLICES_PER_PATIENT"
        --int-channels "$PADIS_PNP_INT_CHANNELS"
        --n-blocks "$PADIS_PNP_N_BLOCKS"
        --patches-per-image "$PADIS_PNP_PATCHES_PER_IMAGE"
        --validation-every "$PADIS_PNP_VALIDATION_EVERY"
        --checkpoint-every "$PADIS_PNP_CHECKPOINT_EVERY"
        --seed "$PADIS_PNP_SEED"
        --device "$PADIS_PNP_DEVICE"
        --num-workers "$PADIS_PNP_NUM_WORKERS"
        --final-name "$PADIS_PNP_FINAL_NAME"
        --checkpoint-pattern "$PADIS_PNP_CHECKPOINT_PATTERN"
        --validation-name "$PADIS_PNP_VALIDATION_NAME"
)

if [ "$PADIS_PNP_FULL_LIDC" = "1" ]; then
        CMD+=(--full-lidc)
fi
if [ -n "$PADIS_PNP_MAX_TRAIN_SAMPLES" ]; then
        CMD+=(--max-train-samples "$PADIS_PNP_MAX_TRAIN_SAMPLES")
fi
if [ -n "$PADIS_PNP_MAX_VALIDATION_SAMPLES" ]; then
        CMD+=(--max-validation-samples "$PADIS_PNP_MAX_VALIDATION_SAMPLES")
fi
if [ "$PADIS_PNP_USE_NOISE_LEVEL" = "1" ]; then
        CMD+=(--use-noise-level)
fi
if [ -n "$PADIS_PNP_PATCH_SIZE" ]; then
        CMD+=(--patch-size "$PADIS_PNP_PATCH_SIZE")
fi
if [ -n "$PADIS_DATA_FOLDER" ]; then
        CMD+=(--data-folder "$PADIS_DATA_FOLDER")
fi

echo "Executing PnP denoiser training:"
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

echo "PnP denoiser training completed at $(date)."
