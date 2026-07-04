#!/bin/bash
#!
#! PaDIS A100 reconstruction array for trained CT priors.
#!
#SBATCH -J PaDIS_recon
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mail-type=NONE
#SBATCH --array=0-0
#SBATCH -p ampere
#SBATCH -o slurm-%x-%A_%a.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$SCRIPT_DIR/padis_a100_common.sh" ]; then
        if [ -n "${PADIS_SLURM_DIR:-}" ] && [ -f "$PADIS_SLURM_DIR/padis_a100_common.sh" ]; then
                SCRIPT_DIR="$(cd "$PADIS_SLURM_DIR" && pwd)"
        elif [ -n "${LION_ROOT:-}" ] && [ -f "$LION_ROOT/scripts/paper_scripts/PaDIS/slurm/padis_a100_common.sh" ]; then
                SCRIPT_DIR="$(cd "$LION_ROOT/scripts/paper_scripts/PaDIS/slurm" && pwd)"
        elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/padis_a100_common.sh" ]; then
                SCRIPT_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
        elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/paper_scripts/PaDIS/slurm/padis_a100_common.sh" ]; then
                SCRIPT_DIR="$(cd "$SLURM_SUBMIT_DIR/scripts/paper_scripts/PaDIS/slurm" && pwd)"
        elif [ -f "$PWD/scripts/paper_scripts/PaDIS/slurm/padis_a100_common.sh" ]; then
                SCRIPT_DIR="$(cd "$PWD/scripts/paper_scripts/PaDIS/slurm" && pwd)"
        else
                echo "Could not locate padis_a100_common.sh. Submit via a PaDIS submit wrapper or set PADIS_SLURM_DIR." >&2
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
PADIS_TRAIN_ROOT="${PADIS_TRAIN_ROOT:?PADIS_TRAIN_ROOT must point at a trained PaDIS A100 training root.}"
PADIS_RECON_ROOT="${PADIS_RECON_ROOT:-$PADIS_RUN_ROOT/final_real_runs/a100_reconstruction_${PADIS_RUN_STAMP:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}}"
PADIS_RECON_MODELS="${PADIS_RECON_MODELS:-method_default}"
PADIS_RECON_METHODS="${PADIS_RECON_METHODS:-all}"
PADIS_RECON_EXPERIMENTS="${PADIS_RECON_EXPERIMENTS:-paper_matrix}"
PADIS_RECON_ABLATIONS="${PADIS_RECON_ABLATIONS:-all}"
PADIS_RECON_ALLOW_OFF_PAPER_EXPERIMENTS="${PADIS_RECON_ALLOW_OFF_PAPER_EXPERIMENTS:-0}"
PADIS_RECON_IMPLEMENTATIONS="${PADIS_RECON_IMPLEMENTATIONS:-method_default}"
PADIS_RECON_GEOMETRIES="${PADIS_RECON_GEOMETRIES:-lion}"
PADIS_RECON_SPLIT="${PADIS_RECON_SPLIT:-test}"
PADIS_RECON_ALGORITHM="${PADIS_RECON_ALGORITHM:-dps_langevin}"
PADIS_RECON_MAX_SAMPLES="${PADIS_RECON_MAX_SAMPLES:-25}"
PADIS_RECON_START_INDEX="${PADIS_RECON_START_INDEX:-0}"
PADIS_RECON_SEED="${PADIS_RECON_SEED:-33}"
PADIS_RECON_DEVICE="${PADIS_RECON_DEVICE:-cuda}"
PADIS_RECON_SAVE_PREVIEWS="${PADIS_RECON_SAVE_PREVIEWS:-1}"
PADIS_RECON_PROG_BAR="${PADIS_RECON_PROG_BAR:-0}"
PADIS_RECON_TRACE_INTERVAL="${PADIS_RECON_TRACE_INTERVAL:-}"
PADIS_RECON_TRACE_IMAGES="${PADIS_RECON_TRACE_IMAGES:-0}"
PADIS_PNP_OUTPUT_ROOT="${PADIS_PNP_OUTPUT_ROOT:-$PADIS_TRAIN_ROOT}"
PADIS_PNP_RUN_NAME="${PADIS_PNP_RUN_NAME:-pnp_lidc_drunet}"
PADIS_PNP_FINAL_NAME="${PADIS_PNP_FINAL_NAME:-pnp_lidc_drunet.pt}"
PADIS_PNP_ROOT="${PADIS_PNP_ROOT:-$PADIS_PNP_OUTPUT_ROOT/$PADIS_PNP_RUN_NAME}"
PADIS_PNP_CHECKPOINT="${PADIS_PNP_CHECKPOINT:-}"
PADIS_PNP_ITERATIONS="${PADIS_PNP_ITERATIONS:-20}"
PADIS_PNP_ETA="${PADIS_PNP_ETA:-1e-5}"
PADIS_PNP_CG_ITERATIONS="${PADIS_PNP_CG_ITERATIONS:-100}"
PADIS_PNP_CG_TOLERANCE="${PADIS_PNP_CG_TOLERANCE:-1e-7}"
PADIS_PNP_NOISE_LEVEL="${PADIS_PNP_NOISE_LEVEL:-}"
PADIS_TV_LAMBDA="${PADIS_TV_LAMBDA:-0.001}"
PADIS_TV_ITERATIONS="${PADIS_TV_ITERATIONS:-500}"
PADIS_DATA_FOLDER="${PADIS_DATA_FOLDER:-}"
PADIS_PUBLIC_IMAGE_DIR="${PADIS_PUBLIC_IMAGE_DIR:-}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$PADIS_RECON_ROOT/matplotlib}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LION_ROOT PADIS_RUN_ROOT PADIS_RECON_ROOT MPLCONFIGDIR
export PYTORCH_CUDA_ALLOC_CONF PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1

mkdir -p "$PADIS_RECON_ROOT" "$MPLCONFIGDIR"
cd "$LION_ROOT"
padis_print_job_header

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
compact_recon_methods="${PADIS_RECON_METHODS//[[:space:]]/}"
if {
        [ "$compact_recon_methods" = "all" ] ||
                [[ ",$compact_recon_methods," == *",pnp_admm,"* ]];
} && [ -z "$PADIS_PNP_CHECKPOINT" ]; then
        PADIS_PNP_CHECKPOINT="$PADIS_PNP_ROOT/$PADIS_PNP_FINAL_NAME"
fi

CMD=(
        python -u scripts/paper_scripts/PaDIS/PaDIS_run_reconstruction_matrix.py
        --training-root "$PADIS_TRAIN_ROOT"
        --output-root "$PADIS_RECON_ROOT"
        --task-index "$TASK_ID"
        --models "$PADIS_RECON_MODELS"
        --methods "$PADIS_RECON_METHODS"
        --experiments "$PADIS_RECON_EXPERIMENTS"
        --ablations "$PADIS_RECON_ABLATIONS"
        --implementations "$PADIS_RECON_IMPLEMENTATIONS"
        --geometries "$PADIS_RECON_GEOMETRIES"
        --split "$PADIS_RECON_SPLIT"
        --algorithm "$PADIS_RECON_ALGORITHM"
        --max-samples "$PADIS_RECON_MAX_SAMPLES"
        --start-index "$PADIS_RECON_START_INDEX"
        --seed "$PADIS_RECON_SEED"
        --device "$PADIS_RECON_DEVICE"
        --pnp-root "$PADIS_PNP_ROOT"
        --pnp-iterations "$PADIS_PNP_ITERATIONS"
        --pnp-eta "$PADIS_PNP_ETA"
        --pnp-cg-iterations "$PADIS_PNP_CG_ITERATIONS"
        --pnp-cg-tolerance "$PADIS_PNP_CG_TOLERANCE"
        --tv-lambda "$PADIS_TV_LAMBDA"
        --tv-iterations "$PADIS_TV_ITERATIONS"
)

if [ -n "$PADIS_PNP_CHECKPOINT" ]; then
        CMD+=(--pnp-checkpoint "$PADIS_PNP_CHECKPOINT")
fi
if [ -n "$PADIS_PNP_NOISE_LEVEL" ]; then
        CMD+=(--pnp-noise-level "$PADIS_PNP_NOISE_LEVEL")
fi
if [ "$PADIS_RECON_ALLOW_OFF_PAPER_EXPERIMENTS" = "1" ]; then
        CMD+=(--allow-off-paper-experiments)
fi
if [ -n "$PADIS_DATA_FOLDER" ]; then
        CMD+=(--data-folder "$PADIS_DATA_FOLDER")
fi
if [ -n "$PADIS_PUBLIC_IMAGE_DIR" ]; then
        CMD+=(--public-padis-image-dir "$PADIS_PUBLIC_IMAGE_DIR")
fi
if [ "$PADIS_RECON_SAVE_PREVIEWS" = "1" ]; then
        CMD+=(--save-previews)
fi
if [ "$PADIS_RECON_PROG_BAR" = "1" ]; then
        CMD+=(--prog-bar)
fi
if [ -n "$PADIS_RECON_TRACE_INTERVAL" ]; then
        CMD+=(--trace-interval "$PADIS_RECON_TRACE_INTERVAL")
fi
if [ "$PADIS_RECON_TRACE_IMAGES" = "1" ]; then
        CMD+=(--trace-images)
fi
if [ -n "${PADIS_RECON_EXTRA_ARGS:-}" ]; then
        read -r -a EXTRA_ARGS <<< "$PADIS_RECON_EXTRA_ARGS"
        for item in "${EXTRA_ARGS[@]}"; do
                CMD+=("--reconstruction-arg=$item")
        done
fi

echo "Executing reconstruction matrix task $TASK_ID:"
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

echo "Reconstruction task $TASK_ID completed at $(date)."
