#!/bin/bash
#!
#! Slurm gate job for PaDIS A100 reproduction checks.
#!
#SBATCH -J PaDIS_checks
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
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
padis_activate_environment

LION_ROOT="$(padis_lion_root)"
PADIS_RUN_ROOT="$(padis_default_run_root)"
PADIS_RUN_STAMP="${PADIS_RUN_STAMP:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
PADIS_CHECK_DIR="${PADIS_CHECK_DIR:-$PADIS_RUN_ROOT/debug_runs/a100_checks_$PADIS_RUN_STAMP}"
PADIS_ROOT="${PADIS_ROOT:-$LION_ROOT/../PaDIS}"
PADIS_GOLDEN="${PADIS_GOLDEN:-$PADIS_CHECK_DIR/padis_lion_golden.pt}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$PADIS_CHECK_DIR/matplotlib}"
WANDB_DIR="${WANDB_DIR:-$PADIS_CHECK_DIR/wandb}"
export LION_ROOT PADIS_RUN_ROOT PADIS_RUN_STAMP PADIS_CHECK_DIR PADIS_ROOT PADIS_GOLDEN
export MPLCONFIGDIR WANDB_DIR PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1

mkdir -p "$PADIS_CHECK_DIR" "$MPLCONFIGDIR" "$WANDB_DIR"
cd "$LION_ROOT"
padis_print_job_header

echo "LION_ROOT=$LION_ROOT"
echo "PADIS_ROOT=$PADIS_ROOT"
echo "PADIS_CHECK_DIR=$PADIS_CHECK_DIR"

python - <<'PY'
import matplotlib
import torch

print("matplotlib", matplotlib.__version__)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible to PyTorch.")
print("cuda_device", torch.cuda.get_device_name(0))
PY

python -m py_compile \
        scripts/dev/check_padis_repo_equivalence.py \
        scripts/dev/check_padis_short_run_reproduction.py \
        scripts/dev/run_padis_machine_preflight.py \
        scripts/dev/run_padis_training_smoke.py \
        scripts/paper_scripts/PaDIS/PaDIS_LIDC_256.py \
        scripts/paper_scripts/PaDIS/PaDIS_LIDC_512.py \
        scripts/paper_scripts/PaDIS/PaDIS_experiments.py

python -m pytest \
        tests/models/test_padis_reconstructor.py \
        tests/models/test_padis_training.py \
        tests/experiments/test_padis_ct_experiments.py

python scripts/dev/check_padis_repo_equivalence.py \
        --padis-root "$PADIS_ROOT" \
        --device cpu \
        --write-golden "$PADIS_GOLDEN"

python scripts/dev/check_padis_repo_equivalence.py \
        --padis-root "$PADIS_ROOT" \
        --device cpu \
        --golden "$PADIS_GOLDEN" \
        > "$PADIS_CHECK_DIR/golden_equivalence.json"

python scripts/dev/check_padis_short_run_reproduction.py \
        --padis-root "$PADIS_ROOT" \
        --device cuda \
        --seeds ${PADIS_CHECK_SHORT_RUN_SEEDS:-2026} \
        --steps "${PADIS_CHECK_SHORT_RUN_STEPS:-3}" \
        --patch-sizes 16 32 56 \
        --relative-tolerance 0.005 \
        --json "$PADIS_CHECK_DIR/short_run_reproduction_cuda.json" \
        > "$PADIS_CHECK_DIR/short_run_reproduction_cuda.stdout.json"

PREFLIGHT_ARGS=(
        --device cuda
        --mode-set all
        --base-batch-size "${PADIS_PREFLIGHT_BASE_BATCH_SIZE:-1}"
        --microbatch-size "${PADIS_PREFLIGHT_MICROBATCH_SIZE:-1}"
        --training-steps "${PADIS_PREFLIGHT_TRAINING_STEPS:-1}"
        --validation-batch-size "${PADIS_PREFLIGHT_VALIDATION_BATCH_SIZE:-1}"
        --validation-batches "${PADIS_PREFLIGHT_VALIDATION_BATCHES:-4}"
        --max-slices-per-patient "${PADIS_PREFLIGHT_MAX_SLICES_PER_PATIENT:-4}"
        --num-workers "${PADIS_PREFLIGHT_NUM_WORKERS:-4}"
        --prefetch-factor "${PADIS_PREFLIGHT_PREFETCH_FACTOR:-2}"
        --padis-root "$PADIS_ROOT"
        --golden "$PADIS_GOLDEN"
        --short-run-relative-tolerance 0.005
        --run-cli-smoke
        --cli-target-patches "${PADIS_PREFLIGHT_CLI_TARGET_PATCHES:-8}"
        --output-dir "$PADIS_CHECK_DIR/preflight"
        --json "$PADIS_CHECK_DIR/preflight/preflight_report.json"
)

if [ "${PADIS_PREFLIGHT_FULL_VALIDATION:-0}" = "1" ]; then
        PREFLIGHT_ARGS+=(--full-validation)
fi

python scripts/dev/run_padis_machine_preflight.py "${PREFLIGHT_ARGS[@]}"

echo "PaDIS A100 checks completed at $(date)."
