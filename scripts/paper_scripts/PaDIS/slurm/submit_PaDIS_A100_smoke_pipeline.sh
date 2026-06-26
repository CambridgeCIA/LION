#!/bin/bash
#
# Lightweight PaDIS smoke runner for login nodes.
#
# Despite the historical submit_* name, this script does not call sbatch. It
# performs cheap local checks of the Python import path, the default cached LIDC
# archive lookup/staging path, and an optional one-step synthetic CPU PaDIS smoke.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LION_ROOT="${LION_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
# shellcheck source=scripts/paper_scripts/PaDIS/slurm/padis_a100_common.sh
. "$SCRIPT_DIR/padis_a100_common.sh"
cd "$LION_ROOT"

export LION_DATA_PATH="${LION_DATA_PATH:-/home/tjh200/rds/hpc-work/Datasets}"
export LION_EXPERIMENTS_PATH="${LION_EXPERIMENTS_PATH:-$LION_DATA_PATH/experiments}"
export PADIS_RUN_ROOT="${PADIS_RUN_ROOT:-$LION_EXPERIMENTS_PATH/PaDIS}"
export PADIS_RUN_STAMP="${PADIS_RUN_STAMP:-login_smoke_$(date +%Y%m%d_%H%M%S)}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

padis_setup_modules
padis_activate_environment

OUT_DIR="$PADIS_RUN_ROOT/debug_runs/$PADIS_RUN_STAMP"
CACHE_FOLDER="${PADIS_LOGIN_CACHE_FOLDER:-/tmp/$USER/lion_padis_login_cache_$PADIS_RUN_STAMP}"
mkdir -p "$OUT_DIR" "$OUT_DIR/matplotlib" "$OUT_DIR/wandb"
export MPLCONFIGDIR="$OUT_DIR/matplotlib"
export WANDB_DIR="$OUT_DIR/wandb"

echo "Running PaDIS login-node smoke"
echo "LION_ROOT=$LION_ROOT"
echo "LION_DATA_PATH=$LION_DATA_PATH"
echo "OUT_DIR=$OUT_DIR"
echo "CACHE_FOLDER=$CACHE_FOLDER"
python - <<'PY'
import sys
import torch
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY

python -W error::SyntaxWarning -m py_compile \
        LION/data_loaders/deteCT.py \
        scripts/dev/check_padis_login_cache_smoke.py \
        scripts/dev/run_padis_training_smoke.py \
        scripts/paper_scripts/PaDIS/PaDIS_LIDC_256.py \
        scripts/paper_scripts/PaDIS/PaDIS_LIDC_512.py

python scripts/dev/check_padis_login_cache_smoke.py \
        --cache-folder "$CACHE_FOLDER" \
        --json "$OUT_DIR/cache_smoke.json"

if [ "${PADIS_LOGIN_RUN_TRAINING_SMOKE:-1}" = "1" ]; then
        python scripts/dev/run_padis_training_smoke.py \
                --device cpu \
                --synthetic-data \
                --model-mode "${PADIS_LOGIN_MODEL_MODE:-padis-paper-ct-p8}" \
                --patch-size "${PADIS_LOGIN_PATCH_SIZE:-8}" \
                --batch-size "${PADIS_LOGIN_BATCH_SIZE:-1}" \
                --microbatch-size "${PADIS_LOGIN_MICROBATCH_SIZE:-1}" \
                --steps "${PADIS_LOGIN_STEPS:-1}" \
                --no-ema \
                --json "$OUT_DIR/synthetic_training_smoke.json"
else
        echo "Skipping synthetic CPU training smoke because PADIS_LOGIN_RUN_TRAINING_SMOKE=$PADIS_LOGIN_RUN_TRAINING_SMOKE"
fi

echo "PaDIS login-node smoke completed. Output: $OUT_DIR"
