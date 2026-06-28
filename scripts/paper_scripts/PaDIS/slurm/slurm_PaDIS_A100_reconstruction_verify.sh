#!/bin/bash
#!
#! Verify PaDIS reconstruction matrix outputs after a Slurm reconstruction array.
#!
#SBATCH -J PaDIS_recon_verify
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --mail-type=NONE
#SBATCH -p icelake
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
PADIS_RECON_ROOT="${PADIS_RECON_ROOT:?PADIS_RECON_ROOT must point at reconstruction outputs.}"
PADIS_RECON_VERIFY_METHODS="${PADIS_RECON_VERIFY_METHODS:-${PADIS_RECON_METHODS:-}}"
PADIS_RECON_VERIFY_EXPERIMENTS="${PADIS_RECON_VERIFY_EXPERIMENTS:-${PADIS_RECON_EXPERIMENTS:-}}"
PADIS_RECON_VERIFY_IMPLEMENTATIONS="${PADIS_RECON_VERIFY_IMPLEMENTATIONS:-}"
PADIS_RECON_VERIFY_GEOMETRIES="${PADIS_RECON_VERIFY_GEOMETRIES:-${PADIS_RECON_GEOMETRIES:-}}"
PADIS_RECON_EXPECTED_RECORDS="${PADIS_RECON_EXPECTED_RECORDS:-}"
PADIS_RECON_EXPECTED_SAMPLES="${PADIS_RECON_EXPECTED_SAMPLES:-}"
PADIS_RECON_EXPECTED_JOBS_JSON="${PADIS_RECON_EXPECTED_JOBS_JSON:-}"
PADIS_RECON_VERIFY_JSON="${PADIS_RECON_VERIFY_JSON:-$PADIS_RECON_ROOT/reconstruction_matrix_verification.json}"
PADIS_RECON_VERIFY_MIN_METHOD_MEAN_PSNR="${PADIS_RECON_VERIFY_MIN_METHOD_MEAN_PSNR:-${PADIS_RECON_VERIFY_MIN_METHOD_PSNR:-}}"
PADIS_RECON_VERIFY_MIN_METHOD_MEAN_SSIM="${PADIS_RECON_VERIFY_MIN_METHOD_MEAN_SSIM:-${PADIS_RECON_VERIFY_MIN_METHOD_SSIM:-}}"
PADIS_RECON_VERIFY_MAX_METHOD_MEAN_MAE="${PADIS_RECON_VERIFY_MAX_METHOD_MEAN_MAE:-${PADIS_RECON_VERIFY_MAX_METHOD_MAE:-}}"
export LION_ROOT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1

cd "$LION_ROOT"
padis_print_job_header

CMD=(
        python -u scripts/paper_scripts/PaDIS/PaDIS_verify_reconstruction_matrix.py
        --root "$PADIS_RECON_ROOT"
        --output-json "$PADIS_RECON_VERIFY_JSON"
)

if [ -n "$PADIS_RECON_VERIFY_METHODS" ] && [ "$PADIS_RECON_VERIFY_METHODS" != "all" ]; then
        CMD+=(--methods "$PADIS_RECON_VERIFY_METHODS")
        CMD+=(--require-methods "$PADIS_RECON_VERIFY_METHODS")
fi
if [ -n "$PADIS_RECON_VERIFY_EXPERIMENTS" ] && [ "$PADIS_RECON_VERIFY_EXPERIMENTS" != "paper_matrix" ]; then
        CMD+=(--experiments "$PADIS_RECON_VERIFY_EXPERIMENTS")
        CMD+=(--require-experiments "$PADIS_RECON_VERIFY_EXPERIMENTS")
fi
if [ -n "$PADIS_RECON_VERIFY_IMPLEMENTATIONS" ] && [ "$PADIS_RECON_VERIFY_IMPLEMENTATIONS" != "method_default" ]; then
        CMD+=(--implementations "$PADIS_RECON_VERIFY_IMPLEMENTATIONS")
fi
if [ -n "$PADIS_RECON_VERIFY_GEOMETRIES" ] && [ "$PADIS_RECON_VERIFY_GEOMETRIES" != "all" ]; then
        CMD+=(--geometries "$PADIS_RECON_VERIFY_GEOMETRIES")
fi
if [ -n "$PADIS_RECON_EXPECTED_RECORDS" ]; then
        CMD+=(--expected-records "$PADIS_RECON_EXPECTED_RECORDS")
fi
if [ -n "$PADIS_RECON_EXPECTED_JOBS_JSON" ]; then
        CMD+=(--expected-jobs-json "$PADIS_RECON_EXPECTED_JOBS_JSON")
fi
if [ -n "$PADIS_RECON_EXPECTED_SAMPLES" ]; then
        CMD+=(--expected-samples "$PADIS_RECON_EXPECTED_SAMPLES")
fi
if [ -n "${PADIS_RECON_VERIFY_MIN_MEAN_PSNR:-}" ]; then
        CMD+=(--min-mean-psnr "$PADIS_RECON_VERIFY_MIN_MEAN_PSNR")
fi
if [ -n "${PADIS_RECON_VERIFY_MIN_MEAN_SSIM:-}" ]; then
        CMD+=(--min-mean-ssim "$PADIS_RECON_VERIFY_MIN_MEAN_SSIM")
fi
if [ -n "${PADIS_RECON_VERIFY_MAX_MEAN_MAE:-}" ]; then
        CMD+=(--max-mean-mae "$PADIS_RECON_VERIFY_MAX_MEAN_MAE")
fi
if [ -n "$PADIS_RECON_VERIFY_MIN_METHOD_MEAN_PSNR" ]; then
        read -r -a METHOD_PSNR <<< "$PADIS_RECON_VERIFY_MIN_METHOD_MEAN_PSNR"
        for item in "${METHOD_PSNR[@]}"; do
                CMD+=(--min-method-mean-psnr "$item")
        done
fi
if [ -n "$PADIS_RECON_VERIFY_MIN_METHOD_MEAN_SSIM" ]; then
        read -r -a METHOD_SSIM <<< "$PADIS_RECON_VERIFY_MIN_METHOD_MEAN_SSIM"
        for item in "${METHOD_SSIM[@]}"; do
                CMD+=(--min-method-mean-ssim "$item")
        done
fi
if [ -n "$PADIS_RECON_VERIFY_MAX_METHOD_MEAN_MAE" ]; then
        read -r -a METHOD_MAE <<< "$PADIS_RECON_VERIFY_MAX_METHOD_MEAN_MAE"
        for item in "${METHOD_MAE[@]}"; do
                CMD+=(--max-method-mean-mae "$item")
        done
fi
if [ "${PADIS_RECON_VERIFY_REQUIRE_MEAN_BETTER_THAN_FDK:-0}" = "1" ]; then
        CMD+=(--require-mean-better-than-fdk)
fi
if [ "${PADIS_RECON_VERIFY_REQUIRE_EACH_BETTER_THAN_FDK:-0}" = "1" ]; then
        CMD+=(--require-each-better-than-fdk)
fi
if [ -n "${PADIS_RECON_VERIFY_REQUIRE_METHOD_MEAN_BETTER_THAN_FDK:-}" ]; then
        read -r -a METHOD_MEAN_FDK <<< "$PADIS_RECON_VERIFY_REQUIRE_METHOD_MEAN_BETTER_THAN_FDK"
        for item in "${METHOD_MEAN_FDK[@]}"; do
                CMD+=(--require-method-mean-better-than-fdk "$item")
        done
fi
if [ -n "${PADIS_RECON_VERIFY_REQUIRE_METHOD_EACH_BETTER_THAN_FDK:-}" ]; then
        read -r -a METHOD_EACH_FDK <<< "$PADIS_RECON_VERIFY_REQUIRE_METHOD_EACH_BETTER_THAN_FDK"
        for item in "${METHOD_EACH_FDK[@]}"; do
                CMD+=(--require-method-each-better-than-fdk "$item")
        done
fi

echo "Executing reconstruction verifier:"
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

echo "Reconstruction verification completed at $(date)."
