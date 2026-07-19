#!/bin/bash
#!
#! Build reusable PaDIS LIDC image-prior caches and optional zstd archives.
#! This is CPU data preparation only; it exits before model construction/training.
#!
#SBATCH -J PaDIS_cache_lidc
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --mail-type=NONE
#SBATCH -p icelake
#SBATCH -o slurm-%x-%j.out

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
PADIS_RUN_STAMP="${PADIS_RUN_STAMP:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
PADIS_DATA_ROOT="${LION_DATA_PATH:-/home/tjh200/rds/hpc-work/Datasets}"
PADIS_CACHE_ROOT="${PADIS_CACHE_ROOT:-$PADIS_DATA_ROOT/processed/LIDC-IDRI-cache}"
PADIS_256_CACHE_BUILD_FOLDER="${PADIS_256_CACHE_BUILD_FOLDER:-${PADIS_CACHE_BUILD_FOLDER:-$PADIS_CACHE_ROOT/padis_256}}"
PADIS_512_CACHE_BUILD_FOLDER="${PADIS_512_CACHE_BUILD_FOLDER:-$PADIS_CACHE_ROOT/padis_512}"
PADIS_256_CACHE_ARCHIVE_FOLDER="${PADIS_256_CACHE_ARCHIVE_FOLDER:-${PADIS_CACHE_ARCHIVE_FOLDER:-$PADIS_256_CACHE_BUILD_FOLDER/archives}}"
PADIS_512_CACHE_ARCHIVE_FOLDER="${PADIS_512_CACHE_ARCHIVE_FOLDER:-$PADIS_512_CACHE_BUILD_FOLDER/archives}"
PADIS_CACHE_PREP_VARIANTS="${PADIS_CACHE_PREP_VARIANTS:-256-default,256-full,512-default}"
PADIS_WRITE_CACHE_ARCHIVE="${PADIS_WRITE_CACHE_ARCHIVE:-1}"
PADIS_DATA_FOLDER="${PADIS_DATA_FOLDER:-}"
PADIS_SEED="${PADIS_SEED:-33}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$PADIS_RUN_ROOT/cache_builds/matplotlib_$PADIS_RUN_STAMP}"
WANDB_DIR="${WANDB_DIR:-$PADIS_RUN_ROOT/cache_builds/wandb_$PADIS_RUN_STAMP}"
export LION_ROOT PADIS_RUN_ROOT PADIS_RUN_STAMP PADIS_DATA_FOLDER
export MPLCONFIGDIR WANDB_DIR PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 PYTHONHASHSEED="$PADIS_SEED"

mkdir -p "$PADIS_256_CACHE_BUILD_FOLDER" "$PADIS_512_CACHE_BUILD_FOLDER" "$PADIS_256_CACHE_ARCHIVE_FOLDER" "$PADIS_512_CACHE_ARCHIVE_FOLDER" "$MPLCONFIGDIR" "$WANDB_DIR"
cd "$LION_ROOT"
padis_print_job_header

echo "Preparing PaDIS LIDC caches"
echo "Variants: $PADIS_CACHE_PREP_VARIANTS"
echo "256 cache folder: $PADIS_256_CACHE_BUILD_FOLDER"
echo "256 archive folder: $PADIS_256_CACHE_ARCHIVE_FOLDER"
echo "512 cache folder: $PADIS_512_CACHE_BUILD_FOLDER"
echo "512 archive folder: $PADIS_512_CACHE_ARCHIVE_FOLDER"
echo "Write archive: $PADIS_WRITE_CACHE_ARCHIVE"

prepare_cache_variant() {
        local variant="$1"
        local engine
        local run_name
        local save_name
        local cache_folder
        local archive_folder
        local script_path
        local cache_args=()

        case "$variant" in
                default|256-default)
                        engine="256"
                        run_name="prepare_default_lidc_cache"
                        save_name="default_lidc_256"
                        cache_folder="$PADIS_256_CACHE_BUILD_FOLDER"
                        archive_folder="$PADIS_256_CACHE_ARCHIVE_FOLDER"
                        script_path="scripts/paper_scripts/PaDIS-Reproduction/training/PaDIS_LIDC_256.py"
                        cache_args=(--max-slices-per-patient 4)
                        ;;
                full|256-full)
                        engine="256"
                        run_name="prepare_full_lidc_cache"
                        save_name="full_lidc_256"
                        cache_folder="$PADIS_256_CACHE_BUILD_FOLDER"
                        archive_folder="$PADIS_256_CACHE_ARCHIVE_FOLDER"
                        script_path="scripts/paper_scripts/PaDIS-Reproduction/training/PaDIS_LIDC_256.py"
                        cache_args=(--full-lidc)
                        ;;
                512-default)
                        engine="512"
                        run_name="prepare_default_lidc_512_cache"
                        save_name="default_lidc_512"
                        cache_folder="$PADIS_512_CACHE_BUILD_FOLDER"
                        archive_folder="$PADIS_512_CACHE_ARCHIVE_FOLDER"
                        script_path="scripts/paper_scripts/PaDIS-Reproduction/training/PaDIS_LIDC_512.py"
                        cache_args=(--max-slices-per-patient 4)
                        ;;
                512-full)
                        echo "Refusing to build a full 512x512 LIDC cache by default; it is intentionally unsupported here." >&2
                        exit 1
                        ;;
                *)
                        echo "Unknown PADIS cache prep variant: $variant" >&2
                        exit 1
                        ;;
        esac

        CMD=(
                python -u "$script_path"
                --device cpu
                --save-folder "$PADIS_RUN_ROOT/cache_builds/${save_name}_$PADIS_RUN_STAMP"
                --run-name "$run_name"
                --cache-dataset ramdisk
                --cache-folder "$cache_folder"
                --prepare-cache-only
                --seed "$PADIS_SEED"
                --no-wandb
                --wandb-mode disabled
        )
        CMD+=("${cache_args[@]}")
        if [ -n "$PADIS_DATA_FOLDER" ]; then
                CMD+=(--data-folder "$PADIS_DATA_FOLDER")
        fi
        if [ "$PADIS_WRITE_CACHE_ARCHIVE" = "1" ]; then
                CMD+=(--cache-archive-folder "$archive_folder" --write-cache-archive)
        fi

        echo "Executing cache preparation command for $engine variant $variant:"
        printf '%q ' "${CMD[@]}"
        printf '\n'
        "${CMD[@]}"
}

IFS=',' read -r -a CACHE_VARIANTS <<< "$PADIS_CACHE_PREP_VARIANTS"
for variant in "${CACHE_VARIANTS[@]}"; do
        prepare_cache_variant "$variant"
done

echo "PaDIS LIDC cache preparation completed at $(date)."
