#!/bin/bash
#
# Build reusable PaDIS LIDC image-prior tensor caches locally, without Slurm.
# By default this mirrors the one-time Slurm cache job and writes .pt caches plus
# zstd-compressed .pt.zst archives for the training variants.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LION_ROOT="${LION_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

choose_python() {
        if [ -n "${PYTHON:-}" ]; then
                printf '%s\n' "$PYTHON"
                return
        fi
        if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
                printf '%s\n' "$CONDA_PREFIX/bin/python"
                return
        fi
        if [ -x /mnt/data/conda/envs/lion/bin/python ]; then
                printf '%s\n' /mnt/data/conda/envs/lion/bin/python
                return
        fi
        if command -v python >/dev/null 2>&1; then
                command -v python
                return
        fi
        if command -v python3 >/dev/null 2>&1; then
                command -v python3
                return
        fi
        echo "Could not find Python. Set PYTHON=/path/to/python." >&2
        exit 1
}

default_data_root() {
        if [ -n "${LION_DATA_PATH:-}" ]; then
                printf '%s\n' "$LION_DATA_PATH"
        elif [ -d /mnt/data/Datasets ]; then
                printf '%s\n' /mnt/data/Datasets
        else
                printf '%s\n' "$LION_ROOT/../Data"
        fi
}

default_run_root() {
        if [ -n "${PADIS_RUN_ROOT:-}" ]; then
                printf '%s\n' "$PADIS_RUN_ROOT"
        elif [ -n "${LION_EXPERIMENTS_PATH:-}" ]; then
                printf '%s\n' "$LION_EXPERIMENTS_PATH/PaDIS"
        elif [ -n "${LION_DATA_PATH:-}" ]; then
                printf '%s\n' "$LION_DATA_PATH/experiments/PaDIS"
        elif [ -d /mnt/data/Datasets ]; then
                printf '%s\n' /mnt/data/Datasets/experiments/PaDIS
        else
                printf '%s\n' "$LION_ROOT/../Data/experiments/PaDIS"
        fi
}

PYTHON_BIN="$(choose_python)"
LION_DATA_PATH="${LION_DATA_PATH:-$(default_data_root)}"
PADIS_RUN_ROOT="$(default_run_root)"
PADIS_RUN_STAMP="${PADIS_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
PADIS_CACHE_ROOT="${PADIS_CACHE_ROOT:-$LION_DATA_PATH/processed/LIDC-IDRI-cache}"
PADIS_256_CACHE_BUILD_FOLDER="${PADIS_256_CACHE_BUILD_FOLDER:-${PADIS_CACHE_BUILD_FOLDER:-$PADIS_CACHE_ROOT/padis_256}}"
PADIS_512_CACHE_BUILD_FOLDER="${PADIS_512_CACHE_BUILD_FOLDER:-$PADIS_CACHE_ROOT/padis_512}"
PADIS_256_CACHE_ARCHIVE_FOLDER="${PADIS_256_CACHE_ARCHIVE_FOLDER:-${PADIS_CACHE_ARCHIVE_FOLDER:-$PADIS_256_CACHE_BUILD_FOLDER/archives}}"
PADIS_512_CACHE_ARCHIVE_FOLDER="${PADIS_512_CACHE_ARCHIVE_FOLDER:-$PADIS_512_CACHE_BUILD_FOLDER/archives}"
PADIS_CACHE_PREP_VARIANTS="${PADIS_CACHE_PREP_VARIANTS:-256-default,256-full,512-default}"
PADIS_WRITE_CACHE_ARCHIVE="${PADIS_WRITE_CACHE_ARCHIVE:-1}"
PADIS_REBUILD_CACHE="${PADIS_REBUILD_CACHE:-0}"
PADIS_DATA_FOLDER="${PADIS_DATA_FOLDER:-}"
PADIS_DEVICE="${PADIS_DEVICE:-cpu}"
PADIS_SEED="${PADIS_SEED:-33}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$PADIS_RUN_ROOT/cache_builds/matplotlib_$PADIS_RUN_STAMP}"
WANDB_DIR="${WANDB_DIR:-$PADIS_RUN_ROOT/cache_builds/wandb_$PADIS_RUN_STAMP}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PADIS_RUN_ROOT/cache_builds/xdg_$PADIS_RUN_STAMP}"

export LION_ROOT LION_DATA_PATH PADIS_RUN_ROOT PADIS_RUN_STAMP
export MPLCONFIGDIR WANDB_DIR XDG_CACHE_HOME PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export PYTHONHASHSEED="$PADIS_SEED"
export PYTHONPATH="$LION_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [ "$PADIS_WRITE_CACHE_ARCHIVE" = "1" ] && ! command -v zstd >/dev/null 2>&1; then
        echo "PADIS_WRITE_CACHE_ARCHIVE=1 requires zstd on PATH." >&2
        exit 1
fi

mkdir -p \
        "$PADIS_256_CACHE_BUILD_FOLDER" \
        "$PADIS_512_CACHE_BUILD_FOLDER" \
        "$PADIS_256_CACHE_ARCHIVE_FOLDER" \
        "$PADIS_512_CACHE_ARCHIVE_FOLDER" \
        "$MPLCONFIGDIR" \
        "$WANDB_DIR" \
        "$XDG_CACHE_HOME"

cd "$LION_ROOT"

echo "Preparing PaDIS LIDC caches locally"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "Python: $PYTHON_BIN"
echo "Data root: $LION_DATA_PATH"
echo "Variants: $PADIS_CACHE_PREP_VARIANTS"
echo "256 cache folder: $PADIS_256_CACHE_BUILD_FOLDER"
echo "256 archive folder: $PADIS_256_CACHE_ARCHIVE_FOLDER"
echo "512 cache folder: $PADIS_512_CACHE_BUILD_FOLDER"
echo "512 archive folder: $PADIS_512_CACHE_ARCHIVE_FOLDER"
echo "Write archive: $PADIS_WRITE_CACHE_ARCHIVE"
echo "Rebuild cache: $PADIS_REBUILD_CACHE"

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
                        script_path="scripts/paper_scripts/PaDIS/PaDIS_LIDC_256.py"
                        cache_args=(--max-slices-per-patient 4)
                        ;;
                full|256-full)
                        engine="256"
                        run_name="prepare_full_lidc_cache"
                        save_name="full_lidc_256"
                        cache_folder="$PADIS_256_CACHE_BUILD_FOLDER"
                        archive_folder="$PADIS_256_CACHE_ARCHIVE_FOLDER"
                        script_path="scripts/paper_scripts/PaDIS/PaDIS_LIDC_256.py"
                        cache_args=(--full-lidc)
                        ;;
                512-default)
                        engine="512"
                        run_name="prepare_default_lidc_512_cache"
                        save_name="default_lidc_512"
                        cache_folder="$PADIS_512_CACHE_BUILD_FOLDER"
                        archive_folder="$PADIS_512_CACHE_ARCHIVE_FOLDER"
                        script_path="scripts/paper_scripts/PaDIS/PaDIS_LIDC_512.py"
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
                "$PYTHON_BIN" -u "$script_path"
                --device "$PADIS_DEVICE"
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
        if [ "$PADIS_REBUILD_CACHE" = "1" ]; then
                CMD+=(--rebuild-cache)
        fi

        echo
        echo "Executing cache preparation command for $engine variant $variant:"
        printf '%q ' "${CMD[@]}"
        printf '\n'
        "${CMD[@]}"
}

IFS=',' read -r -a CACHE_VARIANTS <<< "$PADIS_CACHE_PREP_VARIANTS"
for variant in "${CACHE_VARIANTS[@]}"; do
        variant="${variant//[[:space:]]/}"
        if [ -n "$variant" ]; then
                prepare_cache_variant "$variant"
        fi
done

echo
echo "PaDIS LIDC local cache preparation completed at $(date)."
