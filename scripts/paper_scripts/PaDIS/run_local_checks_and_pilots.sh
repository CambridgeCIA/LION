#!/bin/bash
#
# Run the PaDIS checks and short pilot training jobs locally, without Slurm.
# This is intended to burn down implementation/runtime risk before submitting
# the expensive A100 training array.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/paper_scripts/PaDIS/slurm/padis_a100_common.sh
. "$SCRIPT_DIR/slurm/padis_a100_common.sh"

LION_ROOT="$(padis_lion_root)"
PYTHON="${LION_PYTHON:-/home/thomas/anaconda3/envs/lion-dev/bin/python}"
if [ ! -x "$PYTHON" ]; then
        PYTHON="$(command -v python)"
fi

PADIS_EXPERIMENT_ROOT="${PADIS_EXPERIMENT_ROOT:-$LION_ROOT/../Data/experiments/PaDIS}"
PADIS_LOCAL_RUN_ID="${PADIS_LOCAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DEFAULT_PADIS_LOCAL_ROOT="$PADIS_EXPERIMENT_ROOT/debug_runs/local_checks_$PADIS_LOCAL_RUN_ID"
DEFAULT_PADIS_LOCAL_PILOT_ROOT="$PADIS_EXPERIMENT_ROOT/pilot_runs/local_pilots_$PADIS_LOCAL_RUN_ID"
PADIS_LOCAL_ROOT="${PADIS_LOCAL_ROOT:-$DEFAULT_PADIS_LOCAL_ROOT}"
PADIS_LOCAL_PILOT_ROOT="${PADIS_LOCAL_PILOT_ROOT:-$DEFAULT_PADIS_LOCAL_PILOT_ROOT}"
PADIS_LOCAL_MAX_SECONDS="${PADIS_LOCAL_MAX_SECONDS:-3600}"
PADIS_LOCAL_MIN_COMMAND_SECONDS="${PADIS_LOCAL_MIN_COMMAND_SECONDS:-180}"
PADIS_LOCAL_DEVICE="${PADIS_LOCAL_DEVICE:-cuda}"
PADIS_ROOT="${PADIS_ROOT:-$LION_ROOT/../PaDIS}"
PADIS_GOLDEN="${PADIS_GOLDEN:-$PADIS_LOCAL_ROOT/padis_lion_golden.pt}"
PADIS_LOCAL_PATCH_BATCH_SIZE="${PADIS_LOCAL_PATCH_BATCH_SIZE:-24}"
PADIS_LOCAL_P96_BATCH_SIZE="${PADIS_LOCAL_P96_BATCH_SIZE:-8}"
PADIS_LOCAL_WHOLE_BATCH_SIZE="${PADIS_LOCAL_WHOLE_BATCH_SIZE:-1}"
PADIS_LOCAL_512_BATCH_SIZE="${PADIS_LOCAL_512_BATCH_SIZE:-16}"
PADIS_LOCAL_BATCH_SIZE="${PADIS_LOCAL_BATCH_SIZE:-$PADIS_LOCAL_PATCH_BATCH_SIZE}"
PADIS_LOCAL_PREFLIGHT_BATCH_SIZE="${PADIS_LOCAL_PREFLIGHT_BATCH_SIZE:-8}"
PADIS_LOCAL_PREFLIGHT_MICROBATCH_SIZE="${PADIS_LOCAL_PREFLIGHT_MICROBATCH_SIZE:-$PADIS_LOCAL_PREFLIGHT_BATCH_SIZE}"
PADIS_LOCAL_MICROBATCH_SIZE="${PADIS_LOCAL_MICROBATCH_SIZE:-}"
PADIS_LOCAL_PILOT_TARGET_PATCHES="${PADIS_LOCAL_PILOT_TARGET_PATCHES:-512}"
PADIS_LOCAL_VALIDATION_INTERVAL_PATCHES="${PADIS_LOCAL_VALIDATION_INTERVAL_PATCHES:-512}"
PADIS_LOCAL_CHECKPOINT_INTERVAL_PATCHES="${PADIS_LOCAL_CHECKPOINT_INTERVAL_PATCHES:-512}"
PADIS_LOCAL_LOG_INTERVAL_PATCHES="${PADIS_LOCAL_LOG_INTERVAL_PATCHES:-128}"
PADIS_LOCAL_NUM_WORKERS="${PADIS_LOCAL_NUM_WORKERS:-6}"
PADIS_LOCAL_PREFETCH_FACTOR="${PADIS_LOCAL_PREFETCH_FACTOR:-4}"
PADIS_LOCAL_PERSISTENT_WORKERS="${PADIS_LOCAL_PERSISTENT_WORKERS:-0}"
PADIS_LOCAL_INCLUDE_WHOLE="${PADIS_LOCAL_INCLUDE_WHOLE:-0}"
PADIS_LOCAL_INCLUDE_512="${PADIS_LOCAL_INCLUDE_512:-1}"
PADIS_LOCAL_SKIP_CHECKS="${PADIS_LOCAL_SKIP_CHECKS:-0}"
PADIS_LOCAL_VALIDATE_FULL_LIDC="${PADIS_LOCAL_VALIDATE_FULL_LIDC:-0}"
PADIS_LOCAL_CACHE_DATASET="${PADIS_LOCAL_CACHE_DATASET:-none}"
PADIS_LOCAL_CACHE_FOLDER="${PADIS_LOCAL_CACHE_FOLDER:-$PADIS_EXPERIMENT_ROOT/debug_runs/cache}"
PADIS_LOCAL_CACHE_FULL_LIDC="${PADIS_LOCAL_CACHE_FULL_LIDC:-0}"
PADIS_DATA_FOLDER="${PADIS_DATA_FOLDER:-}"

MPLCONFIGDIR="${MPLCONFIGDIR:-$PADIS_LOCAL_ROOT/matplotlib}"
WANDB_DIR="${WANDB_DIR:-$PADIS_LOCAL_ROOT/wandb}"
export MPLCONFIGDIR WANDB_DIR PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1

mkdir -p "$PADIS_LOCAL_ROOT/logs" "$PADIS_LOCAL_PILOT_ROOT" "$MPLCONFIGDIR" "$WANDB_DIR"
cd "$LION_ROOT"

SECONDS=0

remaining_seconds() {
        local remaining=$((PADIS_LOCAL_MAX_SECONDS - SECONDS))
        if [ "$remaining" -lt 0 ]; then
                remaining=0
        fi
        printf '%s\n' "$remaining"
}

run_logged_with_budget() {
        local label="$1"
        shift
        local remaining
        remaining="$(remaining_seconds)"
        if [ "$remaining" -lt "$PADIS_LOCAL_MIN_COMMAND_SECONDS" ]; then
                echo "Skipping $label: only ${remaining}s remain."
                return 2
        fi

        local log_file="$PADIS_LOCAL_ROOT/logs/${label}.log"
        echo
        echo "[$(date)] Running $label with ${remaining}s remaining"
        echo "Log: $log_file"
        printf '%q ' "$@"
        printf '\n'

        set +e
        if command -v timeout >/dev/null 2>&1; then
                timeout --kill-after=60s "${remaining}s" "$@" >"$log_file" 2>&1
        else
                "$@" >"$log_file" 2>&1
        fi
        local status=$?
        set -e

        if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
                echo "$label reached the local time budget."
                tail -n 40 "$log_file" || true
                exit 0
        fi
        if [ "$status" -ne 0 ]; then
                echo "$label failed with status $status."
                tail -n 80 "$log_file" || true
                exit "$status"
        fi
        tail -n 20 "$log_file" || true
}

echo "Local PaDIS check/pilot run"
echo "LION_ROOT=$LION_ROOT"
echo "PYTHON=$PYTHON"
echo "PADIS_LOCAL_ROOT=$PADIS_LOCAL_ROOT"
echo "PADIS_LOCAL_PILOT_ROOT=$PADIS_LOCAL_PILOT_ROOT"
echo "PADIS_LOCAL_DEVICE=$PADIS_LOCAL_DEVICE"
echo "PADIS_LOCAL_MAX_SECONDS=$PADIS_LOCAL_MAX_SECONDS"
echo "PADIS_ROOT=$PADIS_ROOT"
echo "PADIS_LOCAL_SKIP_CHECKS=$PADIS_LOCAL_SKIP_CHECKS"
echo "PADIS_LOCAL_VALIDATE_FULL_LIDC=$PADIS_LOCAL_VALIDATE_FULL_LIDC"
echo "PADIS_LOCAL_PATCH_BATCH_SIZE=$PADIS_LOCAL_PATCH_BATCH_SIZE"
echo "PADIS_LOCAL_P96_BATCH_SIZE=$PADIS_LOCAL_P96_BATCH_SIZE"
echo "PADIS_LOCAL_512_BATCH_SIZE=$PADIS_LOCAL_512_BATCH_SIZE"
echo "PADIS_LOCAL_MICROBATCH_SIZE=${PADIS_LOCAL_MICROBATCH_SIZE:-none}"
echo "PADIS_LOCAL_CACHE_DATASET=$PADIS_LOCAL_CACHE_DATASET"
echo "PADIS_LOCAL_PERSISTENT_WORKERS=$PADIS_LOCAL_PERSISTENT_WORKERS"

if [ "$PADIS_LOCAL_SKIP_CHECKS" != "1" ]; then
        run_logged_with_budget py_compile \
                "$PYTHON" -m py_compile \
                scripts/dev/check_padis_repo_equivalence.py \
                scripts/dev/check_padis_short_run_reproduction.py \
                scripts/dev/run_padis_machine_preflight.py \
                scripts/dev/run_padis_training_smoke.py \
                scripts/paper_scripts/PaDIS/PaDIS_LIDC_256.py \
                scripts/paper_scripts/PaDIS/PaDIS_LIDC_512.py \
                scripts/paper_scripts/PaDIS/PaDIS_LIDC_generation.py \
                scripts/paper_scripts/PaDIS/PaDIS_LIDC_reconstruction.py \
                scripts/paper_scripts/PaDIS/PaDIS_experiments.py

        run_logged_with_budget focused_pytest \
                "$PYTHON" -m pytest \
                tests/models/test_padis_reconstructor.py \
                tests/models/test_padis_training.py \
                tests/experiments/test_padis_ct_experiments.py

        run_logged_with_budget golden_write \
                "$PYTHON" scripts/dev/check_padis_repo_equivalence.py \
                --padis-root "$PADIS_ROOT" \
                --device cpu \
                --write-golden "$PADIS_GOLDEN"

        run_logged_with_budget golden_compare \
                "$PYTHON" scripts/dev/check_padis_repo_equivalence.py \
                --padis-root "$PADIS_ROOT" \
                --device cpu \
                --golden "$PADIS_GOLDEN"

        run_logged_with_budget short_run_reproduction_cuda \
                "$PYTHON" scripts/dev/check_padis_short_run_reproduction.py \
                --padis-root "$PADIS_ROOT" \
                --device "$PADIS_LOCAL_DEVICE" \
                --seeds 2026 2027 2028 \
                --steps 6 \
                --patch-sizes 16 32 56 \
                --relative-tolerance 0.005 \
                --json "$PADIS_LOCAL_ROOT/short_run_reproduction_cuda.json"

        preflight_args=(
                "$PYTHON" scripts/dev/run_padis_machine_preflight.py
                --device "$PADIS_LOCAL_DEVICE"
                --mode-set all
                --base-batch-size "$PADIS_LOCAL_PREFLIGHT_BATCH_SIZE"
                --microbatch-size "$PADIS_LOCAL_PREFLIGHT_MICROBATCH_SIZE"
                --training-steps 1
                --validation-batch-size 1
                --validation-batches 2
                --max-slices-per-patient 4
                --num-workers "$PADIS_LOCAL_NUM_WORKERS"
                --prefetch-factor "$PADIS_LOCAL_PREFETCH_FACTOR"
                --padis-root "$PADIS_ROOT"
                --golden "$PADIS_GOLDEN"
                --short-run-relative-tolerance 0.005
                --run-cli-smoke
                --cli-target-patches 8
                --output-dir "$PADIS_LOCAL_ROOT/preflight"
                --json "$PADIS_LOCAL_ROOT/preflight/preflight_report.json"
        )
        if [ "$PADIS_LOCAL_INCLUDE_WHOLE" != "1" ]; then
                preflight_args+=(--skip-whole-image)
        fi
        run_logged_with_budget preflight_cuda "${preflight_args[@]}"
else
        echo
        echo "Skipping compile/test/equivalence/preflight checks."
fi

padis_init_training_tasks
if [ -n "${PADIS_LOCAL_TASKS:-}" ]; then
        read -r -a selected_tasks <<< "$PADIS_LOCAL_TASKS"
else
        selected_tasks=(
                patch_lidc_default
                patch_lidc_quarter_default
                patch_lidc_half_default
                patch_lidc_full
                patch_lidc_p8_default
                patch_lidc_p16_default
                patch_lidc_p32_default
                patch_lidc_p96_default
                patch_lidc_no_pos_default
        )
        if [ "$PADIS_LOCAL_INCLUDE_512" = "1" ]; then
                selected_tasks+=(patch_lidc_512)
        fi
        if [ "$PADIS_LOCAL_INCLUDE_WHOLE" = "1" ]; then
                selected_tasks+=(
                        whole_lidc_default
                        whole_lidc_quarter_default
                        whole_lidc_half_default
                        whole_lidc_full
                )
        fi
fi

task_index_for() {
        local requested="$1"
        local index
        for index in "${!PADIS_TASK_NAMES[@]}"; do
                if [ "$requested" = "$index" ] || \
                        [ "$requested" = "${PADIS_TASK_NAMES[$index]}" ]; then
                        printf '%s\n' "$index"
                        return 0
                fi
        done
        echo "Unknown local PaDIS pilot task: $requested" >&2
        return 1
}

for task in "${selected_tasks[@]}"; do
        task_index="$(task_index_for "$task")"
        task_name="${PADIS_TASK_NAMES[$task_index]}"
        task_engine="${PADIS_TASK_ENGINES[$task_index]}"
        read -r -a task_args <<< "${PADIS_TASK_ARGUMENTS[$task_index]}"
        run_name="${task_name}_local_pilot"
        task_validation_interval="$PADIS_LOCAL_VALIDATION_INTERVAL_PATCHES"
        task_batch_size="$PADIS_LOCAL_BATCH_SIZE"
        task_cache_dataset="$PADIS_LOCAL_CACHE_DATASET"
        if [[ "$task_name" == *"_p96_"* ]]; then
                task_batch_size="$PADIS_LOCAL_P96_BATCH_SIZE"
        elif [[ "$task_name" == whole_* ]]; then
                task_batch_size="$PADIS_LOCAL_WHOLE_BATCH_SIZE"
        elif [ "$task_engine" = "lidc512" ]; then
                task_batch_size="$PADIS_LOCAL_512_BATCH_SIZE"
        fi
        if [[ "$task_name" == *"_full" ]] && \
                [ "$PADIS_LOCAL_VALIDATE_FULL_LIDC" != "1" ]; then
                task_validation_interval=$((PADIS_LOCAL_PILOT_TARGET_PATCHES * 1000))
                echo
                echo "Skipping local validation for $task_name; set PADIS_LOCAL_VALIDATE_FULL_LIDC=1 to enable it."
                if [ "$PADIS_LOCAL_CACHE_FULL_LIDC" != "1" ]; then
                        task_cache_dataset="none"
                fi
        fi

        if [ "$task_engine" = "lidc256" ]; then
                cmd=(
                        "$PYTHON" -u scripts/paper_scripts/PaDIS/PaDIS_LIDC_256.py
                        --save-folder "$PADIS_LOCAL_PILOT_ROOT"
                        --device "$PADIS_LOCAL_DEVICE"
                        --target-patches "$PADIS_LOCAL_PILOT_TARGET_PATCHES"
                        --validation-interval-patches "$task_validation_interval"
                        --checkpoint-interval-patches "$PADIS_LOCAL_CHECKPOINT_INTERVAL_PATCHES"
                        --log-interval-patches "$PADIS_LOCAL_LOG_INTERVAL_PATCHES"
                        --batch-size "$task_batch_size"
                        --num-workers "$PADIS_LOCAL_NUM_WORKERS"
                        --prefetch-factor "$PADIS_LOCAL_PREFETCH_FACTOR"
                        --no-wandb
                        --wandb-mode disabled
                )
                if [ "$PADIS_LOCAL_PERSISTENT_WORKERS" != "1" ]; then
                        cmd+=(--no-persistent-workers)
                fi
                if [ -n "$PADIS_LOCAL_MICROBATCH_SIZE" ]; then
                        cmd+=(--microbatch-size "$PADIS_LOCAL_MICROBATCH_SIZE")
                fi
                if [ "$task_cache_dataset" != "none" ]; then
                        cmd+=(
                                --cache-dataset "$task_cache_dataset"
                                --cache-folder "$PADIS_LOCAL_CACHE_FOLDER"
                        )
                fi
        elif [ "$task_engine" = "lidc512" ]; then
                cmd=(
                        "$PYTHON" -u scripts/paper_scripts/PaDIS/PaDIS_LIDC_512.py
                        --save-folder "$PADIS_LOCAL_PILOT_ROOT"
                        --device "$PADIS_LOCAL_DEVICE"
                        --target-patches "$PADIS_LOCAL_PILOT_TARGET_PATCHES"
                        --validation-interval-patches "$task_validation_interval"
                        --checkpoint-interval-patches "$PADIS_LOCAL_CHECKPOINT_INTERVAL_PATCHES"
                        --log-interval-patches "$PADIS_LOCAL_LOG_INTERVAL_PATCHES"
                        --batch-size "$task_batch_size"
                        --num-workers "$PADIS_LOCAL_NUM_WORKERS"
                        --prefetch-factor "$PADIS_LOCAL_PREFETCH_FACTOR"
                        --no-wandb
                        --wandb-mode disabled
                )
                if [ "$PADIS_LOCAL_PERSISTENT_WORKERS" != "1" ]; then
                        cmd+=(--no-persistent-workers)
                fi
                if [ -n "$PADIS_LOCAL_MICROBATCH_SIZE" ]; then
                        cmd+=(--microbatch-size "$PADIS_LOCAL_MICROBATCH_SIZE")
                fi
        else
                echo "Unknown task engine: $task_engine"
                exit 1
        fi
        if [ -n "$PADIS_DATA_FOLDER" ]; then
                cmd+=(--data-folder "$PADIS_DATA_FOLDER")
        fi
        cmd+=("${task_args[@]}")
        cmd+=(--run-name "$run_name")
        run_logged_with_budget "pilot_${task_name}" "${cmd[@]}" || true
done

echo
echo "Local PaDIS checks and pilots completed in ${SECONDS}s."
echo "Debug artifacts: $PADIS_LOCAL_ROOT"
echo "Pilot runs: $PADIS_LOCAL_PILOT_ROOT"
