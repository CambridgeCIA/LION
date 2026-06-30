#!/bin/bash
#!
#! Ten-minute GPU utilisation profile for the default PaDIS A100 training task.
#!
#SBATCH -J PaDIS_profile
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:25:00
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
PADIS_PROFILE_ROOT="${PADIS_PROFILE_ROOT:-$PADIS_RUN_ROOT/profile_runs/default_training_$PADIS_RUN_STAMP}"
PADIS_DATA_FOLDER="${PADIS_DATA_FOLDER:-}"
PADIS_DATA_ROOT="${LION_DATA_PATH:-/home/tjh200/rds/hpc-work/Datasets}"
PADIS_CACHE_ROOT="${PADIS_CACHE_ROOT:-$PADIS_DATA_ROOT/processed/LIDC-IDRI-cache}"
PADIS_256_CACHE_ARCHIVE_FOLDER="${PADIS_256_CACHE_ARCHIVE_FOLDER:-${PADIS_CACHE_ARCHIVE_FOLDER:-$PADIS_CACHE_ROOT/padis_256/archives}}"
PADIS_SEED="${PADIS_SEED:-33}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$PADIS_PROFILE_ROOT/matplotlib}"
WANDB_DIR="${WANDB_DIR:-$PADIS_PROFILE_ROOT/wandb}"
export LION_ROOT PADIS_RUN_ROOT PADIS_RUN_STAMP PADIS_PROFILE_ROOT PADIS_DATA_FOLDER
export MPLCONFIGDIR WANDB_DIR PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 PYTHONHASHSEED="$PADIS_SEED"

mkdir -p "$PADIS_PROFILE_ROOT" "$MPLCONFIGDIR" "$WANDB_DIR"
cd "$LION_ROOT"
padis_print_job_header

padis_init_training_tasks
TASK_ID="${PADIS_PROFILE_TASK_ID:-0}"
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#PADIS_TASK_NAMES[@]}" ]; then
        echo "Invalid profile task id $TASK_ID for ${#PADIS_TASK_NAMES[@]} tasks."
        exit 1
fi

TASK_NAME="${PADIS_TASK_NAMES[$TASK_ID]}"
TASK_ENGINE="${PADIS_TASK_ENGINES[$TASK_ID]}"
TASK_BATCH_SIZE="${PADIS_TASK_BATCH_SIZES[$TASK_ID]}"
read -r -a TASK_ARGS <<< "${PADIS_TASK_ARGUMENTS[$TASK_ID]}"

if [ "$TASK_ENGINE" != "lidc256" ]; then
        echo "This profiler is intended for the default 256x256 PaDIS task."
        echo "Got task $TASK_ID $TASK_NAME with engine $TASK_ENGINE."
        exit 1
fi

PROFILE_SECONDS="${PADIS_PROFILE_SECONDS:-600}"
TARGET_PATCHES="${PADIS_PROFILE_TARGET_PATCHES:-400000000}"
VALIDATION_INTERVAL="${PADIS_PROFILE_VALIDATION_INTERVAL_PATCHES:-$TARGET_PATCHES}"
VALIDATION_MAX_PATCHES="${PADIS_PROFILE_VALIDATION_MAX_PATCHES:-${PADIS_VALIDATION_MAX_PATCHES:-1000}}"
CHECKPOINT_INTERVAL="${PADIS_PROFILE_CHECKPOINT_INTERVAL_PATCHES:-$TARGET_PATCHES}"
MAX_PERIODIC_CHECKPOINTS="${PADIS_PROFILE_MAX_PERIODIC_CHECKPOINTS:-${PADIS_MAX_PERIODIC_CHECKPOINTS:-5}}"
LOG_INTERVAL="${PADIS_PROFILE_LOG_INTERVAL_PATCHES:-128}"
NUM_WORKERS="${PADIS_PROFILE_NUM_WORKERS:-${PADIS_NUM_WORKERS:-16}}"
PREFETCH_FACTOR="${PADIS_PROFILE_PREFETCH_FACTOR:-${PADIS_PREFETCH_FACTOR:-4}}"
SAMPLE_INTERVAL="${PADIS_PROFILE_SAMPLE_INTERVAL:-1}"
WARMUP_SECONDS="${PADIS_PROFILE_WARMUP_SECONDS:-60}"
NO_WANDB_ARTIFACT="${PADIS_NO_WANDB_ARTIFACT:-0}"
PADIS_WANDB_PROJECT="${PADIS_WANDB_PROJECT:-PaDIS-Reproduction}"
PADIS_WANDB_ENTITY="${PADIS_WANDB_ENTITY:-}"
PADIS_WANDB_MODE="${PADIS_WANDB_MODE:-online}"
PADIS_NO_WANDB="${PADIS_NO_WANDB:-0}"
export PADIS_WANDB_PROJECT PADIS_WANDB_ENTITY PADIS_WANDB_MODE PADIS_NO_WANDB

wandb_args=()
if [ "$PADIS_NO_WANDB" = "1" ] || [ "${PADIS_PROFILE_NO_WANDB:-0}" = "1" ]; then
        wandb_args=(--no-wandb --wandb-mode disabled)
else
        wandb_name="${PADIS_PROFILE_WANDB_NAME_PREFIX:-PaDIS_A100_profile_${PADIS_RUN_STAMP}}_${TASK_NAME}"
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

GPU_CSV="$PADIS_PROFILE_ROOT/nvidia_smi.csv"
TRAIN_LOG="$PADIS_PROFILE_ROOT/train.log"
METRICS_JSONL="$PADIS_PROFILE_ROOT/timing_metrics.jsonl"
SUMMARY_JSON="$PADIS_PROFILE_ROOT/profile_summary.json"

CMD=(
        python -u scripts/paper_scripts/PaDIS/PaDIS_LIDC_256.py
        --save-folder "$PADIS_PROFILE_ROOT"
        --device cuda
        --target-patches "$TARGET_PATCHES"
        --validation-interval-patches "$VALIDATION_INTERVAL"
        --validation-max-patches "$VALIDATION_MAX_PATCHES"
        --checkpoint-interval-patches "$CHECKPOINT_INTERVAL"
        --max-periodic-checkpoints "$MAX_PERIODIC_CHECKPOINTS"
        --log-interval-patches "$LOG_INTERVAL"
        --max-train-seconds "$PROFILE_SECONDS"
        --seed "$PADIS_SEED"
        --batch-size "$TASK_BATCH_SIZE"
        --num-workers "$NUM_WORKERS"
        --prefetch-factor "$PREFETCH_FACTOR"
        --metrics-jsonl "$METRICS_JSONL"
)
CMD+=("${wandb_args[@]}")

if [ -n "$PADIS_DATA_FOLDER" ]; then
        CMD+=(--data-folder "$PADIS_DATA_FOLDER")
fi
if [ -n "${PADIS_MICROBATCH_SIZE:-}" ]; then
        CMD+=(--microbatch-size "$PADIS_MICROBATCH_SIZE")
fi

CACHE_DATASET="${PADIS_PROFILE_CACHE_DATASET:-${PADIS_256_CACHE_DATASET:-${PADIS_CACHE_DATASET:-ramdisk}}}"
if [ "$CACHE_DATASET" != "none" ]; then
        CACHE_FOLDER="${PADIS_PROFILE_CACHE_FOLDER:-${PADIS_256_CACHE_FOLDER:-${PADIS_CACHE_FOLDER:-/ramdisks/$USER/lion_lidc_cache}}}"
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

CMD+=("${TASK_ARGS[@]}")

SMI_PID=""
stop_gpu_monitor() {
        if [ -n "${SMI_PID:-}" ] && kill -0 "$SMI_PID" 2>/dev/null; then
                kill "$SMI_PID" 2>/dev/null || true
                wait "$SMI_PID" 2>/dev/null || true
        fi
}
trap stop_gpu_monitor EXIT

echo "Profile output: $PADIS_PROFILE_ROOT"
echo "Profiling task: $TASK_ID $TASK_NAME"
echo "GPU sample interval: ${SAMPLE_INTERVAL}s"
echo "Warmup excluded from steady-state summary: ${WARMUP_SECONDS}s"
echo "Executing profile command:"
printf '%q ' "${CMD[@]}"
printf '\n'

if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi \
                --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu \
                --format=csv,nounits \
                -l "$SAMPLE_INTERVAL" \
                > "$GPU_CSV" &
        SMI_PID="$!"
else
        echo "nvidia-smi is not available; GPU CSV will not be written."
fi

set +e
"${CMD[@]}" 2>&1 | tee "$TRAIN_LOG"
train_status="${PIPESTATUS[0]}"
set -e

stop_gpu_monitor
trap - EXIT

python - "$GPU_CSV" "$TRAIN_LOG" "$METRICS_JSONL" "$SUMMARY_JSON" "$SAMPLE_INTERVAL" "$WARMUP_SECONDS" "$train_status" <<'PY'
import csv
import json
import re
import statistics
import sys
from pathlib import Path

gpu_csv = Path(sys.argv[1])
train_log = Path(sys.argv[2])
metrics_jsonl = Path(sys.argv[3])
summary_json = Path(sys.argv[4])
sample_interval = float(sys.argv[5])
warmup_seconds = float(sys.argv[6])
train_status = int(sys.argv[7])


def number(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A" or "NOT SUPPORTED" in text.upper():
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    return float(match.group(0))


def percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    index = fraction * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def stats(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"count": 0}
    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "avg": sum(sorted_values) / len(sorted_values),
        "median": statistics.median(sorted_values),
        "min": sorted_values[0],
        "p10": percentile(sorted_values, 0.10),
        "p90": percentile(sorted_values, 0.90),
        "max": sorted_values[-1],
    }


def find_column(row, prefix):
    for key in row:
        if key is not None and key.strip().startswith(prefix):
            return key
    return None


def series(rows, prefix):
    values = []
    for row in rows:
        key = find_column(row, prefix)
        if key is not None:
            values.append(number(row.get(key)))
    return [value for value in values if value is not None]


gpu_rows = []
if gpu_csv.is_file() and gpu_csv.stat().st_size > 0:
    with gpu_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        gpu_rows = [row for row in reader if row and any(row.values())]

skip_samples = 0
if sample_interval > 0 and warmup_seconds > 0:
    skip_samples = min(len(gpu_rows), int(warmup_seconds / sample_interval))
steady_rows = gpu_rows[skip_samples:]

util_all = series(gpu_rows, "utilization.gpu")
util_steady = series(steady_rows, "utilization.gpu")
mem_all = series(gpu_rows, "memory.used")
power_all = series(gpu_rows, "power.draw")
power_limit_all = series(gpu_rows, "power.limit")
clock_sm_all = series(gpu_rows, "clocks.sm")
temperature_all = series(gpu_rows, "temperature.gpu")

power_fraction = []
for draw, limit in zip(power_all, power_limit_all):
    if limit and limit > 0:
        power_fraction.append(100.0 * draw / limit)

gpu_summary = {
    "samples": len(gpu_rows),
    "estimated_sampled_seconds": len(gpu_rows) * sample_interval,
    "warmup_excluded_seconds": skip_samples * sample_interval,
    "utilization_gpu_pct": stats(util_all),
    "steady_state_utilization_gpu_pct": stats(util_steady),
    "steady_state_fraction_ge_80_pct": (
        100.0 * sum(value >= 80.0 for value in util_steady) / len(util_steady)
        if util_steady
        else None
    ),
    "memory_used_mib": stats(mem_all),
    "power_draw_w": stats(power_all),
    "power_draw_fraction_of_limit_pct": stats(power_fraction),
    "sm_clock_mhz": stats(clock_sm_all),
    "temperature_c": stats(temperature_all),
}

records = []
if metrics_jsonl.is_file():
    for line in metrics_jsonl.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        metrics = payload.get("metrics", {})
        if "timing/patches_per_second" not in metrics:
            continue
        records.append(
            {
                "seen_patches": int(
                    metrics.get("train/seen_patches", payload.get("step", 0))
                ),
                "patch_size": metrics.get("train/patch_size"),
                "data_wait_s_per_step": number(
                    metrics.get("timing/data_wait_s_per_step")
                ),
                "train_s_per_step": number(
                    metrics.get("timing/train_step_s_per_step")
                ),
                "total_s_per_step": number(metrics.get("timing/total_s_per_step")),
                "data_wait_fraction": number(metrics.get("timing/data_wait_fraction")),
                "train_step_fraction": number(metrics.get("timing/train_step_fraction")),
                "patches_per_second": number(metrics.get("timing/patches_per_second")),
                "steps_per_second": number(metrics.get("timing/steps_per_second")),
            }
        )

if not records:
    training_pattern = re.compile(
        r"Patches\s+(\d+).*?data wait\s+([0-9.]+)s/step\s+-\s+"
        r"train\s+([0-9.]+)s/step\s+-\s+([0-9.]+)\s+patches/s"
    )
    if train_log.is_file():
        for line in train_log.read_text(errors="replace").splitlines():
            match = training_pattern.search(line)
            if match is None:
                continue
            seen, data_wait, train_step, patches_per_second = match.groups()
            data_wait = float(data_wait)
            train_step = float(train_step)
            total_step = data_wait + train_step
            records.append(
                {
                    "seen_patches": int(seen),
                    "patch_size": None,
                    "data_wait_s_per_step": data_wait,
                    "train_s_per_step": train_step,
                    "total_s_per_step": total_step,
                    "data_wait_fraction": (
                        data_wait / total_step if total_step > 0 else None
                    ),
                    "train_step_fraction": (
                        train_step / total_step if total_step > 0 else None
                    ),
                    "patches_per_second": float(patches_per_second),
                    "steps_per_second": None,
                }
            )

data_wait_stats = stats([record["data_wait_fraction"] for record in records])
train_step_stats = stats([record["train_step_fraction"] for record in records])
steady_util_avg = gpu_summary["steady_state_utilization_gpu_pct"].get("avg")
data_wait_avg = data_wait_stats.get("avg")
likely_data_bottleneck = (
    data_wait_avg is not None
    and data_wait_avg >= 0.20
    and (steady_util_avg is None or steady_util_avg < 80.0)
)

training_summary = {
    "metrics_source": "jsonl" if metrics_jsonl.is_file() else "stdout_fallback",
    "log_records": len(records),
    "final_seen_patches": max(
        (record["seen_patches"] for record in records), default=None
    ),
    "patches_per_second": stats(
        [record["patches_per_second"] for record in records]
    ),
    "steps_per_second": stats([record["steps_per_second"] for record in records]),
    "data_wait_s_per_step": stats(
        [record["data_wait_s_per_step"] for record in records]
    ),
    "train_s_per_step": stats([record["train_s_per_step"] for record in records]),
    "total_s_per_step": stats([record["total_s_per_step"] for record in records]),
    "data_wait_fraction": data_wait_stats,
    "train_step_fraction": train_step_stats,
    "likely_data_throughput_bottleneck": likely_data_bottleneck,
}

summary = {
    "train_exit_status": train_status,
    "gpu": gpu_summary,
    "training": training_summary,
    "files": {
        "nvidia_smi_csv": str(gpu_csv),
        "train_log": str(train_log),
        "timing_metrics_jsonl": str(metrics_jsonl),
        "summary_json": str(summary_json),
    },
}
summary_json.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "Profile summary written to $SUMMARY_JSON"
if [ "$train_status" -ne 0 ]; then
        echo "Training command exited with status $train_status."
        exit "$train_status"
fi

echo "Default PaDIS GPU utilisation profile completed at $(date)."
