#!/usr/bin/env bash
#
# Manually run the PaDIS reconstruction matrix on a GCP/Colab machine.
#
# This is the manual counterpart to the GCP spot startup reconstruction phase.
# It mounts the GCS bucket as /mnt/data by default, writes all reconstruction
# outputs and resumability state under that mount, and flushes the mount after
# each completed or failed job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

die() {
        echo "$*" >&2
        exit 1
}

log() {
        printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

mount_gcs_bucket() {
        PADIS_GCS_BUCKET="${PADIS_GCS_BUCKET:-padis-bucket}"
        PADIS_GCS_BUCKET="${PADIS_GCS_BUCKET#gs://}"
        PADIS_GCS_BUCKET="${PADIS_GCS_BUCKET%/}"
        PADIS_DATA_MOUNT="${PADIS_DATA_MOUNT:-/mnt/data}"
        PADIS_MOUNT_BUCKET="${PADIS_MOUNT_BUCKET:-1}"
        PADIS_GCSFUSE_FLAGS="${PADIS_GCSFUSE_FLAGS:---implicit-dirs}"

        if [ "$PADIS_MOUNT_BUCKET" != "1" ]; then
                log "Skipping GCS mount because PADIS_MOUNT_BUCKET=$PADIS_MOUNT_BUCKET."
                mkdir -p "$PADIS_DATA_MOUNT"
                return
        fi

        mkdir -p "$PADIS_DATA_MOUNT"
        if mountpoint -q "$PADIS_DATA_MOUNT"; then
                log "$PADIS_DATA_MOUNT is already mounted."
                return
        fi

        if [ "$PADIS_MANUAL_RECON_DRY_RUN" = "1" ]; then
                log "Dry run: would mount gs://$PADIS_GCS_BUCKET at $PADIS_DATA_MOUNT."
                return
        fi

        if ! command -v gcsfuse >/dev/null 2>&1; then
                die "gcsfuse is required to mount gs://$PADIS_GCS_BUCKET. Install it in Colab, authenticate to GCP, or set PADIS_MOUNT_BUCKET=0 if /mnt/data is already available."
        fi

        log "Mounting gs://$PADIS_GCS_BUCKET at $PADIS_DATA_MOUNT."
        # shellcheck disable=SC2086
        gcsfuse $PADIS_GCSFUSE_FLAGS "$PADIS_GCS_BUCKET" "$PADIS_DATA_MOUNT"
        mountpoint -q "$PADIS_DATA_MOUNT" || die "gcsfuse returned successfully, but $PADIS_DATA_MOUNT is not a mountpoint."
}

activate_environment() {
        local conda_bin conda_sh env_candidates env_name activated conda_lib

        if [ "${PADIS_MANUAL_RECON_SKIP_ENV_ACTIVATE:-0}" = "1" ]; then
                log "Skipping environment activation because PADIS_MANUAL_RECON_SKIP_ENV_ACTIVATE=1."
                return
        fi
        if command -v python >/dev/null 2>&1 && python -c "import torch" >/dev/null 2>&1; then
                log "Using active Python: $(command -v python)."
                return
        fi

        conda_bin=""
        if [ -n "${CONDA_EXE:-}" ] && [ -x "$CONDA_EXE" ]; then
                conda_bin="$CONDA_EXE"
        elif command -v conda >/dev/null 2>&1; then
                conda_bin="$(command -v conda)"
        elif [ -x /mnt/data/conda/miniconda3/bin/conda ]; then
                conda_bin="/mnt/data/conda/miniconda3/bin/conda"
        fi

        [ -n "$conda_bin" ] || die "No usable Python with torch, and no conda executable found. Activate/install the LION environment first or set PADIS_MANUAL_RECON_SKIP_ENV_ACTIVATE=1."
        conda_sh="$(cd "$(dirname "$conda_bin")/.." && pwd -P)/etc/profile.d/conda.sh"
        if [ -f "$conda_sh" ]; then
                # shellcheck source=/dev/null
                . "$conda_sh"
        else
                eval "$("$conda_bin" shell.bash hook)"
        fi

        read -r -a env_candidates <<< "${LION_CONDA_ENV:-lion} ${LION_CONDA_ENV_FALLBACKS:-lion-dev padis-dev}"
        activated=""
        for env_name in "${env_candidates[@]}"; do
                if [ -z "$env_name" ]; then
                        continue
                fi
                if conda activate "$env_name"; then
                        activated="$env_name"
                        break
                fi
        done
        [ -n "$activated" ] || die "Failed to activate any conda environment from: ${env_candidates[*]}"
        conda_lib="${CONDA_PREFIX:-$(dirname "$(dirname "$conda_bin")")/envs/$activated}/lib"
        export LION_CONDA_ENV="$activated" LD_LIBRARY_PATH="$conda_lib:${LD_LIBRARY_PATH:-}"
        log "Activated $activated using conda."
}

discover_gpu_ids() {
        local raw_ids max_gpus gpu id count
        raw_ids="${PADIS_GCP_GPU_IDS:-${PADIS_RECON_GPU_IDS:-}}"
        if [ -z "$raw_ids" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]; then
                raw_ids="$CUDA_VISIBLE_DEVICES"
        fi
        if [ -z "$raw_ids" ] \
                && command -v nvidia-smi >/dev/null 2>&1 \
                && nvidia-smi -L >/dev/null 2>&1; then
                raw_ids="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',')"
        fi
        raw_ids="${raw_ids:-0}"
        raw_ids="${raw_ids//,/ }"

        max_gpus="${PADIS_RECON_MAX_GPUS:-1}"
        GPU_IDS=()
        count=0
        for gpu in $raw_ids; do
                id="${gpu//[[:space:]]/}"
                if [ -z "$id" ]; then
                        continue
                fi
                GPU_IDS+=("$id")
                count=$((count + 1))
                if [ "$count" -ge "$max_gpus" ]; then
                        break
                fi
        done
        [ "${#GPU_IDS[@]}" -gt 0 ] || die "No GPUs selected."
}

gpu_total_memory_mib() {
        local gpu_id="$1"
        local value
        command -v nvidia-smi >/dev/null 2>&1 || return 1
        value="$(
                nvidia-smi \
                        --id="$gpu_id" \
                        --query-gpu=memory.total \
                        --format=csv,noheader,nounits 2>/dev/null \
                        | sed -n '1{s/[^0-9]//g;p;}'
        )"
        if [[ "$value" =~ ^[0-9]+$ ]]; then
                printf '%s\n' "$value"
                return 0
        fi
        return 1
}

resolve_reconstruction_tasks_per_gpu() {
        local requested min_memory memory gpu_id
        requested="${PADIS_RECON_TASKS_PER_GPU:-${PADIS_GCP_RECON_TASKS_PER_GPU:-auto}}"
        if [ "$requested" != "auto" ]; then
                [[ "$requested" =~ ^[1-9][0-9]*$ ]] || die "PADIS_RECON_TASKS_PER_GPU must be a positive integer or auto."
                RECON_TASKS_PER_GPU="$requested"
                return
        fi

        RECON_TASKS_PER_GPU=1
        min_memory=""
        for gpu_id in "${GPU_IDS[@]}"; do
                memory="$(gpu_total_memory_mib "$gpu_id" || true)"
                if [ -z "$memory" ]; then
                        min_memory=""
                        break
                fi
                if [ -z "$min_memory" ] || [ "$memory" -lt "$min_memory" ]; then
                        min_memory="$memory"
                fi
        done

        if [ -n "$min_memory" ] && [ "$min_memory" -ge 90000 ]; then
                RECON_TASKS_PER_GPU=3
        elif [ -n "$min_memory" ] && [ "$min_memory" -ge 40000 ]; then
                RECON_TASKS_PER_GPU=2
        fi
}

build_reconstruction_base_command() {
        RECON_BASE_CMD=(
                python -u scripts/paper_scripts/PaDIS/PaDIS_run_reconstruction_matrix.py
                --training-root "$PADIS_TRAIN_ROOT"
                --output-root "$PADIS_RECON_ROOT"
                --checkpoint-policy "$PADIS_RECON_CHECKPOINT_POLICY"
                --job-order "$PADIS_RECON_JOB_ORDER"
                --hparam-defaults "$PADIS_RECON_HPARAM_DEFAULTS"
                --hparam-defaults-json "$PADIS_RECON_HPARAM_DEFAULTS_JSON"
                --hparam-run-root "$PADIS_RECON_HPARAM_RUN_ROOT"
                --hparam-run-glob "$PADIS_RECON_HPARAM_RUN_GLOB"
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
                --pnp-checkpoint "$PADIS_RECON_PNP_CHECKPOINT"
                --pnp-iterations "$PADIS_PNP_ITERATIONS"
                --pnp-eta "$PADIS_PNP_ETA"
                --pnp-cg-iterations "$PADIS_PNP_CG_ITERATIONS"
                --pnp-cg-tolerance "$PADIS_PNP_CG_TOLERANCE"
                --tv-lambda "$PADIS_TV_LAMBDA"
                --tv-iterations "$PADIS_TV_ITERATIONS"
        )
        if [ -n "$PADIS_PNP_NOISE_LEVEL" ]; then
                RECON_BASE_CMD+=(--pnp-noise-level "$PADIS_PNP_NOISE_LEVEL")
        fi
        if [ "$PADIS_RECON_ALLOW_OFF_PAPER_EXPERIMENTS" = "1" ]; then
                RECON_BASE_CMD+=(--allow-off-paper-experiments)
        fi
        if [ -n "$PADIS_DATA_FOLDER" ]; then
                RECON_BASE_CMD+=(--data-folder "$PADIS_DATA_FOLDER")
        fi
        if [ -n "$PADIS_PUBLIC_IMAGE_DIR" ]; then
                RECON_BASE_CMD+=(--public-padis-image-dir "$PADIS_PUBLIC_IMAGE_DIR")
        fi
        if [ "$PADIS_RECON_SAVE_PREVIEWS" = "1" ]; then
                RECON_BASE_CMD+=(--save-previews)
        fi
        if [ "$PADIS_RECON_PROG_BAR" = "1" ]; then
                RECON_BASE_CMD+=(--prog-bar)
        fi
        if [ -n "$PADIS_RECON_TRACE_INTERVAL" ]; then
                RECON_BASE_CMD+=(--trace-interval "$PADIS_RECON_TRACE_INTERVAL")
        fi
        if [ "$PADIS_RECON_TRACE_IMAGES" = "1" ]; then
                RECON_BASE_CMD+=(--trace-images)
        fi
        if [ "$PADIS_RECON_ALLOW_MISSING_CHECKPOINTS" = "1" ]; then
                RECON_BASE_CMD+=(--allow-missing-checkpoint)
        fi
        if [ -n "$PADIS_RECON_EXTRA_ARGS" ]; then
                read -r -a RECON_EXTRA_ARGS <<< "$PADIS_RECON_EXTRA_ARGS"
                for item in "${RECON_EXTRA_ARGS[@]}"; do
                        RECON_BASE_CMD+=("--reconstruction-arg=$item")
                done
        fi
}

reconstruction_phase_key() {
        local task_index="$1"
        printf 'reconstruction_%06d.reconstruction\n' "$task_index"
}

reconstruction_done_marker() {
        printf '%s/%s.done\n' "$DONE_DIR" "$(reconstruction_phase_key "$1")"
}

reconstruction_running_marker() {
        printf '%s/%s.running\n' "$RUNNING_DIR" "$(reconstruction_phase_key "$1")"
}

reconstruction_failed_marker() {
        printf '%s/%s.failed\n' "$FAILED_DIR" "$(reconstruction_phase_key "$1")"
}

sync_bucket_mount() {
        if [ "${PADIS_RECON_SYNC_AFTER_JOB:-1}" != "1" ]; then
                return
        fi
        if command -v sync >/dev/null 2>&1; then
                sync -f "$PADIS_RECON_ROOT" >/dev/null 2>&1 || sync
        fi
        printf '%s\n' "$(date --iso-8601=seconds)" > "$STATE_DIR/last_sync.txt.tmp"
        mv "$STATE_DIR/last_sync.txt.tmp" "$STATE_DIR/last_sync.txt"
}

prepare_reconstruction_matrix() {
        mkdir -p "$PADIS_RECON_ROOT"
        build_reconstruction_base_command
        if [ "$PADIS_MANUAL_RECON_DRY_RUN" != "1" ] \
                && [ "$PADIS_RECON_ALLOW_MISSING_CHECKPOINTS" != "1" ]; then
                "${RECON_BASE_CMD[@]}" --check-inputs
        fi
        RECON_TASK_COUNT="$("${RECON_BASE_CMD[@]}" --count)"
        if ! [[ "$RECON_TASK_COUNT" =~ ^[0-9]+$ ]] || [ "$RECON_TASK_COUNT" -le 0 ]; then
                die "Reconstruction matrix produced invalid task count: $RECON_TASK_COUNT"
        fi
        "${RECON_BASE_CMD[@]}" --list > "$PADIS_RECON_EXPECTED_JOBS_JSON"
        log "Prepared reconstruction matrix with $RECON_TASK_COUNT jobs at $PADIS_RECON_EXPECTED_JOBS_JSON."
        sync_bucket_mount
}

claim_next_reconstruction_task() {
        local gpu_id="$1"
        local slot_id="${2:-1}"
        local claimed="" task_index marker fd phase_key
        exec {fd}>"$STATE_DIR/reconstruction_queue.lock"
        flock "$fd"
        for ((task_index = 0; task_index < RECON_TASK_COUNT; task_index++)); do
                if [ -f "$(reconstruction_done_marker "$task_index")" ]; then
                        continue
                fi
                marker="$(reconstruction_running_marker "$task_index")"
                if [ -f "$marker" ]; then
                        continue
                fi
                phase_key="$(reconstruction_phase_key "$task_index")"
                {
                        printf 'task=%s\n' "$phase_key"
                        printf 'phase=reconstruction\n'
                        printf 'task_index=%s\n' "$task_index"
                        printf 'gpu=%s\n' "$gpu_id"
                        printf 'slot=%s\n' "$slot_id"
                        printf 'pid=%s\n' "$$"
                        printf 'worker_pid=%s\n' "$BASHPID"
                        printf 'host=%s\n' "$(hostname)"
                        printf 'started=%s\n' "$(date --iso-8601=seconds)"
                } > "$marker"
                claimed="$task_index"
                break
        done
        flock -u "$fd"
        eval "exec $fd>&-"
        printf '%s\n' "$claimed"
}

run_reconstruction_task() {
        local task_index="$1"
        local gpu_id="$2"
        local slot_id="${3:-1}"
        local phase_key log_path command_path done_marker failed_marker rc
        phase_key="$(reconstruction_phase_key "$task_index")"
        done_marker="$(reconstruction_done_marker "$task_index")"
        failed_marker="$(reconstruction_failed_marker "$task_index")"
        log_path="$LOG_DIR/$phase_key.gpu${gpu_id}.slot${slot_id}.log"
        command_path="$LOG_DIR/$phase_key.command.txt"

        if [ -f "$done_marker" ]; then
                rm -f "$(reconstruction_running_marker "$task_index")"
                return 0
        fi

        RECON_CMD=("${RECON_BASE_CMD[@]}" --task-index "$task_index")
        printf '%q ' "${RECON_CMD[@]}" > "$command_path"
        printf '\n' >> "$command_path"
        log "GPU $gpu_id slot $slot_id running $phase_key; log: $log_path"

        if [ "$PADIS_MANUAL_RECON_DRY_RUN" = "1" ]; then
                {
                        printf 'completed=%s\n' "$(date --iso-8601=seconds)"
                        printf 'phase=reconstruction\n'
                        printf 'task_index=%s\n' "$task_index"
                        printf 'gpu=%s\n' "$gpu_id"
                        printf 'slot=%s\n' "$slot_id"
                        printf 'dry_run=1\n'
                        printf 'log_path=%s\n' "$log_path"
                } > "$done_marker"
                rm -f "$failed_marker" "$(reconstruction_running_marker "$task_index")"
                sync_bucket_mount
                return 0
        fi

        set +e
        (
                export CUDA_VISIBLE_DEVICES="$gpu_id"
                "${RECON_CMD[@]}"
        ) > "$log_path" 2>&1
        rc="$?"
        set -e

        if [ "$rc" -eq 0 ]; then
                {
                        printf 'completed=%s\n' "$(date --iso-8601=seconds)"
                        printf 'phase=reconstruction\n'
                        printf 'task_index=%s\n' "$task_index"
                        printf 'gpu=%s\n' "$gpu_id"
                        printf 'slot=%s\n' "$slot_id"
                        printf 'log_path=%s\n' "$log_path"
                } > "$done_marker"
                rm -f "$failed_marker" "$(reconstruction_running_marker "$task_index")"
                sync_bucket_mount
                log "Reconstruction task $phase_key completed."
                return 0
        fi

        {
                printf 'failed=%s\n' "$(date --iso-8601=seconds)"
                printf 'exit_code=%s\n' "$rc"
                printf 'phase=reconstruction\n'
                printf 'task_index=%s\n' "$task_index"
                printf 'gpu=%s\n' "$gpu_id"
                printf 'slot=%s\n' "$slot_id"
                printf 'log_path=%s\n' "$log_path"
        } > "$failed_marker"
        rm -f "$(reconstruction_running_marker "$task_index")"
        sync_bucket_mount
        log "Reconstruction task $phase_key failed with exit code $rc. See $log_path"
        return "$rc"
}

reconstruction_worker_loop() {
        local gpu_id="$1"
        local slot_id="${2:-1}"
        local task_index
        while true; do
                task_index="$(claim_next_reconstruction_task "$gpu_id" "$slot_id")"
                if [ -z "$task_index" ]; then
                        log "GPU $gpu_id slot $slot_id has no remaining reconstruction tasks."
                        return 0
                fi
                run_reconstruction_task "$task_index" "$gpu_id" "$slot_id"
        done
}

write_manifest() {
        local manifest="$STATE_DIR/manifest.txt"
        {
                printf 'gcs_bucket=%s\n' "$PADIS_GCS_BUCKET"
                printf 'data_mount=%s\n' "$PADIS_DATA_MOUNT"
                printf 'mount_bucket=%s\n' "$PADIS_MOUNT_BUCKET"
                printf 'lion_root=%s\n' "$LION_ROOT"
                printf 'lion_data_path=%s\n' "$LION_DATA_PATH"
                printf 'training_root=%s\n' "$PADIS_TRAIN_ROOT"
                printf 'reconstruction_root=%s\n' "$PADIS_RECON_ROOT"
                printf 'state_dir=%s\n' "$STATE_DIR"
                printf 'gpu_ids=%s\n' "${GPU_IDS[*]}"
                printf 'reconstruction_tasks_per_gpu=%s\n' "$RECON_TASKS_PER_GPU"
                printf 'reconstruction_checkpoint_policy=%s\n' "$PADIS_RECON_CHECKPOINT_POLICY"
                printf 'reconstruction_job_order=%s\n' "$PADIS_RECON_JOB_ORDER"
                printf 'reconstruction_hparam_defaults=%s\n' "$PADIS_RECON_HPARAM_DEFAULTS"
                printf 'reconstruction_hparam_defaults_json=%s\n' "$PADIS_RECON_HPARAM_DEFAULTS_JSON"
                printf 'reconstruction_methods=%s\n' "$PADIS_RECON_METHODS"
                printf 'reconstruction_experiments=%s\n' "$PADIS_RECON_EXPERIMENTS"
                printf 'reconstruction_ablations=%s\n' "$PADIS_RECON_ABLATIONS"
                printf 'reconstruction_expected_jobs_json=%s\n' "$PADIS_RECON_EXPECTED_JOBS_JSON"
                printf 'sync_after_job=%s\n' "${PADIS_RECON_SYNC_AFTER_JOB:-1}"
                printf 'dry_run=%s\n' "$PADIS_MANUAL_RECON_DRY_RUN"
        } > "$manifest"
}

clear_stale_running_markers() {
        if [ "${PADIS_RECON_CLEAR_STALE_RUNNING:-1}" = "1" ]; then
                find "$RUNNING_DIR" -maxdepth 1 -type f -name 'reconstruction_*.running' -delete
        fi
}

terminate_runner() {
        log "Termination requested; stopping manual reconstruction workers. Rerun this script to resume."
        kill $(jobs -pr) >/dev/null 2>&1 || true
        wait >/dev/null 2>&1 || true
        sync_bucket_mount || true
        exit 143
}

PADIS_MANUAL_RECON_DRY_RUN="${PADIS_MANUAL_RECON_DRY_RUN:-0}"
mount_gcs_bucket
activate_environment

LION_ROOT="${LION_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd -P)}"
LION_DATA_PATH="${LION_DATA_PATH:-$PADIS_DATA_MOUNT/Datasets}"
LION_EXPERIMENTS_PATH="${LION_EXPERIMENTS_PATH:-$LION_DATA_PATH/experiments}"
PADIS_RUN_ROOT="${PADIS_RUN_ROOT:-$LION_EXPERIMENTS_PATH/PaDIS}"
PADIS_GCP_RUN_NAME="${PADIS_GCP_RUN_NAME:-PaDIS-Reproduction-GCP}"
PADIS_TRAIN_ROOT="${PADIS_TRAIN_ROOT:-$PADIS_RUN_ROOT/final_real_runs/$PADIS_GCP_RUN_NAME}"
PADIS_RECON_ROOT="${PADIS_RECON_ROOT:-$PADIS_RUN_ROOT/final_real_runs/${PADIS_GCP_RUN_NAME}_reconstruction}"
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
PADIS_RECON_EXTRA_ARGS="${PADIS_RECON_EXTRA_ARGS:-}"
PADIS_RECON_CHECKPOINT_POLICY="${PADIS_RECON_CHECKPOINT_POLICY:-min_intense_val}"
PADIS_RECON_JOB_ORDER="${PADIS_RECON_JOB_ORDER:-gcp_spot}"
PADIS_RECON_HPARAM_DEFAULTS="${PADIS_RECON_HPARAM_DEFAULTS:-json}"
PADIS_RECON_HPARAM_DEFAULTS_JSON="${PADIS_RECON_HPARAM_DEFAULTS_JSON:-$LION_ROOT/scripts/paper_scripts/PaDIS/config/reconstruction_hparam_defaults.json}"
PADIS_RECON_HPARAM_RUN_ROOT="${PADIS_RECON_HPARAM_RUN_ROOT:-$PADIS_RUN_ROOT/hparam_tuning/runs}"
PADIS_RECON_HPARAM_RUN_GLOB="${PADIS_RECON_HPARAM_RUN_GLOB:-fixedval_*}"
PADIS_RECON_ALLOW_MISSING_CHECKPOINTS="${PADIS_RECON_ALLOW_MISSING_CHECKPOINTS:-0}"
PADIS_RECON_EXPECTED_JOBS_JSON="${PADIS_RECON_EXPECTED_JOBS_JSON:-$PADIS_RECON_ROOT/reconstruction_matrix_jobs.json}"
PADIS_PNP_OUTPUT_ROOT="${PADIS_PNP_OUTPUT_ROOT:-$PADIS_TRAIN_ROOT}"
PADIS_PNP_RUN_NAME="${PADIS_PNP_RUN_NAME:-pnp_lidc_drunet}"
PADIS_PNP_VALIDATION_NAME="${PADIS_PNP_VALIDATION_NAME:-pnp_lidc_drunet_min_val.pt}"
PADIS_PNP_ROOT="${PADIS_PNP_ROOT:-$PADIS_PNP_OUTPUT_ROOT/$PADIS_PNP_RUN_NAME}"
PADIS_RECON_PNP_CHECKPOINT="${PADIS_RECON_PNP_CHECKPOINT:-${PADIS_PNP_CHECKPOINT:-$PADIS_PNP_ROOT/$PADIS_PNP_VALIDATION_NAME}}"
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

STATE_DIR="${PADIS_MANUAL_RECON_STATE_DIR:-$PADIS_RECON_ROOT/.manual_gcp_reconstruction}"
DONE_DIR="$STATE_DIR/done"
RUNNING_DIR="$STATE_DIR/running"
FAILED_DIR="$STATE_DIR/failed"
LOG_DIR="$STATE_DIR/logs"

export LION_ROOT LION_DATA_PATH LION_EXPERIMENTS_PATH PADIS_RUN_ROOT
export PADIS_TRAIN_ROOT PADIS_RECON_ROOT PADIS_RECON_EXPECTED_JOBS_JSON
export MPLCONFIGDIR PYTORCH_CUDA_ALLOC_CONF PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1

mkdir -p "$PADIS_RECON_ROOT" "$MPLCONFIGDIR" "$STATE_DIR" "$DONE_DIR" "$RUNNING_DIR" "$FAILED_DIR" "$LOG_DIR"
discover_gpu_ids
resolve_reconstruction_tasks_per_gpu
write_manifest
clear_stale_running_markers
trap terminate_runner INT TERM

cd "$LION_ROOT"
log "LION root: $LION_ROOT"
log "Data mount: $PADIS_DATA_MOUNT"
log "Training root: $PADIS_TRAIN_ROOT"
log "Reconstruction root: $PADIS_RECON_ROOT"
log "Selected GPUs: ${GPU_IDS[*]}"
log "Worker slots per GPU: $RECON_TASKS_PER_GPU"

prepare_reconstruction_matrix

worker_pids=()
phase_rc=0
for gpu_id in "${GPU_IDS[@]}"; do
        for ((slot_id = 1; slot_id <= RECON_TASKS_PER_GPU; slot_id++)); do
                reconstruction_worker_loop "$gpu_id" "$slot_id" &
                worker_pids+=("$!")
        done
done

for pid in "${worker_pids[@]}"; do
        if ! wait "$pid"; then
                phase_rc=1
        fi
done

sync_bucket_mount
if [ "$phase_rc" -eq 0 ]; then
        log "Manual PaDIS reconstruction matrix completed."
else
        log "Manual PaDIS reconstruction matrix finished with failures; inspect $FAILED_DIR and $LOG_DIR."
fi
exit "$phase_rc"
