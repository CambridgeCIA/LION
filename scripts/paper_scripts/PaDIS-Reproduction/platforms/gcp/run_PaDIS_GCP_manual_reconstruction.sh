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
        requested="${PADIS_RECON_TASKS_PER_GPU:-${PADIS_GCP_RECON_TASKS_PER_GPU:-2}}"
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

ramdisk_fs_type() {
        local target="$1"
        if command -v findmnt >/dev/null 2>&1; then
                findmnt -n -o FSTYPE --target "$target" 2>/dev/null || true
        else
                stat -f -c %T "$target" 2>/dev/null || true
        fi
}

ensure_training_ramdisk() {
        local fs_type mount_cmd=(mount -t tmpfs)
        if ! mkdir -p "$PADIS_RAM_DISK" 2>/dev/null; then
                command -v sudo >/dev/null 2>&1 || die "Cannot create $PADIS_RAM_DISK and sudo is unavailable."
                sudo mkdir -p "$PADIS_RAM_DISK"
                sudo chown "$(id -u):$(id -g)" "$PADIS_RAM_DISK"
        fi
        fs_type="$(ramdisk_fs_type "$PADIS_RAM_DISK")"
        case "$fs_type" in
                tmpfs|ramfs)
                        if [ ! -w "$PADIS_RAM_DISK" ] && command -v sudo >/dev/null 2>&1; then
                                sudo chown "$(id -u):$(id -g)" "$PADIS_RAM_DISK"
                        fi
                        return
                        ;;
        esac

        if [ "$PADIS_MANUAL_RECON_CREATE_RAMDISK" != "1" ]; then
                die "$PADIS_RAM_DISK is backed by '$fs_type', not tmpfs/ramfs. Mount it first or set PADIS_MANUAL_RECON_USE_RAMDISK_DATA=0."
        fi

        if [ -n "$PADIS_RAM_DISK_SIZE" ]; then
                mount_cmd+=(-o "size=$PADIS_RAM_DISK_SIZE")
        fi
        mount_cmd+=(tmpfs "$PADIS_RAM_DISK")
        log "Mounting tmpfs ramdisk at $PADIS_RAM_DISK."
        if command -v sudo >/dev/null 2>&1; then
                sudo "${mount_cmd[@]}"
                sudo chown "$(id -u):$(id -g)" "$PADIS_RAM_DISK"
        else
                "${mount_cmd[@]}"
        fi
        PADIS_RAMDISK_MOUNTED_BY_RUNNER=1
        fs_type="$(ramdisk_fs_type "$PADIS_RAM_DISK")"
        case "$fs_type" in
                tmpfs|ramfs)
                        ;;
                *)
                        die "Failed to mount $PADIS_RAM_DISK as tmpfs/ramfs; current type is '$fs_type'."
                        ;;
        esac
}

stage_training_data_on_ramdisk() {
        if [ "$PADIS_MANUAL_RECON_USE_RAMDISK_DATA" != "1" ]; then
                PADIS_PNP_TRAIN_DATA_FOLDER="${PADIS_PNP_TRAIN_DATA_FOLDER:-$PADIS_DATA_FOLDER}"
                export PADIS_PNP_TRAIN_DATA_FOLDER
                return
        fi
        if [ "$PADIS_PNP_CACHE_DATASET" = "none" ]; then
                PADIS_PNP_TRAIN_DATA_FOLDER="${PADIS_PNP_TRAIN_DATA_FOLDER:-$PADIS_DATA_FOLDER}"
                export PADIS_PNP_TRAIN_DATA_FOLDER
                return
        fi

        [ -d "$PADIS_PNP_CACHE_ARCHIVE_FOLDER" ] || die "PnP cache archive folder not found: $PADIS_PNP_CACHE_ARCHIVE_FOLDER"
        ensure_training_ramdisk
        mkdir -p "$PADIS_PNP_CACHE_FOLDER"
        log "Using PnP cache archives from $PADIS_PNP_CACHE_ARCHIVE_FOLDER; materialized tensors will be staged in $PADIS_PNP_CACHE_FOLDER."
        PADIS_PNP_TRAIN_DATA_FOLDER="${PADIS_PNP_TRAIN_DATA_FOLDER:-$PADIS_DATA_FOLDER}"
        export PADIS_PNP_TRAIN_DATA_FOLDER
}

cleanup_training_ramdisk() {
        if [ "$PADIS_MANUAL_RECON_USE_RAMDISK_DATA" != "1" ]; then
                return
        fi
        if [ "$PADIS_MANUAL_RECON_REMOVE_RAMDISK_AFTER_TRAINING" != "1" ]; then
                return
        fi

        if [ "${PADIS_RAMDISK_MOUNTED_BY_RUNNER:-0}" = "1" ]; then
                log "Unmounting temporary training ramdisk at $PADIS_RAM_DISK."
                if command -v sudo >/dev/null 2>&1; then
                        sudo umount "$PADIS_RAM_DISK" \
                                || sudo umount -l "$PADIS_RAM_DISK" \
                                || log "WARNING: failed to unmount busy ramdisk $PADIS_RAM_DISK; continuing."
                        sudo rmdir "$PADIS_RAM_DISK" 2>/dev/null || true
                else
                        umount "$PADIS_RAM_DISK" \
                                || umount -l "$PADIS_RAM_DISK" \
                                || log "WARNING: failed to unmount busy ramdisk $PADIS_RAM_DISK; continuing."
                        rmdir "$PADIS_RAM_DISK" 2>/dev/null || true
                fi
                PADIS_RAMDISK_MOUNTED_BY_RUNNER=0
                PADIS_PNP_TRAIN_DATA_FOLDER=""
                export PADIS_PNP_TRAIN_DATA_FOLDER
                return
        fi

        if [ -n "${PADIS_PNP_CACHE_FOLDER:-}" ] \
                && [ -d "$PADIS_PNP_CACHE_FOLDER" ]; then
                log "Removing staged training cache at $PADIS_PNP_CACHE_FOLDER."
                rm -rf "$PADIS_PNP_CACHE_FOLDER"
        fi
        PADIS_PNP_TRAIN_DATA_FOLDER=""
        export PADIS_PNP_TRAIN_DATA_FOLDER
}

build_manual_wandb_args() {
        local run_name="$1"
        WANDB_ARGS=()
        if [ "$PADIS_NO_WANDB" = "1" ]; then
                WANDB_ARGS+=(--no-wandb --wandb-mode disabled)
                return
        fi

        WANDB_ARGS+=(--wandb-project "$PADIS_WANDB_PROJECT")
        if [ -n "$PADIS_WANDB_ENTITY" ]; then
                WANDB_ARGS+=(--wandb-entity "$PADIS_WANDB_ENTITY")
        fi
        WANDB_ARGS+=(--wandb-mode "$PADIS_WANDB_MODE")
        if [ -n "$PADIS_WANDB_NAME_PREFIX" ]; then
                WANDB_ARGS+=(--wandb-name "$PADIS_WANDB_NAME_PREFIX-$run_name")
        fi
        if [ "$PADIS_NO_WANDB_ARTIFACT" = "1" ]; then
                WANDB_ARGS+=(--no-wandb-artifact)
        fi
}

run_trainable_checkpoint_if_missing() {
        local label="$1"
        local checkpoint_path="$2"
        local run_name="$3"
        local final_name="$4"
        local final_full_name="$5"
        local checkpoint_pattern="$6"
        local validation_name="$7"
        local use_noise_level="$8"
        local max_train_seconds="${9:-}"
        local pnp_data_folder
        local cmd=()

        if [ -f "$checkpoint_path" ]; then
                log "Found $label PnP checkpoint: $checkpoint_path"
                return
        fi

        log "Missing required $label checkpoint; training $run_name before reconstruction."
        pnp_data_folder="${PADIS_PNP_TRAIN_DATA_FOLDER:-$PADIS_DATA_FOLDER}"
        build_manual_wandb_args "$run_name"
        cmd=(
                python -u scripts/paper_scripts/PaDIS-Reproduction/training/PaDIS_LIDC_PnP_denoiser.py
                --output-root "$PADIS_PNP_OUTPUT_ROOT"
                --run-name "$run_name"
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
                --checkpoint-interval-seconds "$PADIS_PNP_CHECKPOINT_INTERVAL_SECONDS"
                --max-periodic-checkpoints "$PADIS_PNP_MAX_PERIODIC_CHECKPOINTS"
                --keep-final-periodic-checkpoints "$PADIS_PNP_FINAL_PERIODIC_CHECKPOINTS"
                --seed "$PADIS_PNP_SEED"
                --device cuda
                --num-workers "$PADIS_PNP_NUM_WORKERS"
                --final-name "$final_name"
                --final-full-name "$final_full_name"
                --checkpoint-pattern "$checkpoint_pattern"
                --validation-name "$validation_name"
                --pcg-slices-nodule "$PADIS_PNP_PCG_SLICES_NODULE"
        )
        if [ "$PADIS_PNP_FULL_LIDC" = "1" ]; then
                cmd+=(--full-lidc)
        fi
        if [ -n "$PADIS_PNP_MAX_TRAIN_SAMPLES" ]; then
                cmd+=(--max-train-samples "$PADIS_PNP_MAX_TRAIN_SAMPLES")
        fi
        if [ -n "$PADIS_PNP_MAX_VALIDATION_SAMPLES" ]; then
                cmd+=(--max-validation-samples "$PADIS_PNP_MAX_VALIDATION_SAMPLES")
        fi
        if [ "$use_noise_level" = "1" ]; then
                cmd+=(--use-noise-level)
        fi
        if [ -n "$PADIS_PNP_PATCH_SIZE" ]; then
                cmd+=(--patch-size "$PADIS_PNP_PATCH_SIZE")
        fi
        if [ -n "$pnp_data_folder" ]; then
                cmd+=(--data-folder "$pnp_data_folder")
        fi
        if [ "$PADIS_PNP_CACHE_DATASET" != "none" ]; then
                cmd+=(
                        --cache-dataset "$PADIS_PNP_CACHE_DATASET"
                        --cache-folder "$PADIS_PNP_CACHE_FOLDER"
                        --cache-archive-folder "$PADIS_PNP_CACHE_ARCHIVE_FOLDER"
                )
                if [ -n "$PADIS_PNP_CACHE_SOURCE_FOLDER" ]; then
                        cmd+=(--cache-source-folder "$PADIS_PNP_CACHE_SOURCE_FOLDER")
                fi
                if [ "$PADIS_PNP_REBUILD_CACHE" = "1" ]; then
                        cmd+=(--rebuild-cache)
                fi
                if [ "$PADIS_PNP_REQUIRE_CACHE_HIT" = "1" ]; then
                        cmd+=(--require-cache-hit)
                fi
        fi
        if [ -n "$max_train_seconds" ]; then
                cmd+=(--max-train-seconds "$max_train_seconds")
        fi
        cmd+=("${WANDB_ARGS[@]}")

        if [ "$PADIS_MANUAL_RECON_DRY_RUN" = "1" ]; then
                printf 'Dry run PnP training command:'
                printf ' %q' "${cmd[@]}"
                printf '\n'
                return
        fi

        "${cmd[@]}"
        [ -f "$checkpoint_path" ] || die "$label training finished but did not create expected checkpoint: $checkpoint_path"
        sync_bucket_mount
}

ensure_reconstruction_training_inputs() {
        local needs_training=0
        if [ "$PADIS_RECON_TRAIN_MISSING_CHECKPOINTS" != "1" ]; then
                return
        fi
        if [ ! -f "$PADIS_RECON_PNP_CHECKPOINT" ] \
                || [ ! -f "$PADIS_RECON_PNP_NOISE_COND_CHECKPOINT" ]; then
                needs_training=1
        fi
        if [ "$needs_training" = "1" ]; then
                stage_training_data_on_ramdisk
        fi
        run_trainable_checkpoint_if_missing \
                "standard" \
                "$PADIS_RECON_PNP_CHECKPOINT" \
                "$PADIS_PNP_RUN_NAME" \
                "$PADIS_PNP_FINAL_NAME" \
                "$PADIS_PNP_FINAL_FULL_NAME" \
                "$PADIS_PNP_CHECKPOINT_PATTERN" \
                "$PADIS_PNP_VALIDATION_NAME" \
                "$PADIS_PNP_USE_NOISE_LEVEL" \
                "$PADIS_PNP_MAX_TRAIN_SECONDS"
        run_trainable_checkpoint_if_missing \
                "noise-conditioned" \
                "$PADIS_RECON_PNP_NOISE_COND_CHECKPOINT" \
                "$PADIS_PNP_NOISE_COND_RUN_NAME" \
                "$PADIS_PNP_NOISE_COND_FINAL_NAME" \
                "$PADIS_PNP_NOISE_COND_FINAL_FULL_NAME" \
                "$PADIS_PNP_NOISE_COND_CHECKPOINT_PATTERN" \
                "$PADIS_PNP_NOISE_COND_VALIDATION_NAME" \
                1 \
                "$PADIS_PNP_NOISE_COND_MAX_TRAIN_SECONDS"
        if [ "$needs_training" = "1" ]; then
                cleanup_training_ramdisk
        fi
}

build_reconstruction_base_command() {
        RECON_BASE_CMD=(
                python -u scripts/paper_scripts/PaDIS-Reproduction/reconstruction/PaDIS_run_reconstruction_matrix.py
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
                --pnp-noise-conditioned-root "$PADIS_PNP_NOISE_COND_ROOT"
                --pnp-noise-conditioned-checkpoint "$PADIS_RECON_PNP_NOISE_COND_CHECKPOINT"
                --pnp-noise-conditioned-noise-level "$PADIS_PNP_NOISE_COND_NOISE_LEVEL"
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
        local new_jobs_json reconcile_args=()
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
        new_jobs_json="$PADIS_RECON_EXPECTED_JOBS_JSON.tmp.$BASHPID"
        "${RECON_BASE_CMD[@]}" --list > "$new_jobs_json"
        if [ "$PADIS_RECON_RECONCILE_MANIFEST" = "1" ]; then
                reconcile_args=(
                        python -u scripts/paper_scripts/PaDIS-Reproduction/reconstruction/PaDIS_reconcile_reconstruction_manifest.py
                        --old-json "$PADIS_RECON_EXPECTED_JOBS_JSON"
                        --new-json "$new_jobs_json"
                        --output-json "$PADIS_RECON_EXPECTED_JOBS_JSON"
                        --state-dir "$STATE_DIR"
                        --done-dir "$DONE_DIR"
                        --failed-dir "$FAILED_DIR"
                )
                if [ -n "$PADIS_RECON_VALIDATE_SETTINGS_MATRIX_GROUPS" ]; then
                        reconcile_args+=(
                                --validate-settings-matrix-groups
                                "$PADIS_RECON_VALIDATE_SETTINGS_MATRIX_GROUPS"
                        )
                fi
                if [ "$PADIS_MANUAL_RECON_DRY_RUN" = "1" ]; then
                        reconcile_args+=(--skip-output-check)
                fi
                "${reconcile_args[@]}" > "$STATE_DIR/reconstruction_manifest_reconcile.json"
        else
                mv "$new_jobs_json" "$PADIS_RECON_EXPECTED_JOBS_JSON"
        fi
        rm -f "$new_jobs_json"
        log "Prepared reconstruction matrix with $RECON_TASK_COUNT jobs at $PADIS_RECON_EXPECTED_JOBS_JSON."
        sync_bucket_mount
}

claim_next_reconstruction_task() {
        local gpu_id="$1"
        local slot_id="${2:-1}"
        local claimed="" task_index marker fd phase_key
        exec {fd}>"$STATE_DIR/reconstruction_queue.lock"
        flock "$fd"
        for ((task_index = RECON_CLAIM_START; task_index < RECON_CLAIM_END; task_index++)); do
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

run_reconstruction_workers() {
        local label="$1"
        local start_index="$2"
        local end_index="$3"
        local worker_pids=()
        local gpu_id slot_id pid phase_rc
        if [ "$start_index" -ge "$end_index" ]; then
                log "Skipping $label reconstruction range because it is empty [$start_index, $end_index)."
                return 0
        fi
        RECON_CLAIM_START="$start_index"
        RECON_CLAIM_END="$end_index"
        log "Starting $label reconstruction range [$RECON_CLAIM_START, $RECON_CLAIM_END) with $RECON_TASKS_PER_GPU worker slot(s) per GPU."
        for gpu_id in "${GPU_IDS[@]}"; do
                for ((slot_id = 1; slot_id <= RECON_TASKS_PER_GPU; slot_id++)); do
                        reconstruction_worker_loop "$gpu_id" "$slot_id" &
                        worker_pids+=("$!")
                done
        done
        phase_rc=0
        for pid in "${worker_pids[@]}"; do
                if ! wait "$pid"; then
                        phase_rc=1
                fi
        done
        return "$phase_rc"
}

generation_barrier_task_index() {
        python - "$PADIS_RECON_EXPECTED_JOBS_JSON" <<'PY'
import json
import sys

with open(sys.argv[1]) as file:
    jobs = json.load(file)
for index, job in enumerate(jobs):
    if job.get("method") in {"patch_average", "patch_stitch"}:
        print(index)
        break
else:
    print(len(jobs))
PY
}

checkpoint_name_for_policy() {
        local prefix="$1"
        local policy="$2"
        if [ "$policy" = "model_default" ]; then
                printf '%s.pt\n' "$prefix"
        else
                printf '%s_%s.pt\n' "$prefix" "$policy"
        fi
}

generation_output_folder() {
        local preset="$1"
        printf '%s/lion-paper-protocol/%s\n' "$PADIS_GENERATION_ROOT" "$preset"
}

generation_done_marker() {
        local preset="$1"
        printf '%s/%s.done\n' "$GENERATION_DONE_DIR" "$preset"
}

generation_running_marker() {
        local preset="$1"
        printf '%s/%s.running\n' "$GENERATION_RUNNING_DIR" "$preset"
}

generation_failed_marker() {
        local preset="$1"
        printf '%s/%s.failed\n' "$GENERATION_FAILED_DIR" "$preset"
}

generation_samples_path() {
        local preset="$1"
        printf '%s/samples.pt\n' "$(generation_output_folder "$preset")"
}

build_generation_command() {
        local preset="$1"
        local checkpoint="$PADIS_GENERATION_PATCH_CHECKPOINT"
        local num_steps="$PADIS_GENERATION_NUM_STEPS"
        case "$preset" in
                paper-generation-whole)
                        checkpoint="$PADIS_GENERATION_WHOLE_CHECKPOINT"
                        ;;
                paper-generation-naive-patch|paper-generation) ;;
                paper-generation-langevin-300nfe)
                        num_steps="$PADIS_GENERATION_LANGEVIN_NUM_STEPS"
                        ;;
                paper-generation-patch-stitch|paper-generation-patch-average) ;;
                *)
                        die "Unknown PaDIS generation preset: $preset"
                        ;;
        esac
        GEN_CMD=(
                python -u scripts/paper_scripts/PaDIS-Reproduction/core/PaDIS_experiments.py
                run "$preset"
                --output-root "$PADIS_GENERATION_ROOT"
                --skip-current
                --checkpoint "$checkpoint"
                --device "$PADIS_GENERATION_DEVICE"
                --max-samples "$PADIS_GENERATION_NUM_SAMPLES"
                --seed "$PADIS_GENERATION_SEED"
                --
                --num-steps "$num_steps"
                --inner-steps "$PADIS_GENERATION_INNER_STEPS"
                --sigma-min "$PADIS_GENERATION_SIGMA_MIN"
                --sigma-max "$PADIS_GENERATION_SIGMA_MAX"
                --noise-schedule "$PADIS_GENERATION_NOISE_SCHEDULE"
                --rho "$PADIS_GENERATION_RHO"
        )
        if [ -n "$PADIS_GENERATION_EPSILON" ]; then
                GEN_CMD+=(--generation-epsilon "$PADIS_GENERATION_EPSILON")
        fi
        if [ -n "$PADIS_GENERATION_NOISE_SCALE" ]; then
                GEN_CMD+=(--langevin-noise-scale "$PADIS_GENERATION_NOISE_SCALE")
        fi
        if [ -n "$PADIS_GENERATION_PATCH_BATCH_SIZE" ]; then
                GEN_CMD+=(--patch-batch-size "$PADIS_GENERATION_PATCH_BATCH_SIZE")
        fi
        if [ "$PADIS_GENERATION_PROG_BAR" = "1" ]; then
                GEN_CMD+=(--prog-bar)
        fi
}

run_generation_task() {
        local preset="$1"
        local samples_path done_marker running_marker failed_marker log_path
        samples_path="$(generation_samples_path "$preset")"
        done_marker="$(generation_done_marker "$preset")"
        running_marker="$(generation_running_marker "$preset")"
        failed_marker="$(generation_failed_marker "$preset")"
        log_path="$LOG_DIR/generation_${preset}.log"

        rm -f "$done_marker"

        build_generation_command "$preset"
        printf '%q ' "${GEN_CMD[@]}" > "$LOG_DIR/generation_${preset}.command.txt"
        printf '\n' >> "$LOG_DIR/generation_${preset}.command.txt"
        {
                printf 'phase=generation\n'
                printf 'preset=%s\n' "$preset"
                printf 'started=%s\n' "$(date --iso-8601=seconds)"
                printf 'host=%s\n' "$(hostname)"
                printf 'log_path=%s\n' "$log_path"
        } > "$running_marker"

        log "Running generation preset $preset; log: $log_path"
        if [ "$PADIS_MANUAL_RECON_DRY_RUN" = "1" ]; then
                return 0
        fi

        if "${GEN_CMD[@]}" > "$log_path" 2>&1; then
                {
                        printf 'completed=%s\n' "$(date --iso-8601=seconds)"
                        printf 'phase=generation\n'
                        printf 'preset=%s\n' "$preset"
                        printf 'samples_path=%s\n' "$samples_path"
                        printf 'log_path=%s\n' "$log_path"
                } > "$done_marker"
                rm -f "$running_marker" "$failed_marker"
                sync_bucket_mount
                log "Generation preset $preset completed."
                return 0
        fi

        {
                printf 'failed=%s\n' "$(date --iso-8601=seconds)"
                printf 'phase=generation\n'
                printf 'preset=%s\n' "$preset"
                printf 'samples_path=%s\n' "$samples_path"
                printf 'log_path=%s\n' "$log_path"
        } > "$failed_marker"
        rm -f "$running_marker"
        log "Generation preset $preset failed. See $log_path"
        return 1
}

run_generation_phase() {
        local preset
        if [ "$PADIS_GENERATION_PHASE" != "1" ]; then
                log "Skipping generation phase because PADIS_GENERATION_PHASE=$PADIS_GENERATION_PHASE."
                return 0
        fi
        [ -f "$PADIS_GENERATION_PATCH_CHECKPOINT" ] || die "Missing patch generation checkpoint: $PADIS_GENERATION_PATCH_CHECKPOINT"
        [ -f "$PADIS_GENERATION_WHOLE_CHECKPOINT" ] || die "Missing whole-image generation checkpoint: $PADIS_GENERATION_WHOLE_CHECKPOINT"

        mkdir -p "$PADIS_GENERATION_ROOT" "$GENERATION_DONE_DIR" "$GENERATION_RUNNING_DIR" "$GENERATION_FAILED_DIR"
        rm -f "$GENERATION_RUNNING_DIR"/*.running
        log "Starting generation phase for presets: $PADIS_GENERATION_PRESETS"
        for preset in ${PADIS_GENERATION_PRESETS//,/ }; do
                if ! run_generation_task "$preset"; then
                        return 1
                fi
        done
        return 0
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
                printf 'reconstruction_reconcile_manifest=%s\n' "$PADIS_RECON_RECONCILE_MANIFEST"
                printf 'reconstruction_hparam_defaults=%s\n' "$PADIS_RECON_HPARAM_DEFAULTS"
                printf 'reconstruction_hparam_defaults_json=%s\n' "$PADIS_RECON_HPARAM_DEFAULTS_JSON"
                printf 'reconstruction_methods=%s\n' "$PADIS_RECON_METHODS"
                printf 'reconstruction_experiments=%s\n' "$PADIS_RECON_EXPERIMENTS"
                printf 'reconstruction_ablations=%s\n' "$PADIS_RECON_ABLATIONS"
                printf 'reconstruction_expected_jobs_json=%s\n' "$PADIS_RECON_EXPECTED_JOBS_JSON"
                printf 'generation_enabled=%s\n' "$PADIS_GENERATION_PHASE"
                printf 'generation_root=%s\n' "$PADIS_GENERATION_ROOT"
                printf 'generation_presets=%s\n' "$PADIS_GENERATION_PRESETS"
                printf 'generation_num_steps=%s\n' "$PADIS_GENERATION_NUM_STEPS"
                printf 'generation_inner_steps=%s\n' "$PADIS_GENERATION_INNER_STEPS"
                printf 'generation_sigma_min=%s\n' "$PADIS_GENERATION_SIGMA_MIN"
                printf 'generation_sigma_max=%s\n' "$PADIS_GENERATION_SIGMA_MAX"
                printf 'generation_noise_schedule=%s\n' "$PADIS_GENERATION_NOISE_SCHEDULE"
                printf 'generation_epsilon_override=%s\n' "$PADIS_GENERATION_EPSILON"
                printf 'generation_noise_scale_override=%s\n' "$PADIS_GENERATION_NOISE_SCALE"
                printf 'sync_after_job=%s\n' "${PADIS_RECON_SYNC_AFTER_JOB:-1}"
                printf 'dry_run=%s\n' "$PADIS_MANUAL_RECON_DRY_RUN"
        } > "$manifest"
}

clear_stale_running_markers() {
        if [ "${PADIS_RECON_CLEAR_STALE_RUNNING:-1}" = "1" ]; then
                find "$RUNNING_DIR" -maxdepth 1 -type f -name 'reconstruction_*.running' -delete
                find "$GENERATION_RUNNING_DIR" -maxdepth 1 -type f -name '*.running' -delete 2>/dev/null || true
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

LION_ROOT="${LION_ROOT:-$(cd "$SCRIPT_DIR/../../../../.." && pwd -P)}"
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
PADIS_RECON_HPARAM_DEFAULTS_JSON="${PADIS_RECON_HPARAM_DEFAULTS_JSON:-$LION_ROOT/scripts/paper_scripts/PaDIS-Reproduction/config/reconstruction_hparam_defaults.json}"
PADIS_RECON_HPARAM_RUN_ROOT="${PADIS_RECON_HPARAM_RUN_ROOT:-$PADIS_RUN_ROOT/hparam_tuning/runs}"
PADIS_RECON_HPARAM_RUN_GLOB="${PADIS_RECON_HPARAM_RUN_GLOB:-fixedval_*}"
PADIS_RECON_ALLOW_MISSING_CHECKPOINTS="${PADIS_RECON_ALLOW_MISSING_CHECKPOINTS:-0}"
PADIS_RECON_EXPECTED_JOBS_JSON="${PADIS_RECON_EXPECTED_JOBS_JSON:-$PADIS_RECON_ROOT/reconstruction_matrix_jobs.json}"
PADIS_RECON_RECONCILE_MANIFEST="${PADIS_RECON_RECONCILE_MANIFEST:-1}"
PADIS_RECON_VALIDATE_SETTINGS_MATRIX_GROUPS="${PADIS_RECON_VALIDATE_SETTINGS_MATRIX_GROUPS:-dataset_size_patch_full,dataset_size_whole_full}"
PADIS_GENERATION_PHASE="${PADIS_GENERATION_PHASE:-1}"
PADIS_GENERATION_ROOT="${PADIS_GENERATION_ROOT:-$LION_EXPERIMENTS_PATH/PaDIS/reconstruction_presets}"
PADIS_GENERATION_PRESETS="${PADIS_GENERATION_PRESETS:-paper-generation-whole,paper-generation-naive-patch,paper-generation,paper-generation-langevin-300nfe,paper-generation-patch-stitch,paper-generation-patch-average}"
PADIS_GENERATION_PATCH_CHECKPOINT="${PADIS_GENERATION_PATCH_CHECKPOINT:-$PADIS_TRAIN_ROOT/patch_lidc_default/$(checkpoint_name_for_policy padis_lidc_256 "$PADIS_RECON_CHECKPOINT_POLICY")}"
PADIS_GENERATION_WHOLE_CHECKPOINT="${PADIS_GENERATION_WHOLE_CHECKPOINT:-$PADIS_TRAIN_ROOT/whole_lidc_default/$(checkpoint_name_for_policy whole_image_lidc_256 "$PADIS_RECON_CHECKPOINT_POLICY")}"
PADIS_GENERATION_DEVICE="${PADIS_GENERATION_DEVICE:-$PADIS_RECON_DEVICE}"
PADIS_GENERATION_NUM_SAMPLES="${PADIS_GENERATION_NUM_SAMPLES:-4}"
PADIS_GENERATION_SEED="${PADIS_GENERATION_SEED:-$PADIS_RECON_SEED}"
PADIS_GENERATION_NUM_STEPS="${PADIS_GENERATION_NUM_STEPS:-300}"
PADIS_GENERATION_LANGEVIN_NUM_STEPS="${PADIS_GENERATION_LANGEVIN_NUM_STEPS:-300}"
PADIS_GENERATION_INNER_STEPS="${PADIS_GENERATION_INNER_STEPS:-1}"
PADIS_GENERATION_SIGMA_MIN="${PADIS_GENERATION_SIGMA_MIN:-0.002}"
PADIS_GENERATION_SIGMA_MAX="${PADIS_GENERATION_SIGMA_MAX:-10.0}"
PADIS_GENERATION_NOISE_SCHEDULE="${PADIS_GENERATION_NOISE_SCHEDULE:-geometric}"
PADIS_GENERATION_RHO="${PADIS_GENERATION_RHO:-7.0}"
PADIS_GENERATION_EPSILON="${PADIS_GENERATION_EPSILON:-}"
PADIS_GENERATION_NOISE_SCALE="${PADIS_GENERATION_NOISE_SCALE:-}"
PADIS_GENERATION_PATCH_BATCH_SIZE="${PADIS_GENERATION_PATCH_BATCH_SIZE:-}"
PADIS_GENERATION_PROG_BAR="${PADIS_GENERATION_PROG_BAR:-$PADIS_RECON_PROG_BAR}"
PADIS_PNP_OUTPUT_ROOT="${PADIS_PNP_OUTPUT_ROOT:-$PADIS_TRAIN_ROOT}"
PADIS_RECON_TRAIN_MISSING_CHECKPOINTS="${PADIS_RECON_TRAIN_MISSING_CHECKPOINTS:-${PADIS_RECON_TRAIN_MISSING_PNP:-1}}"
PADIS_RAM_DISK="${PADIS_RAM_DISK:-/mnt/ram-disk}"
PADIS_RAM_DISK_SIZE="${PADIS_RAM_DISK_SIZE:-}"
PADIS_MANUAL_RECON_USE_RAMDISK_DATA="${PADIS_MANUAL_RECON_USE_RAMDISK_DATA:-1}"
PADIS_MANUAL_RECON_CREATE_RAMDISK="${PADIS_MANUAL_RECON_CREATE_RAMDISK:-1}"
PADIS_MANUAL_RECON_REMOVE_RAMDISK_AFTER_TRAINING="${PADIS_MANUAL_RECON_REMOVE_RAMDISK_AFTER_TRAINING:-1}"
PADIS_PNP_TRAIN_DATA_FOLDER="${PADIS_PNP_TRAIN_DATA_FOLDER:-}"
PADIS_CACHE_ROOT="${PADIS_CACHE_ROOT:-$LION_DATA_PATH/processed/LIDC-IDRI-cache}"
PADIS_PNP_CACHE_DATASET="${PADIS_PNP_CACHE_DATASET:-ramdisk}"
PADIS_PNP_CACHE_FOLDER="${PADIS_PNP_CACHE_FOLDER:-$PADIS_RAM_DISK/lion_lidc_cache_256}"
PADIS_PNP_CACHE_ARCHIVE_FOLDER="${PADIS_PNP_CACHE_ARCHIVE_FOLDER:-$PADIS_CACHE_ROOT/padis_256/archives}"
PADIS_PNP_CACHE_SOURCE_FOLDER="${PADIS_PNP_CACHE_SOURCE_FOLDER:-}"
PADIS_PNP_REBUILD_CACHE="${PADIS_PNP_REBUILD_CACHE:-0}"
PADIS_PNP_REQUIRE_CACHE_HIT="${PADIS_PNP_REQUIRE_CACHE_HIT:-1}"
PADIS_NO_WANDB_ARTIFACT="${PADIS_NO_WANDB_ARTIFACT:-0}"
PADIS_WANDB_PROJECT="${PADIS_WANDB_PROJECT:-PaDIS-Reproduction}"
PADIS_WANDB_ENTITY="${PADIS_WANDB_ENTITY:-}"
PADIS_WANDB_MODE="${PADIS_WANDB_MODE:-online}"
PADIS_NO_WANDB="${PADIS_NO_WANDB:-0}"
PADIS_WANDB_NAME_PREFIX="${PADIS_WANDB_NAME_PREFIX:-PaDIS-Reproduction-GCP}"
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
PADIS_PNP_PCG_SLICES_NODULE="${PADIS_PNP_PCG_SLICES_NODULE:-0.5}"
PADIS_PNP_MAX_TRAIN_SAMPLES="${PADIS_PNP_MAX_TRAIN_SAMPLES:-}"
PADIS_PNP_MAX_VALIDATION_SAMPLES="${PADIS_PNP_MAX_VALIDATION_SAMPLES:-}"
PADIS_PNP_FULL_LIDC="${PADIS_PNP_FULL_LIDC:-0}"
PADIS_PNP_USE_NOISE_LEVEL="${PADIS_PNP_USE_NOISE_LEVEL:-0}"
PADIS_PNP_INT_CHANNELS="${PADIS_PNP_INT_CHANNELS:-64}"
PADIS_PNP_N_BLOCKS="${PADIS_PNP_N_BLOCKS:-4}"
PADIS_PNP_PATCH_SIZE="${PADIS_PNP_PATCH_SIZE:-}"
PADIS_PNP_PATCHES_PER_IMAGE="${PADIS_PNP_PATCHES_PER_IMAGE:-1}"
PADIS_PNP_VALIDATION_EVERY="${PADIS_PNP_VALIDATION_EVERY:-1}"
PADIS_PNP_CHECKPOINT_EVERY="${PADIS_PNP_CHECKPOINT_EVERY:-10}"
PADIS_PNP_CHECKPOINT_INTERVAL_SECONDS="${PADIS_PNP_CHECKPOINT_INTERVAL_SECONDS:-300}"
PADIS_PNP_MAX_PERIODIC_CHECKPOINTS="${PADIS_PNP_MAX_PERIODIC_CHECKPOINTS:-2}"
PADIS_PNP_FINAL_PERIODIC_CHECKPOINTS="${PADIS_PNP_FINAL_PERIODIC_CHECKPOINTS:-1}"
PADIS_PNP_MAX_TRAIN_SECONDS="${PADIS_PNP_MAX_TRAIN_SECONDS:-}"
PADIS_PNP_SEED="${PADIS_PNP_SEED:-$PADIS_RECON_SEED}"
PADIS_PNP_NUM_WORKERS="${PADIS_PNP_NUM_WORKERS:-4}"
PADIS_PNP_FINAL_NAME="${PADIS_PNP_FINAL_NAME:-pnp_lidc_drunet.pt}"
PADIS_PNP_FINAL_FULL_NAME="${PADIS_PNP_FINAL_FULL_NAME:-${PADIS_PNP_FINAL_NAME%.pt}_full.pt}"
PADIS_PNP_CHECKPOINT_PATTERN="${PADIS_PNP_CHECKPOINT_PATTERN:-pnp_lidc_drunet_check_*.pt}"
PADIS_PNP_VALIDATION_NAME="${PADIS_PNP_VALIDATION_NAME:-pnp_lidc_drunet_min_val.pt}"
PADIS_PNP_ROOT="${PADIS_PNP_ROOT:-$PADIS_PNP_OUTPUT_ROOT/$PADIS_PNP_RUN_NAME}"
PADIS_RECON_PNP_CHECKPOINT="${PADIS_RECON_PNP_CHECKPOINT:-${PADIS_PNP_CHECKPOINT:-$PADIS_PNP_ROOT/$PADIS_PNP_VALIDATION_NAME}}"
PADIS_PNP_NOISE_COND_RUN_NAME="${PADIS_PNP_NOISE_COND_RUN_NAME:-pnp_lidc_drunet_noise_cond}"
PADIS_PNP_NOISE_COND_FINAL_NAME="${PADIS_PNP_NOISE_COND_FINAL_NAME:-pnp_lidc_drunet_noise_cond.pt}"
PADIS_PNP_NOISE_COND_FINAL_FULL_NAME="${PADIS_PNP_NOISE_COND_FINAL_FULL_NAME:-${PADIS_PNP_NOISE_COND_FINAL_NAME%.pt}_full.pt}"
PADIS_PNP_NOISE_COND_CHECKPOINT_PATTERN="${PADIS_PNP_NOISE_COND_CHECKPOINT_PATTERN:-pnp_lidc_drunet_noise_cond_check_*.pt}"
PADIS_PNP_NOISE_COND_VALIDATION_NAME="${PADIS_PNP_NOISE_COND_VALIDATION_NAME:-pnp_lidc_drunet_noise_cond_min_val.pt}"
PADIS_PNP_NOISE_COND_ROOT="${PADIS_PNP_NOISE_COND_ROOT:-$PADIS_PNP_OUTPUT_ROOT/$PADIS_PNP_NOISE_COND_RUN_NAME}"
PADIS_PNP_NOISE_COND_MAX_TRAIN_SECONDS="${PADIS_PNP_NOISE_COND_MAX_TRAIN_SECONDS:-$PADIS_PNP_MAX_TRAIN_SECONDS}"
PADIS_PNP_NOISE_COND_NOISE_LEVEL="${PADIS_PNP_NOISE_COND_NOISE_LEVEL:-0.03}"
PADIS_RECON_PNP_NOISE_COND_CHECKPOINT="${PADIS_RECON_PNP_NOISE_COND_CHECKPOINT:-${PADIS_PNP_NOISE_COND_CHECKPOINT:-$PADIS_PNP_NOISE_COND_ROOT/$PADIS_PNP_NOISE_COND_VALIDATION_NAME}}"
PADIS_PNP_ITERATIONS="${PADIS_PNP_ITERATIONS:-20}"
PADIS_PNP_ETA="${PADIS_PNP_ETA:-1e-5}"
PADIS_PNP_CG_ITERATIONS="${PADIS_PNP_CG_ITERATIONS:-50}"
PADIS_PNP_CG_TOLERANCE="${PADIS_PNP_CG_TOLERANCE:-1e-7}"
PADIS_PNP_NOISE_LEVEL="${PADIS_PNP_NOISE_LEVEL:-}"
PADIS_TV_LAMBDA="${PADIS_TV_LAMBDA:-0.001}"
PADIS_TV_ITERATIONS="${PADIS_TV_ITERATIONS:-500}"
PADIS_DATA_FOLDER="${PADIS_DATA_FOLDER:-}"
PADIS_PUBLIC_IMAGE_DIR="${PADIS_PUBLIC_IMAGE_DIR:-}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$PADIS_RECON_ROOT/matplotlib}"
MPLBACKEND="${PADIS_MPLBACKEND:-Agg}"
WANDB_DIR="${WANDB_DIR:-$PADIS_TRAIN_ROOT/wandb}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PADIS_WANDB_NETRC="${PADIS_WANDB_NETRC:-/mnt/data/.netrc}"
if [ -z "${NETRC:-}" ] && [ -f "$PADIS_WANDB_NETRC" ]; then
        NETRC="$PADIS_WANDB_NETRC"
fi

STATE_DIR="${PADIS_MANUAL_RECON_STATE_DIR:-$PADIS_RECON_ROOT/.manual_gcp_reconstruction}"
DONE_DIR="$STATE_DIR/done"
RUNNING_DIR="$STATE_DIR/running"
FAILED_DIR="$STATE_DIR/failed"
LOG_DIR="$STATE_DIR/logs"
GENERATION_DONE_DIR="$STATE_DIR/generation_done"
GENERATION_RUNNING_DIR="$STATE_DIR/generation_running"
GENERATION_FAILED_DIR="$STATE_DIR/generation_failed"

export LION_ROOT LION_DATA_PATH LION_EXPERIMENTS_PATH PADIS_RUN_ROOT
export PADIS_TRAIN_ROOT PADIS_RECON_ROOT PADIS_RECON_EXPECTED_JOBS_JSON
export MPLCONFIGDIR MPLBACKEND WANDB_DIR PYTORCH_CUDA_ALLOC_CONF PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export PADIS_WANDB_PROJECT PADIS_WANDB_ENTITY PADIS_WANDB_MODE PADIS_NO_WANDB
export PADIS_WANDB_NETRC
if [ -n "${NETRC:-}" ]; then
        export NETRC
fi

mkdir -p "$PADIS_RECON_ROOT" "$MPLCONFIGDIR" "$WANDB_DIR" "$STATE_DIR" "$DONE_DIR" "$RUNNING_DIR" "$FAILED_DIR" "$LOG_DIR" "$GENERATION_DONE_DIR" "$GENERATION_RUNNING_DIR" "$GENERATION_FAILED_DIR"
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
log "Generation root: $PADIS_GENERATION_ROOT"
log "Selected GPUs: ${GPU_IDS[*]}"
log "Worker slots per GPU: $RECON_TASKS_PER_GPU"

ensure_reconstruction_training_inputs
prepare_reconstruction_matrix

phase_rc=0
GENERATION_BARRIER_INDEX="$(generation_barrier_task_index)"
log "Generation barrier task index: $GENERATION_BARRIER_INDEX"
if [ "$PADIS_GENERATION_PHASE" = "1" ]; then
        if ! run_reconstruction_workers "pre-generation" 0 "$GENERATION_BARRIER_INDEX"; then
                phase_rc=1
        fi
        if [ "$phase_rc" -eq 0 ]; then
                if ! run_generation_phase; then
                        log "One or more PaDIS generation tasks failed; inspect $GENERATION_FAILED_DIR and $LOG_DIR."
                        phase_rc=1
                fi
        fi
        if [ "$phase_rc" -eq 0 ]; then
                if ! run_reconstruction_workers "post-generation" "$GENERATION_BARRIER_INDEX" "$RECON_TASK_COUNT"; then
                        phase_rc=1
                fi
        fi
else
        if ! run_reconstruction_workers "full" 0 "$RECON_TASK_COUNT"; then
                phase_rc=1
        fi
fi

sync_bucket_mount
if [ "$phase_rc" -eq 0 ]; then
        log "Manual PaDIS reconstruction matrix completed."
else
        log "Manual PaDIS reconstruction matrix finished with failures; inspect $FAILED_DIR and $LOG_DIR."
fi
exit "$phase_rc"
