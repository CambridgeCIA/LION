#!/usr/bin/env bash
#
# Run the LION PaDIS training matrix on a single GCP spot VM.
#
# The runner is intentionally resumable: rerunning it with the same
# PADIS_GCP_RUN_NAME/PADIS_TRAIN_ROOT reuses existing run folders, WandB ids,
# runtime ledgers, and checkpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SLURM_HELPER_DIR="$(cd "$SCRIPT_DIR/../slurm" && pwd -P)"
# shellcheck source=scripts/paper_scripts/PaDIS/slurm/padis_a100_common.sh
. "$SLURM_HELPER_DIR/padis_a100_common.sh"

die() {
        echo "$*" >&2
        exit 1
}

log() {
        printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

time_to_seconds() {
        local value="$1"
        if [[ "$value" =~ ^[0-9]+$ ]]; then
                printf '%s\n' "$value"
        else
                padis_time_to_seconds "$value"
        fi
}

activate_environment() {
        local mamba_bin conda_bin conda_sh env_candidates env_name activated conda_lib
        local default_conda_envs_path default_lion_env_prefix
        local env_list

        if [ "${PADIS_GCP_SKIP_ENV_ACTIVATE:-0}" = "1" ]; then
                log "Skipping environment activation because PADIS_GCP_SKIP_ENV_ACTIVATE=1."
                return
        fi
        if [ -n "${CONDA_PREFIX:-}" ] && command -v python >/dev/null 2>&1; then
                log "Using active Python environment at $CONDA_PREFIX."
                return
        fi

        default_conda_envs_path="${CONDA_ENVS_PATH:-/mnt/data/conda/envs}"
        default_lion_env_prefix="${LION_CONDA_ENV:-$default_conda_envs_path/lion}"
        if [ -d "$default_conda_envs_path" ]; then
                export CONDA_ENVS_PATH="$default_conda_envs_path"
        fi

        mamba_bin=""
        if [ -n "${MAMBA_EXE:-}" ] && [ -x "$MAMBA_EXE" ]; then
                mamba_bin="$MAMBA_EXE"
        elif [ -x "${MAMBA_ROOT_PREFIX:-$HOME/miniforge3}/bin/mamba" ]; then
                mamba_bin="${MAMBA_ROOT_PREFIX:-$HOME/miniforge3}/bin/mamba"
        elif command -v mamba >/dev/null 2>&1; then
                mamba_bin="$(command -v mamba)"
        fi

        env_list="$(
                printf '%s %s %s %s %s\n' \
                        "${LION_CONDA_ENV:-}" \
                        "${LION_MAMBA_ENV:-}" \
                        "$default_lion_env_prefix" \
                        "lion" \
                        "${LION_CONDA_ENV_FALLBACKS:-$default_conda_envs_path/lion-dev $default_conda_envs_path/padis-dev lion-dev padis-dev}"
        )"
        read -r -a env_candidates <<< "$env_list"

        if [ -n "$mamba_bin" ]; then
                eval "$("$mamba_bin" shell hook --shell bash)"
                activated=""
                for env_name in "${env_candidates[@]}"; do
                        if [ -z "$env_name" ]; then
                                continue
                        fi
                        if mamba activate "$env_name"; then
                                activated="$env_name"
                                break
                        fi
                done
                [ -n "$activated" ] || die \
                        "Failed to activate any mamba environment from: ${env_candidates[*]}"
                LION_CONDA_ENV="$activated"
                LION_MAMBA_ENV="$activated"
                conda_lib="${CONDA_PREFIX:-${MAMBA_ROOT_PREFIX:-$HOME/miniforge3}/envs/$activated}/lib"
                export LION_CONDA_ENV LION_MAMBA_ENV LD_LIBRARY_PATH="$conda_lib:${LD_LIBRARY_PATH:-}"
                log "Activated $activated using mamba."
                return
        fi

        conda_bin=""
        if [ -n "${CONDA_EXE:-}" ] && [ -x "$CONDA_EXE" ]; then
                conda_bin="$CONDA_EXE"
        elif [ -x /mnt/data/conda/miniconda3/bin/conda ]; then
                conda_bin="/mnt/data/conda/miniconda3/bin/conda"
        elif command -v conda >/dev/null 2>&1; then
                conda_bin="$(command -v conda)"
        fi

        if [ -n "$conda_bin" ]; then
                conda_sh="$(cd "$(dirname "$conda_bin")/.." && pwd -P)/etc/profile.d/conda.sh"
                if [ -f "$conda_sh" ]; then
                        # shellcheck source=/dev/null
                        . "$conda_sh"
                else
                        eval "$("$conda_bin" shell.bash hook)"
                fi
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
                [ -n "$activated" ] || die \
                        "Failed to activate any conda environment from: ${env_candidates[*]}"
                LION_CONDA_ENV="$activated"
                conda_lib="${CONDA_PREFIX:-$(dirname "$(dirname "$conda_bin")")/envs/$activated}/lib"
                export LION_CONDA_ENV LD_LIBRARY_PATH="$conda_lib:${LD_LIBRARY_PATH:-}"
                log "Activated $activated using conda."
                return
        fi

        if ! command -v python >/dev/null 2>&1; then
                die \
                        "No active Python, mamba, or conda found." \
                        "Activate the LION environment or set CONDA_EXE/MAMBA_EXE."
        fi
        if ! python -c "import torch" >/dev/null 2>&1; then
                die \
                        "No conda/mamba environment could be activated, and PATH python lacks torch." \
                        "Activate the LION conda environment before running."
        fi
        log "No conda or mamba found; using python from PATH: $(command -v python)."
}

discover_gpu_ids() {
        local raw_ids max_gpus gpu id count
        raw_ids="${PADIS_GCP_GPU_IDS:-}"
        if [ -z "$raw_ids" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]; then
                raw_ids="$CUDA_VISIBLE_DEVICES"
        fi
        if [ -z "$raw_ids" ] && command -v nvidia-smi >/dev/null 2>&1; then
                raw_ids="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',')"
        fi
        raw_ids="${raw_ids:-0}"
        raw_ids="${raw_ids//,/ }"

        max_gpus="${PADIS_GCP_MAX_GPUS:-1}"
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
        if ! command -v nvidia-smi >/dev/null 2>&1; then
                return 1
        fi
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
        requested="${PADIS_GCP_RECON_TASKS_PER_GPU:-auto}"
        if [ "$requested" != "auto" ]; then
                if ! [[ "$requested" =~ ^[1-9][0-9]*$ ]]; then
                        die "PADIS_GCP_RECON_TASKS_PER_GPU must be a positive integer or auto."
                fi
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

        # The GCP RTX PRO 6000 Blackwell exposes roughly 96 GiB. Reconstruction
        # runs without backprop and the large patch/512 rows use small patch
        # batches, so three concurrent processes is a useful default.
        if [ -n "$min_memory" ] && [ "$min_memory" -ge 90000 ]; then
                RECON_TASKS_PER_GPU=3
        fi
}

task_index_by_name() {
        local task_name="$1"
        local i
        for i in "${!PADIS_TASK_NAMES[@]}"; do
                if [ "${PADIS_TASK_NAMES[$i]}" = "$task_name" ]; then
                        printf '%s\n' "$i"
                        return 0
                fi
        done
        return 1
}

task_category() {
        local task_name="$1"
        if [[ "$task_name" == whole_lidc_* ]]; then
                printf 'whole\n'
        elif [[ "$task_name" == patch_lidc_* ]]; then
                printf 'patch\n'
        elif [ "$task_name" = "$PNP_TASK_NAME" ] \
                || [ "$task_name" = "$PNP_NOISE_COND_TASK_NAME" ]; then
                printf 'pnp\n'
        else
                die "Unknown task category for $task_name"
        fi
}

diffusion_run_prefix_from_engine_args() {
        local engine="$1"
        local task_args="$2"
        if [ "$engine" = "lidc512" ]; then
                printf 'padis_lidc_512\n'
        elif [[ " $task_args " == *" --prior-mode whole-image "* ]]; then
                printf 'whole_image_lidc_256\n'
        else
                printf 'padis_lidc_256\n'
        fi
}

task_budget_seconds() {
        local task_name="$1"
        if [ "${PADIS_GCP_PHASE:-base}" = "validation_heavy" ]; then
                case "$(task_category "$task_name")" in
                        patch|whole)
                                printf '%s\n' "$VALIDATION_HEAVY_SECONDS"
                                ;;
                        pnp)
                                printf '\n'
                                ;;
                esac
                return
        fi
        case "$(task_category "$task_name")" in
                patch)
                        printf '%s\n' "$PATCH_TRAIN_SECONDS"
                        ;;
                whole)
                        printf '%s\n' "$WHOLE_TRAIN_SECONDS"
                        ;;
                pnp)
                        printf '\n'
                        ;;
        esac
}

phase_task_key() {
        local task_name="$1"
        local phase="${2:-${PADIS_GCP_PHASE:-base}}"
        if [ "$phase" = "base" ]; then
                printf '%s\n' "$task_name"
        elif [[ "$task_name" == *".$phase" ]]; then
                printf '%s\n' "$task_name"
        else
                printf '%s.%s\n' "$task_name" "$phase"
        fi
}

task_done_marker() {
        local task_name="$1"
        local phase="${2:-${PADIS_GCP_PHASE:-base}}"
        printf '%s/%s.done\n' "$DONE_DIR" "$(phase_task_key "$task_name" "$phase")"
}

task_running_marker() {
        local task_name="$1"
        local phase="${2:-${PADIS_GCP_PHASE:-base}}"
        printf '%s/%s.running\n' "$RUNNING_DIR" "$(phase_task_key "$task_name" "$phase")"
}

task_elapsed_seconds() {
        local task_name="$1"
        local path="$RUNTIME_DIR/$(phase_task_key "$task_name").seconds"
        if [ -f "$path" ]; then
                cat "$path"
        else
                printf '0\n'
        fi
}

write_task_elapsed_seconds() {
        local task_name="$1"
        local seconds="$2"
        local path="$RUNTIME_DIR/$(phase_task_key "$task_name").seconds"
        printf '%s\n' "$seconds" > "$path.tmp"
        mv "$path.tmp" "$path"
}

read_marker_value() {
        local marker="$1"
        local key="$2"
        local line
        line="$(grep -m 1 "^${key}=" "$marker" 2>/dev/null || true)"
        if [ -n "$line" ]; then
                printf '%s\n' "${line#*=}"
        fi
}

write_running_task_metadata() {
        local task_name="$1"
        local gpu_id="$2"
        local start_total="$3"
        local start_epoch="$4"
        local child_pid="$5"
        local monitor_pid="$6"
        local log_path="$7"
        local marker
        marker="$(task_running_marker "$task_name")"
        local tmp="$marker.tmp.$BASHPID"
        {
                printf 'task=%s\n' "$(phase_task_key "$task_name")"
                printf 'base_task=%s\n' "$task_name"
                printf 'phase=%s\n' "${PADIS_GCP_PHASE:-base}"
                printf 'gpu=%s\n' "$gpu_id"
                printf 'pid=%s\n' "$$"
                printf 'worker_pid=%s\n' "$BASHPID"
                printf 'host=%s\n' "$(hostname)"
                printf 'started=%s\n' "$(date --date="@$start_epoch" --iso-8601=seconds)"
                printf 'start_epoch=%s\n' "$start_epoch"
                printf 'start_elapsed=%s\n' "$start_total"
                printf 'child_pid=%s\n' "$child_pid"
                printf 'monitor_pid=%s\n' "$monitor_pid"
                printf 'log_path=%s\n' "$log_path"
        } > "$tmp"
        mv "$tmp" "$marker"
}

refresh_active_runtimes() {
        local marker task start_elapsed start_epoch now total
        now="$(date +%s)"
        for marker in "$RUNNING_DIR"/*.running; do
                [ -e "$marker" ] || continue
                task="$(read_marker_value "$marker" task)"
                if [ -z "$task" ]; then
                        task="${marker##*/}"
                        task="${task%.running}"
                fi
                start_elapsed="$(read_marker_value "$marker" start_elapsed)"
                start_epoch="$(read_marker_value "$marker" start_epoch)"
                if [[ "$start_elapsed" =~ ^[0-9]+$ && "$start_epoch" =~ ^[0-9]+$ ]]; then
                        total=$((start_elapsed + now - start_epoch))
                        if [ "$total" -lt "$start_elapsed" ]; then
                                total="$start_elapsed"
                        fi
                        write_task_elapsed_seconds "$task" "$total"
                fi
        done
}

terminate_active_children() {
        local marker child_pid monitor_pid
        for marker in "$RUNNING_DIR"/*.running; do
                [ -e "$marker" ] || continue
                child_pid="$(read_marker_value "$marker" child_pid)"
                if [[ "$child_pid" =~ ^[0-9]+$ ]] && kill -0 "$child_pid" >/dev/null 2>&1; then
                        kill -TERM "$child_pid" >/dev/null 2>&1 || true
                fi
                monitor_pid="$(read_marker_value "$marker" monitor_pid)"
                if [[ "$monitor_pid" =~ ^[0-9]+$ ]] && kill -0 "$monitor_pid" >/dev/null 2>&1; then
                        kill -TERM "$monitor_pid" >/dev/null 2>&1 || true
                fi
        done
}

remaining_budget_seconds() {
        local task_name="$1"
        local budget elapsed remaining
        budget="$(task_budget_seconds "$task_name")"
        if [ -z "$budget" ]; then
                printf '\n'
                return
        fi
        elapsed="$(task_elapsed_seconds "$task_name")"
        remaining=$((budget - elapsed))
        if [ "$remaining" -le 0 ]; then
                remaining="$PADIS_GCP_FINALIZE_SECONDS"
        fi
        printf '%s\n' "$remaining"
}

task_final_checkpoint() {
        local task_name="$1"
        local index task_args engine prefix
        if [ "$task_name" = "$PNP_TASK_NAME" ]; then
                printf '%s\n' "$PADIS_TRAIN_ROOT/$PADIS_PNP_RUN_NAME/$PADIS_PNP_FINAL_NAME"
                return
        fi
        if [ "$task_name" = "$PNP_NOISE_COND_TASK_NAME" ]; then
                printf '%s\n' "$PADIS_TRAIN_ROOT/$PADIS_PNP_NOISE_COND_RUN_NAME/$PADIS_PNP_NOISE_COND_FINAL_NAME"
                return
        fi

        index="$(task_index_by_name "$task_name")"
        task_args="${PADIS_TASK_ARGUMENTS[$index]}"
        engine="${PADIS_TASK_ENGINES[$index]}"
        prefix="$(diffusion_run_prefix_from_engine_args "$engine" "$task_args")"
        printf '%s\n' "$PADIS_TRAIN_ROOT/$task_name/$prefix.pt"
}

task_final_full_checkpoint() {
        local task_name="$1"
        local final_checkpoint
        if [ "$task_name" = "$PNP_TASK_NAME" ]; then
                printf '%s\n' "$PADIS_TRAIN_ROOT/$PADIS_PNP_RUN_NAME/$PADIS_PNP_FINAL_FULL_NAME"
                return
        fi
        if [ "$task_name" = "$PNP_NOISE_COND_TASK_NAME" ]; then
                printf '%s\n' "$PADIS_TRAIN_ROOT/$PADIS_PNP_NOISE_COND_RUN_NAME/$PADIS_PNP_NOISE_COND_FINAL_FULL_NAME"
                return
        fi

        final_checkpoint="$(task_final_checkpoint "$task_name")"
        printf '%s\n' "${final_checkpoint%.pt}_full.pt"
}

task_validation_intense_checkpoint() {
        local task_name="$1"
        local index task_args engine prefix
        if [ "$task_name" = "$PNP_TASK_NAME" ] \
                || [ "$task_name" = "$PNP_NOISE_COND_TASK_NAME" ]; then
                printf '\n'
                return
        fi

        index="$(task_index_by_name "$task_name")"
        task_args="${PADIS_TASK_ARGUMENTS[$index]}"
        engine="${PADIS_TASK_ENGINES[$index]}"
        prefix="$(diffusion_run_prefix_from_engine_args "$engine" "$task_args")"
        printf '%s\n' "$PADIS_TRAIN_ROOT/$task_name/${prefix}_min_intense_val.pt"
}

task_validation_intense_full_checkpoint() {
        local checkpoint
        checkpoint="$(task_validation_intense_checkpoint "$1")"
        if [ -n "$checkpoint" ]; then
                printf '%s\n' "${checkpoint%.pt}_full.pt"
        else
                printf '\n'
        fi
}

is_task_done() {
        local task_name="$1"
        local phase="${2:-${PADIS_GCP_PHASE:-base}}"
        local marker validation_intense_checkpoint validation_intense_full_checkpoint
        marker="$(task_done_marker "$task_name" "$phase")"
        if [ "$phase" != "base" ] && ! is_task_done "$task_name" "base"; then
                return 1
        fi
        if [ "$PADIS_GCP_DRY_RUN" = "1" ]; then
                [ -f "$marker" ]
                return
        fi
        [ -f "$marker" ] \
                && [ -f "$(task_final_checkpoint "$task_name")" ] \
                && [ -f "$(task_final_full_checkpoint "$task_name")" ] \
                || return 1
        if [ "$phase" = "validation_heavy" ]; then
                validation_intense_checkpoint="$(task_validation_intense_checkpoint "$task_name")"
                validation_intense_full_checkpoint="$(task_validation_intense_full_checkpoint "$task_name")"
                [ -f "$validation_intense_checkpoint" ] \
                        && [ -f "$validation_intense_full_checkpoint" ]
                return
        fi
        return 0
}

build_wandb_args() {
        local task_name="$1"
        WANDB_ARGS=()
        if [ "$PADIS_NO_WANDB" = "1" ]; then
                WANDB_ARGS=(--no-wandb --wandb-mode disabled)
        else
                WANDB_ARGS=(
                        --wandb-project "$PADIS_WANDB_PROJECT"
                        --wandb-name "${PADIS_WANDB_NAME_PREFIX}_${task_name}"
                        --wandb-mode "$PADIS_WANDB_MODE"
                )
                if [ -n "$PADIS_WANDB_ENTITY" ]; then
                        WANDB_ARGS+=(--wandb-entity "$PADIS_WANDB_ENTITY")
                fi
                if [ "$PADIS_NO_WANDB_ARTIFACT" = "1" ]; then
                        WANDB_ARGS+=(--no-wandb-artifact)
                fi
        fi
}

add_cache_args() {
        local engine="$1"
        local task_name="$2"
        local task_args="$3"
        local cache_dataset cache_folder archive_folder source_folder

        if [ "$engine" = "lidc256" ]; then
                cache_dataset="${PADIS_256_CACHE_DATASET:-${PADIS_CACHE_DATASET:-ramdisk}}"
                if [[ "$task_name" == *full ]] && [ "${PADIS_CACHE_FULL_LIDC:-1}" != "1" ]; then
                        cache_dataset="none"
                fi
                cache_folder="$PADIS_256_CACHE_FOLDER"
                archive_folder="$PADIS_256_CACHE_ARCHIVE_FOLDER"
                source_folder="${PADIS_256_CACHE_SOURCE_FOLDER:-${PADIS_CACHE_SOURCE_FOLDER:-}}"
        else
                cache_dataset="${PADIS_512_CACHE_DATASET:-${PADIS_CACHE_DATASET:-ramdisk}}"
                if [[ " $task_args " == *" --full-lidc "* ]] && [ "${PADIS_CACHE_FULL_512_LIDC:-0}" != "1" ]; then
                        cache_dataset="none"
                fi
                cache_folder="$PADIS_512_CACHE_FOLDER"
                archive_folder="$PADIS_512_CACHE_ARCHIVE_FOLDER"
                source_folder="${PADIS_512_CACHE_SOURCE_FOLDER:-}"
        fi

        if [ "$cache_dataset" != "none" ]; then
                CMD+=(
                        --cache-dataset "$cache_dataset"
                        --cache-folder "$cache_folder"
                        --cache-archive-folder "$archive_folder"
                )
                if [ -n "$source_folder" ]; then
                        CMD+=(--cache-source-folder "$source_folder")
                fi
                if [ "$PADIS_REQUIRE_CACHE_HIT" = "1" ]; then
                        CMD+=(--require-cache-hit)
                fi
                if [ "$PADIS_WRITE_CACHE_ARCHIVE" = "1" ]; then
                        CMD+=(--write-cache-archive)
                fi
        fi
}

build_diffusion_command() {
        local task_name="$1"
        local remaining_seconds="$2"
        local index engine batch_size task_args validation_interval validation_max checkpoint_interval log_interval
        local validation_heavy_interval validation_heavy_max
        local validation_name run_prefix
        local num_workers prefetch_factor script_path

        index="$(task_index_by_name "$task_name")"
        engine="${PADIS_TASK_ENGINES[$index]}"
        batch_size="${PADIS_TASK_BATCH_SIZES[$index]}"
        read -r -a task_args <<< "${PADIS_TASK_ARGUMENTS[$index]}"
        run_prefix="$(diffusion_run_prefix_from_engine_args "$engine" "${PADIS_TASK_ARGUMENTS[$index]}")"
        build_wandb_args "$task_name"

        if [[ "$task_name" == whole_lidc_* ]]; then
                validation_interval="${PADIS_GCP_WHOLE_VALIDATION_INTERVAL_PATCHES:-}"
                if [ -z "$validation_interval" ]; then
                        validation_interval="${PADIS_WHOLE_VALIDATION_INTERVAL_PATCHES:-10000}"
                fi
                validation_max="${PADIS_GCP_WHOLE_VALIDATION_MAX_PATCHES:-${PADIS_WHOLE_VALIDATION_MAX_PATCHES:-128}}"
                checkpoint_interval="${PADIS_GCP_WHOLE_CHECKPOINT_INTERVAL_PATCHES:-}"
                if [ -z "$checkpoint_interval" ]; then
                        checkpoint_interval="${PADIS_WHOLE_CHECKPOINT_INTERVAL_PATCHES:-25000}"
                fi
                log_interval="${PADIS_GCP_WHOLE_LOG_INTERVAL_PATCHES:-${PADIS_WHOLE_LOG_INTERVAL_PATCHES:-128}}"
        else
                validation_interval="${PADIS_GCP_VALIDATION_INTERVAL_PATCHES:-}"
                if [ -z "$validation_interval" ]; then
                        validation_interval="${PADIS_VALIDATION_INTERVAL_PATCHES:-200000}"
                fi
                validation_max="${PADIS_GCP_VALIDATION_MAX_PATCHES:-${PADIS_VALIDATION_MAX_PATCHES:-1000}}"
                checkpoint_interval="${PADIS_GCP_CHECKPOINT_INTERVAL_PATCHES:-}"
                if [ -z "$checkpoint_interval" ]; then
                        checkpoint_interval="${PADIS_CHECKPOINT_INTERVAL_PATCHES:-1000000}"
                fi
                log_interval="${PADIS_GCP_LOG_INTERVAL_PATCHES:-${PADIS_LOG_INTERVAL_PATCHES:-128}}"
        fi
        if [ "${PADIS_GCP_PHASE:-base}" = "validation_heavy" ]; then
                if [[ "$task_name" == whole_lidc_* ]]; then
                        validation_heavy_interval="${PADIS_GCP_WHOLE_VALIDATION_HEAVY_INTERVAL_PATCHES:-${PADIS_WHOLE_VALIDATION_HEAVY_INTERVAL_PATCHES:-2500}}"
                        validation_heavy_max="${PADIS_GCP_WHOLE_VALIDATION_HEAVY_MAX_PATCHES:-${PADIS_WHOLE_VALIDATION_HEAVY_MAX_PATCHES:-328}}"
                else
                        validation_heavy_interval="${PADIS_GCP_PATCH_VALIDATION_HEAVY_INTERVAL_PATCHES:-${PADIS_PATCH_VALIDATION_HEAVY_INTERVAL_PATCHES:-20000}}"
                        validation_heavy_max="${PADIS_GCP_PATCH_VALIDATION_HEAVY_MAX_PATCHES:-${PADIS_PATCH_VALIDATION_HEAVY_MAX_PATCHES:-$PADIS_GCP_VALIDATION_HEAVY_MAX_PATCHES}}"
                fi
                validation_interval="$validation_heavy_interval"
                validation_max="$validation_heavy_max"
                validation_name="${run_prefix}_min_intense_val.pt"
        fi

        if [ "$engine" = "lidc256" ]; then
                script_path="scripts/paper_scripts/PaDIS/PaDIS_LIDC_256.py"
                num_workers="$PADIS_NUM_WORKERS"
                prefetch_factor="$PADIS_PREFETCH_FACTOR"
        elif [ "$engine" = "lidc512" ]; then
                script_path="scripts/paper_scripts/PaDIS/PaDIS_LIDC_512.py"
                num_workers="${PADIS_512_NUM_WORKERS:-$PADIS_NUM_WORKERS}"
                prefetch_factor="${PADIS_512_PREFETCH_FACTOR:-$PADIS_PREFETCH_FACTOR}"
        else
                die "Unknown diffusion task engine: $engine"
        fi

        CMD=(
                python -u "$script_path"
                --save-folder "$PADIS_TRAIN_ROOT"
                --device cuda
                --target-patches "$PADIS_TARGET_PATCHES"
                --validation-interval-patches "$validation_interval"
                --validation-max-patches "$validation_max"
                --checkpoint-interval-patches "$checkpoint_interval"
                --checkpoint-interval-seconds "$PADIS_GCP_CHECKPOINT_INTERVAL_SECONDS"
                --max-periodic-checkpoints "$PADIS_GCP_MAX_PERIODIC_CHECKPOINTS"
                --keep-final-periodic-checkpoints "$PADIS_GCP_FINAL_PERIODIC_CHECKPOINTS"
                --log-interval-patches "$log_interval"
                --seed "$PADIS_SEED"
                --batch-size "$batch_size"
                --num-workers "$num_workers"
                --prefetch-factor "$prefetch_factor"
        )
        CMD+=("${WANDB_ARGS[@]}")
        if [ -n "$PADIS_DATA_FOLDER" ]; then
                CMD+=(--data-folder "$PADIS_DATA_FOLDER")
        fi
        if [ -n "${PADIS_MICROBATCH_SIZE:-}" ]; then
                CMD+=(--microbatch-size "$PADIS_MICROBATCH_SIZE")
        fi
        if [ -n "$remaining_seconds" ]; then
                CMD+=(--max-train-seconds "$remaining_seconds")
        fi
        if [ -n "${validation_name:-}" ]; then
                CMD+=(
                        --validation-name "$validation_name"
                        --validation-summary-key min_intense_validation_loss
                        --validation-checkpoint-summary-key min_intense_validation_checkpoint
                )
        fi
        if [ "${PADIS_GCP_PHASE:-base}" = "validation_heavy" ] \
                && [ "$PADIS_GCP_VALIDATION_HEAVY_REPEAT_UNTIL_MAX_PATCHES" = "1" ] \
                && [ "$validation_max" != "-1" ]; then
                CMD+=(--validation-repeat-until-max-patches)
        fi
        add_cache_args "$engine" "$task_name" "${PADIS_TASK_ARGUMENTS[$index]}"
        CMD+=("${task_args[@]}")
}

build_pnp_command() {
        local task_name="$1"
        local run_name final_name final_full_name checkpoint_pattern validation_name
        local use_noise_level max_train_seconds
        run_name="$PADIS_PNP_RUN_NAME"
        final_name="$PADIS_PNP_FINAL_NAME"
        final_full_name="$PADIS_PNP_FINAL_FULL_NAME"
        checkpoint_pattern="$PADIS_PNP_CHECKPOINT_PATTERN"
        validation_name="$PADIS_PNP_VALIDATION_NAME"
        use_noise_level="$PADIS_PNP_USE_NOISE_LEVEL"
        max_train_seconds="$PADIS_PNP_MAX_TRAIN_SECONDS"
        if [ "$task_name" = "$PNP_NOISE_COND_TASK_NAME" ]; then
                run_name="$PADIS_PNP_NOISE_COND_RUN_NAME"
                final_name="$PADIS_PNP_NOISE_COND_FINAL_NAME"
                final_full_name="$PADIS_PNP_NOISE_COND_FINAL_FULL_NAME"
                checkpoint_pattern="$PADIS_PNP_NOISE_COND_CHECKPOINT_PATTERN"
                validation_name="$PADIS_PNP_NOISE_COND_VALIDATION_NAME"
                use_noise_level=1
                max_train_seconds="$PADIS_PNP_NOISE_COND_MAX_TRAIN_SECONDS"
        fi
        build_wandb_args "$task_name"
        CMD=(
                python -u scripts/paper_scripts/PaDIS/PaDIS_LIDC_PnP_denoiser.py
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
                --checkpoint-interval-seconds "$PADIS_GCP_CHECKPOINT_INTERVAL_SECONDS"
                --max-periodic-checkpoints "$PADIS_PNP_MAX_PERIODIC_CHECKPOINTS"
                --keep-final-periodic-checkpoints "$PADIS_GCP_FINAL_PERIODIC_CHECKPOINTS"
                --seed "$PADIS_PNP_SEED"
                --device cuda
                --num-workers "$PADIS_PNP_NUM_WORKERS"
                --final-name "$final_name"
                --final-full-name "$final_full_name"
                --checkpoint-pattern "$checkpoint_pattern"
                --validation-name "$validation_name"
        )
        if [ "$PADIS_PNP_FULL_LIDC" = "1" ]; then
                CMD+=(--full-lidc)
        fi
        if [ -n "$PADIS_PNP_MAX_TRAIN_SAMPLES" ]; then
                CMD+=(--max-train-samples "$PADIS_PNP_MAX_TRAIN_SAMPLES")
        fi
        if [ -n "$PADIS_PNP_MAX_VALIDATION_SAMPLES" ]; then
                CMD+=(--max-validation-samples "$PADIS_PNP_MAX_VALIDATION_SAMPLES")
        fi
        if [ "$use_noise_level" = "1" ]; then
                CMD+=(--use-noise-level)
        fi
        if [ -n "$PADIS_PNP_PATCH_SIZE" ]; then
                CMD+=(--patch-size "$PADIS_PNP_PATCH_SIZE")
        fi
        if [ -n "$PADIS_DATA_FOLDER" ]; then
                CMD+=(--data-folder "$PADIS_DATA_FOLDER")
        fi
        if [ -n "$max_train_seconds" ]; then
                CMD+=(--max-train-seconds "$max_train_seconds")
        fi
        CMD+=("${WANDB_ARGS[@]}")
}

build_task_command() {
        local task_name="$1"
        local remaining_seconds="$2"
        if [ "$task_name" = "$PNP_TASK_NAME" ] \
                || [ "$task_name" = "$PNP_NOISE_COND_TASK_NAME" ]; then
                build_pnp_command "$task_name"
        else
                build_diffusion_command "$task_name" "$remaining_seconds"
        fi
}

stage_cache_variant() {
        local variant="$1"
        local script_path save_name run_name cache_folder archive_folder cache_args
        cache_args=()
        case "$variant" in
                default|256-default)
                        script_path="scripts/paper_scripts/PaDIS/PaDIS_LIDC_256.py"
                        save_name="default_lidc_256"
                        run_name="gcp_prepare_default_lidc_cache"
                        cache_folder="$PADIS_256_CACHE_FOLDER"
                        archive_folder="$PADIS_256_CACHE_ARCHIVE_FOLDER"
                        cache_args=(--max-slices-per-patient 4)
                        ;;
                full|256-full)
                        script_path="scripts/paper_scripts/PaDIS/PaDIS_LIDC_256.py"
                        save_name="full_lidc_256"
                        run_name="gcp_prepare_full_lidc_cache"
                        cache_folder="$PADIS_256_CACHE_FOLDER"
                        archive_folder="$PADIS_256_CACHE_ARCHIVE_FOLDER"
                        cache_args=(--full-lidc)
                        ;;
                512-default)
                        script_path="scripts/paper_scripts/PaDIS/PaDIS_LIDC_512.py"
                        save_name="default_lidc_512"
                        run_name="gcp_prepare_default_lidc_512_cache"
                        cache_folder="$PADIS_512_CACHE_FOLDER"
                        archive_folder="$PADIS_512_CACHE_ARCHIVE_FOLDER"
                        cache_args=(--max-slices-per-patient 4)
                        ;;
                *)
                        die "Unknown cache variant: $variant"
                        ;;
        esac

        CMD=(
                python -u "$script_path"
                --device cpu
                --save-folder "$STATE_DIR/cache_builds/$save_name"
                --run-name "$run_name"
                --cache-dataset ramdisk
                --cache-folder "$cache_folder"
                --cache-archive-folder "$archive_folder"
                --prepare-cache-only
                --seed "$PADIS_SEED"
                --no-wandb
                --wandb-mode disabled
        )
        CMD+=("${cache_args[@]}")
        if [ -n "$PADIS_DATA_FOLDER" ]; then
                CMD+=(--data-folder "$PADIS_DATA_FOLDER")
        fi
        if [ "$PADIS_REQUIRE_CACHE_HIT" = "1" ]; then
                CMD+=(--require-cache-hit)
        fi
        if [ "$PADIS_WRITE_CACHE_ARCHIVE" = "1" ]; then
                CMD+=(--write-cache-archive)
        fi

        log "Staging cache variant $variant into $cache_folder"
        printf '%q ' "${CMD[@]}"
        printf '\n'
        if [ "$PADIS_GCP_DRY_RUN" != "1" ]; then
                "${CMD[@]}"
        fi
}

ensure_ramdisk_mount() {
        local fs_type
        mkdir -p "$PADIS_RAM_DISK"
        fs_type=""
        if command -v findmnt >/dev/null 2>&1; then
                fs_type="$(findmnt -n -o FSTYPE --target "$PADIS_RAM_DISK" 2>/dev/null || true)"
        else
                fs_type="$(stat -f -c %T "$PADIS_RAM_DISK" 2>/dev/null || true)"
        fi
        case "$fs_type" in
                tmpfs|ramfs)
                        return
                        ;;
        esac
        if [ "${PADIS_GCP_ALLOW_NON_TMPFS_RAMDISK:-0}" = "1" ]; then
                log "WARNING: $PADIS_RAM_DISK is backed by '$fs_type', not tmpfs/ramfs."
                return
        fi
        die \
                "$PADIS_RAM_DISK is not a tmpfs/ramfs mount." \
                "Mount it first, or set PADIS_GCP_ALLOW_NON_TMPFS_RAMDISK=1 to override."
}

stage_ramdisk_caches() {
        local variants variant
        if [ "$PADIS_GCP_STAGE_CACHES" != "1" ]; then
                log "Skipping ramdisk cache staging because PADIS_GCP_STAGE_CACHES=$PADIS_GCP_STAGE_CACHES."
                return
        fi
        ensure_ramdisk_mount
        mkdir -p "$PADIS_RAM_DISK" "$PADIS_256_CACHE_FOLDER" "$PADIS_512_CACHE_FOLDER" "$STATE_DIR/cache_builds"
        variants="${PADIS_GCP_CACHE_VARIANTS//,/ }"
        for variant in $variants; do
                stage_cache_variant "$variant"
        done
}

claim_next_task() {
        local gpu_id="$1"
        local claimed="" task fd marker
        exec {fd}>"$STATE_DIR/queue.lock"
        flock "$fd"
        for task in "${ACTIVE_TASK_NAMES[@]}"; do
                if is_task_done "$task"; then
                        continue
                fi
                marker="$(task_running_marker "$task")"
                if [ -f "$marker" ]; then
                        continue
                fi
                {
                        printf 'task=%s\n' "$(phase_task_key "$task")"
                        printf 'base_task=%s\n' "$task"
                        printf 'phase=%s\n' "${PADIS_GCP_PHASE:-base}"
                        printf 'gpu=%s\n' "$gpu_id"
                        printf 'pid=%s\n' "$$"
                        printf 'worker_pid=%s\n' "$BASHPID"
                        printf 'host=%s\n' "$(hostname)"
                        printf 'started=%s\n' "$(date --iso-8601=seconds)"
                } > "$marker"
                claimed="$task"
                break
        done
        flock -u "$fd"
        eval "exec $fd>&-"
        printf '%s\n' "$claimed"
}

record_runtime_while_running() {
        local task_name="$1"
        local child_pid="$2"
        local start_total="$3"
        local start_epoch="$4"
        local now total
        while kill -0 "$child_pid" >/dev/null 2>&1; do
                now="$(date +%s)"
                total=$((start_total + now - start_epoch))
                write_task_elapsed_seconds "$task_name" "$total"
                sleep "$PADIS_GCP_RUNTIME_HEARTBEAT_SECONDS"
        done
}

run_task_command() {
        local task_name="$1"
        local gpu_id="$2"
        local start_total start_epoch child_pid monitor_pid rc now total log_path phase_key
        start_total="$(task_elapsed_seconds "$task_name")"
        start_epoch="$(date +%s)"
        phase_key="$(phase_task_key "$task_name")"
        log_path="$LOG_DIR/$phase_key.gpu${gpu_id}.log"

        printf '%q ' "${CMD[@]}" > "$LOG_DIR/$phase_key.command.txt"
        printf '\n' >> "$LOG_DIR/$phase_key.command.txt"
        log "GPU $gpu_id running $phase_key; log: $log_path"
        if [ "$PADIS_GCP_DRY_RUN" = "1" ]; then
                return 0
        fi

        (
                export CUDA_VISIBLE_DEVICES="$gpu_id"
                "${CMD[@]}"
        ) > "$log_path" 2>&1 &
        child_pid="$!"
        record_runtime_while_running "$task_name" "$child_pid" "$start_total" "$start_epoch" &
        monitor_pid="$!"
        write_running_task_metadata \
                "$task_name" \
                "$gpu_id" \
                "$start_total" \
                "$start_epoch" \
                "$child_pid" \
                "$monitor_pid" \
                "$log_path"

        set +e
        wait "$child_pid"
        rc="$?"
        set -e
        kill "$monitor_pid" >/dev/null 2>&1 || true
        wait "$monitor_pid" >/dev/null 2>&1 || true
        now="$(date +%s)"
        total=$((start_total + now - start_epoch))
        write_task_elapsed_seconds "$task_name" "$total"
        return "$rc"
}

run_task() {
        local task_name="$1"
        local gpu_id="$2"
        local remaining final_checkpoint final_full_checkpoint rc phase_key done_marker
        local validation_intense_checkpoint validation_intense_full_checkpoint
        phase_key="$(phase_task_key "$task_name")"
        if is_task_done "$task_name"; then
                log "Task $phase_key is already done."
                rm -f "$(task_running_marker "$task_name")"
                return 0
        fi

        remaining="$(remaining_budget_seconds "$task_name")"
        if [ -n "$remaining" ]; then
                log "Task $phase_key remaining wall budget for this invocation: ${remaining}s"
        else
                log "Task $phase_key has no wall-clock cap; it will train to completion."
        fi
        build_task_command "$task_name" "$remaining"
        if run_task_command "$task_name" "$gpu_id"; then
                rc=0
        else
                rc="$?"
        fi
        final_checkpoint="$(task_final_checkpoint "$task_name")"
        final_full_checkpoint="$(task_final_full_checkpoint "$task_name")"
        validation_intense_checkpoint=""
        validation_intense_full_checkpoint=""
        if [ "${PADIS_GCP_PHASE:-base}" = "validation_heavy" ]; then
                validation_intense_checkpoint="$(task_validation_intense_checkpoint "$task_name")"
                validation_intense_full_checkpoint="$(task_validation_intense_full_checkpoint "$task_name")"
        fi
        if [ "$rc" -eq 0 ] && {
                [ "$PADIS_GCP_DRY_RUN" = "1" ] || {
                        [ -f "$final_checkpoint" ] && [ -f "$final_full_checkpoint" ] && {
                                [ -z "$validation_intense_checkpoint" ] || {
                                        [ -f "$validation_intense_checkpoint" ] \
                                                && [ -f "$validation_intense_full_checkpoint" ]
                                }
                        }
                }
        }; then
                done_marker="$(task_done_marker "$task_name")"
                {
                        printf 'completed=%s\n' "$(date --iso-8601=seconds)"
                        printf 'task=%s\n' "$task_name"
                        printf 'phase=%s\n' "${PADIS_GCP_PHASE:-base}"
                        printf 'final_checkpoint=%s\n' "$final_checkpoint"
                        printf 'final_full_checkpoint=%s\n' "$final_full_checkpoint"
                        if [ -n "$validation_intense_checkpoint" ]; then
                                printf 'validation_intense_checkpoint=%s\n' "$validation_intense_checkpoint"
                                printf 'validation_intense_full_checkpoint=%s\n' "$validation_intense_full_checkpoint"
                        fi
                } > "$done_marker"
                rm -f "$FAILED_DIR/$phase_key.failed" "$(task_running_marker "$task_name")"
                log "Task $phase_key completed. Final checkpoint: $final_checkpoint; full state: $final_full_checkpoint"
                return 0
        fi

        {
                printf 'failed=%s\n' "$(date --iso-8601=seconds)"
                printf 'exit_code=%s\n' "$rc"
                printf 'task=%s\n' "$task_name"
                printf 'phase=%s\n' "${PADIS_GCP_PHASE:-base}"
                printf 'final_checkpoint=%s\n' "$final_checkpoint"
                printf 'final_full_checkpoint=%s\n' "$final_full_checkpoint"
                if [ -n "$validation_intense_checkpoint" ]; then
                        printf 'validation_intense_checkpoint=%s\n' "$validation_intense_checkpoint"
                        printf 'validation_intense_full_checkpoint=%s\n' "$validation_intense_full_checkpoint"
                fi
        } > "$FAILED_DIR/$phase_key.failed"
        rm -f "$(task_running_marker "$task_name")"
        log "Task $phase_key failed with exit code $rc. See $LOG_DIR/$phase_key.gpu${gpu_id}.log"
        return "$rc"
}

worker_loop() {
        local gpu_id="$1"
        local task
        while true; do
                task="$(claim_next_task "$gpu_id")"
                if [ -z "$task" ]; then
                        log "GPU $gpu_id has no remaining tasks."
                        return 0
                fi
                run_task "$task" "$gpu_id"
        done
}

run_task_phase() {
        local phase="$1"
        local label="$2"
        shift
        shift
        local worker_pids=()
        local gpu_id pid phase_rc
        if [ "$#" -eq 0 ]; then
                log "Skipping $label phase because it has no tasks."
                return 0
        fi
        PADIS_GCP_PHASE="$phase"
        ACTIVE_TASK_NAMES=("$@")
        log "Starting $label phase for tasks: ${ACTIVE_TASK_NAMES[*]}"
        for gpu_id in "${GPU_IDS[@]}"; do
                worker_loop "$gpu_id" &
                worker_pids+=("$!")
        done
        phase_rc=0
        for pid in "${worker_pids[@]}"; do
                if ! wait "$pid"; then
                        phase_rc=1
                fi
        done
        return "$phase_rc"
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

reconstruction_runtime_path() {
        printf '%s/%s.seconds\n' "$RUNTIME_DIR" "$(reconstruction_phase_key "$1")"
}

reconstruction_elapsed_seconds() {
        local path
        path="$(reconstruction_runtime_path "$1")"
        if [ -f "$path" ]; then
                cat "$path"
        else
                printf '0\n'
        fi
}

write_reconstruction_elapsed_seconds() {
        local task_index="$1"
        local seconds="$2"
        local path
        path="$(reconstruction_runtime_path "$task_index")"
        printf '%s\n' "$seconds" > "$path.tmp"
        mv "$path.tmp" "$path"
}

record_reconstruction_runtime_while_running() {
        local task_index="$1"
        local child_pid="$2"
        local start_total="$3"
        local start_epoch="$4"
        local now total
        while kill -0 "$child_pid" >/dev/null 2>&1; do
                now="$(date +%s)"
                total=$((start_total + now - start_epoch))
                write_reconstruction_elapsed_seconds "$task_index" "$total"
                sleep "$PADIS_GCP_RUNTIME_HEARTBEAT_SECONDS"
        done
}

write_running_reconstruction_metadata() {
        local task_index="$1"
        local gpu_id="$2"
        local slot_id="$3"
        local start_total="$4"
        local start_epoch="$5"
        local child_pid="$6"
        local monitor_pid="$7"
        local log_path="$8"
        local phase_key marker tmp
        phase_key="$(reconstruction_phase_key "$task_index")"
        marker="$(reconstruction_running_marker "$task_index")"
        tmp="$marker.tmp.$BASHPID"
        {
                printf 'task=%s\n' "$phase_key"
                printf 'base_task=reconstruction_%06d\n' "$task_index"
                printf 'phase=reconstruction\n'
                printf 'task_index=%s\n' "$task_index"
                printf 'gpu=%s\n' "$gpu_id"
                printf 'slot=%s\n' "$slot_id"
                printf 'pid=%s\n' "$$"
                printf 'worker_pid=%s\n' "$BASHPID"
                printf 'host=%s\n' "$(hostname)"
                printf 'started=%s\n' "$(date --date="@$start_epoch" --iso-8601=seconds)"
                printf 'start_epoch=%s\n' "$start_epoch"
                printf 'start_elapsed=%s\n' "$start_total"
                printf 'child_pid=%s\n' "$child_pid"
                printf 'monitor_pid=%s\n' "$monitor_pid"
                printf 'log_path=%s\n' "$log_path"
        } > "$tmp"
        mv "$tmp" "$marker"
}

prepare_reconstruction_matrix() {
        local new_jobs_json reconcile_args=()
        mkdir -p "$PADIS_RECON_ROOT"
        build_reconstruction_base_command
        if [ "$PADIS_GCP_DRY_RUN" != "1" ] \
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
                        python -u scripts/paper_scripts/PaDIS/PaDIS_reconcile_reconstruction_manifest.py
                        --old-json "$PADIS_RECON_EXPECTED_JOBS_JSON"
                        --new-json "$new_jobs_json"
                        --output-json "$PADIS_RECON_EXPECTED_JOBS_JSON"
                        --state-dir "$STATE_DIR"
                        --done-dir "$DONE_DIR"
                        --failed-dir "$FAILED_DIR"
                        --runtime-dir "$RUNTIME_DIR"
                )
                if [ "$PADIS_GCP_DRY_RUN" = "1" ]; then
                        reconcile_args+=(--skip-output-check)
                fi
                "${reconcile_args[@]}" > "$STATE_DIR/reconstruction_manifest_reconcile.json"
        else
                mv "$new_jobs_json" "$PADIS_RECON_EXPECTED_JOBS_JSON"
        fi
        rm -f "$new_jobs_json"
        log "Prepared reconstruction matrix with $RECON_TASK_COUNT jobs at $PADIS_RECON_EXPECTED_JOBS_JSON"
}

is_reconstruction_done() {
        [ -f "$(reconstruction_done_marker "$1")" ]
}

claim_next_reconstruction_task() {
        local gpu_id="$1"
        local slot_id="${2:-1}"
        local claimed="" task_index marker fd phase_key
        exec {fd}>"$STATE_DIR/reconstruction_queue.lock"
        flock "$fd"
        for ((task_index = 0; task_index < RECON_TASK_COUNT; task_index++)); do
                if is_reconstruction_done "$task_index"; then
                        continue
                fi
                marker="$(reconstruction_running_marker "$task_index")"
                if [ -f "$marker" ]; then
                        continue
                fi
                phase_key="$(reconstruction_phase_key "$task_index")"
                {
                        printf 'task=%s\n' "$phase_key"
                        printf 'base_task=reconstruction_%06d\n' "$task_index"
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
        local start_total start_epoch child_pid monitor_pid rc now total
        local phase_key log_path done_marker failed_marker
        phase_key="$(reconstruction_phase_key "$task_index")"
        done_marker="$(reconstruction_done_marker "$task_index")"
        failed_marker="$(reconstruction_failed_marker "$task_index")"
        if is_reconstruction_done "$task_index"; then
                log "Reconstruction task $phase_key is already done."
                rm -f "$(reconstruction_running_marker "$task_index")"
                return 0
        fi

        start_total="$(reconstruction_elapsed_seconds "$task_index")"
        start_epoch="$(date +%s)"
        log_path="$LOG_DIR/$phase_key.gpu${gpu_id}.log"
        RECON_CMD=("${RECON_BASE_CMD[@]}" --task-index "$task_index")

        printf '%q ' "${RECON_CMD[@]}" > "$LOG_DIR/$phase_key.command.txt"
        printf '\n' >> "$LOG_DIR/$phase_key.command.txt"
        log "GPU $gpu_id reconstruction slot $slot_id running $phase_key; log: $log_path"
        if [ "$PADIS_GCP_DRY_RUN" = "1" ]; then
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
                return 0
        fi

        (
                export CUDA_VISIBLE_DEVICES="$gpu_id"
                "${RECON_CMD[@]}"
        ) > "$log_path" 2>&1 &
        child_pid="$!"
        record_reconstruction_runtime_while_running \
                "$task_index" \
                "$child_pid" \
                "$start_total" \
                "$start_epoch" &
        monitor_pid="$!"
        write_running_reconstruction_metadata \
                "$task_index" \
                "$gpu_id" \
                "$slot_id" \
                "$start_total" \
                "$start_epoch" \
                "$child_pid" \
                "$monitor_pid" \
                "$log_path"

        set +e
        wait "$child_pid"
        rc="$?"
        set -e
        kill "$monitor_pid" >/dev/null 2>&1 || true
        wait "$monitor_pid" >/dev/null 2>&1 || true
        now="$(date +%s)"
        total=$((start_total + now - start_epoch))
        write_reconstruction_elapsed_seconds "$task_index" "$total"
        if [ "$rc" -eq 0 ]; then
                {
                        printf 'completed=%s\n' "$(date --iso-8601=seconds)"
                        printf 'phase=reconstruction\n'
                        printf 'task_index=%s\n' "$task_index"
                        printf 'gpu=%s\n' "$gpu_id"
                        printf 'slot=%s\n' "$slot_id"
                        printf 'elapsed_seconds=%s\n' "$total"
                        printf 'log_path=%s\n' "$log_path"
                } > "$done_marker"
                rm -f "$failed_marker" "$(reconstruction_running_marker "$task_index")"
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
                printf 'elapsed_seconds=%s\n' "$total"
                printf 'log_path=%s\n' "$log_path"
        } > "$failed_marker"
        rm -f "$(reconstruction_running_marker "$task_index")"
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
                        log "GPU $gpu_id reconstruction slot $slot_id has no remaining reconstruction tasks."
                        return 0
                fi
                run_reconstruction_task "$task_index" "$gpu_id" "$slot_id"
        done
}

run_reconstruction_phase() {
        local worker_pids=()
        local gpu_id slot_id pid phase_rc
        if [ "$PADIS_GCP_RECONSTRUCTION_PHASE" != "1" ]; then
                log "Skipping reconstruction phase because PADIS_GCP_RECONSTRUCTION_PHASE=$PADIS_GCP_RECONSTRUCTION_PHASE."
                return 0
        fi
        PADIS_GCP_PHASE="reconstruction"
        prepare_reconstruction_matrix
        log "Starting reconstruction phase for $RECON_TASK_COUNT matrix jobs with $RECON_TASKS_PER_GPU worker slot(s) per GPU."
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

write_manifest() {
        local manifest="$STATE_DIR/manifest.txt"
        {
                printf 'lion_root=%s\n' "$LION_ROOT"
                printf 'train_root=%s\n' "$PADIS_TRAIN_ROOT"
                printf 'run_name=%s\n' "$PADIS_GCP_RUN_NAME"
                printf 'ram_disk=%s\n' "$PADIS_RAM_DISK"
                printf 'gpu_ids=%s\n' "${GPU_IDS[*]}"
                printf 'patch_train_seconds=%s\n' "$PATCH_TRAIN_SECONDS"
                printf 'whole_train_seconds=%s\n' "$WHOLE_TRAIN_SECONDS"
                printf 'validation_heavy_enabled=%s\n' "$PADIS_GCP_VALIDATION_HEAVY_PHASE"
                printf 'validation_heavy_seconds=%s\n' "$VALIDATION_HEAVY_SECONDS"
                printf 'validation_heavy_max_patches=%s\n' "$PADIS_GCP_VALIDATION_HEAVY_MAX_PATCHES"
                printf 'patch_validation_heavy_interval_patches=%s\n' "${PADIS_GCP_PATCH_VALIDATION_HEAVY_INTERVAL_PATCHES:-${PADIS_PATCH_VALIDATION_HEAVY_INTERVAL_PATCHES:-20000}}"
                printf 'whole_validation_heavy_interval_images=%s\n' "${PADIS_GCP_WHOLE_VALIDATION_HEAVY_INTERVAL_PATCHES:-${PADIS_WHOLE_VALIDATION_HEAVY_INTERVAL_PATCHES:-2500}}"
                printf 'whole_validation_heavy_max_images=%s\n' "${PADIS_GCP_WHOLE_VALIDATION_HEAVY_MAX_PATCHES:-${PADIS_WHOLE_VALIDATION_HEAVY_MAX_PATCHES:-328}}"
                printf 'reconstruction_enabled=%s\n' "$PADIS_GCP_RECONSTRUCTION_PHASE"
                printf 'reconstruction_tasks_per_gpu_requested=%s\n' "$PADIS_GCP_RECON_TASKS_PER_GPU"
                printf 'reconstruction_tasks_per_gpu=%s\n' "$RECON_TASKS_PER_GPU"
                printf 'reconstruction_root=%s\n' "$PADIS_RECON_ROOT"
                printf 'reconstruction_checkpoint_policy=%s\n' "$PADIS_RECON_CHECKPOINT_POLICY"
                printf 'reconstruction_job_order=%s\n' "$PADIS_RECON_JOB_ORDER"
                printf 'reconstruction_reconcile_manifest=%s\n' "$PADIS_RECON_RECONCILE_MANIFEST"
                printf 'reconstruction_hparam_defaults=%s\n' "$PADIS_RECON_HPARAM_DEFAULTS"
                printf 'reconstruction_hparam_defaults_json=%s\n' "$PADIS_RECON_HPARAM_DEFAULTS_JSON"
                printf 'reconstruction_hparam_run_root=%s\n' "$PADIS_RECON_HPARAM_RUN_ROOT"
                printf 'reconstruction_hparam_run_glob=%s\n' "$PADIS_RECON_HPARAM_RUN_GLOB"
                printf 'reconstruction_methods=%s\n' "$PADIS_RECON_METHODS"
                printf 'reconstruction_experiments=%s\n' "$PADIS_RECON_EXPERIMENTS"
                printf 'reconstruction_ablations=%s\n' "$PADIS_RECON_ABLATIONS"
                printf 'checkpoint_interval_seconds=%s\n' "$PADIS_GCP_CHECKPOINT_INTERVAL_SECONDS"
                printf 'tasks=%s\n' "${GCP_TASK_NAMES[*]}"
        } > "$manifest"
}

terminate_runner() {
        log "Termination requested; stopping child jobs. Rerun this script to resume."
        refresh_active_runtimes
        terminate_active_children
        kill $(jobs -pr) >/dev/null 2>&1 || true
        wait >/dev/null 2>&1 || true
        refresh_active_runtimes
        exit 143
}

trap terminate_runner INT TERM

LION_ROOT="${LION_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd -P)}"
LION_DATA_PATH="${LION_DATA_PATH:-/mnt/data/Datasets}"
LION_EXPERIMENTS_PATH="${LION_EXPERIMENTS_PATH:-$LION_DATA_PATH/experiments}"
export LION_DATA_PATH LION_EXPERIMENTS_PATH
PADIS_RUN_ROOT="${PADIS_RUN_ROOT:-$(padis_default_run_root)}"
PADIS_GCP_RUN_NAME="${PADIS_GCP_RUN_NAME:-PaDIS-Reproduction-GCP}"
PADIS_RUN_STAMP="${PADIS_RUN_STAMP:-$PADIS_GCP_RUN_NAME}"
PADIS_TRAIN_ROOT="${PADIS_TRAIN_ROOT:-$PADIS_RUN_ROOT/final_real_runs/$PADIS_GCP_RUN_NAME}"
PADIS_PNP_OUTPUT_ROOT="${PADIS_PNP_OUTPUT_ROOT:-$PADIS_TRAIN_ROOT}"
PNP_TASK_NAME="${PADIS_GCP_PNP_TASK_NAME:-pnp_lidc_drunet}"
PNP_NOISE_COND_TASK_NAME="${PADIS_GCP_PNP_NOISE_COND_TASK_NAME:-pnp_lidc_drunet_noise_cond}"

PADIS_DATA_ROOT="${LION_DATA_PATH:-$LION_ROOT/../Data}"
PADIS_CACHE_ROOT="${PADIS_CACHE_ROOT:-$PADIS_DATA_ROOT/processed/LIDC-IDRI-cache}"
PADIS_RAM_DISK="${PADIS_RAM_DISK:-/mnt/ram-disk}"
PADIS_256_CACHE_FOLDER="${PADIS_256_CACHE_FOLDER:-$PADIS_RAM_DISK/lion_lidc_cache_256}"
PADIS_512_CACHE_FOLDER="${PADIS_512_CACHE_FOLDER:-$PADIS_RAM_DISK/lion_lidc_cache_512}"
PADIS_256_CACHE_ARCHIVE_FOLDER="${PADIS_256_CACHE_ARCHIVE_FOLDER:-}"
if [ -z "$PADIS_256_CACHE_ARCHIVE_FOLDER" ]; then
        PADIS_256_CACHE_ARCHIVE_FOLDER="${PADIS_CACHE_ARCHIVE_FOLDER:-$PADIS_CACHE_ROOT/padis_256/archives}"
fi
PADIS_512_CACHE_ARCHIVE_FOLDER="${PADIS_512_CACHE_ARCHIVE_FOLDER:-$PADIS_CACHE_ROOT/padis_512/archives}"
PADIS_GCP_CACHE_VARIANTS="${PADIS_GCP_CACHE_VARIANTS:-256-default,256-full,512-default}"
PADIS_GCP_STAGE_CACHES="${PADIS_GCP_STAGE_CACHES:-1}"
PADIS_REQUIRE_CACHE_HIT="${PADIS_REQUIRE_CACHE_HIT:-0}"
PADIS_WRITE_CACHE_ARCHIVE="${PADIS_WRITE_CACHE_ARCHIVE:-0}"
PADIS_DATA_FOLDER="${PADIS_DATA_FOLDER:-}"

PATCH_TRAIN_SECONDS="$(time_to_seconds "${PADIS_GCP_PATCH_TRAIN_TIME:-06:00:00}")"
WHOLE_TRAIN_SECONDS="$(time_to_seconds "${PADIS_GCP_WHOLE_TRAIN_TIME:-18:00:00}")"
PADIS_GCP_FINALIZE_SECONDS="${PADIS_GCP_FINALIZE_SECONDS:-900}"
VALIDATION_HEAVY_SECONDS="$(time_to_seconds "${PADIS_GCP_VALIDATION_HEAVY_TIME:-06:00:00}")"
PADIS_GCP_VALIDATION_HEAVY_PHASE="${PADIS_GCP_VALIDATION_HEAVY_PHASE:-1}"
PADIS_GCP_VALIDATION_HEAVY_MAX_PATCHES="${PADIS_GCP_VALIDATION_HEAVY_MAX_PATCHES:-4000}"
PADIS_GCP_VALIDATION_HEAVY_REPEAT_UNTIL_MAX_PATCHES="${PADIS_GCP_VALIDATION_HEAVY_REPEAT_UNTIL_MAX_PATCHES:-1}"
PADIS_GCP_CHECKPOINT_INTERVAL_SECONDS="${PADIS_GCP_CHECKPOINT_INTERVAL_SECONDS:-300}"
PADIS_GCP_MAX_PERIODIC_CHECKPOINTS="${PADIS_GCP_MAX_PERIODIC_CHECKPOINTS:-2}"
PADIS_GCP_FINAL_PERIODIC_CHECKPOINTS="${PADIS_GCP_FINAL_PERIODIC_CHECKPOINTS:-1}"
PADIS_GCP_RUNTIME_HEARTBEAT_SECONDS="${PADIS_GCP_RUNTIME_HEARTBEAT_SECONDS:-60}"
PADIS_GCP_DRY_RUN="${PADIS_GCP_DRY_RUN:-0}"

PADIS_TARGET_PATCHES="${PADIS_TARGET_PATCHES:-400000000}"
PADIS_SEED="${PADIS_SEED:-33}"
PADIS_NUM_WORKERS="${PADIS_NUM_WORKERS:-16}"
PADIS_PREFETCH_FACTOR="${PADIS_PREFETCH_FACTOR:-4}"
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
PADIS_PNP_MAX_PERIODIC_CHECKPOINTS="${PADIS_PNP_MAX_PERIODIC_CHECKPOINTS:-$PADIS_GCP_MAX_PERIODIC_CHECKPOINTS}"
PADIS_PNP_MAX_TRAIN_SECONDS="${PADIS_PNP_MAX_TRAIN_SECONDS:-}"
PADIS_PNP_SEED="${PADIS_PNP_SEED:-$PADIS_SEED}"
PADIS_PNP_NUM_WORKERS="${PADIS_PNP_NUM_WORKERS:-4}"
PADIS_PNP_FINAL_NAME="${PADIS_PNP_FINAL_NAME:-pnp_lidc_drunet.pt}"
PADIS_PNP_FINAL_FULL_NAME="${PADIS_PNP_FINAL_FULL_NAME:-${PADIS_PNP_FINAL_NAME%.pt}_full.pt}"
PADIS_PNP_CHECKPOINT_PATTERN="${PADIS_PNP_CHECKPOINT_PATTERN:-pnp_lidc_drunet_check_*.pt}"
PADIS_PNP_VALIDATION_NAME="${PADIS_PNP_VALIDATION_NAME:-pnp_lidc_drunet_min_val.pt}"
PADIS_PNP_ROOT="${PADIS_PNP_ROOT:-$PADIS_PNP_OUTPUT_ROOT/$PADIS_PNP_RUN_NAME}"
PADIS_PNP_NOISE_COND_RUN_NAME="${PADIS_PNP_NOISE_COND_RUN_NAME:-pnp_lidc_drunet_noise_cond}"
PADIS_PNP_NOISE_COND_FINAL_NAME="${PADIS_PNP_NOISE_COND_FINAL_NAME:-pnp_lidc_drunet_noise_cond.pt}"
PADIS_PNP_NOISE_COND_FINAL_FULL_NAME="${PADIS_PNP_NOISE_COND_FINAL_FULL_NAME:-${PADIS_PNP_NOISE_COND_FINAL_NAME%.pt}_full.pt}"
PADIS_PNP_NOISE_COND_CHECKPOINT_PATTERN="${PADIS_PNP_NOISE_COND_CHECKPOINT_PATTERN:-pnp_lidc_drunet_noise_cond_check_*.pt}"
PADIS_PNP_NOISE_COND_VALIDATION_NAME="${PADIS_PNP_NOISE_COND_VALIDATION_NAME:-pnp_lidc_drunet_noise_cond_min_val.pt}"
PADIS_PNP_NOISE_COND_ROOT="${PADIS_PNP_NOISE_COND_ROOT:-$PADIS_PNP_OUTPUT_ROOT/$PADIS_PNP_NOISE_COND_RUN_NAME}"
PADIS_PNP_NOISE_COND_MAX_TRAIN_SECONDS="${PADIS_PNP_NOISE_COND_MAX_TRAIN_SECONDS:-$PADIS_PNP_MAX_TRAIN_SECONDS}"
PADIS_PNP_NOISE_COND_NOISE_LEVEL="${PADIS_PNP_NOISE_COND_NOISE_LEVEL:-0.03}"

PADIS_GCP_RECONSTRUCTION_PHASE="${PADIS_GCP_RECONSTRUCTION_PHASE:-1}"
PADIS_GCP_RECON_TASKS_PER_GPU="${PADIS_GCP_RECON_TASKS_PER_GPU:-auto}"
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
PADIS_RECON_SEED="${PADIS_RECON_SEED:-$PADIS_SEED}"
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
PADIS_RECON_RECONCILE_MANIFEST="${PADIS_RECON_RECONCILE_MANIFEST:-1}"
PADIS_RECON_PNP_CHECKPOINT="${PADIS_RECON_PNP_CHECKPOINT:-${PADIS_PNP_CHECKPOINT:-$PADIS_PNP_ROOT/$PADIS_PNP_VALIDATION_NAME}}"
PADIS_RECON_PNP_NOISE_COND_CHECKPOINT="${PADIS_RECON_PNP_NOISE_COND_CHECKPOINT:-${PADIS_PNP_NOISE_COND_CHECKPOINT:-$PADIS_PNP_NOISE_COND_ROOT/$PADIS_PNP_NOISE_COND_VALIDATION_NAME}}"
PADIS_PNP_ITERATIONS="${PADIS_PNP_ITERATIONS:-20}"
PADIS_PNP_ETA="${PADIS_PNP_ETA:-1e-5}"
PADIS_PNP_CG_ITERATIONS="${PADIS_PNP_CG_ITERATIONS:-100}"
PADIS_PNP_CG_TOLERANCE="${PADIS_PNP_CG_TOLERANCE:-1e-7}"
PADIS_PNP_NOISE_LEVEL="${PADIS_PNP_NOISE_LEVEL:-}"
PADIS_TV_LAMBDA="${PADIS_TV_LAMBDA:-0.001}"
PADIS_TV_ITERATIONS="${PADIS_TV_ITERATIONS:-500}"
PADIS_PUBLIC_IMAGE_DIR="${PADIS_PUBLIC_IMAGE_DIR:-}"

MPLCONFIGDIR="${MPLCONFIGDIR:-$PADIS_TRAIN_ROOT/matplotlib}"
WANDB_DIR="${WANDB_DIR:-$PADIS_TRAIN_ROOT/wandb}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PADIS_WANDB_NETRC="${PADIS_WANDB_NETRC:-/mnt/data/.netrc}"
if [ -z "${NETRC:-}" ] && [ -f "$PADIS_WANDB_NETRC" ]; then
        NETRC="$PADIS_WANDB_NETRC"
fi
if [ "$PADIS_GCP_DRY_RUN" = "1" ]; then
        STATE_DIR="$PADIS_TRAIN_ROOT/.gcp_spot_dry_run"
else
        STATE_DIR="$PADIS_TRAIN_ROOT/.gcp_spot"
fi
DONE_DIR="$STATE_DIR/done"
RUNNING_DIR="$STATE_DIR/running"
FAILED_DIR="$STATE_DIR/failed"
LOG_DIR="$STATE_DIR/logs"
RUNTIME_DIR="$STATE_DIR/runtime"
export LION_ROOT PADIS_RUN_ROOT PADIS_RUN_STAMP PADIS_TRAIN_ROOT
export PADIS_DATA_FOLDER MPLCONFIGDIR WANDB_DIR PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF OMP_NUM_THREADS=1 PYTHONHASHSEED="$PADIS_SEED"
export PADIS_WANDB_PROJECT PADIS_WANDB_ENTITY PADIS_WANDB_MODE PADIS_NO_WANDB
export PADIS_WANDB_NETRC
export PADIS_RECON_ROOT PADIS_RECON_EXPECTED_JOBS_JSON
export PADIS_RECON_HPARAM_DEFAULTS PADIS_RECON_HPARAM_DEFAULTS_JSON
if [ -n "${NETRC:-}" ]; then
        export NETRC
fi

mkdir -p \
        "$PADIS_TRAIN_ROOT" \
        "$MPLCONFIGDIR" \
        "$WANDB_DIR" \
        "$STATE_DIR" \
        "$DONE_DIR" \
        "$RUNNING_DIR" \
        "$FAILED_DIR" \
        "$LOG_DIR" \
        "$RUNTIME_DIR"
exec 200>"$STATE_DIR/runner.lock"
flock -n 200 || die "Another GCP PaDIS runner is already active for $PADIS_TRAIN_ROOT."
rm -f "$RUNNING_DIR"/*.running

activate_environment
cd "$LION_ROOT"
padis_init_training_tasks
discover_gpu_ids
resolve_reconstruction_tasks_per_gpu

default_task_order=(
        whole_lidc_full
        whole_lidc_default
        "$PNP_TASK_NAME"
        "$PNP_NOISE_COND_TASK_NAME"
        patch_lidc_p96_default
        patch_lidc_full
        patch_lidc_512
        patch_lidc_default
        patch_lidc_p32_default
        patch_lidc_p16_default
        patch_lidc_p8_default
        patch_lidc_no_pos_default
)

GCP_TASK_NAMES=()
if [ -n "${PADIS_GCP_TASK_ORDER:-}" ]; then
        for task in ${PADIS_GCP_TASK_ORDER//,/ }; do
                GCP_TASK_NAMES+=("$task")
        done
else
        for task in "${default_task_order[@]}"; do
                if { [ "$task" = "$PNP_TASK_NAME" ] \
                        || [ "$task" = "$PNP_NOISE_COND_TASK_NAME" ]; } \
                        && [ "${PADIS_GCP_INCLUDE_PNP:-1}" != "1" ]; then
                        continue
                fi
                GCP_TASK_NAMES+=("$task")
        done
fi

for task in "${GCP_TASK_NAMES[@]}"; do
        if [ "$task" != "$PNP_TASK_NAME" ] \
                && [ "$task" != "$PNP_NOISE_COND_TASK_NAME" ]; then
                task_index_by_name "$task" >/dev/null || die "Unknown PaDIS training task: $task"
        fi
done

write_manifest
log "LION root: $LION_ROOT"
log "Training root: $PADIS_TRAIN_ROOT"
log "Selected GPUs: ${GPU_IDS[*]}"
log "Reconstruction worker slots per GPU: $RECON_TASKS_PER_GPU (requested: $PADIS_GCP_RECON_TASKS_PER_GPU)"
log "Task order: ${GCP_TASK_NAMES[*]}"
log "Patch task budget: ${PATCH_TRAIN_SECONDS}s; whole-image task budget: ${WHOLE_TRAIN_SECONDS}s"
log "Validation-heavy phase: enabled=$PADIS_GCP_VALIDATION_HEAVY_PHASE, budget=${VALIDATION_HEAVY_SECONDS}s, max_patches=$PADIS_GCP_VALIDATION_HEAVY_MAX_PATCHES"
log "Checkpoint interval: ${PADIS_GCP_CHECKPOINT_INTERVAL_SECONDS}s"
log "Final periodic checkpoints kept: $PADIS_GCP_FINAL_PERIODIC_CHECKPOINTS"
if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi || true
fi

stage_ramdisk_caches

overall_rc=0
if ! run_task_phase "base" "base training" "${GCP_TASK_NAMES[@]}"; then
        overall_rc=1
fi

if [ "$overall_rc" -ne 0 ]; then
        log "One or more PaDIS GCP training tasks failed. Rerun after fixing the failure to resume remaining tasks."
        exit "$overall_rc"
fi

VALIDATION_HEAVY_TASK_NAMES=()
if [ "$PADIS_GCP_VALIDATION_HEAVY_PHASE" = "1" ]; then
        for task in "${GCP_TASK_NAMES[@]}"; do
                if [ "$task" = "$PNP_TASK_NAME" ] \
                        || [ "$task" = "$PNP_NOISE_COND_TASK_NAME" ]; then
                        continue
                fi
                VALIDATION_HEAVY_TASK_NAMES+=("$task")
        done
        if ! run_task_phase \
                "validation_heavy" \
                "validation-heavy continuation" \
                "${VALIDATION_HEAVY_TASK_NAMES[@]}"; then
                overall_rc=1
        fi
fi

if [ "$overall_rc" -ne 0 ]; then
        log "One or more PaDIS GCP validation-heavy continuation tasks failed. Rerun after fixing the failure to resume remaining tasks."
        exit "$overall_rc"
fi

if ! run_reconstruction_phase; then
        overall_rc=1
fi

if [ "$overall_rc" -ne 0 ]; then
        log "One or more PaDIS GCP reconstruction tasks failed. Rerun after fixing the failure to resume remaining tasks."
        exit "$overall_rc"
fi

log "All selected PaDIS GCP training, validation-heavy, and reconstruction tasks completed."
