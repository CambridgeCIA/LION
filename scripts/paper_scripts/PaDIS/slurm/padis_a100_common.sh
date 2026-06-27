#!/bin/bash
#
# Shared helpers for the PaDIS A100 Slurm scripts. This file is meant to be
# sourced by the job scripts in this directory.

padis_common_dir() {
        if [ -n "${PADIS_SLURM_DIR:-}" ]; then
                cd "$PADIS_SLURM_DIR" && pwd
        else
                cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
        fi
}

padis_lion_root() {
        if [ -n "${LION_ROOT:-}" ]; then
                cd "$LION_ROOT" && pwd
        else
                cd "$(padis_common_dir)/../../../.." && pwd
        fi
}

padis_setup_modules() {
        if [ -f /etc/profile.d/modules.sh ]; then
                # shellcheck source=/dev/null
                . /etc/profile.d/modules.sh
                module purge
                module load rhel8/default-amp
        fi
}

padis_activate_environment() {
        local env_candidates env_name activated
        MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/tjh200/miniforge3}"
        LION_MAMBA_ENV="${LION_MAMBA_ENV:-lion}"
        export MAMBA_ROOT_PREFIX
        export PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"

        if [ ! -x "$MAMBA_ROOT_PREFIX/bin/mamba" ]; then
                echo "Could not find mamba under $MAMBA_ROOT_PREFIX."
                exit 1
        fi
        eval "$("$MAMBA_ROOT_PREFIX/bin/mamba" shell hook --shell bash)"

        env_candidates=("$LION_MAMBA_ENV")
        if [ -n "${LION_MAMBA_ENV_FALLBACKS:-}" ]; then
                read -r -a env_candidates <<< "$LION_MAMBA_ENV ${LION_MAMBA_ENV_FALLBACKS}"
        fi
        activated=""
        for env_name in "${env_candidates[@]}"; do
                if [ -z "$env_name" ]; then
                        continue
                fi
                if mamba activate "$env_name"; then
                        activated="$env_name"
                        break
                fi
                echo "Could not activate mamba environment $env_name."
        done
        if [ -z "$activated" ]; then
                echo "Failed to activate any requested mamba environment: ${env_candidates[*]}"
                exit 1
        fi
        LION_MAMBA_ENV="$activated"
        export LION_MAMBA_ENV
        CONDA_LIB="${CONDA_PREFIX:-$MAMBA_ROOT_PREFIX/envs/$LION_MAMBA_ENV}/lib"
        export LD_LIBRARY_PATH="$CONDA_LIB:${LD_LIBRARY_PATH:-}"
        echo "Activated $LION_MAMBA_ENV using mamba."
}

padis_default_run_root() {
        local lion_root
        lion_root="$(padis_lion_root)"
        if [ -n "${PADIS_RUN_ROOT:-}" ]; then
                printf '%s\n' "$PADIS_RUN_ROOT"
        elif [ -n "${LION_EXPERIMENTS_PATH:-}" ]; then
                printf '%s\n' "$LION_EXPERIMENTS_PATH/PaDIS"
        elif [ -n "${LION_DATA_PATH:-}" ]; then
                printf '%s\n' "$LION_DATA_PATH/experiments/PaDIS"
        else
                printf '%s\n' "$lion_root/../Data/experiments/PaDIS"
        fi
}

padis_time_to_seconds() {
        local time_spec days first second third seconds
        time_spec="$1"
        days=0
        if [[ "$time_spec" == *-* ]]; then
                days="${time_spec%%-*}"
                time_spec="${time_spec#*-}"
        fi

        IFS=: read -r first second third <<< "$time_spec"
        if [ -n "${third:-}" ]; then
                seconds=$((10#${first:-0} * 3600 + 10#${second:-0} * 60 + 10#${third:-0}))
        elif [ -n "${second:-}" ]; then
                seconds=$((10#${first:-0} * 60 + 10#${second:-0}))
        elif [ -n "${first:-}" ]; then
                seconds=$((10#${first:-0} * 60))
        else
                return 1
        fi
        printf '%s\n' $((10#${days:-0} * 86400 + seconds))
}

padis_configure_real_training_defaults() {
        local real_time run_stamp real_seconds max_seconds_buffer
        real_time="$1"
        run_stamp="$2"

        export PADIS_TARGET_PATCHES="${PADIS_TARGET_PATCHES:-400000000}"
        export PADIS_WANDB_PROJECT="${PADIS_WANDB_PROJECT:-PaDIS-Reproduction}"
        export PADIS_WANDB_ENTITY="${PADIS_WANDB_ENTITY:-}"
        export PADIS_WANDB_MODE="${PADIS_WANDB_MODE:-online}"
        export PADIS_NO_WANDB="${PADIS_NO_WANDB:-0}"
        export PADIS_NO_WANDB_ARTIFACT="${PADIS_NO_WANDB_ARTIFACT:-1}"
        export PADIS_WANDB_NAME_PREFIX="${PADIS_WANDB_NAME_PREFIX:-PaDIS_A100_${run_stamp}}"

        if [ -z "${PADIS_MAX_TRAIN_SECONDS:-}" ]; then
                max_seconds_buffer="${PADIS_MAX_TRAIN_SECONDS_BUFFER:-1800}"
                real_seconds="$(padis_time_to_seconds "$real_time" || true)"
                if [ -n "$real_seconds" ] && [ "$real_seconds" -gt "$max_seconds_buffer" ]; then
                        export PADIS_MAX_TRAIN_SECONDS=$((real_seconds - max_seconds_buffer))
                fi
        else
                export PADIS_MAX_TRAIN_SECONDS
        fi
}

padis_print_job_header() {
        echo "JobID: ${SLURM_JOB_ID:-not-slurm}"
        echo "ArrayTaskID: ${SLURM_ARRAY_TASK_ID:-not-array}"
        echo "Time: $(date)"
        echo "Host: $(hostname)"
        echo "Submit dir: ${SLURM_SUBMIT_DIR:-not-slurm}"
        echo "Working dir: $(pwd)"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
        nvidia-smi || true
}

padis_init_training_tasks() {
        local patch_batch p96_batch whole_batch native512_batch
        patch_batch="${PADIS_PATCH_BATCH_SIZE:-128}"
        p96_batch="${PADIS_P96_BATCH_SIZE:-${PADIS_PATCH_BATCH_SIZE:-120}}"
        whole_batch="${PADIS_WHOLE_BATCH_SIZE:-8}"
        native512_batch="${PADIS_512_BATCH_SIZE:-128}"

        PADIS_TASK_NAMES=(
                patch_lidc_default
                patch_lidc_full
                patch_lidc_p8_default
                patch_lidc_p16_default
                patch_lidc_p32_default
                patch_lidc_p96_default
                patch_lidc_no_pos_default
                whole_lidc_default
                whole_lidc_full
                patch_lidc_512
        )
        PADIS_TASK_ENGINES=(
                lidc256
                lidc256
                lidc256
                lidc256
                lidc256
                lidc256
                lidc256
                lidc256
                lidc256
                lidc512
        )
        PADIS_TASK_BATCH_SIZES=(
                "$patch_batch"
                "$patch_batch"
                "$patch_batch"
                "$patch_batch"
                "$patch_batch"
                "$p96_batch"
                "$patch_batch"
                "$whole_batch"
                "$whole_batch"
                "$native512_batch"
        )
        PADIS_TASK_ARGUMENTS=(
                "--run-name patch_lidc_default --max-slices-per-patient 4"
                "--run-name patch_lidc_full --full-lidc"
                "--run-name patch_lidc_p8_default --patch-size-preset 8 --max-slices-per-patient 4"
                "--run-name patch_lidc_p16_default --patch-size-preset 16 --max-slices-per-patient 4"
                "--run-name patch_lidc_p32_default --patch-size-preset 32 --max-slices-per-patient 4"
                "--run-name patch_lidc_p96_default --patch-size-preset 96 --max-slices-per-patient 4"
                "--run-name patch_lidc_no_pos_default --no-position-channels --max-slices-per-patient 4"
                "--run-name whole_lidc_default --prior-mode whole-image --max-slices-per-patient 4"
                "--run-name whole_lidc_full --prior-mode whole-image --full-lidc"
                "--run-name patch_lidc_512 --max-slices-per-patient 4"
        )
}

padis_training_task_count() {
        padis_init_training_tasks
        printf '%s\n' "${#PADIS_TASK_NAMES[@]}"
}
