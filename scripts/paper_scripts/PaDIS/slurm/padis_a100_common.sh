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
        MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/tjh200/miniforge3}"
        LION_MAMBA_ENV="${LION_MAMBA_ENV:-lion}"
        export MAMBA_ROOT_PREFIX
        export PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"

        if [ ! -x "$MAMBA_ROOT_PREFIX/bin/mamba" ]; then
                echo "Could not find mamba under $MAMBA_ROOT_PREFIX."
                exit 1
        fi
        eval "$("$MAMBA_ROOT_PREFIX/bin/mamba" shell hook --shell bash)"
        mamba activate "$LION_MAMBA_ENV" || {
                echo "Could not activate mamba environment $LION_MAMBA_ENV."
                exit 1
        }
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
        local patch_batch whole_batch native512_batch
        patch_batch="${PADIS_PATCH_BATCH_SIZE:-128}"
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
                "$patch_batch"
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
