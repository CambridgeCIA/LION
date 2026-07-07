#!/bin/bash
#
# Submit the PaDIS A100 reproduction pipeline as:
#   checks -> LIDC cache preparation -> pilot training array
#          -> real PaDIS training array + PnP denoiser training
#          -> optional reconstruction matrix -> optional verifier
#
# Useful overrides:
#   PADIS_SLURM_ACCOUNT=MPHIL-DIS-SL2-GPU
#   PADIS_CACHE_SLURM_ACCOUNT=MPHIL-DIS-SL2-CPU
#   PADIS_CHECK_TIME=00:20:00
#   PADIS_CACHE_SLURM_TIME=08:00:00
#   PADIS_CACHE_PARTITION=icelake
#   PADIS_CACHE_CPUS_PER_TASK=8
#   PADIS_CACHE_MEM=128G
#   PADIS_CACHE_PREP_VARIANTS=256-default,256-full,512-default
#   PADIS_PILOT_TIME=00:15:00
#   PADIS_REAL_TIME=24:00:00
#   PADIS_PNP_TIME=24:00:00
#   PADIS_SUBMIT_PNP_TRAINING=1
#   PADIS_SUBMIT_RECONSTRUCTION=0
#   PADIS_RECON_TIME=12:00:00
#   PADIS_RECON_ARRAY_LIMIT=10
#   PADIS_RECON_METHODS=all
#   PADIS_RECON_MAX_SAMPLES=25
#   PADIS_RECON_VERIFY=1
#   PADIS_TASK_ARRAY=0-9
#   PADIS_PILOT_ARRAY_LIMIT=10
#   PADIS_REAL_ARRAY_LIMIT=10
#   PADIS_RUN_ROOT=/path/to/experiments/PaDIS
#   PADIS_TARGET_PATCHES=400000000
#   PADIS_MAX_TRAIN_SECONDS_BUFFER=1800
#   PADIS_WANDB_PROJECT=PaDIS-Reproduction
#   PADIS_WANDB_MODE=online
#   PADIS_WANDB_ENTITY=optional-wandb-entity

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/paper_scripts/PaDIS/slurm/padis_a100_common.sh
. "$SCRIPT_DIR/padis_a100_common.sh"

LION_ROOT="$(padis_lion_root)"
cd "$LION_ROOT"

task_count="$(padis_training_task_count)"
last_task=$((task_count - 1))

account="${PADIS_SLURM_ACCOUNT:-MPHIL-DIS-SL2-GPU}"
cache_account="${PADIS_CACHE_SLURM_ACCOUNT:-MPHIL-DIS-SL2-CPU}"
check_time="${PADIS_CHECK_TIME:-00:20:00}"
cache_time="${PADIS_CACHE_SLURM_TIME:-08:00:00}"
cache_partition="${PADIS_CACHE_PARTITION:-icelake}"
cache_cpus="${PADIS_CACHE_CPUS_PER_TASK:-8}"
cache_mem="${PADIS_CACHE_MEM:-128G}"
cache_variants="${PADIS_CACHE_PREP_VARIANTS:-256-default,256-full,512-default}"
pilot_time="${PADIS_PILOT_TIME:-00:15:00}"
real_time="${PADIS_REAL_TIME:-24:00:00}"
pnp_time="${PADIS_PNP_TIME:-24:00:00}"
recon_time="${PADIS_RECON_TIME:-12:00:00}"
recon_limit="${PADIS_RECON_ARRAY_LIMIT:-10}"
verify_account="${PADIS_RECON_VERIFY_ACCOUNT:-${PADIS_CACHE_SLURM_ACCOUNT:-MPHIL-DIS-SL2-CPU}}"
verify_partition="${PADIS_RECON_VERIFY_PARTITION:-icelake}"
verify_cpus="${PADIS_RECON_VERIFY_CPUS_PER_TASK:-4}"
verify_mem="${PADIS_RECON_VERIFY_MEM:-16G}"
verify_time="${PADIS_RECON_VERIFY_TIME:-00:20:00}"
task_array="${PADIS_TASK_ARRAY:-0-${last_task}}"
pilot_limit="${PADIS_PILOT_ARRAY_LIMIT:-10}"
real_limit="${PADIS_REAL_ARRAY_LIMIT:-10}"
pilot_array="${PADIS_PILOT_ARRAY:-${task_array}%${pilot_limit}}"
real_array="${PADIS_REAL_ARRAY:-${task_array}%${real_limit}}"
run_root="$(padis_default_run_root)"
run_stamp="${PADIS_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
data_root="${LION_DATA_PATH:-/home/tjh200/rds/hpc-work/Datasets}"
cache_root="${PADIS_CACHE_ROOT:-$data_root/processed/LIDC-IDRI-cache}"

mkdir -p "$run_root/debug_runs/slurm_logs" "$run_root/pilot_runs" "$run_root/final_real_runs" "$cache_root"

export PADIS_RUN_STAMP="$run_stamp"
export PADIS_SLURM_DIR="$SCRIPT_DIR"
export LION_ROOT
export PADIS_TRAIN_ROOT="$run_root/final_real_runs/a100_training_$run_stamp"
export PADIS_CACHE_ROOT="$cache_root"
export PADIS_CACHE_PREP_VARIANTS="$cache_variants"
export PADIS_PNP_OUTPUT_ROOT="${PADIS_PNP_OUTPUT_ROOT:-$PADIS_TRAIN_ROOT}"
export PADIS_PNP_RUN_NAME="${PADIS_PNP_RUN_NAME:-pnp_lidc_drunet}"
export PADIS_PNP_FINAL_NAME="${PADIS_PNP_FINAL_NAME:-pnp_lidc_drunet.pt}"
export PADIS_PNP_ROOT="${PADIS_PNP_ROOT:-$PADIS_PNP_OUTPUT_ROOT/$PADIS_PNP_RUN_NAME}"

if [ "${PADIS_SUBMIT_RECONSTRUCTION:-0}" = "1" ]; then
        PADIS_RECON_ROOT="${PADIS_RECON_ROOT:-$run_root/final_real_runs/a100_reconstruction_$run_stamp}"
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
        PADIS_RECON_HPARAM_DEFAULTS="${PADIS_RECON_HPARAM_DEFAULTS:-json}"
        PADIS_RECON_HPARAM_DEFAULTS_JSON="${PADIS_RECON_HPARAM_DEFAULTS_JSON:-$LION_ROOT/scripts/paper_scripts/PaDIS/config/reconstruction_hparam_defaults.json}"
        PADIS_RECON_HPARAM_RUN_ROOT="${PADIS_RECON_HPARAM_RUN_ROOT:-$run_root/hparam_tuning/runs}"
        PADIS_RECON_HPARAM_RUN_GLOB="${PADIS_RECON_HPARAM_RUN_GLOB:-fixedval_*}"
        PADIS_PNP_CHECKPOINT="${PADIS_PNP_CHECKPOINT:-}"
        PADIS_PNP_ITERATIONS="${PADIS_PNP_ITERATIONS:-20}"
        PADIS_PNP_ETA="${PADIS_PNP_ETA:-1e-5}"
        PADIS_PNP_CG_ITERATIONS="${PADIS_PNP_CG_ITERATIONS:-100}"
        PADIS_PNP_CG_TOLERANCE="${PADIS_PNP_CG_TOLERANCE:-1e-7}"
        PADIS_PNP_NOISE_LEVEL="${PADIS_PNP_NOISE_LEVEL:-}"
        PADIS_TV_LAMBDA="${PADIS_TV_LAMBDA:-0.001}"
        PADIS_TV_ITERATIONS="${PADIS_TV_ITERATIONS:-500}"
        PADIS_RECON_EXTRA_ARGS="${PADIS_RECON_EXTRA_ARGS:-}"

        off_paper_args=()
        if [ "$PADIS_RECON_ALLOW_OFF_PAPER_EXPERIMENTS" = "1" ]; then
                off_paper_args=(--allow-off-paper-experiments)
        fi
        recon_extra_args=()
        if [ -n "$PADIS_RECON_EXTRA_ARGS" ]; then
                read -r -a RECON_EXTRA_ARGS <<< "$PADIS_RECON_EXTRA_ARGS"
                for item in "${RECON_EXTRA_ARGS[@]}"; do
                        recon_extra_args+=("--reconstruction-arg=$item")
                done
        fi

        python scripts/paper_scripts/PaDIS/PaDIS_run_reconstruction_matrix.py \
                --training-root "$PADIS_TRAIN_ROOT" \
                --output-root "$PADIS_RECON_ROOT" \
                --hparam-defaults "$PADIS_RECON_HPARAM_DEFAULTS" \
                --hparam-defaults-json "$PADIS_RECON_HPARAM_DEFAULTS_JSON" \
                --hparam-run-root "$PADIS_RECON_HPARAM_RUN_ROOT" \
                --hparam-run-glob "$PADIS_RECON_HPARAM_RUN_GLOB" \
                --models "$PADIS_RECON_MODELS" \
                --methods "$PADIS_RECON_METHODS" \
                --experiments "$PADIS_RECON_EXPERIMENTS" \
                --ablations "$PADIS_RECON_ABLATIONS" \
                "${off_paper_args[@]}" \
                --implementations "$PADIS_RECON_IMPLEMENTATIONS" \
                --geometries "$PADIS_RECON_GEOMETRIES" \
                --split "$PADIS_RECON_SPLIT" \
                --algorithm "$PADIS_RECON_ALGORITHM" \
                --max-samples "$PADIS_RECON_MAX_SAMPLES" \
                --start-index "$PADIS_RECON_START_INDEX" \
                --seed "$PADIS_RECON_SEED" \
                --device "$PADIS_RECON_DEVICE" \
                --pnp-root "$PADIS_PNP_ROOT" \
                --pnp-iterations "$PADIS_PNP_ITERATIONS" \
                --pnp-eta "$PADIS_PNP_ETA" \
                --pnp-cg-iterations "$PADIS_PNP_CG_ITERATIONS" \
                --pnp-cg-tolerance "$PADIS_PNP_CG_TOLERANCE" \
                --tv-lambda "$PADIS_TV_LAMBDA" \
                --tv-iterations "$PADIS_TV_ITERATIONS" \
                "${recon_extra_args[@]}" \
                --count >/dev/null

        compact_recon_methods="${PADIS_RECON_METHODS//[[:space:]]/}"
        recon_needs_pnp="0"
        if [ "$compact_recon_methods" = "all" ] || [[ ",$compact_recon_methods," == *",pnp_admm,"* ]]; then
                recon_needs_pnp="1"
        fi
        if [ "$recon_needs_pnp" = "1" ] && [ -z "$PADIS_PNP_CHECKPOINT" ]; then
                PADIS_PNP_CHECKPOINT="$PADIS_PNP_ROOT/$PADIS_PNP_FINAL_NAME"
        fi
        if [ "${PADIS_SUBMIT_PNP_TRAINING:-1}" != "1" ] && {
                [ "$compact_recon_methods" = "all" ] ||
                        [[ ",$compact_recon_methods," == *",pnp_admm,"* ]];
        }; then
                pnp_checkpoint="$PADIS_PNP_CHECKPOINT"
                if [ ! -f "$pnp_checkpoint" ]; then
                        cat >&2 <<EOF
PADIS_SUBMIT_RECONSTRUCTION=1 selected a matrix containing pnp_admm, but
PADIS_SUBMIT_PNP_TRAINING=0 and no existing PnP checkpoint was found.

Either:
  - leave PADIS_SUBMIT_PNP_TRAINING=1,
  - set PADIS_PNP_CHECKPOINT to an existing DRUNet checkpoint, or
  - exclude pnp_admm from PADIS_RECON_METHODS.
EOF
                        exit 1
                fi
        fi
fi

checks_job="$(
        sbatch \
                --parsable \
                -A "$account" \
                --time "$check_time" \
                --export=ALL \
                --output "$run_root/debug_runs/slurm_logs/%x-%j.out" \
                "$SCRIPT_DIR/slurm_PaDIS_A100_checks.sh"
)"
echo "Submitted checks job: $checks_job"

cache_job="$(
        sbatch \
                --parsable \
                -A "$cache_account" \
                -p "$cache_partition" \
                --cpus-per-task "$cache_cpus" \
                --mem "$cache_mem" \
                --time "$cache_time" \
                --dependency "afterok:$checks_job" \
                --export=ALL \
                --output "$run_root/debug_runs/slurm_logs/%x-%j.out" \
                "$SCRIPT_DIR/slurm_PaDIS_A100_prepare_full_cache.sh"
)"
echo "Submitted cache preparation job: $cache_job after $checks_job"

pilot_job="$(
        sbatch \
                --parsable \
                -A "$account" \
                --time "$pilot_time" \
                --array "$pilot_array" \
                --dependency "afterok:$cache_job" \
                --export=ALL \
                --output "$run_root/debug_runs/slurm_logs/%x-%A_%a.out" \
                "$SCRIPT_DIR/slurm_PaDIS_A100_pilot_array.sh"
)"
echo "Submitted pilot array: $pilot_job after $cache_job"

padis_configure_real_training_defaults "$real_time" "$run_stamp"

real_job="$(
        sbatch \
                --parsable \
                -A "$account" \
                --time "$real_time" \
                --array "$real_array" \
                --dependency "afterok:$pilot_job" \
                --export=ALL \
                --output "$run_root/debug_runs/slurm_logs/%x-%A_%a.out" \
                "$SCRIPT_DIR/slurm_PaDIS_A100_training_array.sh"
)"
echo "Submitted real training array: $real_job after $pilot_job"

pnp_job=""
if [ "${PADIS_SUBMIT_PNP_TRAINING:-1}" = "1" ]; then
        pnp_job="$(
                sbatch \
                        --parsable \
                        -A "$account" \
                        --time "$pnp_time" \
                        --dependency "afterok:$pilot_job" \
                        --export=ALL \
                        --output "$run_root/debug_runs/slurm_logs/%x-%j.out" \
                        "$SCRIPT_DIR/slurm_PaDIS_A100_pnp_training.sh"
        )"
        echo "Submitted PnP denoiser training: $pnp_job after $pilot_job"
else
        echo "Skipping PnP denoiser training because PADIS_SUBMIT_PNP_TRAINING=0"
fi

recon_job=""
verify_job=""
if [ "${PADIS_SUBMIT_RECONSTRUCTION:-0}" = "1" ]; then
        export PADIS_RECON_ROOT="${PADIS_RECON_ROOT:-$run_root/final_real_runs/a100_reconstruction_$run_stamp}"
        export PADIS_RECON_MODELS="${PADIS_RECON_MODELS:-method_default}"
        export PADIS_RECON_METHODS="${PADIS_RECON_METHODS:-all}"
        export PADIS_RECON_EXPERIMENTS="${PADIS_RECON_EXPERIMENTS:-paper_matrix}"
        export PADIS_RECON_ABLATIONS="${PADIS_RECON_ABLATIONS:-all}"
        export PADIS_RECON_ALLOW_OFF_PAPER_EXPERIMENTS="${PADIS_RECON_ALLOW_OFF_PAPER_EXPERIMENTS:-0}"
        export PADIS_RECON_IMPLEMENTATIONS="${PADIS_RECON_IMPLEMENTATIONS:-method_default}"
        export PADIS_RECON_GEOMETRIES="${PADIS_RECON_GEOMETRIES:-lion}"
        export PADIS_RECON_SPLIT="${PADIS_RECON_SPLIT:-test}"
        export PADIS_RECON_ALGORITHM="${PADIS_RECON_ALGORITHM:-dps_langevin}"
        export PADIS_RECON_MAX_SAMPLES="${PADIS_RECON_MAX_SAMPLES:-25}"
        export PADIS_RECON_START_INDEX="${PADIS_RECON_START_INDEX:-0}"
        export PADIS_RECON_SEED="${PADIS_RECON_SEED:-33}"
        export PADIS_RECON_DEVICE="${PADIS_RECON_DEVICE:-cuda}"
        export PADIS_RECON_HPARAM_DEFAULTS="${PADIS_RECON_HPARAM_DEFAULTS:-json}"
        export PADIS_RECON_HPARAM_DEFAULTS_JSON="${PADIS_RECON_HPARAM_DEFAULTS_JSON:-$LION_ROOT/scripts/paper_scripts/PaDIS/config/reconstruction_hparam_defaults.json}"
        export PADIS_RECON_HPARAM_RUN_ROOT="${PADIS_RECON_HPARAM_RUN_ROOT:-$run_root/hparam_tuning/runs}"
        export PADIS_RECON_HPARAM_RUN_GLOB="${PADIS_RECON_HPARAM_RUN_GLOB:-fixedval_*}"
        export PADIS_PNP_ROOT
        export PADIS_PNP_CHECKPOINT="${PADIS_PNP_CHECKPOINT:-}"
        export PADIS_PNP_ITERATIONS="${PADIS_PNP_ITERATIONS:-20}"
        export PADIS_PNP_ETA="${PADIS_PNP_ETA:-1e-5}"
        export PADIS_PNP_CG_ITERATIONS="${PADIS_PNP_CG_ITERATIONS:-100}"
        export PADIS_PNP_CG_TOLERANCE="${PADIS_PNP_CG_TOLERANCE:-1e-7}"
        export PADIS_PNP_NOISE_LEVEL="${PADIS_PNP_NOISE_LEVEL:-}"
        export PADIS_TV_LAMBDA="${PADIS_TV_LAMBDA:-0.001}"
        export PADIS_TV_ITERATIONS="${PADIS_TV_ITERATIONS:-500}"
        export PADIS_RECON_VERIFY="${PADIS_RECON_VERIFY:-1}"
        export PADIS_RECON_VERIFY_METHODS="${PADIS_RECON_VERIFY_METHODS:-$PADIS_RECON_METHODS}"
        export PADIS_RECON_VERIFY_EXPERIMENTS="${PADIS_RECON_VERIFY_EXPERIMENTS:-$PADIS_RECON_EXPERIMENTS}"
        export PADIS_RECON_VERIFY_GEOMETRIES="${PADIS_RECON_VERIFY_GEOMETRIES:-$PADIS_RECON_GEOMETRIES}"
        export PADIS_RECON_EXTRA_ARGS="${PADIS_RECON_EXTRA_ARGS:-}"
        recon_needs_pnp="0"
        compact_recon_methods="${PADIS_RECON_METHODS//[[:space:]]/}"
        if [ "$compact_recon_methods" = "all" ] || [[ ",$compact_recon_methods," == *",pnp_admm,"* ]]; then
                recon_needs_pnp="1"
        fi
        if [ "$recon_needs_pnp" = "1" ] && [ -z "$PADIS_PNP_CHECKPOINT" ]; then
                export PADIS_PNP_CHECKPOINT="$PADIS_PNP_ROOT/$PADIS_PNP_FINAL_NAME"
        fi
        if [ "$recon_needs_pnp" = "1" ] && [ -z "$pnp_job" ]; then
                pnp_checkpoint="$PADIS_PNP_CHECKPOINT"
                if [ ! -f "$pnp_checkpoint" ]; then
                        cat >&2 <<EOF
PADIS_SUBMIT_RECONSTRUCTION=1 selected a matrix containing pnp_admm, but
PADIS_SUBMIT_PNP_TRAINING=0 and no existing PnP checkpoint was found.

Either:
  - leave PADIS_SUBMIT_PNP_TRAINING=1,
  - set PADIS_PNP_CHECKPOINT to an existing DRUNet checkpoint, or
  - exclude pnp_admm from PADIS_RECON_METHODS.
EOF
                        exit 1
                fi
        fi
        mkdir -p "$PADIS_RECON_ROOT"

        pnp_checkpoint_args=()
        if [ -n "$PADIS_PNP_CHECKPOINT" ]; then
                pnp_checkpoint_args=(--pnp-checkpoint "$PADIS_PNP_CHECKPOINT")
        fi
        pnp_noise_args=()
        if [ -n "$PADIS_PNP_NOISE_LEVEL" ]; then
                pnp_noise_args=(--pnp-noise-level "$PADIS_PNP_NOISE_LEVEL")
        fi
        off_paper_args=()
        if [ "$PADIS_RECON_ALLOW_OFF_PAPER_EXPERIMENTS" = "1" ]; then
                off_paper_args=(--allow-off-paper-experiments)
        fi
        recon_extra_args=()
        if [ -n "$PADIS_RECON_EXTRA_ARGS" ]; then
                read -r -a RECON_EXTRA_ARGS <<< "$PADIS_RECON_EXTRA_ARGS"
                for item in "${RECON_EXTRA_ARGS[@]}"; do
                        recon_extra_args+=("--reconstruction-arg=$item")
                done
        fi

        recon_task_count="$(
                python scripts/paper_scripts/PaDIS/PaDIS_run_reconstruction_matrix.py \
                        --training-root "$PADIS_TRAIN_ROOT" \
                        --output-root "$PADIS_RECON_ROOT" \
                        --hparam-defaults "$PADIS_RECON_HPARAM_DEFAULTS" \
                        --hparam-defaults-json "$PADIS_RECON_HPARAM_DEFAULTS_JSON" \
                        --hparam-run-root "$PADIS_RECON_HPARAM_RUN_ROOT" \
                        --hparam-run-glob "$PADIS_RECON_HPARAM_RUN_GLOB" \
                        --models "$PADIS_RECON_MODELS" \
                        --methods "$PADIS_RECON_METHODS" \
                        --experiments "$PADIS_RECON_EXPERIMENTS" \
                        --ablations "$PADIS_RECON_ABLATIONS" \
                        "${off_paper_args[@]}" \
                        --implementations "$PADIS_RECON_IMPLEMENTATIONS" \
                        --geometries "$PADIS_RECON_GEOMETRIES" \
                        --split "$PADIS_RECON_SPLIT" \
                        --algorithm "$PADIS_RECON_ALGORITHM" \
                        --max-samples "$PADIS_RECON_MAX_SAMPLES" \
                        --start-index "$PADIS_RECON_START_INDEX" \
                        --seed "$PADIS_RECON_SEED" \
                        --device "$PADIS_RECON_DEVICE" \
                        --pnp-root "$PADIS_PNP_ROOT" \
                        "${pnp_checkpoint_args[@]}" \
                        --pnp-iterations "$PADIS_PNP_ITERATIONS" \
                        --pnp-eta "$PADIS_PNP_ETA" \
                        --pnp-cg-iterations "$PADIS_PNP_CG_ITERATIONS" \
                        --pnp-cg-tolerance "$PADIS_PNP_CG_TOLERANCE" \
                        "${pnp_noise_args[@]}" \
                        --tv-lambda "$PADIS_TV_LAMBDA" \
                        --tv-iterations "$PADIS_TV_ITERATIONS" \
                        "${recon_extra_args[@]}" \
                        --count
        )"
        export PADIS_RECON_EXPECTED_JOBS_JSON="${PADIS_RECON_EXPECTED_JOBS_JSON:-$PADIS_RECON_ROOT/reconstruction_matrix_jobs.json}"
        python scripts/paper_scripts/PaDIS/PaDIS_run_reconstruction_matrix.py \
                --training-root "$PADIS_TRAIN_ROOT" \
                --output-root "$PADIS_RECON_ROOT" \
                --hparam-defaults "$PADIS_RECON_HPARAM_DEFAULTS" \
                --hparam-defaults-json "$PADIS_RECON_HPARAM_DEFAULTS_JSON" \
                --hparam-run-root "$PADIS_RECON_HPARAM_RUN_ROOT" \
                --hparam-run-glob "$PADIS_RECON_HPARAM_RUN_GLOB" \
                --models "$PADIS_RECON_MODELS" \
                --methods "$PADIS_RECON_METHODS" \
                --experiments "$PADIS_RECON_EXPERIMENTS" \
                --ablations "$PADIS_RECON_ABLATIONS" \
                "${off_paper_args[@]}" \
                --implementations "$PADIS_RECON_IMPLEMENTATIONS" \
                --geometries "$PADIS_RECON_GEOMETRIES" \
                --split "$PADIS_RECON_SPLIT" \
                --algorithm "$PADIS_RECON_ALGORITHM" \
                --max-samples "$PADIS_RECON_MAX_SAMPLES" \
                --start-index "$PADIS_RECON_START_INDEX" \
                --seed "$PADIS_RECON_SEED" \
                --device "$PADIS_RECON_DEVICE" \
                --pnp-root "$PADIS_PNP_ROOT" \
                "${pnp_checkpoint_args[@]}" \
                --pnp-iterations "$PADIS_PNP_ITERATIONS" \
                --pnp-eta "$PADIS_PNP_ETA" \
                --pnp-cg-iterations "$PADIS_PNP_CG_ITERATIONS" \
                --pnp-cg-tolerance "$PADIS_PNP_CG_TOLERANCE" \
                "${pnp_noise_args[@]}" \
                --tv-lambda "$PADIS_TV_LAMBDA" \
                --tv-iterations "$PADIS_TV_ITERATIONS" \
                "${recon_extra_args[@]}" \
                --list > "$PADIS_RECON_EXPECTED_JOBS_JSON"
        export PADIS_RECON_EXPECTED_RECORDS="${PADIS_RECON_EXPECTED_RECORDS:-$recon_task_count}"
        export PADIS_RECON_EXPECTED_SAMPLES="${PADIS_RECON_EXPECTED_SAMPLES:-$PADIS_RECON_MAX_SAMPLES}"
        last_recon_task=$((recon_task_count - 1))
        recon_array="${PADIS_RECON_ARRAY:-0-${last_recon_task}%${recon_limit}}"
        recon_dependency="afterok:$real_job"
        if [ -n "$pnp_job" ]; then
                recon_dependency="$recon_dependency:$pnp_job"
        fi
        recon_job="$(
                sbatch \
                        --parsable \
                        -A "$account" \
                        --time "$recon_time" \
                        --array "$recon_array" \
                        --dependency "$recon_dependency" \
                        --export=ALL \
                        --output "$run_root/debug_runs/slurm_logs/%x-%A_%a.out" \
                        "$SCRIPT_DIR/slurm_PaDIS_A100_reconstruction_array.sh"
        )"
        echo "Submitted reconstruction array: $recon_job after $recon_dependency"

        if [ "$PADIS_RECON_VERIFY" = "1" ]; then
                verify_job="$(
                        sbatch \
                                --parsable \
                                -A "$verify_account" \
                                -p "$verify_partition" \
                                --cpus-per-task "$verify_cpus" \
                                --mem "$verify_mem" \
                                --time "$verify_time" \
                                --dependency "afterok:$recon_job" \
                                --export=ALL \
                                --output "$run_root/debug_runs/slurm_logs/%x-%j.out" \
                                "$SCRIPT_DIR/slurm_PaDIS_A100_reconstruction_verify.sh"
                )"
                echo "Submitted reconstruction verifier: $verify_job after $recon_job"
        fi
else
        echo "Skipping reconstruction submission because PADIS_SUBMIT_RECONSTRUCTION=0"
fi

cat <<EOF

Pipeline submitted.

Run root: $run_root
Run stamp: $run_stamp
Cache root: $cache_root
Debug/checks: $run_root/debug_runs/a100_checks_$run_stamp
Pilots: $run_root/pilot_runs/a100_pilots_$run_stamp
Real training: $run_root/final_real_runs/a100_training_$run_stamp
PnP denoiser: $PADIS_PNP_ROOT/$PADIS_PNP_FINAL_NAME
Reconstruction: ${PADIS_RECON_ROOT:-not submitted}

Checks: $checks_job
Cache prep: $cache_job
Pilots: $pilot_job
Real training: $real_job
PnP training: ${pnp_job:-not submitted}
Reconstruction: ${recon_job:-not submitted}
Verifier: ${verify_job:-not submitted}

Monitor:
  squeue -j $checks_job,$cache_job,$pilot_job,$real_job${pnp_job:+,$pnp_job}${recon_job:+,$recon_job}${verify_job:+,$verify_job}

Cancel all:
  scancel $checks_job $cache_job $pilot_job $real_job${pnp_job:+ $pnp_job}${recon_job:+ $recon_job}${verify_job:+ $verify_job}
EOF
