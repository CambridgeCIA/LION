#!/bin/bash
#
# Submit the PaDIS reconstruction matrix for models trained by
# submit_PaDIS_A100_training_only.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/paper_scripts/PaDIS-Reproduction/platforms/slurm/padis_a100_common.sh
. "$SCRIPT_DIR/padis_a100_common.sh"

LION_ROOT="$(padis_lion_root)"
cd "$LION_ROOT"

account="${PADIS_SLURM_ACCOUNT:-MPHIL-DIS-SL2-GPU}"
time_limit="${PADIS_RECON_TIME:-12:00:00}"
array_limit="${PADIS_RECON_ARRAY_LIMIT:-10}"
verify_account="${PADIS_RECON_VERIFY_ACCOUNT:-${PADIS_CACHE_SLURM_ACCOUNT:-MPHIL-DIS-SL2-CPU}}"
verify_partition="${PADIS_RECON_VERIFY_PARTITION:-icelake}"
verify_cpus="${PADIS_RECON_VERIFY_CPUS_PER_TASK:-4}"
verify_mem="${PADIS_RECON_VERIFY_MEM:-16G}"
verify_time="${PADIS_RECON_VERIFY_TIME:-00:20:00}"
run_root="$(padis_default_run_root)"
run_stamp="${PADIS_RUN_STAMP:-}"

if [ -z "${PADIS_TRAIN_ROOT:-}" ]; then
        if [ -n "$run_stamp" ]; then
                PADIS_TRAIN_ROOT="$run_root/final_real_runs/a100_training_$run_stamp"
        else
                latest_train_root=""
                if [ -d "$run_root/final_real_runs" ]; then
                        latest_train_root="$(
                                find "$run_root/final_real_runs" \
                                        -maxdepth 1 \
                                        -type d \
                                        -name 'a100_training_*' \
                                        -printf '%T@ %p\n' \
                                        | sort -nr \
                                        | sed -n '1s/^[^ ]* //p'
                        )"
                fi
                if [ -z "$latest_train_root" ]; then
                        echo "Could not infer PADIS_TRAIN_ROOT. Set PADIS_TRAIN_ROOT or PADIS_RUN_STAMP." >&2
                        exit 1
                fi
                PADIS_TRAIN_ROOT="$latest_train_root"
                run_stamp="${PADIS_TRAIN_ROOT##*a100_training_}"
        fi
fi

run_stamp="${run_stamp:-$(date +%Y%m%d_%H%M%S)}"
PADIS_RECON_ROOT="${PADIS_RECON_ROOT:-$run_root/final_real_runs/a100_reconstruction_$run_stamp}"
PADIS_RECON_MODELS="${PADIS_RECON_MODELS:-method_default}"
PADIS_RECON_METHODS="${PADIS_RECON_METHODS:-all}"
PADIS_RECON_EXPERIMENTS="${PADIS_RECON_EXPERIMENTS:-paper_matrix}"
PADIS_RECON_ABLATIONS="${PADIS_RECON_ABLATIONS:-all}"
PADIS_RECON_CHECKPOINT_POLICY="${PADIS_RECON_CHECKPOINT_POLICY:-min_intense_val}"
PADIS_RECON_JOB_ORDER="${PADIS_RECON_JOB_ORDER:-gcp_spot}"
PADIS_RECON_HPARAM_DEFAULTS="${PADIS_RECON_HPARAM_DEFAULTS:-json}"
PADIS_RECON_HPARAM_DEFAULTS_JSON="${PADIS_RECON_HPARAM_DEFAULTS_JSON:-$LION_ROOT/scripts/paper_scripts/PaDIS-Reproduction/config/reconstruction_hparam_defaults.json}"
PADIS_RECON_HPARAM_RUN_ROOT="${PADIS_RECON_HPARAM_RUN_ROOT:-$run_root/hparam_tuning/runs}"
PADIS_RECON_HPARAM_RUN_GLOB="${PADIS_RECON_HPARAM_RUN_GLOB:-fixedval_*}"
PADIS_RECON_ALLOW_OFF_PAPER_EXPERIMENTS="${PADIS_RECON_ALLOW_OFF_PAPER_EXPERIMENTS:-0}"
PADIS_RECON_IMPLEMENTATIONS="${PADIS_RECON_IMPLEMENTATIONS:-method_default}"
PADIS_RECON_GEOMETRIES="${PADIS_RECON_GEOMETRIES:-lion}"
PADIS_RECON_SPLIT="${PADIS_RECON_SPLIT:-test}"
PADIS_RECON_ALGORITHM="${PADIS_RECON_ALGORITHM:-dps_langevin}"
PADIS_RECON_MAX_SAMPLES="${PADIS_RECON_MAX_SAMPLES:-25}"
PADIS_RECON_START_INDEX="${PADIS_RECON_START_INDEX:-0}"
PADIS_RECON_SEED="${PADIS_RECON_SEED:-33}"
PADIS_RECON_DEVICE="${PADIS_RECON_DEVICE:-cuda}"
PADIS_PNP_OUTPUT_ROOT="${PADIS_PNP_OUTPUT_ROOT:-$PADIS_TRAIN_ROOT}"
PADIS_PNP_RUN_NAME="${PADIS_PNP_RUN_NAME:-pnp_lidc_drunet}"
PADIS_PNP_VALIDATION_NAME="${PADIS_PNP_VALIDATION_NAME:-pnp_lidc_drunet_min_val.pt}"
PADIS_PNP_FINAL_NAME="${PADIS_PNP_FINAL_NAME:-$PADIS_PNP_VALIDATION_NAME}"
PADIS_PNP_ROOT="${PADIS_PNP_ROOT:-$PADIS_PNP_OUTPUT_ROOT/$PADIS_PNP_RUN_NAME}"
PADIS_PNP_CHECKPOINT="${PADIS_PNP_CHECKPOINT:-}"
PADIS_PNP_ITERATIONS="${PADIS_PNP_ITERATIONS:-20}"
PADIS_PNP_ETA="${PADIS_PNP_ETA:-1e-5}"
PADIS_PNP_CG_ITERATIONS="${PADIS_PNP_CG_ITERATIONS:-100}"
PADIS_PNP_CG_TOLERANCE="${PADIS_PNP_CG_TOLERANCE:-1e-7}"
PADIS_PNP_NOISE_LEVEL="${PADIS_PNP_NOISE_LEVEL:-}"
PADIS_TV_LAMBDA="${PADIS_TV_LAMBDA:-0.001}"
PADIS_TV_ITERATIONS="${PADIS_TV_ITERATIONS:-500}"
PADIS_RECON_VERIFY="${PADIS_RECON_VERIFY:-0}"
PADIS_RECON_VERIFY_METHODS="${PADIS_RECON_VERIFY_METHODS:-$PADIS_RECON_METHODS}"
PADIS_RECON_VERIFY_EXPERIMENTS="${PADIS_RECON_VERIFY_EXPERIMENTS:-$PADIS_RECON_EXPERIMENTS}"
PADIS_RECON_VERIFY_GEOMETRIES="${PADIS_RECON_VERIFY_GEOMETRIES:-$PADIS_RECON_GEOMETRIES}"
PADIS_RECON_EXTRA_ARGS="${PADIS_RECON_EXTRA_ARGS:-}"

export PADIS_RUN_STAMP="$run_stamp"
export PADIS_SLURM_DIR="$SCRIPT_DIR"
export LION_ROOT PADIS_TRAIN_ROOT PADIS_RECON_ROOT
export PADIS_RECON_MODELS PADIS_RECON_METHODS PADIS_RECON_EXPERIMENTS
export PADIS_RECON_ABLATIONS
export PADIS_RECON_CHECKPOINT_POLICY PADIS_RECON_JOB_ORDER
export PADIS_RECON_HPARAM_DEFAULTS PADIS_RECON_HPARAM_RUN_ROOT
export PADIS_RECON_HPARAM_DEFAULTS_JSON PADIS_RECON_HPARAM_RUN_GLOB
export PADIS_RECON_ALLOW_OFF_PAPER_EXPERIMENTS PADIS_RECON_IMPLEMENTATIONS
export PADIS_RECON_GEOMETRIES PADIS_RECON_SPLIT PADIS_RECON_ALGORITHM
export PADIS_RECON_MAX_SAMPLES PADIS_RECON_START_INDEX PADIS_RECON_SEED
export PADIS_RECON_DEVICE
export PADIS_PNP_OUTPUT_ROOT PADIS_PNP_RUN_NAME PADIS_PNP_FINAL_NAME
export PADIS_PNP_ROOT PADIS_PNP_CHECKPOINT PADIS_PNP_ITERATIONS PADIS_PNP_ETA
export PADIS_PNP_CG_ITERATIONS PADIS_PNP_CG_TOLERANCE PADIS_PNP_NOISE_LEVEL
export PADIS_TV_LAMBDA PADIS_TV_ITERATIONS
export PADIS_RECON_VERIFY PADIS_RECON_VERIFY_METHODS PADIS_RECON_VERIFY_EXPERIMENTS
export PADIS_RECON_VERIFY_GEOMETRIES
export PADIS_RECON_EXTRA_ARGS

mkdir -p "$run_root/debug_runs/slurm_logs" "$PADIS_RECON_ROOT"

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

recon_needs_pnp="0"
compact_recon_methods="${PADIS_RECON_METHODS//[[:space:]]/}"
if [ "$compact_recon_methods" = "all" ] || [[ ",$compact_recon_methods," == *",pnp_admm,"* ]]; then
        recon_needs_pnp="1"
fi
if [ "$recon_needs_pnp" = "1" ]; then
        if [ -z "$PADIS_PNP_CHECKPOINT" ]; then
                export PADIS_PNP_CHECKPOINT="$PADIS_PNP_ROOT/$PADIS_PNP_FINAL_NAME"
        fi
        pnp_checkpoint="$PADIS_PNP_CHECKPOINT"
        if [ ! -f "$pnp_checkpoint" ]; then
                cat >&2 <<EOF
PADIS_RECON_METHODS selected a matrix containing pnp_admm, but no PnP
checkpoint was found.

Either:
  - run scripts/paper_scripts/PaDIS-Reproduction/platforms/slurm/submit_PaDIS_A100_pnp_training.sh first,
  - set PADIS_PNP_CHECKPOINT to an existing DRUNet checkpoint, or
  - exclude pnp_admm from PADIS_RECON_METHODS.
EOF
                exit 1
        fi
fi

pnp_checkpoint_args=()
if [ -n "$PADIS_PNP_CHECKPOINT" ]; then
        pnp_checkpoint_args=(--pnp-checkpoint "$PADIS_PNP_CHECKPOINT")
fi

python scripts/paper_scripts/PaDIS-Reproduction/reconstruction/PaDIS_run_reconstruction_matrix.py \
        --training-root "$PADIS_TRAIN_ROOT" \
        --output-root "$PADIS_RECON_ROOT" \
        --checkpoint-policy "$PADIS_RECON_CHECKPOINT_POLICY" \
        --job-order "$PADIS_RECON_JOB_ORDER" \
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
        --check-inputs >/dev/null

task_count="$(
        python scripts/paper_scripts/PaDIS-Reproduction/reconstruction/PaDIS_run_reconstruction_matrix.py \
                --training-root "$PADIS_TRAIN_ROOT" \
                --output-root "$PADIS_RECON_ROOT" \
                --checkpoint-policy "$PADIS_RECON_CHECKPOINT_POLICY" \
                --job-order "$PADIS_RECON_JOB_ORDER" \
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
python scripts/paper_scripts/PaDIS-Reproduction/reconstruction/PaDIS_run_reconstruction_matrix.py \
        --training-root "$PADIS_TRAIN_ROOT" \
        --output-root "$PADIS_RECON_ROOT" \
        --checkpoint-policy "$PADIS_RECON_CHECKPOINT_POLICY" \
        --job-order "$PADIS_RECON_JOB_ORDER" \
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
export PADIS_RECON_EXPECTED_RECORDS="${PADIS_RECON_EXPECTED_RECORDS:-$task_count}"
export PADIS_RECON_EXPECTED_SAMPLES="${PADIS_RECON_EXPECTED_SAMPLES:-$PADIS_RECON_MAX_SAMPLES}"
last_task=$((task_count - 1))
array_spec="${PADIS_RECON_ARRAY:-0-${last_task}%${array_limit}}"

recon_job="$(
        sbatch \
                --parsable \
                -A "$account" \
                --time "$time_limit" \
                --array "$array_spec" \
                --export=ALL \
                --output "$run_root/debug_runs/slurm_logs/%x-%A_%a.out" \
                "$SCRIPT_DIR/slurm_PaDIS_A100_reconstruction_array.sh"
)"

verify_job=""
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
fi

cat <<EOF
Submitted PaDIS reconstruction array: $recon_job

Training root: $PADIS_TRAIN_ROOT
Reconstruction root: $PADIS_RECON_ROOT
Run stamp: $run_stamp
Array: $array_spec
Tasks: $task_count
Models: $PADIS_RECON_MODELS
Methods: $PADIS_RECON_METHODS
Experiments: $PADIS_RECON_EXPERIMENTS
Ablations: $PADIS_RECON_ABLATIONS
Checkpoint policy: $PADIS_RECON_CHECKPOINT_POLICY
Job order: $PADIS_RECON_JOB_ORDER
Hparam defaults: $PADIS_RECON_HPARAM_DEFAULTS
Hparam defaults JSON: $PADIS_RECON_HPARAM_DEFAULTS_JSON
Hparam run root: $PADIS_RECON_HPARAM_RUN_ROOT
Hparam run glob: $PADIS_RECON_HPARAM_RUN_GLOB
Implementations: $PADIS_RECON_IMPLEMENTATIONS
Geometries: $PADIS_RECON_GEOMETRIES
Slurm logs: $run_root/debug_runs/slurm_logs
Verification job: ${verify_job:-not submitted}

Monitor:
  squeue -j $recon_job${verify_job:+,$verify_job}

Cancel:
  scancel $recon_job${verify_job:+ $verify_job}
EOF
