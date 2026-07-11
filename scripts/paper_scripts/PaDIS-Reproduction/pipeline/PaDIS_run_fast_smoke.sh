#!/usr/bin/env bash
# Exercise every PaDIS pipeline stage with representative one-sample/one-NFE work.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
LION_ROOT="${LION_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd -P)}"
cd "$LION_ROOT"

PYTHON="${PYTHON:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
PYTHON="${PYTHON:-python}"
LION_DATA_PATH="${LION_DATA_PATH:?Set LION_DATA_PATH}"
LION_EXPERIMENTS_PATH="${LION_EXPERIMENTS_PATH:-$LION_DATA_PATH/experiments}"
PADIS_FAST_SMOKE_TRAINING_ROOT="${PADIS_FAST_SMOKE_TRAINING_ROOT:-${PADIS_TRAIN_ROOT:-}}"
[ -n "$PADIS_FAST_SMOKE_TRAINING_ROOT" ] || {
        echo "Set PADIS_FAST_SMOKE_TRAINING_ROOT or PADIS_TRAIN_ROOT." >&2
        exit 2
}
PADIS_FAST_SMOKE_ROOT="${PADIS_FAST_SMOKE_ROOT:-$LION_EXPERIMENTS_PATH/PaDIS/debug_runs/fast_smoke_$(date +%Y%m%d_%H%M%S)}"
registry="$LION_ROOT/scripts/paper_scripts/PaDIS-Reproduction/config/reconstruction_hparam_defaults.json"
registry_before="$(sha256sum "$registry" | awk '{print $1}')"

export LION_ROOT LION_DATA_PATH LION_EXPERIMENTS_PATH
mkdir -p "$PADIS_FAST_SMOKE_ROOT"/{reconstruction,generation,tuning,reporting/tables,reporting/figures,logs}

"$PYTHON" -u scripts/paper_scripts/PaDIS-Reproduction/tuning/PaDIS_run_reproduction_tuning.py \
        --training-root "$PADIS_FAST_SMOKE_TRAINING_ROOT" \
        --output-root "$PADIS_FAST_SMOKE_ROOT/tuning" \
        --max-samples 1 --start-index 4 --seed 33 --device cuda \
        --fast-smoke --stop-on-failure

matrix=(
        "$PYTHON" -u scripts/paper_scripts/PaDIS-Reproduction/reconstruction/PaDIS_run_reconstruction_matrix.py
        --training-root "$PADIS_FAST_SMOKE_TRAINING_ROOT"
        --output-root "$PADIS_FAST_SMOKE_ROOT/reconstruction"
        --checkpoint-policy min_intense_val --hparam-defaults json
        --models method_default
        --methods baseline,admm_tv,pnp_admm,whole_image_diffusion,padis_dps
        --experiments ct_20 --ablations none --implementations method_default
        --geometries lion --split test --max-samples 1 --start-index 0 --seed 33
        --device cuda --pnp-cg-iterations 50 --pnp-cg-tolerance 1e-7
        --reconstruction-arg=--stop-after-outer-steps
        --reconstruction-arg=1
        --reconstruction-arg=--inner-steps
        --reconstruction-arg=1
        --reconstruction-arg=--tv-iterations
        --reconstruction-arg=1
        --reconstruction-arg=--pnp-iterations
        --reconstruction-arg=1
        --reconstruction-arg=--pnp-cg-iterations
        --reconstruction-arg=1
        --reconstruction-arg=--patch-batch-size
        --reconstruction-arg=1
)
"${matrix[@]}" --list > "$PADIS_FAST_SMOKE_ROOT/reconstruction/reconstruction_matrix_jobs.json"
job_count="$($PYTHON -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$PADIS_FAST_SMOKE_ROOT/reconstruction/reconstruction_matrix_jobs.json")"
for ((index=0; index<job_count; index++)); do
        "${matrix[@]}" --task-index "$index" \
                > "$PADIS_FAST_SMOKE_ROOT/logs/reconstruction_$(printf '%06d' "$index").log" 2>&1
done

patch_checkpoint="$PADIS_FAST_SMOKE_TRAINING_ROOT/patch_lidc_default/padis_lidc_256_min_intense_val.pt"
whole_checkpoint="$PADIS_FAST_SMOKE_TRAINING_ROOT/whole_lidc_default/whole_image_lidc_256_min_intense_val.pt"
generation_root="$PADIS_FAST_SMOKE_ROOT/generation/lion-paper-protocol"
mkdir -p "$generation_root/paper-generation" "$generation_root/paper-generation-whole"
"$PYTHON" -u scripts/paper_scripts/PaDIS-Reproduction/reconstruction/PaDIS_LIDC_generation.py \
        --checkpoint "$patch_checkpoint" --output-folder "$generation_root/paper-generation" \
        --device cuda --num-samples 1 --seed 33 --num-steps 1 --inner-steps 1 \
        --sigma-min 0.002 --sigma-max 10 --noise-schedule geometric \
        --generation-epsilon 1 --prior-mode patch --generation-mode padis --patch-batch-size 1
"$PYTHON" -u scripts/paper_scripts/PaDIS-Reproduction/reconstruction/PaDIS_LIDC_generation.py \
        --checkpoint "$whole_checkpoint" --output-folder "$generation_root/paper-generation-whole" \
        --device cuda --num-samples 1 --seed 33 --num-steps 1 --inner-steps 1 \
        --sigma-min 0.002 --sigma-max 10 --noise-schedule geometric \
        --generation-epsilon 1 --prior-mode whole-image

"$PYTHON" -u scripts/paper_scripts/PaDIS-Reproduction/reconstruction/PaDIS_verify_reconstruction_matrix.py \
        --root "$PADIS_FAST_SMOKE_ROOT/reconstruction" \
        --expected-jobs-json "$PADIS_FAST_SMOKE_ROOT/reconstruction/reconstruction_matrix_jobs.json" \
        --expected-samples 1 --bootstrap-resamples 10 \
        --output-json "$PADIS_FAST_SMOKE_ROOT/reconstruction/reconstruction_matrix_verification.json" \
        --output-csv "$PADIS_FAST_SMOKE_ROOT/reconstruction/reconstruction_matrix_verification.csv"
"$PYTHON" -u scripts/paper_scripts/PaDIS-Reproduction/reporting/PaDIS_make_tables.py \
        --csv-path "$PADIS_FAST_SMOKE_ROOT/reconstruction/reconstruction_matrix_verification.csv" \
        --tex-path "$PADIS_FAST_SMOKE_ROOT/reporting/tables/reconstruction_tables.tex" \
        --csv-output-dir "$PADIS_FAST_SMOKE_ROOT/reporting/tables/csv" \
        --timing-mode gcp --timing-log-root "$PADIS_FAST_SMOKE_ROOT/logs" \
        --timing-jobs-json "$PADIS_FAST_SMOKE_ROOT/reconstruction/reconstruction_matrix_jobs.json" \
        --allow-missing
"$PYTHON" -u scripts/paper_scripts/PaDIS-Reproduction/reporting/PaDIS_make_paper_figures.py \
        --reconstruction-root "$PADIS_FAST_SMOKE_ROOT/reconstruction" \
        --generation-root "$PADIS_FAST_SMOKE_ROOT/generation" \
        --output-folder "$PADIS_FAST_SMOKE_ROOT/reporting/figures" \
        --figures all --allow-missing

registry_after="$(sha256sum "$registry" | awk '{print $1}')"
[ "$registry_before" = "$registry_after" ] || {
        echo "Hyperparameter registry changed during fast smoke." >&2
        exit 1
}
printf 'Fast smoke completed: %s\nRegistry SHA256: %s\n' "$PADIS_FAST_SMOKE_ROOT" "$registry_after"
