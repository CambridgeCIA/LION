#!/usr/bin/env bash
# Generate unconditional samples, verify reconstructions, and build paper outputs.
set -euo pipefail

LION_ROOT="${LION_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd -P)}"
cd "$LION_ROOT"
LION_EXPERIMENTS_PATH="${LION_EXPERIMENTS_PATH:?Set LION_EXPERIMENTS_PATH}"
PADIS_RUN_ROOT="${PADIS_RUN_ROOT:-$LION_EXPERIMENTS_PATH/PaDIS}"
PADIS_TRAIN_ROOT="${PADIS_TRAIN_ROOT:?Set PADIS_TRAIN_ROOT}"
PADIS_RECON_ROOT="${PADIS_RECON_ROOT:?Set PADIS_RECON_ROOT}"
PADIS_GENERATION_ROOT="${PADIS_GENERATION_ROOT:-$PADIS_RUN_ROOT/reconstruction_presets}"
PADIS_PAPER_FIGURE_ROOT="${PADIS_PAPER_FIGURE_ROOT:-$PADIS_RUN_ROOT/paper_figures}"
PADIS_PAPER_TABLE_ROOT="${PADIS_PAPER_TABLE_ROOT:-$PADIS_RUN_ROOT/paper_tables}"
policy="${PADIS_RECON_CHECKPOINT_POLICY:-min_intense_val}"
checkpoint_name_for_policy() {
        local stem="$1"
        if [ "$policy" = "model_default" ]; then
                printf '%s.pt\n' "$stem"
        else
                printf '%s_%s.pt\n' "$stem" "$policy"
        fi
}
patch_checkpoint="${PADIS_GENERATION_PATCH_CHECKPOINT:-$PADIS_TRAIN_ROOT/patch_lidc_default/$(checkpoint_name_for_policy padis_lidc_256)}"
whole_checkpoint="${PADIS_GENERATION_WHOLE_CHECKPOINT:-$PADIS_TRAIN_ROOT/whole_lidc_default/$(checkpoint_name_for_policy whole_image_lidc_256)}"
presets="${PADIS_GENERATION_PRESETS:-paper-generation-whole,paper-generation-naive-patch,paper-generation,paper-generation-langevin-300nfe,paper-generation-patch-stitch,paper-generation-patch-average}"

mkdir -p "$PADIS_GENERATION_ROOT" "$PADIS_PAPER_FIGURE_ROOT" "$PADIS_PAPER_TABLE_ROOT"

for preset in ${presets//,/ }; do
        output="$PADIS_GENERATION_ROOT/lion-paper-protocol/$preset"
        checkpoint="$patch_checkpoint"
        num_steps="${PADIS_GENERATION_NUM_STEPS:-300}"
        case "$preset" in
                paper-generation-whole) checkpoint="$whole_checkpoint" ;;
                paper-generation-naive-patch|paper-generation) ;;
                paper-generation-langevin-300nfe) num_steps="${PADIS_GENERATION_LANGEVIN_NUM_STEPS:-300}" ;;
                paper-generation-patch-stitch|paper-generation-patch-average) ;;
                *) echo "Unknown generation preset: $preset" >&2; exit 2 ;;
        esac
        generation_overrides=(
                --num-steps "$num_steps"
                --inner-steps "${PADIS_GENERATION_INNER_STEPS:-1}"
                --sigma-min "${PADIS_GENERATION_SIGMA_MIN:-0.002}"
                --sigma-max "${PADIS_GENERATION_SIGMA_MAX:-10.0}"
                --noise-schedule "${PADIS_GENERATION_NOISE_SCHEDULE:-geometric}"
                --rho "${PADIS_GENERATION_RHO:-7.0}"
        )
        if [ -n "${PADIS_GENERATION_EPSILON:-}" ]; then
                generation_overrides+=(--generation-epsilon "$PADIS_GENERATION_EPSILON")
        fi
        if [ -n "${PADIS_GENERATION_NOISE_SCALE:-}" ]; then
                generation_overrides+=(--langevin-noise-scale "$PADIS_GENERATION_NOISE_SCALE")
        fi
        if [ -n "${PADIS_GENERATION_PATCH_BATCH_SIZE:-}" ]; then
                generation_overrides+=(--patch-batch-size "$PADIS_GENERATION_PATCH_BATCH_SIZE")
        fi
        if [ "${PADIS_GENERATION_PROG_BAR:-0}" = "1" ]; then
                generation_overrides+=(--prog-bar)
        fi
        python -u scripts/paper_scripts/PaDIS-Reproduction/core/PaDIS_experiments.py \
                run "$preset" --output-root "$PADIS_GENERATION_ROOT" \
                --skip-current \
                --checkpoint "$checkpoint" \
                --device "${PADIS_GENERATION_DEVICE:-cuda}" \
                --max-samples "${PADIS_GENERATION_NUM_SAMPLES:-4}" \
                --seed "${PADIS_GENERATION_SEED:-33}" \
                -- "${generation_overrides[@]}"
done

python -u scripts/paper_scripts/PaDIS-Reproduction/reconstruction/PaDIS_verify_reconstruction_matrix.py \
        --root "$PADIS_RECON_ROOT" \
        --expected-jobs-json "$PADIS_RECON_ROOT/reconstruction_matrix_jobs.json" \
        --output-json "$PADIS_RECON_ROOT/reconstruction_matrix_verification.json" \
        --output-csv "$PADIS_RECON_ROOT/reconstruction_matrix_verification.csv"

timing_mode="${PADIS_TIMING_MODE:-gcp}"
timing_root="${PADIS_TIMING_LOG_ROOT:-$PADIS_RECON_ROOT/.manual_gcp_reconstruction/logs}"
python -u scripts/paper_scripts/PaDIS-Reproduction/reporting/PaDIS_make_tables.py \
        --csv-path "$PADIS_RECON_ROOT/reconstruction_matrix_verification.csv" \
        --tex-path "$PADIS_PAPER_TABLE_ROOT/reconstruction_tables.tex" \
        --csv-output-dir "$PADIS_PAPER_TABLE_ROOT/csv" \
        --timing-mode "$timing_mode" --timing-log-root "$timing_root" \
        --timing-jobs-json "$PADIS_RECON_ROOT/reconstruction_matrix_jobs.json"
python -u scripts/paper_scripts/PaDIS-Reproduction/reporting/PaDIS_make_paper_figures.py \
        --reconstruction-root "$PADIS_RECON_ROOT" \
        --generation-root "$PADIS_GENERATION_ROOT" \
        --output-folder "$PADIS_PAPER_FIGURE_ROOT" --figures all
