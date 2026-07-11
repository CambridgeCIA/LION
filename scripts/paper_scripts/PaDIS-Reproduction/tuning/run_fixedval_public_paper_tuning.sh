#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/thomas/DiS/Project/LION"
RUN_ROOT="/home/thomas/DiS/Project/Data/experiments/PaDIS/hparam_tuning/runs"
FIXEDVAL_RUNS="fixedval_smoke_validation_20260705,fixedval_consensus_ct20_ct8_validation_20260705,fixedval_lion_physics_dps_refinement_ct20_ct8_validation_20260705,fixedval_lion_physics_refinement_ct20_ct8_validation_20260705,fixedval_public_paper_dps_ct20_ct8_validation_20260705,fixedval_public_paper_default_anchors_ct20_ct8_validation_20260705,fixedval_public_paper_sampler_refinement_ct20_ct8_validation_20260705"

RUN_PUBLIC_PAPER_DPS="${RUN_PUBLIC_PAPER_DPS:-1}"
RUN_PUBLIC_PAPER_DEFAULT_ANCHORS="${RUN_PUBLIC_PAPER_DEFAULT_ANCHORS:-1}"
RUN_PUBLIC_PAPER_SAMPLER_REFINEMENT="${RUN_PUBLIC_PAPER_SAMPLER_REFINEMENT:-0}"

PUBLIC_PAPER_DPS_CANDIDATES=(
        current_defaults
        zeta_0p15__eps_0p5
        zeta_0p15__eps_0p75
        zeta_0p2__eps_0p5
        zeta_0p2__eps_0p75
        zeta_0p0075__eps_0p5
        zeta_0p0075__eps_1
        zeta_0p01__eps_0p5
        zeta_0p01__eps_1
        zeta_0p015__eps_0p5
        zeta_0p015__eps_1
)
PUBLIC_PAPER_DPS_CANDIDATES_CSV="$(IFS=,; echo "${PUBLIC_PAPER_DPS_CANDIDATES[*]}")"

PUBLIC_PAPER_SAMPLER_CANDIDATES=(
        current_defaults
        zeta_0p2__eps_0p5
        zeta_0p2__eps_0p75
        zeta_0p3__eps_0p5
        zeta_0p3__eps_0p75
        zeta_0p01__eps_0p5
        zeta_0p01__eps_0p75
        zeta_0p03__eps_0p5
        zeta_0p03__eps_0p75
        zeta_0p5__snr_0p08
        zeta_0p5__snr_0p16
        zeta_0p01__snr_0p08
        zeta_0p01__snr_0p16
        zeta_0p03__snr_0p08
        zeta_0p03__snr_0p16
        sampling_eps_0p05
        sampling_eps_0p1
        sampling_eps_0p2
)
PUBLIC_PAPER_SAMPLER_CANDIDATES_CSV="$(IFS=,; echo "${PUBLIC_PAPER_SAMPLER_CANDIDATES[*]}")"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-padis}"
mkdir -p "$MPLCONFIGDIR"

cd "$PROJECT_ROOT"

conda run --no-capture-output -n lion-dev python scripts/paper_scripts/PaDIS-Reproduction/tuning/PaDIS_summarize_hparam_tuning.py \
        --run-names "$FIXEDVAL_RUNS" \
        --expected-experiments ct_20,ct_8 \
        --top-k 8 \
        --output-csv "$RUN_ROOT/fixedval_hparam_recommendations.csv"

if [[ "$RUN_PUBLIC_PAPER_DPS" == "1" ]]; then
        conda run --no-capture-output -n lion-dev python scripts/paper_scripts/PaDIS-Reproduction/tuning/PaDIS_tune_reconstruction_hyperparameters.py \
                --candidate-set consensus_24h \
                --run-name fixedval_public_paper_dps_ct20_ct8_validation_20260705 \
                --methods all \
                --implementations method_default \
                --only-implementations public_repo,paper \
                --only-methods padis_dps \
                --only-groups padis_dps__public_repo__any,padis_dps__paper__any \
                --only-candidates "$PUBLIC_PAPER_DPS_CANDIDATES_CSV" \
                --experiments ct_20,ct_8 \
                --ablations none \
                --max-samples 1 \
                --start-index 4 \
                --device cuda \
                --prog-bar
fi

if [[ "$RUN_PUBLIC_PAPER_DEFAULT_ANCHORS" == "1" ]]; then
        conda run --no-capture-output -n lion-dev python scripts/paper_scripts/PaDIS-Reproduction/tuning/PaDIS_tune_reconstruction_hyperparameters.py \
                --candidate-set smoke \
                --run-name fixedval_public_paper_default_anchors_ct20_ct8_validation_20260705 \
                --methods all \
                --implementations method_default \
                --only-implementations public_repo,paper \
                --only-methods langevin,predictor_corrector,ve_ddnm,whole_image_diffusion \
                --only-groups langevin__public_repo__patch,langevin__paper__patch,predictor_corrector__public_repo__patch,predictor_corrector__paper__patch,ve_ddnm__public_repo__patch,ve_ddnm__paper__patch,whole_image_diffusion__paper__any \
                --only-candidates current_defaults \
                --experiments ct_20,ct_8 \
                --ablations none \
                --max-samples 1 \
                --start-index 4 \
                --device cuda \
                --prog-bar
fi

if [[ "$RUN_PUBLIC_PAPER_SAMPLER_REFINEMENT" == "1" ]]; then
        conda run --no-capture-output -n lion-dev python scripts/paper_scripts/PaDIS-Reproduction/tuning/PaDIS_tune_reconstruction_hyperparameters.py \
                --candidate-set consensus_24h \
                --run-name fixedval_public_paper_sampler_refinement_ct20_ct8_validation_20260705 \
                --methods all \
                --implementations method_default \
                --only-implementations public_repo,paper \
                --only-methods langevin,predictor_corrector,ve_ddnm \
                --only-groups langevin__public_repo__patch,langevin__paper__patch,predictor_corrector__public_repo__patch,predictor_corrector__paper__patch,ve_ddnm__public_repo__patch,ve_ddnm__paper__patch \
                --only-candidates "$PUBLIC_PAPER_SAMPLER_CANDIDATES_CSV" \
                --experiments ct_20,ct_8 \
                --ablations none \
                --max-samples 1 \
                --start-index 4 \
                --device cuda \
                --prog-bar
fi

conda run --no-capture-output -n lion-dev python scripts/paper_scripts/PaDIS-Reproduction/tuning/PaDIS_summarize_hparam_tuning.py \
        --run-names "$FIXEDVAL_RUNS" \
        --expected-experiments ct_20,ct_8 \
        --top-k 8 \
        --output-csv "$RUN_ROOT/fixedval_hparam_recommendations.csv"
