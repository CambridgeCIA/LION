#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/thomas/DiS/Project/LION"
RUN_ROOT="/home/thomas/DiS/Project/Data/experiments/PaDIS/hparam_tuning/runs"
FIXEDVAL_RUNS="fixedval_smoke_validation_20260705,fixedval_consensus_ct20_ct8_validation_20260705,fixedval_lion_physics_dps_refinement_ct20_ct8_validation_20260705,fixedval_lion_physics_refinement_ct20_ct8_validation_20260705"
RUN_DPS_REFINEMENT="${RUN_DPS_REFINEMENT:-1}"
RUN_TARGETED_REFINEMENT="${RUN_TARGETED_REFINEMENT:-1}"
RUN_SMOKE_ANCHORS="${RUN_SMOKE_ANCHORS:-0}"
DPS_REFINEMENT_CANDIDATES=(
        core_zeta_3p5__eps_0p3
        core_zeta_3p5__eps_0p5
        core_zeta_3p75__eps_0p3
        core_zeta_3p75__eps_0p5
        core_zeta_4__eps_0p3
        core_zeta_4__eps_0p5
        core_zeta_4p25__eps_0p3
)
DPS_REFINEMENT_CANDIDATES_CSV="$(IFS=,; echo "${DPS_REFINEMENT_CANDIDATES[*]}")"
REFINEMENT_CANDIDATES=(
        tv_lam_0p0005__iters_1000
        tv_lam_0p001__iters_1000
        tv_lam_0p002__iters_1000
        eta_5e_06__iters_100
        eta_1e_05__iters_100
        eta_2e_05__iters_100
        eta_3e_05__iters_100
        eta_5e_05__iters_60
        eta_5e_05__iters_100
        zeta_3p75__snr_0p01
        zeta_3p75__snr_0p015
        zeta_4__snr_0p01
        zeta_4__snr_0p015
        zeta_4p25__snr_0p01
        zeta_4p25__snr_0p015
        zeta_4p5__snr_0p01
        zeta_4p5__snr_0p015
        zeta_4p75__snr_0p01
        zeta_4p75__snr_0p015
)
REFINEMENT_CANDIDATES_CSV="$(IFS=,; echo "${REFINEMENT_CANDIDATES[*]}")"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-padis}"
mkdir -p "$MPLCONFIGDIR"

cd "$PROJECT_ROOT"

conda run --no-capture-output -n lion-dev python scripts/paper_scripts/PaDIS-Reproduction/tuning/PaDIS_summarize_hparam_tuning.py \
        --run-names "$FIXEDVAL_RUNS" \
        --expected-experiments ct_20,ct_8 \
        --top-k 8 \
        --output-csv "$RUN_ROOT/fixedval_hparam_recommendations.csv"

if [[ "$RUN_DPS_REFINEMENT" == "1" ]]; then
        conda run --no-capture-output -n lion-dev python scripts/paper_scripts/PaDIS-Reproduction/tuning/PaDIS_tune_reconstruction_hyperparameters.py \
                --candidate-set lion_physics_full \
                --run-name fixedval_lion_physics_dps_refinement_ct20_ct8_validation_20260705 \
                --methods all \
                --implementations method_default \
                --only-implementations lion_physics \
                --only-methods padis_dps \
                --only-groups padis_dps__lion_physics__any \
                --only-candidates "$DPS_REFINEMENT_CANDIDATES_CSV" \
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

if [[ "$RUN_TARGETED_REFINEMENT" == "1" ]]; then
        conda run --no-capture-output -n lion-dev python scripts/paper_scripts/PaDIS-Reproduction/tuning/PaDIS_tune_reconstruction_hyperparameters.py \
                --candidate-set lion_physics_full \
                --run-name fixedval_lion_physics_refinement_ct20_ct8_validation_20260705 \
                --methods all \
                --implementations method_default \
                --only-implementations lion_physics \
                --only-methods admm_tv,pnp_admm,predictor_corrector \
                --only-groups admm_tv__lion_physics__any,pnp_admm__lion_physics__any,predictor_corrector__lion_physics__patch \
                --only-candidates "$REFINEMENT_CANDIDATES_CSV" \
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

if [[ "$RUN_SMOKE_ANCHORS" == "1" ]]; then
        conda run --no-capture-output -n lion-dev python scripts/paper_scripts/PaDIS-Reproduction/tuning/PaDIS_tune_reconstruction_hyperparameters.py \
                --candidate-set smoke \
                --run-name fixedval_smoke_validation_20260705 \
                --methods all \
                --implementations method_default \
                --experiments paper_matrix \
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
