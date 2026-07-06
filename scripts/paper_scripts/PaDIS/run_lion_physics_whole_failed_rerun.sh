#!/usr/bin/env bash
set -euo pipefail

echo "This pre-fixed-validation helper is obsolete." >&2
echo "Use scripts/paper_scripts/PaDIS/run_fixedval_reconstruction_tuning.sh instead." >&2
echo "Refusing to write contaminated/non-fixedval run names or summarize --run-names all." >&2
exit 1

SOCKET="/home/thomas/DiS/Project/Data/experiments/PaDIS/hparam_tuning/padis_tune.tmux"
PROJECT_ROOT="/home/thomas/DiS/Project/LION"

while tmux -S "$SOCKET" has-session -t paper_public_consensus_queue 2>/dev/null; do
        sleep 300
done

cd "$PROJECT_ROOT"

conda run --no-capture-output -n lion-dev python scripts/paper_scripts/PaDIS/PaDIS_tune_reconstruction_hyperparameters.py \
        --candidate-set consensus_24h_no_defaults \
        --run-name lion_physics_whole_sampler_failed_rerun_20260705 \
        --methods predictor_corrector,ve_ddnm \
        --only-methods predictor_corrector,ve_ddnm \
        --only-implementations lion_physics \
        --only-groups predictor_corrector__lion_physics__whole_image,ve_ddnm__lion_physics__whole_image \
        --only-candidates zeta_4p25__snr_0p08,zeta_4p5__snr_0p02,zeta_4p5__snr_0p04,zeta_4p5__snr_0p08,sampling_eps_0p05,sampling_eps_0p1,sampling_eps_0p2 \
        --experiments paper_matrix \
        --ablations none \
        --max-samples 1 \
        --start-index 4 \
        --device cuda \
        --prog-bar

conda run --no-capture-output -n lion-dev python scripts/paper_scripts/PaDIS/PaDIS_summarize_hparam_tuning.py \
        --run-names all \
        --expected-experiments ct_20,ct_8 \
        --top-k 8 \
        --output-csv /home/thomas/DiS/Project/Data/experiments/PaDIS/hparam_tuning/runs/hparam_recommendations_with_public_deltas.csv
