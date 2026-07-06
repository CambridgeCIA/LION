#!/usr/bin/env bash
set -euo pipefail

SOCKET="/home/thomas/DiS/Project/Data/experiments/PaDIS/hparam_tuning/padis_tune.tmux"
PROJECT_ROOT="/home/thomas/DiS/Project/LION"

while tmux -S "$SOCKET" has-session -t lion_physics_focus_queue 2>/dev/null; do
        sleep 300
done

cd "$PROJECT_ROOT"

conda run --no-capture-output -n lion-dev python scripts/paper_scripts/PaDIS/PaDIS_tune_reconstruction_hyperparameters.py \
        --candidate-set lion_physics_full \
        --run-name lion_physics_pc_low_snr_confirm_validation_20260705 \
        --methods predictor_corrector \
        --only-methods predictor_corrector \
        --only-implementations lion_physics \
        --only-candidates zeta_3p75__snr_0p01,zeta_3p75__snr_0p015,zeta_4__snr_0p01,zeta_4__snr_0p015,zeta_4p25__snr_0p01,zeta_4p5__snr_0p01,zeta_4p75__snr_0p01 \
        --experiments ct_20,ct_8 \
        --max-samples 3 \
        --start-index 4 \
        --device cuda \
        --prog-bar

conda run --no-capture-output -n lion-dev python scripts/paper_scripts/PaDIS/PaDIS_tune_reconstruction_hyperparameters.py \
        --candidate-set public_paper_sampler \
        --run-name public_repo_pc_default_anchor_validation_20260705 \
        --methods predictor_corrector \
        --only-methods predictor_corrector \
        --only-implementations public_repo \
        --only-groups predictor_corrector__public_repo__patch \
        --only-candidates zeta_0p5__snr_0p16 \
        --experiments ct_20,ct_8 \
        --max-samples 3 \
        --start-index 4 \
        --device cuda \
        --prog-bar

conda run --no-capture-output -n lion-dev python scripts/paper_scripts/PaDIS/PaDIS_tune_reconstruction_hyperparameters.py \
        --candidate-set lion_physics_full \
        --run-name lion_physics_full_admm_validation_20260705 \
        --methods admm_tv \
        --only-methods admm_tv \
        --only-implementations lion_physics \
        --experiments ct_20,ct_8 \
        --max-samples 1 \
        --start-index 4 \
        --device cuda \
        --prog-bar

conda run --no-capture-output -n lion-dev python scripts/paper_scripts/PaDIS/PaDIS_tune_reconstruction_hyperparameters.py \
        --candidate-set lion_physics_full \
        --run-name lion_physics_full_pnp_validation_20260705 \
        --methods pnp_admm \
        --only-methods pnp_admm \
        --only-implementations lion_physics \
        --experiments ct_20,ct_8 \
        --max-samples 1 \
        --start-index 4 \
        --device cuda \
        --prog-bar

conda run --no-capture-output -n lion-dev python scripts/paper_scripts/PaDIS/PaDIS_tune_reconstruction_hyperparameters.py \
        --candidate-set smoke \
        --run-name lion_physics_whole_sampler_defaults_validation_20260705 \
        --methods langevin,predictor_corrector,ve_ddnm \
        --only-methods langevin,predictor_corrector,ve_ddnm \
        --only-implementations lion_physics \
        --only-groups langevin__lion_physics__whole_image,predictor_corrector__lion_physics__whole_image,ve_ddnm__lion_physics__whole_image \
        --experiments paper_matrix \
        --ablations none \
        --max-samples 1 \
        --start-index 4 \
        --device cuda \
        --prog-bar

conda run --no-capture-output -n lion-dev python scripts/paper_scripts/PaDIS/PaDIS_tune_reconstruction_hyperparameters.py \
        --candidate-set consensus_24h_no_defaults \
        --run-name lion_physics_whole_sampler_consensus_validation_20260705 \
        --methods langevin,predictor_corrector,ve_ddnm \
        --only-methods langevin,predictor_corrector,ve_ddnm \
        --only-implementations lion_physics \
        --only-groups langevin__lion_physics__whole_image,predictor_corrector__lion_physics__whole_image,ve_ddnm__lion_physics__whole_image \
        --experiments paper_matrix \
        --ablations none \
        --max-samples 1 \
        --start-index 4 \
        --device cuda \
        --prog-bar
