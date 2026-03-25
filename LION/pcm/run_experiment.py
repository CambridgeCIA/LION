from __future__ import annotations

from datetime import datetime
from functools import partial
from pathlib import Path

from tqdm import tqdm as std_tqdm

from LION.pcm.config import ExperimentConfig
from LION.pcm.data import prepare_data
from LION.pcm.denoiser import build_denoiser
from LION.pcm.experiment import make_csv, make_test_cases, resolve_device
from LION.pcm.reconstruct import (
    ReconFn,
    make_pnp_admm_reconstructor,
    make_spgl1_reconstructor,
)
from LION.pcm.run_demo import run_pcm_demo

tqdm = partial(std_tqdm, dynamic_ncols=True)


def run_experiment(config: ExperimentConfig) -> Path:
    """Run the full PCM experiment defined by ``config``.

    Parameters
    ----------
    config : ExperimentConfig
        Full experiment configuration.

    Returns
    -------
    Path
        Output directory for the experiment.
    """
    device = resolve_device(config.runtime.device)
    print(f"Using device: {device}")

    config.runtime.root_output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_image, measurement_vector = prepare_data(config.data, device)
    test_cases = make_test_cases(config)

    current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_log_dir = (
        config.runtime.root_output_dir
        / f"{current_datetime_str}_{config.data.data_name}_{config.sampling.randomising_scheme}_{config.runtime.num_trials}_trials"
    )
    experiment_log_dir.mkdir(parents=True, exist_ok=True)

    pnp_recon_fn: ReconFn | None = None
    if config.pnp.enabled:
        denoiser = build_denoiser(config.pnp, device)
        pnp_recon_fn = make_pnp_admm_reconstructor(config.pnp, config.data, denoiser)

    spgl1_recon_fn: ReconFn | None = None
    if config.spgl1.enabled:
        spgl1_recon_fn = make_spgl1_reconstructor(config.spgl1, device)

    for i_seed in tqdm(
        range(config.runtime.num_trials_skip, config.runtime.num_trials),
        desc="Running trials",
    ):
        print(f"\n=== Trial {i_seed} ===")
        trial_log_dir = experiment_log_dir / f"trial_{i_seed}"
        trial_log_dir.mkdir(parents=True, exist_ok=True)

        if config.pnp.enabled and pnp_recon_fn is not None:
            method_name = (
                f"pnp_admm_{config.pnp.denoiser_name}_iters={config.pnp.iters}_"
                f"eta={config.pnp.eta}_cg_iters={config.pnp.cg_iters}_"
                f"drunet_sigma={config.pnp.drunet_sigma}"
            )
            make_csv(method_name=method_name, log_dir=trial_log_dir)
            for sampling_ratio, coarse_j in tqdm(
                test_cases, desc="Running PnP-ADMM experiments"
            ):
                run_pcm_demo(
                    config=config,
                    recon_description=(
                        f"PnP-ADMM ({config.pnp.iters} iters, eta={config.pnp.eta}, "
                        f"cg_iters={config.pnp.cg_iters}, sigma={config.pnp.drunet_sigma})"
                    ),
                    recon_fn=pnp_recon_fn,
                    ground_truth_image=ground_truth_image,
                    method_name=method_name,
                    sampling_ratio=sampling_ratio,
                    coarse_j=coarse_j,
                    measurement_vector=measurement_vector,
                    log_dir=trial_log_dir,
                    device=device,
                    seed=i_seed,
                )

        if config.spgl1.enabled and spgl1_recon_fn is not None:
            method_name = f"spgl1_factor={config.spgl1.factor}"
            make_csv(method_name=method_name, log_dir=trial_log_dir)
            for sampling_ratio, coarse_j in tqdm(
                test_cases, desc="Running SPGL1 experiments"
            ):
                run_pcm_demo(
                    config=config,
                    recon_description="SPGL1",
                    recon_fn=spgl1_recon_fn,
                    ground_truth_image=ground_truth_image,
                    method_name=method_name,
                    sampling_ratio=sampling_ratio,
                    coarse_j=coarse_j,
                    measurement_vector=measurement_vector,
                    log_dir=trial_log_dir,
                    device=device,
                    seed=i_seed,
                )

    return experiment_log_dir
