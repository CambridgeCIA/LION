from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm as std_tqdm

from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.config import ExperimentConfig
from LION.pcm.data import prepare_data
from LION.pcm.plotting import build_plot_helper
from LION.pcm.reconstruct import (
    ReconFn,
    build_denoiser,
    make_pnp_admm_reconstructor,
    make_spgl1_reconstructor,
)
from LION.pcm.types import GrayscaleImage2D, Measurement1D
from LION.utils.pcm_sampling import multilevel_sample, uniform_random_sample
from LION.utils.plot_helper import show_images_with_inset

tqdm = partial(std_tqdm, dynamic_ncols=True)


METRICS_HEADER = (
    "sampling_percentage,coarse_J,"
    "mse_zero_filled,psnr_zero_filled,ssim_zero_filled,pearson_corr_zero_filled,"
    "mse_recon,psnr_recon,ssim_recon,pearson_corr_recon\n"
)


def resolve_device(device_name: str) -> torch.device:
    """Resolve the torch device from a string.

    Parameters
    ----------
    device_name : str
        Requested device string, or ``'auto'``.

    Returns
    -------
    torch.device
        Resolved torch device.
    """
    if device_name != "auto":
        return torch.device(device_name)

    return torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def make_csv(method_name: str, log_dir: Path | str) -> None:
    """Create the metrics CSV file for one reconstruction method.

    Parameters
    ----------
    method_name : str
        Method-specific output directory name.
    log_dir : Path | str
        Trial log directory.
    """
    method_dir = Path(log_dir) / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    csv_path = method_dir / "metrics.csv"
    csv_path.write_text(METRICS_HEADER)


def make_test_cases(config: ExperimentConfig) -> list[tuple[float, int]]:
    """Generate sampling/coarse-level test cases.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.

    Returns
    -------
    list[tuple[float, int]]
        List of ``(sampling_ratio, coarse_J)`` pairs.
    """
    test_cases: list[tuple[float, int]] = []
    j_order = config.data.j_order

    if config.sampling.coarse_j_values is not None:
        coarse_j_values = list(config.sampling.coarse_j_values)
    elif config.sampling.coarse_j_offset_from_j_order is not None:
        coarse_j_values = [j_order - config.sampling.coarse_j_offset_from_j_order]
    else:
        coarse_j_values = list(range(j_order))

    for sampling_ratio in config.sampling.sampling_ratios:
        for coarse_j in coarse_j_values:
            test_cases.append((sampling_ratio, coarse_j))

    if config.sampling.reverse_test_cases:
        test_cases.reverse()
    return test_cases


def sample_indices(
    j_order: int,
    sampling_ratio: float,
    coarse_j: int,
    randomising_scheme: str,
    seed: int,
) -> np.ndarray:
    """Generate sampling indices for one test case.

    Parameters
    ----------
    j_order : int
        Walsh-Hadamard order.
    sampling_ratio : float
        Ratio of sampled measurements.
    coarse_j : int
        Number of coarse levels to keep deterministically.
    randomising_scheme : str
        Sampling scheme name.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Sampled measurement indices.
    """
    n = 1 << j_order
    num_samples = int(sampling_ratio * n * n)
    rng = np.random.default_rng(seed)

    if randomising_scheme == "multilevel":
        return multilevel_sample(
            J=j_order,
            num_samples=num_samples,
            coarse_J=coarse_j,
            alpha=1.0,
            rng=rng,
        )
    if randomising_scheme == "uniform":
        return uniform_random_sample(
            J=j_order,
            num_samples=num_samples,
            coarse_J=coarse_j,
            rng=rng,
        )

    raise ValueError(f"Unknown sampling scheme '{randomising_scheme}'.")


def maybe_add_noise(
    measurement: Measurement1D,
    noise_std: float,
    seed: int,
) -> Measurement1D:
    """Add optional homoscedastic Gaussian noise to a measurement vector.

    Parameters
    ----------
    measurement : Measurement1D
        Clean measurement vector.
    noise_std : float
        Standard deviation of the additive Gaussian noise.
    seed : int
        Random seed.

    Returns
    -------
    Measurement1D
        Noisy measurement vector.
    """
    if noise_std <= 0.0:
        return measurement

    generator = torch.Generator(device=measurement.device.type)
    generator.manual_seed(seed)
    noise = torch.randn(
        measurement.shape,
        generator=generator,
        device=measurement.device,
        dtype=measurement.dtype,
    )
    return measurement + noise_std * noise


def run_pcm_demo(
    *,
    config: ExperimentConfig,
    recon_description: str,
    recon_fn: ReconFn,
    ground_truth_image: GrayscaleImage2D,
    method_name: str,
    sampling_ratio: float,
    coarse_j: int,
    measurement_vector: Measurement1D | None,
    log_dir: Path | str,
    device: torch.device,
    seed: int,
) -> None:
    """Run one PCM reconstruction case and save its outputs.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    recon_description : str
        Human-readable reconstruction label.
    recon_fn : ReconFn
        Reconstruction function.
    ground_truth_image : GrayscaleImage2D
        Ground truth image.
    method_name : str
        Method-specific output directory name.
    sampling_ratio : float
        Fraction of sampled measurements.
    coarse_j : int
        Number of coarse levels kept deterministically.
    measurement_vector : Measurement1D | None
        Optional full measurement vector.
    log_dir : Path | str
        Trial output directory.
    device : torch.device
        Torch device.
    seed : int
        Random seed.
    """
    zero_filled_dir = Path(log_dir) / "zero_filled"
    zero_filled_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = Path(log_dir) / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    method_dir = Path(log_dir) / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    images_dir = method_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    recons_dir = method_dir / "recons"
    recons_dir.mkdir(parents=True, exist_ok=True)

    plot_helper = build_plot_helper(config.plot)

    j_order = config.data.j_order
    n = 1 << j_order
    im_tensor = ground_truth_image.unsqueeze(0).unsqueeze(0)

    sampling_percentage = sampling_ratio * 100.0
    in_order_measurements_percentage = (1 << (2 * coarse_j)) / (n * n) * 100.0
    print()
    print(f"Sampling rate: {sampling_percentage}%")
    print(f"Coarse levels to keep: {coarse_j} ({in_order_measurements_percentage}%)")

    sampled_indices = sample_indices(
        j_order=j_order,
        sampling_ratio=sampling_ratio,
        coarse_j=coarse_j,
        randomising_scheme=config.sampling.randomising_scheme,
        seed=seed,
    )
    pcm_op = PhotocurrentMapOp(
        J=j_order, sampled_indices=sampled_indices, device=device
    )

    if measurement_vector is not None:
        print(
            f"Using provided measurement vector with shape {measurement_vector.shape}."
        )
        y_subsampled_tensor_noiseless = measurement_vector[sampled_indices]
    else:
        y_subsampled_tensor_noiseless = pcm_op(im_tensor)

    y_subsampled_tensor = maybe_add_noise(
        measurement=y_subsampled_tensor_noiseless,
        noise_std=config.runtime.noise_std,
        seed=config.runtime.noise_seed + seed,
    )

    zero_filled_recon_tensor = (
        pcm_op.pseudo_inv(y_subsampled_tensor).unsqueeze(0).unsqueeze(0)
    )
    recon_tensor = (
        recon_fn(
            pcm_op=pcm_op,
            pcm_measurement=y_subsampled_tensor,
            initial_image=zero_filled_recon_tensor.squeeze(),
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )

    data_range = (im_tensor.max() - im_tensor.min()).item()
    if data_range <= 0.0:
        data_range = 1.0
    psnr = PeakSignalNoiseRatio(data_range=data_range).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)

    psnr_zero_filled = float(psnr(zero_filled_recon_tensor, im_tensor).item())
    psnr_recon = float(psnr(recon_tensor, im_tensor).item())
    ssim_zero_filled = float(ssim(zero_filled_recon_tensor, im_tensor).item())
    ssim_recon = float(ssim(recon_tensor, im_tensor).item())

    mse_zero_filled = float(
        torch.mean((zero_filled_recon_tensor - im_tensor) ** 2).item()
    )
    mse_recon = float(torch.mean((recon_tensor - im_tensor) ** 2).item())

    pearson_corr_zero_filled = float(
        torch.corrcoef(
            torch.stack([zero_filled_recon_tensor.flatten(), im_tensor.flatten()])
        )[0, 1].item()
    )
    pearson_corr_recon = float(
        torch.corrcoef(torch.stack([recon_tensor.flatten(), im_tensor.flatten()]))[
            0, 1
        ].item()
    )

    csv_path = method_dir / "metrics.csv"
    with csv_path.open("a") as f:
        f.write(
            f"{sampling_percentage},{coarse_j},"
            f"{mse_zero_filled},{psnr_zero_filled},{ssim_zero_filled},{pearson_corr_zero_filled},"
            f"{mse_recon},{psnr_recon},{ssim_recon},{pearson_corr_recon}\n"
        )

    filename = (
        f"{config.data.data_name}_{method_name}_sample_{sampling_percentage}_percent_"
        f"coarse_J={coarse_j}_{config.sampling.randomising_scheme}_random"
    )
    zero_filled_filename = (
        f"{config.data.data_name}_sample_{sampling_percentage}_percent_"
        f"coarse_J={coarse_j}_{config.sampling.randomising_scheme}_random"
    )

    np.save(
        zero_filled_dir / f"{zero_filled_filename}.npy",
        zero_filled_recon_tensor.squeeze().cpu().numpy(),
    )
    np.save(recons_dir / f"{filename}.npy", recon_tensor.squeeze().cpu().numpy())

    mask_of_masks_np = np.zeros(n * n, dtype=bool)
    mask_of_masks_np[sampled_indices] = True
    np.save(
        masks_dir
        / f"sample_{sampling_percentage}_percent_coarse_J={coarse_j}_{config.sampling.randomising_scheme}_random.npy",
        mask_of_masks_np.reshape(n, n),
    )

    show_images_with_inset(
        [im_tensor, zero_filled_recon_tensor, recon_tensor],
        fig_filepath=images_dir / f"{filename}.png",
        plot_helper=plot_helper,
        titles=[
            "Original Image",
            (
                "Inverse WHT (Zero-filled)\n"
                f"PSNR: {psnr_zero_filled:.2f} dB, SSIM: {ssim_zero_filled:.4f}\n"
                f"MSE: {mse_zero_filled:.3e}, Pearson Corr.: {pearson_corr_zero_filled:.4f}"
            ),
            (
                f"{recon_description}\n"
                f"PSNR: {psnr_recon:.2f} dB, SSIM: {ssim_recon:.4f}\n"
                f"MSE: {mse_recon:.3e}, Pearson Corr.: {pearson_corr_recon:.4f}"
            ),
        ],
        suptitle=(
            f"PCM Reconstructions, J={j_order} ({n}x{n} image)\n"
            f"Sample {sampling_percentage:.2f}%, keep {coarse_j} coarse levels "
            f"({in_order_measurements_percentage}% here), the rest: {config.sampling.randomising_scheme} random"
        ),
        adds_insets=config.plot.adds_insets,
        saves_fig=True,
    )


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


def override_device(
    config: ExperimentConfig, device_name: str | None
) -> ExperimentConfig:
    """Return a copy of ``config`` with an optional device override.

    Parameters
    ----------
    config : ExperimentConfig
        Original configuration.
    device_name : str | None
        Override for the runtime device.

    Returns
    -------
    ExperimentConfig
        Possibly modified configuration.
    """
    if device_name is None:
        return config
    return replace(config, runtime=replace(config.runtime, device=device_name))
