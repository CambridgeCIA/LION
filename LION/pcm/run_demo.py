from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.config import ExperimentConfig
from LION.pcm.experiment import sample_indices
from LION.pcm.reconstruct import ReconFn
from LION.pcm.types import GrayscaleImage2D, Measurement1D


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

    plot_helper = config.plot

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

    # TODO: Implement noise model
    y_subsampled_tensor = y_subsampled_tensor_noiseless

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

    plot_helper.show_images_with_inset(
        [im_tensor, zero_filled_recon_tensor, recon_tensor],
        fig_filepath=images_dir / f"{filename}.png",
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
        saves_fig=True,
    )
