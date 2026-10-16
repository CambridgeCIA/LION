"""Metrics utilities."""

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torchmetrics.functional.image import structural_similarity_index_measure


def compute_psnr_torch(image: torch.Tensor, reference: torch.Tensor) -> float:
    """Compute PSNR using PyTorch.

    Parameters
    ----------
    image : torch.Tensor
        Reconstructed image.
    reference : torch.Tensor
        Reference image.

    Returns:
    -------
    float
        PSNR in dB.
    """
    data_range = torch.max(reference) - torch.min(reference)
    mse = torch.mean((image - reference) ** 2)

    mse_float = float(mse.detach().cpu())
    if mse_float == 0.0:
        return float("inf")

    psnr = 20.0 * torch.log10(data_range) - 10.0 * torch.log10(mse)
    return float(psnr.detach().cpu())


def compute_ssim_cpu(
    image: torch.Tensor, reference_np: np.ndarray, pixel_min: float
) -> float:
    """Compute SSIM using scikit-image on CPU.

    Parameters
    ----------
    image : torch.Tensor
        Reconstructed image.
    reference_np : np.ndarray
        Reference image as a CPU array.
    pixel_min : float
        Minimum pixel value used for MATLAB-style scaling.

    Returns:
    -------
    float
        SSIM value.
    """
    image_np = image.detach().cpu().numpy()

    image_ssim = image_np / pixel_min
    reference_ssim = reference_np / pixel_min
    data_range = reference_ssim.max() - reference_ssim.min()

    return float(
        structural_similarity(image_ssim, reference_ssim, data_range=data_range)
    )


def compute_ssim_torch(
    image: torch.Tensor, reference: torch.Tensor, pixel_min: float
) -> float:
    """Compute SSIM using TorchMetrics.

    Parameters
    ----------
    image : torch.Tensor
        Reconstructed image.
    reference : torch.Tensor
        Reference image.
    pixel_min : float
        Minimum pixel value used for MATLAB-style scaling.

    Returns:
    -------
    float
        SSIM value.
    """
    image_ssim = image / pixel_min
    reference_ssim = reference / pixel_min

    data_range = torch.max(reference_ssim) - torch.min(reference_ssim)

    # TorchMetrics image metrics normally expect NCHW-like tensors.
    image_ssim = image_ssim[None, None, :, :]
    reference_ssim = reference_ssim[None, None, :, :]

    ssim = structural_similarity_index_measure(
        preds=image_ssim, target=reference_ssim, data_range=data_range
    )

    return float(ssim.detach().cpu())


def compute_metrics_hybrid(
    image: torch.Tensor,
    reference: torch.Tensor,
    pixel_min: float,
) -> tuple[float, float]:
    """Compute SSIM and PSNR.

    Parameters
    ----------
    image : torch.Tensor
        Reconstructed image.
    reference : torch.Tensor
        Reference image.
    reference_np : np.ndarray
        Reference image as a CPU array.
    pixel_min : float
        Minimum pixel value used for MATLAB-style scaling.

    Returns:
    -------
    tuple[float, float]
        SSIM and PSNR.
    """
    ssim_val = compute_ssim_torch(image=image, reference=reference, pixel_min=pixel_min)
    psnr_val = compute_psnr_torch(image=image, reference=reference)
    return ssim_val, psnr_val
