"""Metrics computation utilities."""

import torch
from piq import haarpsi
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


def compute_mse(recon: torch.Tensor, *, target: torch.Tensor) -> float:
    """Computes mean squared error between recon and target."""
    return torch.mean((recon.reshape(-1) - target.reshape(-1)) ** 2).item()


def compute_psnr(recon: torch.Tensor, target: torch.Tensor, data_range: float) -> float:
    """Computes peak signal-to-noise ratio between recon and target."""
    return peak_signal_noise_ratio(
        preds=recon, target=target, data_range=data_range
    ).item()


def compute_ssim(
    recon: torch.Tensor, *, target: torch.Tensor, data_range: float
) -> float:
    """Computes structural similarity index measure between recon and target."""
    return structural_similarity_index_measure(
        preds=recon,
        target=target,
        data_range=data_range,
        gaussian_kernel=False,  # scikit-image compatible
        kernel_size=7,  # scikit-image compatible
    ).item()


def compute_haarpsi(
    recon: torch.Tensor, *, target: torch.Tensor, data_range: float
) -> float:
    """Computes HAARPsi between recon and target."""
    return haarpsi(x=recon, y=target, data_range=data_range).item()


def compute_pearson_correlation(recon: torch.Tensor, *, target: torch.Tensor) -> float:
    """Computes Pearson correlation coefficient between recon and target."""
    return torch.corrcoef(torch.stack([recon.flatten(), target.flatten()]))[0, 1].item()


def compute_metrics_2d(
    recon: torch.Tensor, *, target: torch.Tensor, data_range: float = 1.0
) -> tuple[float, float, float, float]:
    """Computes MSE, PSNR, SSIM, and HAARPsi between recon and target."""
    assert (
        recon.shape == target.shape
    ), f"Shape mismatch: recon {recon.shape}, target {target.shape}"
    assert (recon.dim() == 4) and (
        recon.shape[0] == 1
    ), f"Expected 4D tensor with shape (1, c, y, x), but got {recon.shape}"
    # ssim and haarpsi expect `preds` and `target` to have BxCxHxW or BxCxDxHxW shape
    recon = recon.detach().cpu().contiguous()
    target = target.detach().cpu().contiguous()
    mse = compute_mse(recon=recon, target=target)
    psnr = compute_psnr(recon=recon, target=target, data_range=data_range)
    ssim = compute_ssim(recon=recon, target=target, data_range=data_range)
    haarpsi = compute_haarpsi(
        recon=recon.clip(0.0, data_range), target=target, data_range=data_range
    )
    return mse, psnr, ssim, haarpsi


def compute_metrics(
    *, recon: torch.Tensor, target: torch.Tensor, data_range: float
) -> dict[str, float]:
    assert (
        recon.shape == target.shape
    ), f"Shape mismatch: recon {recon.shape}, target {target.shape}"
    while recon.dim() < 4:
        recon = recon.unsqueeze(0)
        target = target.unsqueeze(0)
    mse = compute_mse(recon=recon, target=target)
    psnr = compute_psnr(recon=recon, target=target, data_range=data_range)
    ssim = compute_ssim(recon=recon, target=target, data_range=data_range)
    haarpsi = compute_haarpsi(
        recon=recon.clamp_min(0),
        target=target.clamp_min(0),
        data_range=max(target.max().item(), recon.max().item()),
    )
    pearson_corr = float(
        torch.corrcoef(torch.stack([recon.flatten(), target.flatten()]))[0, 1].item()
    )
    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "haarpsi": haarpsi,
        "pearson_corr": pearson_corr,
    }
