"""Image metrics and reconstruction diagnostics for PaDIS LIDC runs."""

from __future__ import annotations

import math

import numpy as np
import torch

from LION.reconstructors import PaDIS


def psnr(recon: torch.Tensor, target: torch.Tensor, data_range: float) -> float:
    """Handle psnr for the PaDIS workflow."""
    mse = torch.mean((recon - target).square()).item()
    if mse == 0:
        return float("inf")
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse))


def psnr_from_mse(mse: float, data_range: float) -> float:
    """Handle psnr from mse for the PaDIS workflow."""
    if mse == 0:
        return float("inf")
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse))


def mean_metric(metrics: list[dict], key: str):
    """Return the mean metric."""
    values = [item[key] for item in metrics if key in item]
    if not values:
        return None
    return sum(values) / len(values)


def min_metric(metrics: list[dict], key: str):
    """Return the minimum metric."""
    values = [item[key] for item in metrics if key in item]
    if not values:
        return None
    return min(values)


def masked_mse(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Handle masked mse for the PaDIS workflow."""
    mask = mask.to(device=recon.device, dtype=torch.bool)
    if not torch.any(mask):
        return None
    return float(torch.mean((recon[mask] - target[mask]).square()).item())


def masked_mae(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Handle masked mae for the PaDIS workflow."""
    mask = mask.to(device=recon.device, dtype=torch.bool)
    if not torch.any(mask):
        return None
    return float(torch.mean(torch.abs(recon[mask] - target[mask])).item())


def ssim_or_none(recon: torch.Tensor, target: torch.Tensor, data_range: float):
    """Calculate SSIM for or none."""
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return None
    recon_np = recon.detach().cpu().squeeze().numpy()
    target_np = target.detach().cpu().squeeze().numpy()
    return float(structural_similarity(target_np, recon_np, data_range=data_range))


def normal_to_hu(image: torch.Tensor) -> torch.Tensor:
    """Convert normalised intensity to to hu."""
    return 3000.0 * image - 1000.0


def hu_window(image: torch.Tensor, *, level: float, width: float) -> torch.Tensor:
    """Handle hu window for the PaDIS workflow."""
    lower = level - width / 2.0
    return ((normal_to_hu(image) - lower) / width).clamp(0.0, 1.0)


def mask_bbox(mask: torch.Tensor, pad: int = 8):
    """Return the mask bbox."""
    mask_2d = mask.detach().cpu().squeeze().bool()
    if not torch.any(mask_2d):
        return None
    rows = torch.where(torch.any(mask_2d, dim=1))[0]
    cols = torch.where(torch.any(mask_2d, dim=0))[0]
    top = max(int(rows[0]) - pad, 0)
    bottom = min(int(rows[-1]) + pad + 1, mask_2d.shape[0])
    left = max(int(cols[0]) - pad, 0)
    right = min(int(cols[-1]) + pad + 1, mask_2d.shape[1])
    return top, bottom, left, right


def crop_bbox(image: torch.Tensor, bbox):
    """Crop bbox."""
    if bbox is None:
        return image
    top, bottom, left, right = bbox
    return image[..., top:bottom, left:right]


def ssim_on_bbox_or_none(
    recon: torch.Tensor,
    target: torch.Tensor,
    data_range: float,
    bbox,
):
    """Calculate SSIM for on bbox or none."""
    return ssim_or_none(crop_bbox(recon, bbox), crop_bbox(target, bbox), data_range)


def edge_ssim_or_none(recon: torch.Tensor, target: torch.Tensor):
    """Handle edge ssim or none for the PaDIS workflow."""
    try:
        from skimage.filters import sobel
        from skimage.metrics import structural_similarity
    except ImportError:
        return None

    recon_np = recon.detach().cpu().squeeze().numpy()
    target_np = target.detach().cpu().squeeze().numpy()
    recon_edges = sobel(recon_np)
    target_edges = sobel(target_np)
    data_range = float(target_edges.max() - target_edges.min())
    if data_range == 0:
        return 1.0 if float(np.max(np.abs(recon_edges - target_edges))) == 0 else 0.0
    return float(
        structural_similarity(target_edges, recon_edges, data_range=data_range)
    )


def add_image_similarity_metrics(
    item: dict,
    *,
    prefix: str,
    image: torch.Tensor,
    reference: torch.Tensor,
    data_range: float,
) -> None:
    """Add image similarity metrics."""
    key = "" if prefix == "" else f"{prefix}_"
    abs_error = torch.abs(image - reference)
    mse = float(torch.mean((image - reference).square()).item())
    item[f"{key}mse"] = mse
    item[f"{key}psnr"] = psnr_from_mse(mse, data_range)
    item[f"{key}mae"] = float(torch.mean(abs_error).item())
    item[f"{key}abs_error_p95"] = float(
        torch.quantile(abs_error.flatten(), 0.95).item()
    )
    item[f"{key}abs_error_p99"] = float(
        torch.quantile(abs_error.flatten(), 0.99).item()
    )
    item[f"{key}mean_delta"] = float((image.mean() - reference.mean()).item())
    ssim_value = ssim_or_none(image, reference, data_range)
    if ssim_value is not None:
        item[f"{key}ssim"] = ssim_value
    edge_ssim_value = edge_ssim_or_none(image, reference)
    if edge_ssim_value is not None:
        item[f"{key}edge_ssim"] = edge_ssim_value


def image_tensor_from_array(array) -> torch.Tensor:
    """Return the image tensor from array."""
    tensor = torch.as_tensor(array).detach().cpu().float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[0] in (1, 3):
            pass
        elif tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        else:
            raise ValueError(
                f"Cannot infer image channel axis for shape {tensor.shape}."
            )
    else:
        raise ValueError(
            f"Expected an image tensor with 2 or 3 dimensions, got {tensor.ndim}."
        )
    return tensor.contiguous()


def forward_project_normal_image(
    image: torch.Tensor,
    reconstructor: PaDIS,
    params,
) -> torch.Tensor:
    """Apply the forward operation to project normal image."""
    measurement_image = float(params.measurement_scale) * image + float(
        params.measurement_offset
    )
    return reconstructor.op(measurement_image)


def relative_sinogram_residual(
    image: torch.Tensor,
    sinogram: torch.Tensor,
    reconstructor: PaDIS,
    params,
) -> float:
    """Calculate the relative sinogram residual."""
    with torch.no_grad():
        predicted = forward_project_normal_image(image, reconstructor, params)
        residual = predicted.to(dtype=sinogram.dtype) - sinogram
        return float(
            torch.linalg.norm(residual)
            .div(torch.linalg.norm(sinogram).clamp_min(1e-12))
            .item()
        )


def add_reconstruction_metrics(
    item: dict,
    *,
    prefix: str,
    recon: torch.Tensor,
    target: torch.Tensor,
    sinogram: torch.Tensor,
    reconstructor: PaDIS,
    params,
    body_mask: torch.Tensor,
    nonair_mask: torch.Tensor,
    body_bbox,
    data_range: float,
) -> None:
    """Add reconstruction metrics."""
    key = "" if prefix == "" else f"{prefix}_"
    abs_error = torch.abs(recon - target)
    mse = float(torch.mean((recon - target).square()).item())
    mae = float(torch.mean(abs_error).item())
    item[f"{key}mse"] = mse
    item[f"{key}mae"] = mae
    item[f"{key}abs_error_p95"] = float(
        torch.quantile(abs_error.flatten(), 0.95).item()
    )
    item[f"{key}abs_error_p99"] = float(
        torch.quantile(abs_error.flatten(), 0.99).item()
    )
    item[f"{key}psnr"] = psnr_from_mse(mse, data_range)
    item[f"{key}min"] = float(recon.detach().amin().cpu())
    item[f"{key}max"] = float(recon.detach().amax().cpu())
    item[f"{key}mean"] = float(recon.detach().mean().cpu())
    item[f"{key}hu_mae"] = float(
        torch.mean(torch.abs(normal_to_hu(recon) - normal_to_hu(target))).item()
    )
    item[f"{key}relative_sinogram_residual"] = relative_sinogram_residual(
        recon, sinogram, reconstructor, params
    )

    for mask_name, mask in (("body", body_mask), ("nonair", nonair_mask)):
        mse_value = masked_mse(recon, target, mask)
        mae_value = masked_mae(normal_to_hu(recon), normal_to_hu(target), mask)
        if mse_value is not None:
            item[f"{key}{mask_name}_mse"] = mse_value
            item[f"{key}{mask_name}_psnr"] = psnr_from_mse(mse_value, data_range)
        if mae_value is not None:
            item[f"{key}{mask_name}_hu_mae"] = mae_value

    ssim_value = ssim_or_none(recon, target, data_range)
    if ssim_value is not None:
        item[f"{key}ssim"] = ssim_value
    edge_ssim_value = edge_ssim_or_none(recon, target)
    if edge_ssim_value is not None:
        item[f"{key}edge_ssim"] = edge_ssim_value
    body_ssim = ssim_on_bbox_or_none(recon, target, data_range, body_bbox)
    if body_ssim is not None:
        item[f"{key}body_bbox_ssim"] = body_ssim

    for window_name, level, width in (
        ("lung", -600.0, 1500.0),
        ("soft_tissue", 40.0, 400.0),
        ("bone", 400.0, 1800.0),
    ):
        recon_window = hu_window(recon, level=level, width=width)
        target_window = hu_window(target, level=level, width=width)
        window_mse = float(torch.mean((recon_window - target_window).square()).item())
        item[f"{key}{window_name}_window_mse"] = window_mse
        item[f"{key}{window_name}_window_psnr"] = psnr_from_mse(window_mse, 1.0)
        window_ssim = ssim_on_bbox_or_none(recon_window, target_window, 1.0, body_bbox)
        if window_ssim is not None:
            item[f"{key}{window_name}_window_body_bbox_ssim"] = window_ssim


def add_ddnm_pseudoinverse_diagnostics(
    item: dict,
    *,
    target: torch.Tensor,
    sinogram: torch.Tensor,
    reconstructor: PaDIS,
    params,
    body_mask: torch.Tensor,
    nonair_mask: torch.Tensor,
    body_bbox,
    data_range: float,
) -> None:
    """Measure the LION pseudoinverse terms used by the DDNM correction."""
    with torch.no_grad():
        measured_pinv = reconstructor.pseudoinverse_reconstruction(
            sinogram,
            params,
        )
        target_sinogram = forward_project_normal_image(target, reconstructor, params)
        projected_target_pinv = reconstructor.pseudoinverse_reconstruction(
            target_sinogram.to(dtype=sinogram.dtype),
            params,
            clip=bool(getattr(params, "ddnm_projected_pseudoinverse_clip", False)),
        )
        perfect_denoiser_corrected = measured_pinv + target - projected_target_pinv
        corrected_clipped = perfect_denoiser_corrected.clamp(0.0, 1.0)

    add_reconstruction_metrics(
        item,
        prefix="ddnm_measured_pseudoinverse",
        recon=measured_pinv,
        target=target,
        sinogram=sinogram,
        reconstructor=reconstructor,
        params=params,
        body_mask=body_mask,
        nonair_mask=nonair_mask,
        body_bbox=body_bbox,
        data_range=data_range,
    )
    add_reconstruction_metrics(
        item,
        prefix="ddnm_projected_target_pseudoinverse",
        recon=projected_target_pinv,
        target=target,
        sinogram=sinogram,
        reconstructor=reconstructor,
        params=params,
        body_mask=body_mask,
        nonair_mask=nonair_mask,
        body_bbox=body_bbox,
        data_range=data_range,
    )
    add_reconstruction_metrics(
        item,
        prefix="ddnm_perfect_denoiser_corrected",
        recon=perfect_denoiser_corrected,
        target=target,
        sinogram=sinogram,
        reconstructor=reconstructor,
        params=params,
        body_mask=body_mask,
        nonair_mask=nonair_mask,
        body_bbox=body_bbox,
        data_range=data_range,
    )
    add_reconstruction_metrics(
        item,
        prefix="ddnm_perfect_denoiser_corrected_clipped",
        recon=corrected_clipped,
        target=target,
        sinogram=sinogram,
        reconstructor=reconstructor,
        params=params,
        body_mask=body_mask,
        nonair_mask=nonair_mask,
        body_bbox=body_bbox,
        data_range=data_range,
    )
    item["ddnm_pseudoinverse_diagnostic"] = {
        "formula": "A^dagger y + x - A^dagger A(x), evaluated with x=target",
        "measured_pseudoinverse_clip": bool(
            getattr(params, "ddnm_pseudoinverse_clip", False)
        ),
        "projected_pseudoinverse_clip": bool(
            getattr(params, "ddnm_projected_pseudoinverse_clip", False)
        ),
        "corrected_min": float(perfect_denoiser_corrected.detach().amin().cpu()),
        "corrected_max": float(perfect_denoiser_corrected.detach().amax().cpu()),
        "corrected_clipped_min": float(corrected_clipped.detach().amin().cpu()),
        "corrected_clipped_max": float(corrected_clipped.detach().amax().cpu()),
    }
