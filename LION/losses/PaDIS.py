"""PaDIS patch-based denoising loss and training utilities."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def validate_patch_schedule(
    patch_sizes: Sequence[int], patch_probabilities: Sequence[float]
) -> None:
    if len(patch_sizes) != len(patch_probabilities):
        raise ValueError("patch_sizes and patch_probabilities must have same length.")
    if not patch_sizes:
        raise ValueError("At least one patch size is required.")
    if any(size % 8 != 0 for size in patch_sizes):
        raise ValueError("All PaDIS patch sizes must be divisible by 8.")
    prob_sum = float(sum(patch_probabilities))
    if abs(prob_sum - 1.0) > 1e-6:
        raise ValueError("patch_probabilities must sum to 1.")


def build_position_grid(
    batch_size: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((xx, yy), dim=0).unsqueeze(0)
    return grid.expand(batch_size, -1, -1, -1)


def zero_pad_images(images: torch.Tensor, pad_width: int) -> torch.Tensor:
    if pad_width == 0:
        return images
    return F.pad(images, (pad_width, pad_width, pad_width, pad_width), mode="constant")


def sample_patch_size(
    patch_sizes: Sequence[int],
    patch_probabilities: Sequence[float],
    *,
    device: torch.device,
) -> int:
    probs = torch.as_tensor(patch_probabilities, device=device, dtype=torch.float32)
    index = torch.multinomial(probs, num_samples=1).item()
    return int(patch_sizes[index])


def sample_patch_pair(
    images: torch.Tensor,
    positions: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, _, height, width = images.shape
    if patch_size > height or patch_size > width:
        raise ValueError("patch_size cannot exceed padded image dimensions.")
    top = torch.randint(0, height - patch_size + 1, (batch_size,), device=images.device)
    left = torch.randint(0, width - patch_size + 1, (batch_size,), device=images.device)
    rows = top[:, None] + torch.arange(patch_size, device=images.device)[None, :]
    cols = left[:, None] + torch.arange(patch_size, device=images.device)[None, :]
    batch = torch.arange(batch_size, device=images.device)[:, None, None]
    image_patch = images.permute(1, 0, 2, 3)[
        :, batch, rows[:, :, None], cols[:, None, :]
    ].permute(1, 0, 2, 3)
    position_patch = positions.permute(1, 0, 2, 3)[
        :, batch, rows[:, :, None], cols[:, None, :]
    ].permute(1, 0, 2, 3)
    return image_patch, position_patch


def sample_image_patch_with_position_channels(
    images: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, _, height, width = images.shape
    if patch_size > height or patch_size > width:
        raise ValueError("patch_size cannot exceed padded image dimensions.")
    top = torch.randint(0, height - patch_size + 1, (batch_size,), device=images.device)
    left = torch.randint(0, width - patch_size + 1, (batch_size,), device=images.device)
    rows = top[:, None] + torch.arange(patch_size, device=images.device)[None, :]
    cols = left[:, None] + torch.arange(patch_size, device=images.device)[None, :]
    batch = torch.arange(batch_size, device=images.device)[:, None, None]
    image_patch = images.permute(1, 0, 2, 3)[
        :, batch, rows[:, :, None], cols[:, None, :]
    ].permute(1, 0, 2, 3)

    x_pos = (cols.to(images.dtype) / (width - 1) - 0.5) * 2.0
    y_pos = (rows.to(images.dtype) / (height - 1) - 0.5) * 2.0
    x_pos = x_pos[:, None, None, :].expand(-1, 1, patch_size, -1)
    y_pos = y_pos[:, None, :, None].expand(-1, 1, -1, patch_size)
    position_patch = torch.cat((x_pos, y_pos), dim=1)
    return image_patch, position_patch


def score_from_denoiser(
    noisy_image_patch: torch.Tensor, denoised_patch: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    while sigma.ndim < noisy_image_patch.ndim:
        sigma = sigma.unsqueeze(-1)
    return (denoised_patch - noisy_image_patch) / sigma.square()


class PaDISDenoisingLoss(nn.Module):
    """PaDIS repository Patch_EDMLoss mechanics with paper-level parameters."""

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 40.0,
        sigma_distribution: str = "edm_lognormal",
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        reduction: str = "batch_mean_sum",
        augment_pipe: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if sigma_min <= 0 or sigma_max <= sigma_min:
            raise ValueError("Require 0 < sigma_min < sigma_max.")
        if sigma_distribution not in ("edm_lognormal", "log_uniform"):
            raise ValueError("sigma_distribution must be edm_lognormal or log_uniform.")
        if reduction not in ("batch_mean_sum", "mean"):
            raise ValueError("reduction must be batch_mean_sum or mean.")
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_distribution = sigma_distribution
        self.P_mean = float(P_mean)
        self.P_std = float(P_std)
        self.sigma_data = float(sigma_data)
        self.reduction = reduction
        self.augment_pipe = augment_pipe

    def sample_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.sigma_distribution == "edm_lognormal":
            rnd_normal = torch.randn(batch_size, device=device)
            return torch.exp(rnd_normal * self.P_std + self.P_mean)
        log_min = torch.log(torch.tensor(self.sigma_min, device=device))
        log_max = torch.log(torch.tensor(self.sigma_max, device=device))
        log_sigma = log_min + torch.rand(batch_size, device=device) * (
            log_max - log_min
        )
        return torch.exp(log_sigma)

    def forward(
        self,
        model: nn.Module,
        clean_patch: torch.Tensor,
        position_patch: torch.Tensor | None = None,
        augment_pipe: nn.Module | None = None,
    ) -> torch.Tensor:
        pipe = self.augment_pipe if augment_pipe is None else augment_pipe
        target_patch, augment_labels = (
            pipe(clean_patch) if pipe is not None else (clean_patch, None)
        )
        sigma = self.sample_sigma(target_patch.shape[0], target_patch.device)
        sigma_view = sigma.reshape(
            target_patch.shape[0], *([1] * (target_patch.ndim - 1))
        )
        noisy_patch = target_patch + sigma_view * torch.randn_like(target_patch)
        sigma_data = torch.as_tensor(
            self.sigma_data, device=target_patch.device, dtype=target_patch.dtype
        )
        c_skip = sigma_data.square() / (sigma_view.square() + sigma_data.square())
        c_out = (
            sigma_view * sigma_data / (sigma_view.square() + sigma_data.square()).sqrt()
        )
        c_in = 1 / (sigma_data.square() + sigma_view.square()).sqrt()
        c_noise = sigma.log() / 4
        if position_patch is not None:
            model_input = torch.cat((c_in * noisy_patch, position_patch), dim=1)
        else:
            model_input = c_in * noisy_patch
        if augment_labels is None:
            model_output = model(model_input, c_noise)
        else:
            model_output = model(model_input, c_noise, augment_labels=augment_labels)
        denoised = c_skip * noisy_patch + c_out * model_output
        weight = (sigma_view.square() + sigma_data.square()) / (
            sigma_view * sigma_data
        ).square()
        loss = weight * (denoised - target_patch).square()
        if self.reduction == "batch_mean_sum":
            return loss.flatten(1).sum(dim=1).mean()
        return torch.mean(loss)
