"""Shared adjoint data-consistency updates for diffusion reconstructors."""

from __future__ import annotations

import torch


class AdjointDataConsistency:
    """Apply adjoint measurement updates shared by diffusion samplers."""

    def apply_adjoint_correction(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        step_size: torch.Tensor,
        params,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        """Apply one scheduled adjoint correction to a padded sampler state."""
        raw_correction = self.adjoint_project(residual).unsqueeze(0)
        data_normalizer = self.data_consistency_normalizer(
            params, raw_correction.device
        )
        data_scale = self.scheduled_adjoint_data_consistency_scale(
            params, sigma, raw_correction.device
        )
        correction = data_scale * raw_correction / data_normalizer
        if bool(params.disable_data_consistency):
            return x, correction, raw_correction, data_normalizer, data_scale

        pad = int(params.pad_width)
        if pad == 0:
            x = x + step_size * correction
        else:
            x = x.clone()
            x[:, :, pad:-pad, pad:-pad] += step_size * correction
        return x, correction, raw_correction, data_normalizer, data_scale

    def adjoint_data_step_size(
        self,
        residual: torch.Tensor,
        sigma: torch.Tensor,
        params,
        *,
        public_repo_multiplier: bool,
    ) -> torch.Tensor:
        """Compute the paper or public-repository adjoint step size."""
        if getattr(params, "data_consistency_gradient", "norm") == "least_squares":
            return torch.as_tensor(
                float(params.zeta), device=residual.device, dtype=residual.dtype
            )
        residual_norm = torch.linalg.norm(residual).clamp_min(1e-12)
        step_size = float(params.zeta) / residual_norm
        schedule = getattr(params, "adjoint_data_step_schedule", "public_repo")
        if schedule == "paper":
            return step_size
        if schedule == "public_repo":
            if public_repo_multiplier:
                step_size = step_size * min(40.0, float(sigma.item()) * 200.0)
            return step_size
        raise ValueError("adjoint_data_step_schedule must be 'paper' or 'public_repo'.")
