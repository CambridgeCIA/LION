"""Measurement-domain transformations for the PaDIS reconstructor."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from LION.utils.math import power_method


class PaDISPhysics:
    """Implement measurement transforms and data-consistency gradients."""

    def _crop(self, x: torch.Tensor, params) -> torch.Tensor:
        pad = int(params.pad_width)
        if pad == 0:
            return x
        return x[:, :, pad:-pad, pad:-pad]

    def _pad(self, x: torch.Tensor, params) -> torch.Tensor:
        pad = int(params.pad_width)
        if pad == 0:
            return x
        return F.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)

    def _clip_model_range(self, x: torch.Tensor, params) -> torch.Tensor:
        if bool(getattr(params, "clip_denoised", False)):
            x = x.clamp(0.0, 1.0)
            pad = int(params.pad_width)
            if pad > 0:
                x = x.clone()
                x[:, :, :pad, :] = 0.0
                x[:, :, -pad:, :] = 0.0
                x[:, :, :, :pad] = 0.0
                x[:, :, :, -pad:] = 0.0
        return x

    def _clip_state_range(self, x: torch.Tensor, params) -> torch.Tensor:
        if bool(getattr(params, "clip_state", False)):
            x = x.clamp(0.0, 1.0)
            pad = int(params.pad_width)
            if pad > 0:
                x = x.clone()
                x[:, :, :pad, :] = 0.0
                x[:, :, -pad:, :] = 0.0
                x[:, :, :, :pad] = 0.0
                x[:, :, :, -pad:] = 0.0
        return x

    def _to_measurement_image(self, x: torch.Tensor, params) -> torch.Tensor:
        return float(params.measurement_scale) * x + float(params.measurement_offset)

    def _from_measurement_image(self, x: torch.Tensor, params) -> torch.Tensor:
        scale = float(params.measurement_scale)
        if scale == 0:
            raise ValueError("measurement_scale must be non-zero.")
        return (x - float(params.measurement_offset)) / scale

    def forward_project(self, x: torch.Tensor) -> torch.Tensor:
        """Project a normalized model-domain image into measurement space."""
        params = getattr(self, "_active_params", None)
        if params is not None:
            x = self._to_measurement_image(x, params)
        if self.geometry is not None:
            return self.op_autograd(x)
        return self.op(x)

    def adjoint_project(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the measurement adjoint in normalized model-domain units."""
        params = getattr(self, "_active_params", None)
        scale = 1.0 if params is None else float(params.measurement_scale)
        return scale * self.op.adjoint(y)

    def operator_norm(self, params, device: torch.device) -> float:
        """Return or estimate the measurement operator norm."""
        provided_norm = getattr(params, "operator_norm", None)
        if provided_norm is not None:
            norm = float(provided_norm)
            if norm <= 0:
                raise ValueError("operator_norm must be positive when provided.")
            return norm

        cache_key = (device.type, device.index)
        cache = getattr(self, "_operator_norm_cache", {})
        if cache_key not in cache:
            with torch.no_grad():
                estimate = power_method(
                    self.op,
                    maxiter=int(params.operator_norm_iterations),
                    tol=float(params.operator_norm_tolerance),
                    device=device,
                )
            cache[cache_key] = float(estimate.detach().cpu())
            self._operator_norm_cache = cache
        return cache[cache_key]

    def data_consistency_normalizer(self, params, device: torch.device) -> float:
        """Return the selected measurement-gradient normalization factor."""
        method = getattr(params, "data_consistency_normalization", "none")
        if method in (None, "none", False):
            return 1.0
        if method not in ("operator_norm", "operator_lipschitz"):
            raise ValueError(
                "data_consistency_normalization must be 'operator_norm', "
                "'operator_lipschitz', or 'none'."
            )

        # The sampler state is in the diffusion model's normalized image units,
        # while the forward model may first map it to attenuation units. The
        # Lipschitz scale of that composed measurement map is |scale| * ||A||.
        normalizer = abs(float(params.measurement_scale)) * self.operator_norm(
            params, device
        )
        if method == "operator_lipschitz":
            normalizer = normalizer**2
        return max(normalizer, 1e-12)

    def normalise_data_gradient(
        self,
        gradient: torch.Tensor,
        params,
        sigma: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float, float]:
        """Normalize and schedule a raw measurement gradient."""
        normalizer = self.data_consistency_normalizer(params, gradient.device)
        scaled = gradient / normalizer
        scale = self.scheduled_data_consistency_scale(params, sigma, gradient.device)
        scaled = scale * scaled
        return scaled, normalizer, scale

    def scheduled_data_consistency_scale(
        self,
        params,
        sigma: torch.Tensor | None,
        device: torch.device,
        *,
        base_override: float | None = None,
    ) -> float:
        """Evaluate the configured DPS data-consistency scale schedule."""
        base = (
            float(base_override)
            if base_override is not None
            else float(getattr(params, "data_consistency_scale", 1.0))
        )
        schedule = getattr(params, "data_consistency_scale_schedule", "constant")
        if sigma is None or schedule in (None, "constant"):
            return base

        sigma_value = max(float(sigma.detach().cpu()), 1e-12)
        power = float(getattr(params, "data_consistency_scale_power", 1.0))
        floor = float(getattr(params, "data_consistency_scale_floor", 0.0))
        if schedule == "edm":
            sigma_data = float(getattr(params, "sigma_data", 0.5))
            factor = sigma_data**2 / (sigma_value**2 + sigma_data**2)
        elif schedule == "inverse_sigma":
            sigma_min = max(float(getattr(params, "sigma_min", 1e-12)), 1e-12)
            factor = min(1.0, sigma_min / sigma_value)
        else:
            raise ValueError(
                "data_consistency_scale_schedule must be 'constant', 'edm', or 'inverse_sigma'."
            )
        factor = max(float(factor) ** power, floor)
        return base * factor

    def scheduled_adjoint_data_consistency_scale(
        self,
        params,
        sigma: torch.Tensor | None,
        device: torch.device,
    ) -> float:
        """Evaluate the adjoint-specific data-consistency scale schedule."""
        adjoint_scale = getattr(params, "adjoint_data_consistency_scale", None)
        if adjoint_scale is None:
            return self.scheduled_data_consistency_scale(params, sigma, device)
        return self.scheduled_data_consistency_scale(
            params, sigma, device, base_override=float(adjoint_scale)
        )

    def measurement_gradient(
        self,
        measurement: torch.Tensor,
        x: torch.Tensor,
        x0hat: torch.Tensor,
        params,
    ) -> torch.Tensor:
        """Return only the normalized DPS measurement gradient."""
        grad, *_ = self.dps_data_gradient(measurement, x, x0hat, params, sigma=None)
        return grad

    def dps_data_gradient(
        self,
        measurement: torch.Tensor,
        x: torch.Tensor,
        x0hat: torch.Tensor,
        params,
        sigma: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
        """Compute the DPS residual, raw gradient, and scheduled update terms."""
        predicted = self.forward_project(self._crop(x0hat, params).squeeze(0))
        residual = measurement - predicted.to(dtype=measurement.dtype)
        residual_norm = torch.linalg.norm(residual).clamp_min(1e-12)
        gradient_mode = getattr(params, "data_consistency_gradient", "norm")
        if gradient_mode == "least_squares":
            objective = 0.5 * residual.square().sum()
            step_size = float(params.zeta)
        elif gradient_mode == "paper_squared_residual":
            objective = residual.square().sum()
            step_size = float(params.zeta) / float(residual_norm.detach().cpu())
        elif gradient_mode == "norm":
            objective = residual_norm
            step_size = float(params.zeta)
        else:
            raise ValueError(
                "data_consistency_gradient must be 'norm', 'least_squares', "
                "or 'paper_squared_residual'."
            )
        raw_gradient = torch.autograd.grad(outputs=objective, inputs=x)[0]
        gradient, data_normalizer, data_scale = self.normalise_data_gradient(
            raw_gradient, params, sigma
        )
        return (
            gradient,
            raw_gradient,
            residual,
            data_normalizer,
            data_scale,
            step_size,
        )
