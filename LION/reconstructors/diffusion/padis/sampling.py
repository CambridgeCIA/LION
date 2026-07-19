"""Shared stochastic sampling and trace support for PaDIS."""

from __future__ import annotations

import torch


class PaDISSampling:
    """Provide the shared random-noise and sampler-tracing mechanics."""

    def _append_trace(
        self,
        params,
        *,
        algorithm: str,
        step_index: int,
        inner_index: int,
        sigma: torch.Tensor,
        x: torch.Tensor,
        denoised: torch.Tensor,
        projected: torch.Tensor | None = None,
        score: torch.Tensor,
        residual: torch.Tensor | None = None,
        gradient: torch.Tensor | None = None,
        raw_gradient: torch.Tensor | None = None,
        data_normalizer: float | None = None,
        data_scale: float | None = None,
        measurement_norm: torch.Tensor | float | None = None,
        z: torch.Tensor | None = None,
        x_next: torch.Tensor | None = None,
    ) -> None:
        interval = int(getattr(params, "trace_interval", 0))
        if interval <= 0:
            return
        if inner_index != 0 and inner_index != int(params.inner_steps) - 1:
            return
        if step_index % interval != 0 and step_index != int(params.num_steps) - 1:
            return
        item = {
            "algorithm": algorithm,
            "step": int(step_index),
            "inner": int(inner_index),
            "sigma": float(sigma.detach().cpu()),
            "x_min": float(x.detach().amin().cpu()),
            "x_max": float(x.detach().amax().cpu()),
            "x_mean": float(x.detach().mean().cpu()),
            "x_std": float(x.detach().std().cpu()),
            "x_norm": float(torch.linalg.norm(x.detach()).cpu()),
            "denoised_min": float(denoised.detach().amin().cpu()),
            "denoised_max": float(denoised.detach().amax().cpu()),
            "denoised_mean": float(denoised.detach().mean().cpu()),
            "denoised_std": float(denoised.detach().std().cpu()),
            "denoised_norm": float(torch.linalg.norm(denoised.detach()).cpu()),
            "score_norm": float(torch.linalg.norm(score.detach()).cpu()),
        }
        if projected is not None:
            item["projected_min"] = float(projected.detach().amin().cpu())
            item["projected_max"] = float(projected.detach().amax().cpu())
            item["projected_mean"] = float(projected.detach().mean().cpu())
            item["projected_std"] = float(projected.detach().std().cpu())
            item["projected_norm"] = float(torch.linalg.norm(projected.detach()).cpu())
        if residual is not None:
            residual_norm = torch.linalg.norm(residual.detach())
            item["residual_norm"] = float(residual_norm.cpu())
            item["residual_min"] = float(residual.detach().amin().cpu())
            item["residual_max"] = float(residual.detach().amax().cpu())
            item["residual_mean"] = float(residual.detach().mean().cpu())
            if measurement_norm is not None:
                measurement_norm_tensor = torch.as_tensor(
                    measurement_norm, device=residual_norm.device
                ).clamp_min(1e-12)
                item["measurement_norm"] = float(measurement_norm_tensor.detach().cpu())
                item["relative_residual_norm"] = float(
                    (residual_norm / measurement_norm_tensor).detach().cpu()
                )
        if gradient is not None:
            item["gradient_norm"] = float(torch.linalg.norm(gradient.detach()).cpu())
        if raw_gradient is not None:
            item["raw_gradient_norm"] = float(
                torch.linalg.norm(raw_gradient.detach()).cpu()
            )
        if data_normalizer is not None:
            item["data_consistency_normalizer"] = float(data_normalizer)
        if data_scale is not None:
            item["data_consistency_scale"] = float(data_scale)
        if z is not None:
            item["z_norm"] = float(torch.linalg.norm(z.detach()).cpu())
        if x_next is not None:
            item["x_next_min"] = float(x_next.detach().amin().cpu())
            item["x_next_max"] = float(x_next.detach().amax().cpu())
            item["x_next_mean"] = float(x_next.detach().mean().cpu())
            item["x_next_std"] = float(x_next.detach().std().cpu())
            item["x_next_norm"] = float(torch.linalg.norm(x_next.detach()).cpu())
        if bool(getattr(params, "trace_images", False)):
            image_index = len(getattr(self, "last_trace_images", []))
            item["trace_image_index"] = int(image_index)
            if projected is None:
                projected = x
            x_crop = self._crop(x.detach(), params).squeeze(0)
            denoised_crop = self._crop(denoised.detach(), params).squeeze(0)
            projected_crop = self._crop(projected.detach(), params).squeeze(0)
            if x_next is None:
                x_next = projected
            x_next_crop = self._crop(x_next.detach(), params).squeeze(0)
            with torch.no_grad():
                forward_projected = self.forward_project(
                    projected_crop.to(dtype=torch.float32)
                )
            self.last_trace_images.append(
                {
                    "algorithm": algorithm,
                    "step": int(step_index),
                    "inner": int(inner_index),
                    "sigma": float(sigma.detach().cpu()),
                    "x": x_crop.detach().cpu(),
                    "denoised": denoised_crop.detach().cpu(),
                    "projected": projected_crop.detach().cpu(),
                    "x_next": x_next_crop.detach().cpu(),
                    "forward_projected": forward_projected.detach().cpu(),
                }
            )
        self.last_trace.append(item)

    def _sample_noise(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if generator is None:
            return torch.randn_like(x)
        return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
