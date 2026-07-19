"""Diffusion posterior sampling with Langevin dynamics."""

from __future__ import annotations

import torch
from tqdm import tqdm


class DPSLangevin:
    """Reconstruct with diffusion posterior sampling and Langevin updates."""

    def dps_langevin(
        self,
        measurement: torch.Tensor,
        params,
        *,
        prog_bar: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Reconstruct a measurement using the DPS-Langevin sampler."""
        x_init = self.initial_reconstruction(measurement, params).unsqueeze(0)
        x = (
            self.initial_padded_state(measurement, params, generator)
            .detach()
            .requires_grad_(True)
        )
        t_steps = self.noise_schedule(params, x.device)
        iterator = zip(t_steps[:-1], t_steps[1:])
        stop_after = getattr(params, "stop_after_outer_steps", None)
        if stop_after is not None:
            stop_after = int(stop_after)
            if stop_after <= 0:
                raise ValueError("stop_after_outer_steps must be positive or None.")
        if prog_bar:
            total_steps = int(params.num_steps)
            if stop_after is not None:
                total_steps = min(total_steps, stop_after)
            iterator = tqdm(list(iterator), desc="PaDIS DPS", total=total_steps)

        measurement_norm = torch.linalg.norm(measurement.detach()).clamp_min(1e-12)
        for step_index, (t_cur, _t_next) in enumerate(iterator):
            if stop_after is not None and step_index >= stop_after:
                break
            alpha = float(getattr(params, "dps_epsilon", 1.0)) * t_cur.square()
            for inner_index in range(int(params.inner_steps)):
                x_current = x
                if bool(params.disable_data_consistency):
                    with torch.no_grad():
                        denoised = self.denoise_prior(
                            x_current,
                            t_cur.reshape(1),
                            params,
                            tuple(x_init.shape[-2:]),
                            generator,
                        )
                        denoised = self._clip_model_range(denoised, params)
                else:
                    denoised = self.denoise_prior(
                        x_current,
                        t_cur.reshape(1),
                        params,
                        tuple(x_init.shape[-2:]),
                        generator,
                    )
                    denoised = self._clip_model_range(denoised, params)
                score = (denoised - x_current) / t_cur.square()
                if bool(params.disable_data_consistency):
                    with torch.no_grad():
                        predicted = self.forward_project(
                            self._crop(denoised, params).squeeze(0)
                        )
                        residual = measurement - predicted.to(dtype=measurement.dtype)
                    data_gradient = torch.zeros_like(x_current)
                    raw_data_gradient = data_gradient
                    data_normalizer = 1.0
                    data_scale = 0.0
                    data_step_size = 0.0
                    projected = x_current
                else:
                    (
                        data_gradient,
                        raw_data_gradient,
                        residual,
                        data_normalizer,
                        data_scale,
                        data_step_size,
                    ) = self.dps_data_gradient(
                        measurement, x_current, denoised, params, sigma=t_cur
                    )
                    projected = x_current - data_step_size * data_gradient
                z = self._sample_noise(x_current, generator)
                score_step = (
                    0 if bool(params.disable_prior_score) else alpha / 2 * score
                )
                if step_index < int(params.num_steps) - 1:
                    noise_step = (
                        0
                        if bool(params.disable_langevin_noise)
                        else float(params.langevin_noise_scale) * torch.sqrt(alpha) * z
                    )
                    x_next = projected + score_step + noise_step
                else:
                    x_next = projected + score_step
                self._append_trace(
                    params,
                    algorithm="dps_langevin",
                    step_index=step_index,
                    inner_index=inner_index,
                    sigma=t_cur,
                    x=x_current,
                    denoised=denoised,
                    projected=projected,
                    score=score,
                    residual=residual,
                    gradient=data_gradient,
                    raw_gradient=raw_data_gradient,
                    data_normalizer=data_normalizer,
                    data_scale=data_scale,
                    measurement_norm=measurement_norm,
                    z=z,
                    x_next=x_next,
                )
                x = self._clip_state_range(x_next, params).detach().requires_grad_(True)

        reconstruction = self._crop(x.detach(), params).squeeze(0)
        if bool(params.clip_output):
            reconstruction = reconstruction.clamp(0.0, 1.0)
        return reconstruction
