"""Annealed Langevin sampling for diffusion reconstructors."""

from __future__ import annotations

import torch
from tqdm import tqdm


class AnnealedLangevin:
    """Reconstruct with the annealed Langevin sampler, optionally using DDNM."""

    def langevin(
        self,
        measurement: torch.Tensor,
        params,
        *,
        prog_bar: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Reconstruct a measurement using annealed Langevin dynamics."""
        x_init = self.initial_reconstruction(measurement, params).unsqueeze(0)
        if (
            params.initial_reconstruction == "noise"
            and getattr(params, "noise_initialization", "padded") == "central_then_pad"
        ):
            x = self._pad(
                float(params.sigma_max) * self._sample_noise(x_init, generator),
                params,
            )
        else:
            x = float(params.sigma_max) * self._sample_noise(
                self._pad(x_init, params), generator
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
            iterator = tqdm(list(iterator), desc="PaDIS Langevin", total=total_steps)

        with torch.no_grad():
            for step_index, (t_cur, _t_next) in enumerate(iterator):
                if stop_after is not None and step_index >= stop_after:
                    break
                alpha = float(getattr(params, "sampling_epsilon", 1.0)) * t_cur.square()
                for inner_index in range(int(params.inner_steps)):
                    denoised = self.denoise_prior(
                        x,
                        t_cur.reshape(1),
                        params,
                        tuple(x_init.shape[-2:]),
                        generator,
                    )
                    denoised = self._clip_model_range(denoised, params)
                    if bool(params.langevin_ddnm):
                        denoised_crop = self._crop(denoised, params).squeeze(0)
                        backprojected = self.pseudoinverse_reconstruction(
                            measurement, params
                        )
                        projected_denoised = self.forward_project(denoised_crop)
                        corrected = (
                            backprojected
                            + denoised_crop
                            - self.pseudoinverse_reconstruction(
                                projected_denoised,
                                params,
                                clip=bool(
                                    getattr(
                                        params,
                                        "ddnm_projected_pseudoinverse_clip",
                                        False,
                                    )
                                ),
                            )
                        )
                        if bool(getattr(params, "ddnm_corrected_clip", False)):
                            corrected = corrected.clamp(0.0, 1.0)
                        x0hat = self._pad(corrected.unsqueeze(0), params)
                        score = (x0hat - x) / t_cur.square()
                        self._append_trace(
                            params,
                            algorithm="langevin_ddnm",
                            step_index=step_index,
                            inner_index=inner_index,
                            sigma=t_cur,
                            x=x,
                            denoised=denoised,
                            projected=x0hat,
                            score=score,
                        )
                    else:
                        score = (denoised - x) / t_cur.square()
                        residual = measurement - self.forward_project(
                            self._crop(x, params).squeeze(0).to(torch.float32)
                        ).to(dtype=measurement.dtype)
                        step_size = self.adjoint_data_step_size(
                            residual, t_cur, params, public_repo_multiplier=True
                        )
                        (
                            projected,
                            correction,
                            raw_correction,
                            data_normalizer,
                            data_scale,
                        ) = self.apply_adjoint_correction(
                            x, residual, step_size, params, t_cur
                        )
                        self._append_trace(
                            params,
                            algorithm="langevin",
                            step_index=step_index,
                            inner_index=inner_index,
                            sigma=t_cur,
                            x=x,
                            denoised=denoised,
                            projected=projected,
                            score=score,
                            residual=residual,
                            gradient=correction,
                            raw_gradient=raw_correction,
                            data_normalizer=data_normalizer,
                            data_scale=data_scale,
                        )
                        x = projected

                    z = self._sample_noise(x, generator)
                    score_step = (
                        0 if bool(params.disable_prior_score) else alpha / 2 * score
                    )
                    if step_index < int(params.num_steps) - 1:
                        noise_step = (
                            0
                            if bool(params.disable_langevin_noise)
                            else float(params.langevin_noise_scale)
                            * torch.sqrt(alpha)
                            * z
                        )
                        x = x + score_step + noise_step
                    else:
                        x = x + score_step
                    x = self._clip_state_range(x, params)

        reconstruction = self._crop(x, params).squeeze(0)
        if bool(params.clip_output):
            reconstruction = reconstruction.clamp(0.0, 1.0)
        return reconstruction
