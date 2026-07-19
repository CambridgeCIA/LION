"""Predictor-corrector sampling for diffusion reconstructors."""

from __future__ import annotations

import torch
from tqdm import tqdm


class PredictorCorrector:
    """Reconstruct with predictor and Langevin-corrector updates."""

    @staticmethod
    def pc_corrector_step_size(
        noise: torch.Tensor,
        score: torch.Tensor,
        snr: float,
        rule: str = "paper_linear",
    ) -> torch.Tensor:
        """Compute the selected predictor-corrector Langevin step size."""
        ratio = (
            float(snr)
            * torch.linalg.norm(noise)
            / torch.linalg.norm(score).clamp_min(1e-12)
        )
        if rule == "paper_linear":
            return 2.0 * ratio
        if rule == "score_sde_squared":
            return 2.0 * ratio.square()
        raise ValueError(
            "pc_corrector_step_rule must be 'paper_linear' or 'score_sde_squared'."
        )

    def predictor_corrector(
        self,
        measurement: torch.Tensor,
        params,
        *,
        prog_bar: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Reconstruct a measurement using predictor-corrector sampling."""
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
            total_steps = max(int(params.num_steps) - 1, 0)
            if stop_after is not None:
                total_steps = min(total_steps, stop_after)
            iterator = tqdm(
                list(iterator),
                desc="PaDIS predictor-corrector",
                total=total_steps,
            )

        with torch.no_grad():
            for step_index, (t_cur, t_next) in enumerate(iterator):
                if stop_after is not None and step_index >= stop_after:
                    break
                if step_index == int(params.num_steps) - 1:
                    break
                pc_layout = None
                if (
                    bool(getattr(params, "pc_reuse_predictor_layout", False))
                    and getattr(params, "prior_mode", "patch") == "patch"
                    and getattr(params, "patch_assembly", "padis") == "padis"
                ):
                    pc_layout = self.patch_layout(
                        tuple(x_init.shape[-2:]), params, x.device, generator
                    )
                denoise_kwargs = (
                    {"layout_override": pc_layout} if pc_layout is not None else {}
                )

                denoised = self.denoise_prior(
                    x,
                    t_cur.reshape(1),
                    params,
                    tuple(x_init.shape[-2:]),
                    generator,
                    **denoise_kwargs,
                )
                denoised = self._clip_model_range(denoised, params)
                score = (denoised - x) / t_cur.square()
                predictor_delta = t_cur.square() - t_next.square()
                if not bool(params.disable_prior_score):
                    x = x + predictor_delta * score
                if not bool(params.disable_langevin_noise):
                    z = self._sample_noise(x, generator)
                    x = x + torch.sqrt(predictor_delta.clamp_min(0.0)) * z

                residual = measurement - self.forward_project(
                    self._crop(x, params).squeeze(0).to(torch.float32)
                ).to(dtype=measurement.dtype)
                step_size = self.adjoint_data_step_size(
                    residual, t_cur, params, public_repo_multiplier=False
                )
                (
                    x,
                    correction,
                    raw_correction,
                    data_normalizer,
                    data_scale,
                ) = self.apply_adjoint_correction(x, residual, step_size, params, t_cur)
                self._append_trace(
                    params,
                    algorithm="pc_predictor",
                    step_index=step_index,
                    inner_index=0,
                    sigma=t_cur,
                    x=x,
                    denoised=denoised,
                    projected=x,
                    score=score,
                    residual=residual,
                    gradient=correction,
                    raw_gradient=raw_correction,
                    data_normalizer=data_normalizer,
                    data_scale=data_scale,
                )

                if step_index < int(params.num_steps) - 1:
                    z = self._sample_noise(x, generator)
                    corrector_sigma = (
                        t_cur
                        if getattr(params, "pc_corrector_denoise_sigma", "next")
                        == "current"
                        else t_next
                    )
                    denoised = self.denoise_prior(
                        x,
                        corrector_sigma.reshape(1),
                        params,
                        tuple(x_init.shape[-2:]),
                        generator,
                        **denoise_kwargs,
                    )
                    denoised = self._clip_model_range(denoised, params)
                    score = (denoised - x) / t_next.square().clamp_min(1e-12)
                    eps = self.pc_corrector_step_size(
                        z,
                        score,
                        params.pc_snr,
                        getattr(params, "pc_corrector_step_rule", "paper_linear"),
                    )
                    if not bool(params.disable_prior_score):
                        x = x + eps * score
                    if not bool(params.disable_langevin_noise):
                        x = x + torch.sqrt(2.0 * eps) * z

                    residual = measurement - self.forward_project(
                        self._crop(x, params).squeeze(0).to(torch.float32)
                    ).to(dtype=measurement.dtype)
                    step_size = self.adjoint_data_step_size(
                        residual, t_cur, params, public_repo_multiplier=True
                    )
                    (
                        x,
                        correction,
                        raw_correction,
                        data_normalizer,
                        data_scale,
                    ) = self.apply_adjoint_correction(
                        x, residual, step_size, params, t_cur
                    )
                    self._append_trace(
                        params,
                        algorithm="pc_corrector",
                        step_index=step_index,
                        inner_index=1,
                        sigma=t_cur,
                        x=x,
                        denoised=denoised,
                        projected=x,
                        score=score,
                        residual=residual,
                        gradient=correction,
                        raw_gradient=raw_correction,
                        data_normalizer=data_normalizer,
                        data_scale=data_scale,
                    )
                x = self._clip_state_range(x, params)

        reconstruction = self._crop(x, params).squeeze(0)
        if bool(params.clip_output):
            reconstruction = reconstruction.clamp(0.0, 1.0)
        return reconstruction
