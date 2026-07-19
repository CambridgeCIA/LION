"""Unconditional image-generation routines for PaDIS."""

from __future__ import annotations

import torch


class PaDISGeneration:
    """Generate images with either the assembled prior or independent patches."""

    def generate_sample(
        self,
        params,
        *,
        channels: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Generate one image using the assembled PaDIS score."""
        central = torch.zeros((1, channels, height, width), device=device, dtype=dtype)
        x = float(params.sigma_max) * self._sample_noise(
            self._pad(central, params), generator
        )
        t_steps = self.noise_schedule(params, x.device)
        epsilon = float(getattr(params, "generation_epsilon", 1.0))

        with torch.no_grad():
            for step_index, t_cur in enumerate(t_steps[:-1]):
                alpha = epsilon * t_cur.square()
                for _ in range(int(params.inner_steps)):
                    denoised = self.denoise_prior(
                        x, t_cur.reshape(1), params, (height, width), generator
                    )
                    denoised = self._clip_model_range(denoised, params)
                    score = (denoised - x) / t_cur.square()
                    if not bool(params.disable_prior_score):
                        x = x + alpha / 2 * score
                    if step_index < int(params.num_steps) - 1 and not bool(
                        params.disable_langevin_noise
                    ):
                        z = self._sample_noise(x, generator)
                        x = (
                            x
                            + float(params.langevin_noise_scale) * torch.sqrt(alpha) * z
                        )
                    x = self._clip_state_range(x, params)

        sample = self._crop(x, params).squeeze(0)
        if bool(params.clip_output):
            sample = sample.clamp(0.0, 1.0)
        return sample

    def generate_naive_patch_sample(
        self,
        params,
        *,
        channels: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Generate and stitch one partition of independently sampled patches."""
        self._validate_prior_configuration(params, (height, width))
        layout = self.patch_layout(
            (height, width),
            params,
            device,
            generator,
            fixed_offset=bool(getattr(params, "naive_patch_fixed_layout", True)),
        )
        patch_size = int(params.patch_size)
        pad = int(params.pad_width)
        padded = torch.zeros(
            (1, channels, height + 2 * pad, width + 2 * pad),
            device=device,
            dtype=dtype,
        )
        positions = (
            self._position_grid(padded)
            if int(getattr(self.model.model_parameters, "input_position_channels", 2))
            > 0
            else None
        )
        position_batch = None
        if positions is not None:
            position_batch = torch.cat(
                [
                    positions[:, :, top:bottom, left:right]
                    for top, bottom, left, right in layout.indices
                ],
                dim=0,
            )
        x = float(params.sigma_max) * self._sample_noise(
            torch.zeros(
                (len(layout.indices), channels, patch_size, patch_size),
                device=device,
                dtype=dtype,
            ),
            generator,
        )
        t_steps = self.noise_schedule(params, x.device)
        epsilon = float(getattr(params, "generation_epsilon", 1.0))
        final_denoised = None

        with torch.no_grad():
            for step_index, t_cur in enumerate(t_steps[:-1]):
                alpha = epsilon * t_cur.square()
                for _ in range(int(params.inner_steps)):
                    denoised = self.edm_denoise_batch(
                        x, position_batch, t_cur.reshape(1), params
                    )
                    final_denoised = denoised
                    score = (denoised - x) / t_cur.square()
                    if not bool(params.disable_prior_score):
                        x = x + alpha / 2 * score
                    if step_index < int(params.num_steps) - 1 and not bool(
                        params.disable_langevin_noise
                    ):
                        z = self._sample_noise(x, generator)
                        x = (
                            x
                            + float(params.langevin_noise_scale) * torch.sqrt(alpha) * z
                        )

        patches = x
        if getattr(params, "naive_patch_output", "sampler_state") == "denoised":
            if final_denoised is None:
                raise RuntimeError("No denoised patches were produced.")
            patches = final_denoised

        output = torch.zeros_like(padded)
        for index, (top, bottom, left, right) in enumerate(layout.indices):
            output[:, :, top:bottom, left:right] = patches[index : index + 1]
        sample = self._crop(output, params).squeeze(0)
        if bool(params.clip_output):
            sample = sample.clamp(0.0, 1.0)
        return sample
