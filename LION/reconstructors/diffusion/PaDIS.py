"""PaDIS diffusion-prior reconstructors."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from tqdm import tqdm

from LION.classical_algorithms.fdk import fdk
from LION.CTtools.ct_geometry import Geometry
from LION.models.diffusion import NCSNpp
from LION.operators import Operator
from LION.reconstructors.diffusion.data_consistency import AdjointDataConsistency
from LION.reconstructors.diffusion.dps_langevin import DPSLangevin
from LION.reconstructors.diffusion.langevin import AnnealedLangevin
from LION.reconstructors.diffusion.padis import (  # noqa: F401
    PaDISCitations,
    PaDISGeneration,
    PaDISParameters,
    PaDISPhysics,
    PaDISPrior,
    PaDISSampling,
    PatchLayout,
    PUBLIC_REPO_CT_ADJOINT_SCALE,
    PUBLIC_REPO_CT_GRADIENT_SCALE,
)
from LION.reconstructors.diffusion.predictor_corrector import PredictorCorrector
from LION.reconstructors.LIONreconstructor import LIONReconstructor
from LION.utils.parameter import LIONParameter


class PaDIS(
    PaDISCitations,
    PaDISParameters,
    PaDISPrior,
    PaDISPhysics,
    PaDISSampling,
    PaDISGeneration,
    AdjointDataConsistency,
    DPSLangevin,
    PredictorCorrector,
    AnnealedLangevin,
    LIONReconstructor,
):
    """Patch-aware diffusion inverse solver for CT reconstruction.

    The sampler follows the PaDIS reference implementation: a whole-image score
    is assembled from position-aware denoised patches, then combined with either
    DPS measurement conditioning or Langevin dynamics with adjoint data steps.
    """

    def __init__(
        self,
        physics: Geometry | Operator | None,
        model: NCSNpp,
        parameters: LIONParameter | None = None,
        algorithm: Literal["dps_langevin", "dps", "langevin", "pc"] = "dps_langevin",
    ) -> None:
        if physics is None:
            self.geometry = None
            self.op = None
            self.op_autograd = None
        elif isinstance(physics, Operator):
            self.geometry = None
            self.op = physics
            self.op_autograd = physics
        else:
            super().__init__(physics)
        if algorithm not in ("dps_langevin", "dps", "langevin", "pc"):
            raise ValueError("algorithm must be 'dps_langevin', 'langevin', or 'pc'.")
        self.model = model
        self.parameters = parameters or self.default_parameters(model)
        self.algorithm = self._canonical_algorithm(algorithm)

    def reconstruct_sample(
        self,
        sino: torch.Tensor,
        *,
        algorithm: Literal["dps_langevin", "dps", "langevin", "pc"] | None = None,
        prog_bar: bool = False,
        generator: torch.Generator | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Reconstruct one image from a channel-first sinogram.

        Parameters
        ----------
        sino : torch.Tensor
            Measurement tensor with shape ``(channels, angles, detector)``.
        algorithm : {"dps_langevin", "dps", "langevin", "pc"}, optional
            Per-call algorithm override.
        prog_bar : bool, optional
            Display outer sampler progress.
        generator : torch.Generator, optional
            Random generator controlling initialisation, patch offsets, and
            stochastic sampler updates.
        **kwargs
            Temporary overrides of sampler parameters.

        Returns
        -------
        torch.Tensor
            Reconstructed image in normalised-intensity units.
        """
        if not isinstance(sino, torch.Tensor):
            raise TypeError("Sinogram must be a torch.Tensor.")
        if sino.dim() != 3:
            raise ValueError(
                f"Sinogram must be a 3D tensor (channels, angles, detector), got {sino.dim()}D."
            )

        params = self._merged_parameters(kwargs)
        algorithm = self._canonical_algorithm(algorithm or self.algorithm)
        self.model.eval()
        self.last_trace = []
        self.last_trace_images = []
        if bool(getattr(params, "consume_unused_latents", False)):
            latent_shape = (1, *tuple(int(value) for value in self.op.domain_shape))
            _ = self._sample_noise(sino.new_empty(latent_shape), generator)
        previous_params = getattr(self, "_active_params", None)
        self._active_params = params
        try:
            if algorithm == "dps_langevin":
                return self.dps_langevin(
                    sino, params, prog_bar=prog_bar, generator=generator
                )
            if algorithm == "langevin":
                return self.langevin(
                    sino, params, prog_bar=prog_bar, generator=generator
                )
            if algorithm == "pc":
                return self.predictor_corrector(
                    sino, params, prog_bar=prog_bar, generator=generator
                )
            raise ValueError(f"Unknown PaDIS algorithm: {algorithm}")
        finally:
            self._active_params = previous_params

    def generate_samples(
        self,
        *,
        num_samples: int = 1,
        image_shape: tuple[int, int, int] | None = None,
        prog_bar: bool = False,
        generator: torch.Generator | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sample images from the PaDIS prior without measurement conditioning."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        params = self._merged_parameters(kwargs)
        if image_shape is None:
            geometry = getattr(self.model, "geometry", None)
            if geometry is None:
                raise ValueError(
                    "image_shape is required when the model has no geometry."
                )
            image_shape = tuple(int(value) for value in geometry.image_shape)
        if len(image_shape) != 3:
            raise ValueError("image_shape must be (channels, height, width).")

        channels, height, width = (int(value) for value in image_shape)
        if channels <= 0 or height <= 0 or width <= 0:
            raise ValueError("image_shape values must be positive.")

        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        samples = []
        iterator = range(num_samples)
        if prog_bar:
            iterator = tqdm(iterator, desc="PaDIS generation", total=num_samples)

        previous_params = (
            self._active_params if hasattr(self, "_active_params") else None
        )
        self._active_params = params
        try:
            for _ in iterator:
                sample = self.generate_sample(
                    params,
                    channels=channels,
                    height=height,
                    width=width,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                samples.append(sample)
        finally:
            self._active_params = previous_params
        return torch.stack(samples, dim=0)

    def generate_naive_patch_samples(
        self,
        *,
        num_samples: int = 1,
        image_shape: tuple[int, int, int] | None = None,
        prog_bar: bool = False,
        generator: torch.Generator | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sample patches independently and stitch one partition into each image."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        params = self._merged_parameters(kwargs)
        if image_shape is None:
            geometry = getattr(self.model, "geometry", None)
            if geometry is None:
                raise ValueError(
                    "image_shape is required when the model has no geometry."
                )
            image_shape = tuple(int(value) for value in geometry.image_shape)
        if len(image_shape) != 3:
            raise ValueError("image_shape must be (channels, height, width).")

        channels, height, width = (int(value) for value in image_shape)
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        samples = []
        iterator = range(num_samples)
        if prog_bar:
            iterator = tqdm(
                iterator, desc="PaDIS naive patch generation", total=num_samples
            )

        previous_params = getattr(self, "_active_params", None)
        self._active_params = params
        try:
            for _ in iterator:
                sample = self.generate_naive_patch_sample(
                    params,
                    channels=channels,
                    height=height,
                    width=width,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                samples.append(sample)
        finally:
            self._active_params = previous_params
        return torch.stack(samples, dim=0)

    def initial_reconstruction(self, measurement: torch.Tensor, params) -> torch.Tensor:
        """Construct the configured unpadded sampler initialization."""
        if params.initial_reconstruction == "fdk" and self.geometry is not None:
            x = fdk(
                measurement,
                self.op,
                clip=bool(params.clip_initial),
                padded=bool(getattr(params, "initial_fdk_padded", True)),
                filter_type=getattr(params, "initial_fdk_filter_type", None),
                frequency_scaling=float(
                    getattr(params, "initial_fdk_frequency_scaling", 1.0)
                ),
                batch_size=int(getattr(params, "initial_fdk_batch_size", 10)),
            )
        elif params.initial_reconstruction == "noise":
            x = measurement.new_zeros(self.op.domain_shape)
            return x.to(device=measurement.device, dtype=measurement.dtype)
        else:
            try:
                x = self.op.inverse(measurement)
            except NotImplementedError:
                x = self.op.adjoint(measurement)
        x = self._from_measurement_image(x, params)
        if bool(params.clip_initial):
            x = x.clamp(0.0, 1.0)
        return x.to(device=measurement.device, dtype=measurement.dtype)

    def pseudoinverse_reconstruction(
        self, measurement: torch.Tensor, params, *, clip: bool | None = None
    ) -> torch.Tensor:
        """Compute the pseudoinverse image used by DDNM corrections."""
        clip_pseudoinverse = (
            bool(getattr(params, "ddnm_pseudoinverse_clip", False))
            if clip is None
            else bool(clip)
        )
        if self.geometry is not None:
            x = fdk(
                measurement,
                self.op,
                clip=clip_pseudoinverse,
                padded=bool(getattr(params, "initial_fdk_padded", True)),
                filter_type=getattr(params, "initial_fdk_filter_type", None),
                frequency_scaling=float(
                    getattr(params, "initial_fdk_frequency_scaling", 1.0)
                ),
                batch_size=int(getattr(params, "initial_fdk_batch_size", 10)),
            )
        else:
            try:
                x = self.op.inverse(measurement)
            except NotImplementedError:
                x = self.op.adjoint(measurement)
        x = self._from_measurement_image(x, params)
        if clip_pseudoinverse:
            x = x.clamp(0.0, 1.0)
        return x.to(device=measurement.device, dtype=measurement.dtype)

    def initial_padded_state(
        self,
        measurement: torch.Tensor,
        params,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Construct the configured padded initial sampler state."""
        x_init = self.initial_reconstruction(measurement, params).unsqueeze(0)
        if (
            params.initial_reconstruction == "noise"
            and getattr(params, "noise_initialization", "padded") == "central_then_pad"
        ):
            return self._pad(
                float(params.sigma_max) * self._sample_noise(x_init, generator),
                params,
            )
        if params.initial_reconstruction == "noise":
            return float(params.sigma_max) * self._sample_noise(
                self._pad(x_init, params), generator
            )
        if bool(getattr(params, "consume_discarded_initial_noise", False)):
            _ = float(params.sigma_max) * self._sample_noise(x_init, generator)
        return self._pad(x_init, params)

    def noise_schedule(self, params, device: torch.device) -> torch.Tensor:
        """Build the configured descending diffusion-noise schedule."""
        if int(params.num_steps) < 1:
            raise ValueError("num_steps must be positive.")
        schedule_dtype = (
            torch.float64
            if getattr(params, "noise_schedule_dtype", "float32") == "float64"
            else torch.float32
        )
        if int(params.num_steps) == 1:
            t_steps = torch.tensor(
                [float(params.sigma_max)], dtype=torch.float64, device=device
            )
        else:
            sigma_min = float(params.sigma_min)
            sigma_max = float(params.sigma_max)
            schedule = getattr(params, "noise_schedule", "edm")
            if schedule == "geometric":
                log_steps = torch.linspace(
                    math.log(sigma_max),
                    math.log(sigma_min),
                    int(params.num_steps),
                    dtype=torch.float64,
                    device=device,
                )
                t_steps = torch.exp(log_steps)
            elif schedule == "edm":
                step_indices = torch.arange(
                    int(params.num_steps), dtype=torch.float64, device=device
                )
                rho = float(params.rho)
                t_steps = (
                    sigma_max ** (1.0 / rho)
                    + step_indices
                    / (int(params.num_steps) - 1)
                    * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
                ) ** rho
            else:
                raise ValueError("noise_schedule must be 'edm' or 'geometric'.")
        t_steps = t_steps.to(schedule_dtype)
        return torch.cat([t_steps, torch.zeros(1, dtype=schedule_dtype, device=device)])

    def edm_denoise_batch(
        self,
        image_batch: torch.Tensor,
        position_batch: torch.Tensor | None,
        sigma: torch.Tensor,
        params,
        *,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        """Denoise an image or patch batch using EDM preconditioning."""
        batch_size = image_batch.shape[0]
        patch_batch_size = params.patch_batch_size
        if patch_batch_size is None:
            patch_batch_size = batch_size
        patch_batch_size = int(patch_batch_size)
        if patch_batch_size <= 0:
            raise ValueError("patch_batch_size must be positive or None.")

        outputs = []
        try:
            model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            model_dtype = image_batch.dtype

        def denoise_chunk(
            image: torch.Tensor, position: torch.Tensor | None
        ) -> torch.Tensor:
            sigma_vec = sigma.expand(image.shape[0]).to(
                device=image.device, dtype=model_dtype
            )
            sigma_view = sigma_vec.reshape(image.shape[0], 1, 1, 1)
            sigma_data = torch.as_tensor(
                float(params.sigma_data), device=image.device, dtype=image.dtype
            )
            c_skip = sigma_data.square() / (sigma_view.square() + sigma_data.square())
            c_out = (
                sigma_view
                * sigma_data
                / (sigma_view.square() + sigma_data.square()).sqrt()
            )
            c_in = 1 / (sigma_data.square() + sigma_view.square()).sqrt()
            c_noise = sigma_vec.log() / 4
            if position is not None:
                model_input = torch.cat((c_in * image, position), dim=1)
            else:
                model_input = c_in * image
            model_output = self.model(model_input, c_noise)
            return c_skip * image + c_out * model_output

        for start in range(0, batch_size, patch_batch_size):
            stop = min(start + patch_batch_size, batch_size)
            image = image_batch[start:stop].to(model_dtype)
            position = (
                position_batch[start:stop].to(model_dtype)
                if position_batch is not None
                else None
            )
            if use_checkpoint and torch.is_grad_enabled() and image.requires_grad:
                if position is None:
                    output = activation_checkpoint(
                        lambda image_arg: denoise_chunk(image_arg, None),
                        image,
                        use_reentrant=False,
                    )
                else:
                    output = activation_checkpoint(
                        lambda image_arg, position_arg: denoise_chunk(
                            image_arg, position_arg
                        ),
                        image,
                        position,
                        use_reentrant=False,
                    )
            else:
                output = denoise_chunk(image, position)
            outputs.append(output)
        return torch.cat(outputs, dim=0)
