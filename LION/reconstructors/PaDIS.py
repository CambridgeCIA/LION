"""PaDIS diffusion-prior reconstructors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn.functional as F
from tqdm import tqdm

from LION.classical_algorithms.fdk import fdk
from LION.CTtools.ct_geometry import Geometry
from LION.models.diffusion import NCSNpp
from LION.operators import Operator
from LION.reconstructors.LIONreconstructor import LIONReconstructor
from LION.utils.math import power_method
from LION.utils.parameter import LIONParameter


@dataclass(frozen=True)
class _PatchLayout:
    indices: list[tuple[int, int, int, int]]
    image_height: int
    image_width: int


class PaDIS(LIONReconstructor):
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
        else:
            super().__init__(physics)
        if algorithm not in ("dps_langevin", "dps", "langevin", "pc"):
            raise ValueError("algorithm must be 'dps_langevin', 'langevin', or 'pc'.")
        self.model = model
        self.parameters = parameters or self.default_parameters(model)
        self.algorithm = self._canonical_algorithm(algorithm)

    @staticmethod
    def default_parameters(model: NCSNpp | None = None) -> LIONParameter:
        params = LIONParameter()
        model_params = getattr(model, "model_parameters", None)
        params.num_steps = 18
        params.inner_steps = 10
        params.sigma_min = 0.005
        params.sigma_max = 0.05
        params.noise_schedule = "edm"
        params.rho = 7.0
        params.zeta = 0.3
        params.sampling_epsilon = 1.0
        params.dps_epsilon = 0.5
        params.generation_epsilon = 1.0
        params.prior_mode = getattr(model_params, "prior_mode", "patch")
        params.pad_width = int(getattr(model_params, "pad_width", 24))
        params.patch_size = int(getattr(model_params, "largest_patch_size", 56))
        params.sigma_data = 0.5
        params.initial_reconstruction = "fdk"
        params.clip_initial = True
        params.clip_output = True
        params.clip_denoised = False
        params.clip_state = False
        params.patch_batch_size = None
        params.langevin_ddnm = False
        params.pc_snr = 0.16
        params.langevin_noise_scale = 1.0
        params.measurement_scale = 1.0
        params.measurement_offset = 0.0
        params.data_consistency_gradient = "norm"
        params.adjoint_data_step_schedule = "public_repo"
        params.data_consistency_normalization = "none"
        params.data_consistency_scale = 1.0
        params.data_consistency_scale_schedule = "constant"
        params.data_consistency_scale_power = 1.0
        params.data_consistency_scale_floor = 0.0
        params.operator_norm_iterations = 20
        params.operator_norm_tolerance = 1e-4
        params.operator_norm = None
        params.disable_data_consistency = False
        params.disable_langevin_noise = False
        params.disable_prior_score = False
        params.naive_patch_fixed_layout = True
        params.naive_patch_output = "sampler_state"
        params.trace_interval = 0
        return params

    @staticmethod
    def paper_ct_parameters(
        model: NCSNpp | None = None, *, views: int = 20
    ) -> LIONParameter:
        """Return the strict CT reconstruction sampler described in the paper."""
        params = PaDIS.default_parameters(model)
        params.num_steps = 100
        params.inner_steps = 10
        if int(views) == 20:
            params.sigma_min = 0.002
        elif int(views) == 8:
            params.sigma_min = 0.003
        else:
            raise ValueError("PaDIS paper CT sampler only specifies 20 or 8 views.")
        params.sigma_max = 10.0
        params.noise_schedule = "geometric"
        params.zeta = 0.3
        params.sampling_epsilon = 1.0
        params.dps_epsilon = 1.0
        params.data_consistency_gradient = "paper_squared_residual"
        params.data_consistency_normalization = "none"
        params.data_consistency_scale = 1.0
        params.data_consistency_scale_schedule = "constant"
        params.adjoint_data_step_schedule = "paper"
        params.initial_reconstruction = "noise"
        params.clip_initial = False
        params.clip_output = False
        params.clip_denoised = False
        params.clip_state = False
        params.langevin_noise_scale = 1.0
        return params

    @staticmethod
    def padis_repo_ct_parameters(model: NCSNpp | None = None) -> LIONParameter:
        """Return the public PaDIS CT-script sampler settings for compatibility."""
        params = PaDIS.default_parameters(model)
        params.num_steps = 100
        params.inner_steps = 10
        params.sigma_min = 0.003
        params.sigma_max = 10.0
        params.noise_schedule = "edm"
        params.zeta = 0.3
        params.dps_epsilon = 0.5
        params.sampling_epsilon = 1.0
        params.data_consistency_gradient = "norm"
        params.adjoint_data_step_schedule = "public_repo"
        params.initial_reconstruction = "fdk"
        params.clip_initial = True
        params.clip_output = True
        return params

    def reconstruct_sample(
        self,
        sino: torch.Tensor,
        *,
        algorithm: Literal["dps_langevin", "dps", "langevin", "pc"] | None = None,
        prog_bar: bool = False,
        generator: torch.Generator | None = None,
        **kwargs,
    ) -> torch.Tensor:
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
        previous_params = getattr(self, "_active_params", None)
        self._active_params = params
        try:
            if algorithm == "dps_langevin":
                return self._dps_langevin(
                    sino, params, prog_bar=prog_bar, generator=generator
                )
            if algorithm == "langevin":
                return self._langevin(
                    sino, params, prog_bar=prog_bar, generator=generator
                )
            if algorithm == "pc":
                return self._predictor_corrector(
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
                sample = self._generate_one_sample(
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
                sample = self._generate_one_naive_patch_sample(
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

    def _merged_parameters(self, overrides: dict) -> LIONParameter:
        params = LIONParameter()
        for key, value in self.parameters.__dict__.items():
            if not key.startswith("_"):
                setattr(params, key, value)
        for key, value in overrides.items():
            if value is not None:
                setattr(params, key, value)
        self._validate_sampler_parameters(params)
        return params

    @staticmethod
    def _validate_sampler_parameters(params) -> None:
        prior_mode = getattr(params, "prior_mode", "patch")
        if prior_mode not in ("patch", "whole_image"):
            raise ValueError("prior_mode must be 'patch' or 'whole_image'.")
        if int(params.num_steps) < 1:
            raise ValueError("num_steps must be positive.")
        if int(params.inner_steps) < 1:
            raise ValueError("inner_steps must be positive.")
        if float(params.sigma_min) <= 0 or float(params.sigma_max) <= 0:
            raise ValueError("sigma_min and sigma_max must be positive.")
        if float(params.sigma_max) < float(params.sigma_min):
            raise ValueError("sigma_max must be greater than or equal to sigma_min.")
        if getattr(params, "noise_schedule", "edm") not in ("edm", "geometric"):
            raise ValueError("noise_schedule must be 'edm' or 'geometric'.")
        if float(getattr(params, "sampling_epsilon", 1.0)) <= 0:
            raise ValueError("sampling_epsilon must be positive.")
        if float(getattr(params, "dps_epsilon", 1.0)) <= 0:
            raise ValueError("dps_epsilon must be positive.")
        if float(getattr(params, "generation_epsilon", 1.0)) <= 0:
            raise ValueError("generation_epsilon must be positive.")
        if getattr(params, "data_consistency_gradient", "norm") not in (
            "norm",
            "paper_squared_residual",
        ):
            raise ValueError(
                "data_consistency_gradient must be 'norm' or "
                "'paper_squared_residual'."
            )
        if getattr(params, "adjoint_data_step_schedule", "public_repo") not in (
            "paper",
            "public_repo",
        ):
            raise ValueError(
                "adjoint_data_step_schedule must be 'paper' or 'public_repo'."
            )
        if getattr(params, "naive_patch_output", "sampler_state") not in (
            "sampler_state",
            "denoised",
        ):
            raise ValueError(
                "naive_patch_output must be 'sampler_state' or 'denoised'."
            )

    @staticmethod
    def _canonical_algorithm(algorithm: str) -> str:
        if algorithm == "dps":
            return "dps_langevin"
        if algorithm == "predictor_corrector":
            return "pc"
        return algorithm

    def _initial_reconstruction(
        self, measurement: torch.Tensor, params
    ) -> torch.Tensor:
        if params.initial_reconstruction == "fdk" and self.geometry is not None:
            x = fdk(measurement, self.op, clip=bool(params.clip_initial))
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

    def _initial_padded_state(
        self,
        measurement: torch.Tensor,
        params,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        x_init = self._initial_reconstruction(measurement, params).unsqueeze(0)
        if params.initial_reconstruction == "noise":
            return float(params.sigma_max) * self._sample_noise(
                self._pad(x_init, params), generator
            )
        return self._pad(x_init, params)

    def _noise_schedule(self, params, device: torch.device) -> torch.Tensor:
        if int(params.num_steps) < 1:
            raise ValueError("num_steps must be positive.")
        if int(params.num_steps) == 1:
            t_steps = torch.tensor([float(params.sigma_max)], device=device)
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
        return torch.cat([t_steps.to(torch.float32), torch.zeros(1, device=device)])

    def _patch_layout(
        self,
        image_shape: tuple[int, int],
        params,
        device: torch.device,
        generator: torch.Generator | None = None,
        *,
        fixed_offset: bool = False,
    ) -> _PatchLayout:
        height, width = image_shape
        patch_size = int(params.patch_size)
        pad = int(params.pad_width)
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if pad < 0:
            raise ValueError("pad_width must be non-negative.")
        n_rows = height // patch_size + 1
        n_cols = width // patch_size + 1
        row_offset = self._random_offset(pad, device, generator, fixed_offset)
        col_offset = self._random_offset(pad, device, generator, fixed_offset)
        row_spaced = torch.arange(n_rows, device=device, dtype=torch.int64) * patch_size
        col_spaced = torch.arange(n_cols, device=device, dtype=torch.int64) * patch_size
        indices = []
        padded_height = height + 2 * pad
        padded_width = width + 2 * pad
        for row_start in row_spaced.tolist():
            for col_start in col_spaced.tolist():
                top = int(row_start) + row_offset
                left = int(col_start) + col_offset
                bottom = top + patch_size
                right = left + patch_size
                if bottom <= padded_height and right <= padded_width:
                    indices.append((top, bottom, left, right))
        if not indices:
            raise ValueError("Patch layout produced no valid patches.")
        return _PatchLayout(indices, height, width)

    @staticmethod
    def _random_offset(
        pad: int,
        device: torch.device,
        generator: torch.Generator | None = None,
        fixed_offset: bool = False,
    ) -> int:
        if pad <= 0 or fixed_offset:
            return 0
        return int(
            torch.randint(0, pad, (1,), device=device, generator=generator).item()
        )

    def _position_grid(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        y = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype)
        x_coord = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(y, x_coord, indexing="ij")
        grid = torch.stack((xx, yy), dim=0).unsqueeze(0)
        return grid.expand(batch_size, -1, -1, -1)

    def _denoise_patches(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        layout: _PatchLayout,
        params,
    ) -> torch.Tensor:
        if x.dim() != 4 or x.shape[0] != 1:
            raise ValueError(
                "PaDIS patch denoising expects a single padded image batch."
            )

        image_patches = []
        position_patches = []
        positions = (
            self._position_grid(x)
            if int(getattr(self.model.model_parameters, "input_position_channels", 2))
            > 0
            else None
        )
        for top, bottom, left, right in layout.indices:
            image_patches.append(x[:, :, top:bottom, left:right])
            if positions is not None:
                position_patches.append(positions[:, :, top:bottom, left:right])

        image_batch = torch.cat(image_patches, dim=0)
        position_batch = (
            torch.cat(position_patches, dim=0) if positions is not None else None
        )
        denoised_batch = self._edm_denoise_batch(
            image_batch, position_batch, sigma, params
        )

        output = torch.zeros_like(x)
        cursor = 0
        for top, bottom, left, right in layout.indices:
            output[:, :, top:bottom, left:right] += denoised_batch[cursor : cursor + 1]
            output[:, :, top:bottom, left:right] -= x[:, :, top:bottom, left:right]
            cursor += 1
        denoised = x + output
        pad = int(params.pad_width)
        if pad == 0:
            return denoised
        zero_border = torch.zeros_like(denoised)
        zero_border[
            :, :, pad : pad + layout.image_height, pad : pad + layout.image_width
        ] = denoised[
            :, :, pad : pad + layout.image_height, pad : pad + layout.image_width
        ]
        return zero_border

    def _denoise_whole_image(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        params,
    ) -> torch.Tensor:
        if x.dim() != 4 or x.shape[0] != 1:
            raise ValueError("Whole-image denoising expects a single image batch.")
        if int(params.pad_width) != 0:
            raise ValueError(
                "Whole-image diffusion reconstruction expects pad_width=0."
            )
        positions = (
            self._position_grid(x)
            if int(getattr(self.model.model_parameters, "input_position_channels", 2))
            > 0
            else None
        )
        return self._edm_denoise_batch(x, positions, sigma, params)

    def _validate_prior_configuration(
        self, params, image_shape: tuple[int, int]
    ) -> None:
        height, width = image_shape
        prior_mode = getattr(params, "prior_mode", "patch")
        patch_size = int(params.patch_size)
        pad_width = int(params.pad_width)
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if pad_width < 0:
            raise ValueError("pad_width must be non-negative.")
        if prior_mode == "whole_image":
            if pad_width != 0:
                raise ValueError(
                    "Whole-image diffusion reconstruction expects pad_width=0."
                )
            if patch_size != height or patch_size != width:
                raise ValueError(
                    "Whole-image diffusion reconstruction expects patch_size to "
                    "match the image height and width."
                )
            model_prior_mode = getattr(
                getattr(self.model, "model_parameters", None),
                "prior_mode",
                "patch",
            )
            if model_prior_mode != "whole_image":
                raise ValueError(
                    "Whole-image reconstruction requires a whole-image model preset."
                )
        elif prior_mode == "patch":
            if (
                patch_size > height + 2 * pad_width
                or patch_size > width + 2 * pad_width
            ):
                raise ValueError("patch_size cannot exceed padded image dimensions.")
        else:
            raise ValueError("prior_mode must be 'patch' or 'whole_image'.")

    def _denoise_prior(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        params,
        image_shape: tuple[int, int],
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        self._validate_prior_configuration(params, image_shape)
        prior_mode = getattr(params, "prior_mode", "patch")
        if prior_mode == "whole_image":
            return self._denoise_whole_image(x, sigma, params)
        if prior_mode != "patch":
            raise ValueError("prior_mode must be 'patch' or 'whole_image'.")
        layout = self._patch_layout(image_shape, params, x.device, generator)
        return self._denoise_patches(x, sigma, layout, params)

    def _edm_denoise_batch(
        self,
        image_batch: torch.Tensor,
        position_batch: torch.Tensor | None,
        sigma: torch.Tensor,
        params,
    ) -> torch.Tensor:
        batch_size = image_batch.shape[0]
        patch_batch_size = params.patch_batch_size
        if patch_batch_size is None:
            patch_batch_size = batch_size
        patch_batch_size = int(patch_batch_size)
        if patch_batch_size <= 0:
            raise ValueError("patch_batch_size must be positive or None.")

        outputs = []
        for start in range(0, batch_size, patch_batch_size):
            stop = min(start + patch_batch_size, batch_size)
            image = image_batch[start:stop]
            position = (
                position_batch[start:stop] if position_batch is not None else None
            )
            sigma_vec = sigma.expand(image.shape[0]).to(
                device=image.device, dtype=image.dtype
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
            outputs.append(c_skip * image + c_out * model_output)
        return torch.cat(outputs, dim=0)

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

    def _forward_project(self, x: torch.Tensor) -> torch.Tensor:
        params = getattr(self, "_active_params", None)
        if params is not None:
            x = self._to_measurement_image(x, params)
        if self.geometry is not None:
            return self.op_autograd(x)
        return self.op(x)

    def _adjoint_project(self, y: torch.Tensor) -> torch.Tensor:
        params = getattr(self, "_active_params", None)
        scale = 1.0 if params is None else float(params.measurement_scale)
        return scale * self.op.adjoint(y)

    def _operator_norm(self, params, device: torch.device) -> float:
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

    def _data_consistency_normalizer(self, params, device: torch.device) -> float:
        method = getattr(params, "data_consistency_normalization", "none")
        if method in (None, "none", False):
            return 1.0
        if method != "operator_norm":
            raise ValueError(
                "data_consistency_normalization must be 'operator_norm' or 'none'."
            )

        # The sampler state is in the diffusion model's normalized image units,
        # while the forward model may first map it to attenuation units. The
        # Lipschitz scale of that composed measurement map is |scale| * ||A||.
        normalizer = abs(float(params.measurement_scale)) * self._operator_norm(
            params, device
        )
        return max(normalizer, 1e-12)

    def _normalise_data_gradient(
        self,
        gradient: torch.Tensor,
        params,
        sigma: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float, float]:
        normalizer = self._data_consistency_normalizer(params, gradient.device)
        scaled = gradient / normalizer
        scale = self._scheduled_data_consistency_scale(params, sigma, gradient.device)
        scaled = scale * scaled
        return scaled, normalizer, scale

    def _scheduled_data_consistency_scale(
        self,
        params,
        sigma: torch.Tensor | None,
        device: torch.device,
    ) -> float:
        base = float(getattr(params, "data_consistency_scale", 1.0))
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

    def _measurement_gradient(
        self,
        measurement: torch.Tensor,
        x: torch.Tensor,
        x0hat: torch.Tensor,
        params,
    ) -> torch.Tensor:
        grad, *_ = self._dps_data_gradient(measurement, x, x0hat, params, sigma=None)
        return grad

    def _dps_data_gradient(
        self,
        measurement: torch.Tensor,
        x: torch.Tensor,
        x0hat: torch.Tensor,
        params,
        sigma: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
        predicted = self._forward_project(self._crop(x0hat, params).squeeze(0))
        residual = measurement - predicted.to(dtype=measurement.dtype)
        residual_norm = torch.linalg.norm(residual).clamp_min(1e-12)
        gradient_mode = getattr(params, "data_consistency_gradient", "norm")
        if gradient_mode == "paper_squared_residual":
            objective = residual.square().sum()
            step_size = float(params.zeta) / float(residual_norm.detach().cpu())
        elif gradient_mode == "norm":
            objective = residual_norm
            step_size = float(params.zeta)
        else:
            raise ValueError(
                "data_consistency_gradient must be 'norm' or "
                "'paper_squared_residual'."
            )
        raw_gradient = torch.autograd.grad(outputs=objective, inputs=x)[0]
        gradient, data_normalizer, data_scale = self._normalise_data_gradient(
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
        score: torch.Tensor,
        residual: torch.Tensor | None = None,
        gradient: torch.Tensor | None = None,
        raw_gradient: torch.Tensor | None = None,
        data_normalizer: float | None = None,
        data_scale: float | None = None,
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
            "denoised_min": float(denoised.detach().amin().cpu()),
            "denoised_max": float(denoised.detach().amax().cpu()),
            "denoised_mean": float(denoised.detach().mean().cpu()),
            "score_norm": float(torch.linalg.norm(score.detach()).cpu()),
        }
        if residual is not None:
            item["residual_norm"] = float(torch.linalg.norm(residual.detach()).cpu())
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
        self.last_trace.append(item)

    def _sample_noise(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if generator is None:
            return torch.randn_like(x)
        return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)

    def _generate_one_sample(
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
        central = torch.zeros((1, channels, height, width), device=device, dtype=dtype)
        x = float(params.sigma_max) * self._sample_noise(
            self._pad(central, params), generator
        )
        t_steps = self._noise_schedule(params, x.device)
        epsilon = float(getattr(params, "generation_epsilon", 1.0))

        with torch.no_grad():
            for step_index, t_cur in enumerate(t_steps[:-1]):
                alpha = epsilon * t_cur.square()
                for _ in range(int(params.inner_steps)):
                    denoised = self._denoise_prior(
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

    def _generate_one_naive_patch_sample(
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
        self._validate_prior_configuration(params, (height, width))
        layout = self._patch_layout(
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
        t_steps = self._noise_schedule(params, x.device)
        epsilon = float(getattr(params, "generation_epsilon", 1.0))
        final_denoised = None

        with torch.no_grad():
            for step_index, t_cur in enumerate(t_steps[:-1]):
                alpha = epsilon * t_cur.square()
                for _ in range(int(params.inner_steps)):
                    denoised = self._edm_denoise_batch(
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

    def _dps_langevin(
        self,
        measurement: torch.Tensor,
        params,
        *,
        prog_bar: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        x_init = self._initial_reconstruction(measurement, params).unsqueeze(0)
        x = (
            self._initial_padded_state(measurement, params, generator)
            .detach()
            .requires_grad_(True)
        )
        t_steps = self._noise_schedule(params, x.device)
        iterator = zip(t_steps[:-1], t_steps[1:])
        if prog_bar:
            iterator = tqdm(
                list(iterator), desc="PaDIS DPS", total=int(params.num_steps)
            )

        for step_index, (t_cur, _t_next) in enumerate(iterator):
            alpha = float(getattr(params, "dps_epsilon", 1.0)) * t_cur.square()
            for inner_index in range(int(params.inner_steps)):
                denoised = self._denoise_prior(
                    x,
                    t_cur.reshape(1),
                    params,
                    tuple(x_init.shape[-2:]),
                    generator,
                )
                denoised = self._clip_model_range(denoised, params)
                score = (denoised - x) / t_cur.square()
                (
                    data_gradient,
                    raw_data_gradient,
                    residual,
                    data_normalizer,
                    data_scale,
                    data_step_size,
                ) = self._dps_data_gradient(
                    measurement, x, denoised, params, sigma=t_cur
                )
                self._append_trace(
                    params,
                    algorithm="dps_langevin",
                    step_index=step_index,
                    inner_index=inner_index,
                    sigma=t_cur,
                    x=x,
                    denoised=denoised,
                    score=score,
                    residual=residual,
                    gradient=data_gradient,
                    raw_gradient=raw_data_gradient,
                    data_normalizer=data_normalizer,
                    data_scale=data_scale,
                )
                z = self._sample_noise(x, generator)
                if bool(params.disable_data_consistency):
                    x = x
                else:
                    x = x - data_step_size * data_gradient
                score_step = (
                    0 if bool(params.disable_prior_score) else alpha / 2 * score
                )
                if step_index < int(params.num_steps) - 1:
                    noise_step = (
                        0
                        if bool(params.disable_langevin_noise)
                        else float(params.langevin_noise_scale) * torch.sqrt(alpha) * z
                    )
                    x = x + score_step + noise_step
                else:
                    x = x + score_step
                x = self._clip_state_range(x, params).detach().requires_grad_(True)

        reconstruction = self._crop(x.detach(), params).squeeze(0)
        if bool(params.clip_output):
            reconstruction = reconstruction.clamp(0.0, 1.0)
        return reconstruction

    def _apply_adjoint_correction(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        step_size: torch.Tensor,
        params,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        raw_correction = self._adjoint_project(residual).unsqueeze(0)
        data_normalizer = self._data_consistency_normalizer(
            params, raw_correction.device
        )
        data_scale = self._scheduled_data_consistency_scale(
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

    def _adjoint_data_step_size(
        self,
        residual: torch.Tensor,
        sigma: torch.Tensor,
        params,
        *,
        public_repo_multiplier: bool,
    ) -> torch.Tensor:
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

    def _predictor_corrector(
        self,
        measurement: torch.Tensor,
        params,
        *,
        prog_bar: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        x_init = self._initial_reconstruction(measurement, params).unsqueeze(0)
        x = float(params.sigma_max) * self._sample_noise(
            self._pad(x_init, params), generator
        )
        t_steps = self._noise_schedule(params, x.device)
        iterator = zip(t_steps[:-1], t_steps[1:])
        if prog_bar:
            iterator = tqdm(
                list(iterator),
                desc="PaDIS predictor-corrector",
                total=max(int(params.num_steps) - 1, 0),
            )

        with torch.no_grad():
            for step_index, (t_cur, t_next) in enumerate(iterator):
                if step_index == int(params.num_steps) - 1:
                    break

                denoised = self._denoise_prior(
                    x,
                    t_cur.reshape(1),
                    params,
                    tuple(x_init.shape[-2:]),
                    generator,
                )
                denoised = self._clip_model_range(denoised, params)
                score = (denoised - x) / t_cur.square()
                predictor_delta = t_cur.square() - t_next.square()
                if not bool(params.disable_prior_score):
                    x = x + predictor_delta * score
                if not bool(params.disable_langevin_noise):
                    z = self._sample_noise(x, generator)
                    x = x + torch.sqrt(predictor_delta.clamp_min(0.0)) * z

                residual = measurement - self._forward_project(
                    self._crop(x, params).squeeze(0).to(torch.float32)
                ).to(dtype=measurement.dtype)
                step_size = self._adjoint_data_step_size(
                    residual, t_cur, params, public_repo_multiplier=False
                )
                (
                    x,
                    correction,
                    raw_correction,
                    data_normalizer,
                    data_scale,
                ) = self._apply_adjoint_correction(
                    x, residual, step_size, params, t_cur
                )
                self._append_trace(
                    params,
                    algorithm="pc_predictor",
                    step_index=step_index,
                    inner_index=0,
                    sigma=t_cur,
                    x=x,
                    denoised=denoised,
                    score=score,
                    residual=residual,
                    gradient=correction,
                    raw_gradient=raw_correction,
                    data_normalizer=data_normalizer,
                    data_scale=data_scale,
                )

                if step_index < int(params.num_steps) - 1:
                    z = self._sample_noise(x, generator)
                    denoised = self._denoise_prior(
                        x,
                        t_cur.reshape(1),
                        params,
                        tuple(x_init.shape[-2:]),
                        generator,
                    )
                    denoised = self._clip_model_range(denoised, params)
                    score = (denoised - x) / t_next.square().clamp_min(1e-12)
                    eps = (
                        2.0
                        * float(params.pc_snr)
                        * torch.linalg.norm(z)
                        / torch.linalg.norm(score).clamp_min(1e-12)
                    )
                    if not bool(params.disable_prior_score):
                        x = x + eps * score
                    if not bool(params.disable_langevin_noise):
                        x = x + torch.sqrt(2.0 * eps) * z

                    residual = measurement - self._forward_project(
                        self._crop(x, params).squeeze(0).to(torch.float32)
                    ).to(dtype=measurement.dtype)
                    step_size = self._adjoint_data_step_size(
                        residual, t_cur, params, public_repo_multiplier=True
                    )
                    (
                        x,
                        correction,
                        raw_correction,
                        data_normalizer,
                        data_scale,
                    ) = self._apply_adjoint_correction(
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

    def _langevin(
        self,
        measurement: torch.Tensor,
        params,
        *,
        prog_bar: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        x_init = self._initial_reconstruction(measurement, params).unsqueeze(0)
        x = float(params.sigma_max) * self._sample_noise(
            self._pad(x_init, params), generator
        )
        t_steps = self._noise_schedule(params, x.device)
        iterator = zip(t_steps[:-1], t_steps[1:])
        if prog_bar:
            iterator = tqdm(
                list(iterator), desc="PaDIS Langevin", total=int(params.num_steps)
            )

        with torch.no_grad():
            for step_index, (t_cur, _t_next) in enumerate(iterator):
                alpha = float(getattr(params, "sampling_epsilon", 1.0)) * t_cur.square()
                for inner_index in range(int(params.inner_steps)):
                    denoised = self._denoise_prior(
                        x,
                        t_cur.reshape(1),
                        params,
                        tuple(x_init.shape[-2:]),
                        generator,
                    )
                    denoised = self._clip_model_range(denoised, params)
                    if bool(params.langevin_ddnm):
                        denoised_crop = self._crop(denoised, params).squeeze(0)
                        backprojected = self._initial_reconstruction(
                            measurement, params
                        )
                        projected_denoised = self._forward_project(denoised_crop)
                        corrected = (
                            backprojected
                            + denoised_crop
                            - self._initial_reconstruction(projected_denoised, params)
                        )
                        x0hat = self._pad(corrected.unsqueeze(0), params)
                        score = (x0hat - x) / t_cur.square()
                    else:
                        score = (denoised - x) / t_cur.square()
                        residual = measurement - self._forward_project(
                            self._crop(x, params).squeeze(0).to(torch.float32)
                        ).to(dtype=measurement.dtype)
                        step_size = self._adjoint_data_step_size(
                            residual, t_cur, params, public_repo_multiplier=True
                        )
                        raw_correction = self._adjoint_project(residual).unsqueeze(0)
                        data_normalizer = self._data_consistency_normalizer(
                            params, raw_correction.device
                        )
                        data_scale = self._scheduled_data_consistency_scale(
                            params, t_cur, raw_correction.device
                        )
                        correction = data_scale * raw_correction / data_normalizer
                        self._append_trace(
                            params,
                            algorithm="langevin",
                            step_index=step_index,
                            inner_index=inner_index,
                            sigma=t_cur,
                            x=x,
                            denoised=denoised,
                            score=score,
                            residual=residual,
                            gradient=correction,
                            raw_gradient=raw_correction,
                            data_normalizer=data_normalizer,
                            data_scale=data_scale,
                        )
                        pad = int(params.pad_width)
                        if not bool(params.disable_data_consistency):
                            if pad == 0:
                                x = x + step_size * correction
                            else:
                                x[:, :, pad:-pad, pad:-pad] += step_size * correction

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
