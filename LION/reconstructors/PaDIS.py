"""PaDIS diffusion-prior reconstructors."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from tqdm import tqdm

from LION.classical_algorithms.fdk import fdk
from LION.CTtools.ct_geometry import Geometry
from LION.models.diffusion import NCSNpp
from LION.operators import Operator
from LION.reconstructors.LIONreconstructor import LIONReconstructor
from LION.utils.math import power_method
from LION.utils.parameter import LIONParameter


PUBLIC_REPO_CT_GRADIENT_SCALE = 0.0405
PUBLIC_REPO_CT_ADJOINT_SCALE = 0.1022


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

    @staticmethod
    def default_parameters(model: NCSNpp | None = None) -> LIONParameter:
        params = LIONParameter()
        model_params = getattr(model, "model_parameters", None)
        params.num_steps = 18
        params.inner_steps = 10
        params.sigma_min = 0.005
        params.sigma_max = 0.05
        params.noise_schedule = "edm"
        params.noise_schedule_dtype = "float32"
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
        params.noise_initialization = "padded"
        params.initial_fdk_filter_type = None
        params.initial_fdk_frequency_scaling = 1.0
        params.initial_fdk_padded = True
        params.initial_fdk_batch_size = 10
        params.clip_initial = True
        params.clip_output = True
        params.clip_denoised = False
        params.clip_state = False
        params.patch_batch_size = None
        params.patch_checkpoint_denoiser = False
        params.langevin_ddnm = False
        params.ddnm_pseudoinverse_clip = False
        params.ddnm_projected_pseudoinverse_clip = False
        params.ddnm_corrected_clip = False
        params.pc_snr = 0.16
        params.pc_corrector_step_rule = "paper_linear"
        params.pc_corrector_denoise_sigma = "next"
        params.pc_reuse_predictor_layout = False
        params.langevin_noise_scale = 1.0
        params.measurement_scale = 1.0
        params.measurement_offset = 0.0
        params.data_consistency_gradient = "norm"
        params.adjoint_data_step_schedule = "public_repo"
        params.data_consistency_normalization = "none"
        params.data_consistency_scale = 1.0
        params.adjoint_data_consistency_scale = None
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
        params.patch_assembly = "padis"
        params.patch_overlap = 8
        params.fixed_overlap_layout = "lion_clipped"
        params.fixed_overlap_checkpoint_denoiser = False
        params.trace_interval = 0
        params.trace_images = False
        params.stop_after_outer_steps = None
        params.patch_offset_rng = "torch"
        params.consume_discarded_measurement_noise = False
        params.consume_discarded_initial_noise = False
        params.consume_unused_latents = False
        params.consume_denoise_output_noise = False
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
    def lion_physics_ct_parameters(
        model: NCSNpp | None = None, *, views: int = 20
    ) -> LIONParameter:
        """Return the LION-native CT sampler with physical operator scaling.

        The paper sigma schedule is kept unchanged, but the measurement update
        is expressed in LION CT units: residual gradients use the LION forward
        operator and adjoint, normalized by the composed measurement operator
        Lipschitz constant rather than by public-repository matching constants.
        """
        params = PaDIS.paper_ct_parameters(model, views=views)
        params.initial_reconstruction = "fdk"
        params.initial_fdk_filter_type = "hann"
        params.initial_fdk_frequency_scaling = 0.3
        params.initial_fdk_padded = False
        params.initial_fdk_batch_size = 10
        params.clip_initial = True
        params.clip_output = True
        params.zeta = 3.0
        params.data_consistency_gradient = "least_squares"
        params.data_consistency_normalization = "operator_lipschitz"
        params.data_consistency_scale = 1.0
        params.adjoint_data_consistency_scale = None
        params.adjoint_data_step_schedule = "paper"
        params.pc_snr = 0.08
        params.patch_offset_rng = "torch"
        params.consume_discarded_measurement_noise = False
        params.consume_discarded_initial_noise = False
        params.consume_unused_latents = False
        params.consume_denoise_output_noise = False
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
        params.noise_schedule_dtype = "float64"
        params.zeta = 0.3
        params.dps_epsilon = 0.5
        params.sampling_epsilon = 1.0
        params.data_consistency_gradient = "norm"
        params.adjoint_data_step_schedule = "public_repo"
        params.data_consistency_scale = PUBLIC_REPO_CT_GRADIENT_SCALE
        params.adjoint_data_consistency_scale = PUBLIC_REPO_CT_ADJOINT_SCALE
        params.initial_reconstruction = "fdk"
        params.initial_fdk_filter_type = "hann"
        params.initial_fdk_frequency_scaling = 0.3
        params.initial_fdk_padded = False
        params.clip_initial = True
        params.clip_output = True
        params.patch_offset_rng = "python"
        params.consume_discarded_measurement_noise = True
        params.consume_discarded_initial_noise = True
        params.consume_unused_latents = True
        params.consume_denoise_output_noise = True
        params.pc_corrector_denoise_sigma = "current"
        params.pc_reuse_predictor_layout = True
        return params

    @staticmethod
    def lion_quality_ct_parameters(
        model: NCSNpp | None = None, *, views: int = 20
    ) -> LIONParameter:
        """Return the preferred LION CT sampler for PaDIS reconstruction checks.

        This keeps the paper's CT sigma schedule and squared-residual data
        objective, but uses FDK initialization and operator-norm measurement
        scaling for stable LION-native CT operations.
        """
        params = PaDIS.paper_ct_parameters(model, views=views)
        params.initial_reconstruction = "fdk"
        params.initial_fdk_filter_type = "hann"
        params.initial_fdk_frequency_scaling = 0.9
        params.initial_fdk_padded = False
        params.initial_fdk_batch_size = 10
        params.clip_initial = True
        params.clip_output = True
        params.data_consistency_normalization = "operator_norm"
        params.patch_offset_rng = "python"
        params.consume_discarded_measurement_noise = True
        params.consume_discarded_initial_noise = True
        params.consume_unused_latents = True
        params.consume_denoise_output_noise = True
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
        self.last_trace_images = []
        if bool(getattr(params, "consume_unused_latents", False)):
            latent_shape = (1, *tuple(int(value) for value in self.op.domain_shape))
            _ = self._sample_noise(sino.new_empty(latent_shape), generator)
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
        if getattr(params, "noise_schedule_dtype", "float32") not in (
            "float32",
            "float64",
        ):
            raise ValueError("noise_schedule_dtype must be 'float32' or 'float64'.")
        if float(getattr(params, "sampling_epsilon", 1.0)) <= 0:
            raise ValueError("sampling_epsilon must be positive.")
        if float(getattr(params, "dps_epsilon", 1.0)) <= 0:
            raise ValueError("dps_epsilon must be positive.")
        if float(getattr(params, "generation_epsilon", 1.0)) <= 0:
            raise ValueError("generation_epsilon must be positive.")
        if getattr(params, "data_consistency_gradient", "norm") not in (
            "norm",
            "least_squares",
            "paper_squared_residual",
        ):
            raise ValueError(
                "data_consistency_gradient must be 'norm', 'least_squares', "
                "or 'paper_squared_residual'."
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
        if getattr(params, "patch_offset_rng", "torch") not in ("torch", "python"):
            raise ValueError("patch_offset_rng must be 'torch' or 'python'.")
        if getattr(params, "pc_corrector_step_rule", "paper_linear") not in (
            "paper_linear",
            "score_sde_squared",
        ):
            raise ValueError(
                "pc_corrector_step_rule must be 'paper_linear' or "
                "'score_sde_squared'."
            )
        if getattr(params, "pc_corrector_denoise_sigma", "next") not in (
            "next",
            "current",
        ):
            raise ValueError("pc_corrector_denoise_sigma must be 'next' or 'current'.")
        if getattr(params, "noise_initialization", "padded") not in (
            "padded",
            "central_then_pad",
        ):
            raise ValueError(
                "noise_initialization must be 'padded' or 'central_then_pad'."
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

    def _pseudoinverse_reconstruction(
        self, measurement: torch.Tensor, params, *, clip: bool | None = None
    ) -> torch.Tensor:
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

    def _initial_padded_state(
        self,
        measurement: torch.Tensor,
        params,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        x_init = self._initial_reconstruction(measurement, params).unsqueeze(0)
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

    def _noise_schedule(self, params, device: torch.device) -> torch.Tensor:
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
        offset_rng = getattr(params, "patch_offset_rng", "torch")
        row_offset = self._random_offset(
            pad, device, generator, fixed_offset, rng_source=offset_rng
        )
        col_offset = self._random_offset(
            pad, device, generator, fixed_offset, rng_source=offset_rng
        )
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
        rng_source: str = "torch",
    ) -> int:
        if pad <= 0 or fixed_offset:
            return 0
        if rng_source == "python":
            return random.randint(0, pad - 1)
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
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if x.dim() != 4 or x.shape[0] != 1:
            raise ValueError(
                "PaDIS patch denoising expects a single padded image batch."
            )

        positions = (
            self._position_grid(x)
            if int(getattr(self.model.model_parameters, "input_position_channels", 2))
            > 0
            else None
        )
        patch_batch_size = getattr(params, "patch_batch_size", None)
        if patch_batch_size is None:
            patch_batch_size = len(layout.indices)
        patch_batch_size = int(patch_batch_size)
        if patch_batch_size <= 0:
            raise ValueError("patch_batch_size must be positive or None.")
        use_checkpoint = self._use_patch_checkpoint_denoiser(params)

        output = torch.zeros_like(x)
        for chunk_start in range(0, len(layout.indices), patch_batch_size):
            chunk_indices = layout.indices[chunk_start : chunk_start + patch_batch_size]
            image_batch = torch.cat(
                [
                    x[:, :, top:bottom, left:right]
                    for top, bottom, left, right in chunk_indices
                ],
                dim=0,
            )
            position_batch = None
            if positions is not None:
                position_batch = torch.cat(
                    [
                        positions[:, :, top:bottom, left:right]
                        for top, bottom, left, right in chunk_indices
                    ],
                    dim=0,
                )
            denoised_batch = self._edm_denoise_batch(
                image_batch,
                position_batch,
                sigma,
                params,
                use_checkpoint=use_checkpoint,
            )
            for offset, (top, bottom, left, right) in enumerate(chunk_indices):
                output[:, :, top:bottom, left:right] += denoised_batch[
                    offset : offset + 1
                ]
                output[:, :, top:bottom, left:right] -= x[:, :, top:bottom, left:right]
        denoised = x + output
        if bool(getattr(params, "consume_denoise_output_noise", False)):
            _ = self._sample_noise(denoised, generator)
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

    @staticmethod
    def _fixed_patch_starts(
        padded_length: int,
        *,
        pad: int,
        patch_size: int,
        overlap: int,
        layout: str = "lion_clipped",
    ) -> list[int]:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if overlap < 0 or overlap >= patch_size:
            raise ValueError("patch_overlap must satisfy 0 <= overlap < patch_size.")
        if patch_size > padded_length:
            raise ValueError("patch_size cannot exceed padded image dimensions.")
        stride = patch_size - overlap
        last_valid_start = padded_length - patch_size
        if layout == "lion_clipped":
            starts = [pad]
            while starts[-1] < last_valid_start:
                starts.append(starts[-1] + stride)
            starts[-1] = min(starts[-1], last_valid_start)
        elif layout in ("public_overlap", "public_tile"):
            start = pad if layout == "public_overlap" else 4
            if start < 0 or start + patch_size > padded_length:
                raise ValueError(
                    f"fixed_overlap_layout={layout!r} cannot place its first "
                    "patch inside the padded image."
                )
            starts = [start]
            public_stop = padded_length - pad - patch_size
            while starts[-1] < public_stop:
                next_start = starts[-1] + stride
                if next_start + patch_size > padded_length:
                    next_start = last_valid_start
                if next_start <= starts[-1]:
                    break
                starts.append(next_start)
        else:
            raise ValueError(
                "fixed_overlap_layout must be 'lion_clipped', "
                "'public_overlap', or 'public_tile'."
            )
        return sorted(set(int(start) for start in starts))

    def _fixed_overlap_patch_layout(
        self,
        image_shape: tuple[int, int],
        params,
    ) -> _PatchLayout:
        height, width = image_shape
        pad = int(params.pad_width)
        patch_size = int(params.patch_size)
        overlap = int(getattr(params, "patch_overlap", 8))
        fixed_overlap_layout = getattr(params, "fixed_overlap_layout", "lion_clipped")
        padded_height = height + 2 * pad
        padded_width = width + 2 * pad
        row_starts = self._fixed_patch_starts(
            padded_height,
            pad=pad,
            patch_size=patch_size,
            overlap=overlap,
            layout=fixed_overlap_layout,
        )
        col_starts = self._fixed_patch_starts(
            padded_width,
            pad=pad,
            patch_size=patch_size,
            overlap=overlap,
            layout=fixed_overlap_layout,
        )
        indices = [
            (top, top + patch_size, left, left + patch_size)
            for top in row_starts
            for left in col_starts
        ]
        if not indices:
            raise ValueError("Fixed-overlap patch layout produced no valid patches.")
        return _PatchLayout(indices, height, width)

    @staticmethod
    def _use_patch_checkpoint_denoiser(params) -> bool:
        return bool(
            getattr(params, "patch_checkpoint_denoiser", False)
            or getattr(params, "fixed_overlap_checkpoint_denoiser", False)
        )

    def _denoise_fixed_overlap_patches(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        layout: _PatchLayout,
        params,
        *,
        assembly: Literal["fixed_average", "fixed_stitch"],
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if x.dim() != 4 or x.shape[0] != 1:
            raise ValueError(
                "Fixed-overlap patch denoising expects a single padded image batch."
            )

        positions = (
            self._position_grid(x)
            if int(getattr(self.model.model_parameters, "input_position_channels", 2))
            > 0
            else None
        )
        patch_batch_size = getattr(params, "patch_batch_size", None)
        if patch_batch_size is None:
            patch_batch_size = len(layout.indices)
        patch_batch_size = int(patch_batch_size)
        if patch_batch_size <= 0:
            raise ValueError("patch_batch_size must be positive or None.")
        use_checkpoint = self._use_patch_checkpoint_denoiser(params)

        output = torch.zeros_like(x)
        if assembly == "fixed_average":
            counts = torch.zeros_like(x)
        else:
            counts = None
        for chunk_start in range(0, len(layout.indices), patch_batch_size):
            chunk_indices = layout.indices[chunk_start : chunk_start + patch_batch_size]
            image_batch = torch.cat(
                [
                    x[:, :, top:bottom, left:right]
                    for top, bottom, left, right in chunk_indices
                ],
                dim=0,
            )
            position_batch = None
            if positions is not None:
                position_batch = torch.cat(
                    [
                        positions[:, :, top:bottom, left:right]
                        for top, bottom, left, right in chunk_indices
                    ],
                    dim=0,
                )
            denoised_batch = self._edm_denoise_batch(
                image_batch,
                position_batch,
                sigma,
                params,
                use_checkpoint=use_checkpoint,
            )

            for offset, (top, bottom, left, right) in enumerate(chunk_indices):
                patch = denoised_batch[offset : offset + 1]
                if assembly == "fixed_average":
                    output[:, :, top:bottom, left:right] += patch
                    counts[:, :, top:bottom, left:right] += 1
                elif assembly == "fixed_stitch":
                    output[:, :, top:bottom, left:right] = patch
                else:
                    raise ValueError(
                        "patch_assembly must be 'padis', 'fixed_average', "
                        "or 'fixed_stitch'."
                    )

        if assembly == "fixed_average":
            output = torch.where(counts > 0, output / counts.clamp_min(1), x)

        if bool(getattr(params, "consume_denoise_output_noise", False)):
            _ = self._sample_noise(output, generator)
        pad = int(params.pad_width)
        if pad == 0:
            return output
        zero_border = torch.zeros_like(output)
        zero_border[
            :, :, pad : pad + layout.image_height, pad : pad + layout.image_width
        ] = output[
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
            patch_assembly = getattr(params, "patch_assembly", "padis")
            if patch_assembly not in ("padis", "fixed_average", "fixed_stitch"):
                raise ValueError(
                    "patch_assembly must be 'padis', 'fixed_average', or 'fixed_stitch'."
                )
            if patch_assembly in ("fixed_average", "fixed_stitch"):
                patch_overlap = int(getattr(params, "patch_overlap", 8))
                if patch_overlap < 0 or patch_overlap >= patch_size:
                    raise ValueError(
                        "patch_overlap must satisfy 0 <= patch_overlap < patch_size."
                    )
                fixed_overlap_layout = getattr(
                    params, "fixed_overlap_layout", "lion_clipped"
                )
                if fixed_overlap_layout not in (
                    "lion_clipped",
                    "public_overlap",
                    "public_tile",
                ):
                    raise ValueError(
                        "fixed_overlap_layout must be 'lion_clipped', "
                        "'public_overlap', or 'public_tile'."
                    )
        else:
            raise ValueError("prior_mode must be 'patch' or 'whole_image'.")

    def _denoise_prior(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        params,
        image_shape: tuple[int, int],
        generator: torch.Generator | None = None,
        *,
        layout_override: _PatchLayout | None = None,
    ) -> torch.Tensor:
        self._validate_prior_configuration(params, image_shape)
        prior_mode = getattr(params, "prior_mode", "patch")
        if prior_mode == "whole_image":
            if layout_override is not None:
                raise ValueError("layout_override is only valid for patch priors.")
            return self._denoise_whole_image(x, sigma, params)
        if prior_mode != "patch":
            raise ValueError("prior_mode must be 'patch' or 'whole_image'.")
        patch_assembly = getattr(params, "patch_assembly", "padis")
        if patch_assembly in ("fixed_average", "fixed_stitch"):
            if layout_override is not None:
                raise ValueError(
                    "layout_override is only valid for padis patch assembly."
                )
            layout = self._fixed_overlap_patch_layout(image_shape, params)
            return self._denoise_fixed_overlap_patches(
                x,
                sigma,
                layout,
                params,
                assembly=patch_assembly,
                generator=generator,
            )
        layout = (
            layout_override
            if layout_override is not None
            else self._patch_layout(image_shape, params, x.device, generator)
        )
        return self._denoise_patches(x, sigma, layout, params, generator)

    def _edm_denoise_batch(
        self,
        image_batch: torch.Tensor,
        position_batch: torch.Tensor | None,
        sigma: torch.Tensor,
        params,
        *,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
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
        if method not in ("operator_norm", "operator_lipschitz"):
            raise ValueError(
                "data_consistency_normalization must be 'operator_norm', "
                "'operator_lipschitz', or 'none'."
            )

        # The sampler state is in the diffusion model's normalized image units,
        # while the forward model may first map it to attenuation units. The
        # Lipschitz scale of that composed measurement map is |scale| * ||A||.
        normalizer = abs(float(params.measurement_scale)) * self._operator_norm(
            params, device
        )
        if method == "operator_lipschitz":
            normalizer = normalizer**2
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
        *,
        base_override: float | None = None,
    ) -> float:
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

    def _scheduled_adjoint_data_consistency_scale(
        self,
        params,
        sigma: torch.Tensor | None,
        device: torch.device,
    ) -> float:
        adjoint_scale = getattr(params, "adjoint_data_consistency_scale", None)
        if adjoint_scale is None:
            return self._scheduled_data_consistency_scale(params, sigma, device)
        return self._scheduled_data_consistency_scale(
            params, sigma, device, base_override=float(adjoint_scale)
        )

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
                forward_projected = self._forward_project(
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
                        denoised = self._denoise_prior(
                            x_current,
                            t_cur.reshape(1),
                            params,
                            tuple(x_init.shape[-2:]),
                            generator,
                        )
                        denoised = self._clip_model_range(denoised, params)
                else:
                    denoised = self._denoise_prior(
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
                        predicted = self._forward_project(
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
                    ) = self._dps_data_gradient(
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
        data_scale = self._scheduled_adjoint_data_consistency_scale(
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

    @staticmethod
    def _pc_corrector_step_size(
        noise: torch.Tensor,
        score: torch.Tensor,
        snr: float,
        rule: str = "paper_linear",
    ) -> torch.Tensor:
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

    def _predictor_corrector(
        self,
        measurement: torch.Tensor,
        params,
        *,
        prog_bar: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        x_init = self._initial_reconstruction(measurement, params).unsqueeze(0)
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
        t_steps = self._noise_schedule(params, x.device)
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
                    pc_layout = self._patch_layout(
                        tuple(x_init.shape[-2:]), params, x.device, generator
                    )
                denoise_kwargs = (
                    {"layout_override": pc_layout} if pc_layout is not None else {}
                )

                denoised = self._denoise_prior(
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
                    denoised = self._denoise_prior(
                        x,
                        corrector_sigma.reshape(1),
                        params,
                        tuple(x_init.shape[-2:]),
                        generator,
                        **denoise_kwargs,
                    )
                    denoised = self._clip_model_range(denoised, params)
                    score = (denoised - x) / t_next.square().clamp_min(1e-12)
                    eps = self._pc_corrector_step_size(
                        z,
                        score,
                        params.pc_snr,
                        getattr(params, "pc_corrector_step_rule", "paper_linear"),
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

    def _langevin(
        self,
        measurement: torch.Tensor,
        params,
        *,
        prog_bar: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        x_init = self._initial_reconstruction(measurement, params).unsqueeze(0)
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
        t_steps = self._noise_schedule(params, x.device)
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
                        backprojected = self._pseudoinverse_reconstruction(
                            measurement, params
                        )
                        projected_denoised = self._forward_project(denoised_crop)
                        corrected = (
                            backprojected
                            + denoised_crop
                            - self._pseudoinverse_reconstruction(
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
                        residual = measurement - self._forward_project(
                            self._crop(x, params).squeeze(0).to(torch.float32)
                        ).to(dtype=measurement.dtype)
                        step_size = self._adjoint_data_step_size(
                            residual, t_cur, params, public_repo_multiplier=True
                        )
                        (
                            projected,
                            correction,
                            raw_correction,
                            data_normalizer,
                            data_scale,
                        ) = self._apply_adjoint_correction(
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
