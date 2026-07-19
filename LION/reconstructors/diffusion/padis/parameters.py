"""Parameter presets and validation for the PaDIS reconstructor."""

from __future__ import annotations

from LION.models.diffusion import NCSNpp
from LION.utils.parameter import LIONParameter

PUBLIC_REPO_CT_GRADIENT_SCALE = 0.0405
PUBLIC_REPO_CT_ADJOINT_SCALE = 0.1022


class PaDISParameters:
    """Define PaDIS parameter presets, merging, and validation."""

    @staticmethod
    def default_parameters(model: NCSNpp | None = None) -> LIONParameter:
        """Return conservative generic sampler parameters.

        Study-specific reconstruction should normally start from one of the CT
        preset constructors rather than these diagnostic defaults.
        """
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
        params = PaDISParameters.default_parameters(model)
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
        params = PaDISParameters.paper_ct_parameters(model, views=views)
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
        params.pc_snr = 0.01
        params.patch_offset_rng = "torch"
        params.consume_discarded_measurement_noise = False
        params.consume_discarded_initial_noise = False
        params.consume_unused_latents = False
        params.consume_denoise_output_noise = False
        return params

    @staticmethod
    def padis_repo_ct_parameters(model: NCSNpp | None = None) -> LIONParameter:
        """Return the public PaDIS CT-script sampler settings for compatibility."""
        params = PaDISParameters.default_parameters(model)
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
        params = PaDISParameters.paper_ct_parameters(model, views=views)
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
