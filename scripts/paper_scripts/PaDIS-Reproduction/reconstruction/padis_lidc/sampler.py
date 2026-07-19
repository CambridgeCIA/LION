"""Sampler parameter construction for PaDIS LIDC reconstruction."""

from __future__ import annotations

import copy
import os
import random

import numpy as np
import torch

from LION.reconstructors import PaDIS
from LION.utils.parameter import LIONParameter

from padis_lidc.data import canonical_experiment_name, experiment_spec_from_args
from padis_lidc.experiments import LIDC_NORMAL_TO_MU_OFFSET, LIDC_NORMAL_TO_MU_SCALE


def set_run_seed(seed: int) -> None:
    """Set run seed."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_sampler_params(args, model, *, measurement_source: str) -> LIONParameter:
    """Build sampler params."""
    paper_views = args.paper_ct_views
    spec = experiment_spec_from_args(args)
    if spec is not None:
        paper_views = spec.paper_sampler_views

    if args.implementation == "paper":
        sampler_params = PaDIS.paper_ct_parameters(model, views=paper_views)
    elif args.implementation == "public_repo":
        sampler_params = PaDIS.padis_repo_ct_parameters(model)
    elif args.implementation == "lion_physics":
        sampler_params = PaDIS.lion_physics_ct_parameters(model, views=paper_views)
    elif args.implementation == "lion_quality":
        sampler_params = PaDIS.lion_quality_ct_parameters(model, views=paper_views)
    else:
        sampler_params = PaDIS.default_parameters(model)
    if args.implementation == "paper":
        paper_params = PaDIS.paper_ct_parameters(model, views=paper_views)
        sampler_params.num_steps = paper_params.num_steps
        sampler_params.inner_steps = paper_params.inner_steps
        sampler_params.sigma_min = paper_params.sigma_min
        sampler_params.sigma_max = paper_params.sigma_max
        sampler_params.noise_schedule = paper_params.noise_schedule
        sampler_params.zeta = paper_params.zeta
        sampler_params.initial_reconstruction = paper_params.initial_reconstruction
        sampler_params.clip_initial = paper_params.clip_initial
        sampler_params.clip_output = paper_params.clip_output
        sampler_params.dps_epsilon = paper_params.dps_epsilon
        sampler_params.sampling_epsilon = paper_params.sampling_epsilon
        sampler_params.data_consistency_gradient = (
            paper_params.data_consistency_gradient
        )
        sampler_params.adjoint_data_step_schedule = (
            paper_params.adjoint_data_step_schedule
        )
        if args.method == "padis_dps":
            sampler_params.zeta = 0.0075
            sampler_params.dps_epsilon = 0.5
        elif args.method == "langevin":
            sampler_params.zeta = 0.03
            sampler_params.sampling_epsilon = 0.5
        elif args.method == "predictor_corrector":
            sampler_params.zeta = 0.03
            sampler_params.pc_snr = 0.08
        elif args.method == "ve_ddnm":
            sampler_params.sampling_epsilon = 0.1
    elif args.implementation == "public_repo":
        public_params = PaDIS.padis_repo_ct_parameters(model)
        if args.public_repo_sigma_schedule == "paper":
            sigma_schedule_params = PaDIS.paper_ct_parameters(model, views=paper_views)
        else:
            sigma_schedule_params = public_params
        sampler_params.num_steps = public_params.num_steps
        sampler_params.inner_steps = public_params.inner_steps
        sampler_params.sigma_min = sigma_schedule_params.sigma_min
        sampler_params.sigma_max = sigma_schedule_params.sigma_max
        sampler_params.noise_schedule = sigma_schedule_params.noise_schedule
        sampler_params.noise_schedule_dtype = sigma_schedule_params.noise_schedule_dtype
        sampler_params.zeta = public_params.zeta
        sampler_params.initial_reconstruction = public_params.initial_reconstruction
        sampler_params.clip_initial = public_params.clip_initial
        sampler_params.clip_output = public_params.clip_output
        sampler_params.dps_epsilon = public_params.dps_epsilon
        sampler_params.sampling_epsilon = public_params.sampling_epsilon
        sampler_params.data_consistency_gradient = (
            public_params.data_consistency_gradient
        )
        sampler_params.adjoint_data_step_schedule = (
            public_params.adjoint_data_step_schedule
        )
        sampler_params.data_consistency_scale = public_params.data_consistency_scale
        sampler_params.adjoint_data_consistency_scale = (
            public_params.adjoint_data_consistency_scale
        )
        sampler_params.pc_corrector_denoise_sigma = (
            public_params.pc_corrector_denoise_sigma
        )
        sampler_params.pc_reuse_predictor_layout = (
            public_params.pc_reuse_predictor_layout
        )
        if args.method == "langevin":
            sampler_params.zeta = 0.2
            sampler_params.sampling_epsilon = 0.5
        elif args.method == "ve_ddnm":
            sampler_params.sampling_epsilon = 0.2
        if args.public_repo_helper_initialization and args.method in (
            "predictor_corrector",
            "langevin",
            "ve_ddnm",
        ):
            sampler_params.initial_reconstruction = "noise"
            sampler_params.noise_initialization = (
                "central_then_pad" if args.method == "predictor_corrector" else "padded"
            )
            sampler_params.initial_fdk_filter_type = None
            sampler_params.initial_fdk_frequency_scaling = 1.0
            sampler_params.initial_fdk_padded = True
            sampler_params.clip_initial = False
        if args.method == "padis_dps":
            sampler_params.zeta = 0.2
        elif args.method == "predictor_corrector":
            sampler_params.zeta = 0.5
    elif args.implementation == "lion_physics":
        physics_params = PaDIS.lion_physics_ct_parameters(model, views=paper_views)
        sampler_params.num_steps = physics_params.num_steps
        sampler_params.inner_steps = physics_params.inner_steps
        sampler_params.sigma_min = physics_params.sigma_min
        sampler_params.sigma_max = physics_params.sigma_max
        sampler_params.noise_schedule = physics_params.noise_schedule
        sampler_params.zeta = physics_params.zeta
        sampler_params.initial_reconstruction = physics_params.initial_reconstruction
        sampler_params.initial_fdk_filter_type = physics_params.initial_fdk_filter_type
        sampler_params.initial_fdk_frequency_scaling = (
            physics_params.initial_fdk_frequency_scaling
        )
        sampler_params.initial_fdk_padded = physics_params.initial_fdk_padded
        sampler_params.initial_fdk_batch_size = physics_params.initial_fdk_batch_size
        sampler_params.clip_initial = physics_params.clip_initial
        sampler_params.clip_output = physics_params.clip_output
        sampler_params.dps_epsilon = physics_params.dps_epsilon
        sampler_params.sampling_epsilon = physics_params.sampling_epsilon
        sampler_params.data_consistency_gradient = (
            physics_params.data_consistency_gradient
        )
        sampler_params.adjoint_data_step_schedule = (
            physics_params.adjoint_data_step_schedule
        )
        sampler_params.data_consistency_normalization = (
            physics_params.data_consistency_normalization
        )
        sampler_params.data_consistency_scale = physics_params.data_consistency_scale
        sampler_params.adjoint_data_consistency_scale = (
            physics_params.adjoint_data_consistency_scale
        )
    elif args.implementation == "lion_quality":
        quality_params = PaDIS.lion_quality_ct_parameters(model, views=paper_views)
        sampler_params.num_steps = quality_params.num_steps
        sampler_params.inner_steps = quality_params.inner_steps
        sampler_params.sigma_min = quality_params.sigma_min
        sampler_params.sigma_max = quality_params.sigma_max
        sampler_params.noise_schedule = quality_params.noise_schedule
        sampler_params.zeta = quality_params.zeta
        sampler_params.initial_reconstruction = quality_params.initial_reconstruction
        sampler_params.initial_fdk_filter_type = quality_params.initial_fdk_filter_type
        sampler_params.initial_fdk_frequency_scaling = (
            quality_params.initial_fdk_frequency_scaling
        )
        sampler_params.initial_fdk_padded = quality_params.initial_fdk_padded
        sampler_params.initial_fdk_batch_size = quality_params.initial_fdk_batch_size
        sampler_params.clip_initial = quality_params.clip_initial
        sampler_params.clip_output = quality_params.clip_output
        sampler_params.dps_epsilon = quality_params.dps_epsilon
        sampler_params.sampling_epsilon = quality_params.sampling_epsilon
        sampler_params.data_consistency_gradient = (
            quality_params.data_consistency_gradient
        )
        sampler_params.adjoint_data_step_schedule = (
            quality_params.adjoint_data_step_schedule
        )
        sampler_params.data_consistency_normalization = (
            quality_params.data_consistency_normalization
        )
        sampler_params.data_consistency_scale = quality_params.data_consistency_scale
    if args.implementation == "lion_physics":
        experiment_key = (
            spec.key if spec is not None else canonical_experiment_name(args.experiment)
        )
        if args.method == "predictor_corrector":
            sampler_params.zeta = 4.25
            sampler_params.pc_snr = 0.01
        elif args.method == "langevin":
            sampler_params.zeta = 4.0
            sampler_params.sampling_epsilon = 0.5
        elif args.method == "whole_image_diffusion":
            sampler_params.zeta = 4.0
            sampler_params.dps_epsilon = 0.5
        elif args.method == "padis_dps":
            sampler_params.zeta = 4.25
            sampler_params.dps_epsilon = 0.5
            sampler_params.initial_reconstruction = "noise"
            sampler_params.clip_initial = False
            sampler_params.clip_output = False
            sampler_params.initial_fdk_filter_type = None
            sampler_params.initial_fdk_frequency_scaling = 1.0
            sampler_params.initial_fdk_padded = True
        elif args.method in ("patch_average", "patch_stitch"):
            sampler_params.dps_epsilon = 0.5
    if args.method == "ve_ddnm":
        ve_ddnm_layout = args.ve_ddnm_nfe_layout
        if ve_ddnm_layout is None:
            ve_ddnm_layout = (
                "public_inner"
                if args.implementation == "public_repo"
                else "paper_1000x1"
            )
        if ve_ddnm_layout == "paper_1000x1":
            sampler_params.num_steps = 1000
            sampler_params.inner_steps = 1
        elif ve_ddnm_layout == "public_inner":
            sampler_params.num_steps = 100
            sampler_params.inner_steps = 10
        sampler_params.ve_ddnm_nfe_layout = ve_ddnm_layout
        if args.implementation in ("lion_physics", "lion_quality"):
            # LION fan-beam FDK pseudoinverses make strict paper VE-DDNM unstable;
            # these presets keep the NFE layout of Hu et al. but project the corrected
            # clean DDNM estimate back to the physically valid model support.
            sampler_params.initial_reconstruction = "noise"
            sampler_params.initial_fdk_filter_type = None
            sampler_params.initial_fdk_frequency_scaling = 1.0
            sampler_params.initial_fdk_padded = True
            sampler_params.clip_initial = False
            sampler_params.clip_output = False
            sampler_params.sampling_epsilon = 0.1
            sampler_params.ddnm_corrected_clip = True
    if args.initial_reconstruction is not None:
        sampler_params.initial_reconstruction = args.initial_reconstruction
        if args.initial_reconstruction == "noise":
            sampler_params.initial_fdk_filter_type = None
            sampler_params.initial_fdk_frequency_scaling = 1.0
            sampler_params.initial_fdk_padded = True
    if args.initial_fdk_filter_type is not None:
        sampler_params.initial_fdk_filter_type = (
            None
            if args.initial_fdk_filter_type == "none"
            else args.initial_fdk_filter_type
        )
    if args.initial_fdk_frequency_scaling is not None:
        sampler_params.initial_fdk_frequency_scaling = (
            args.initial_fdk_frequency_scaling
        )
    if args.initial_fdk_padded is not None:
        sampler_params.initial_fdk_padded = args.initial_fdk_padded
    if args.initial_fdk_batch_size is not None:
        sampler_params.initial_fdk_batch_size = args.initial_fdk_batch_size
    sampler_params.patch_batch_size = args.patch_batch_size
    sampler_params.langevin_ddnm = args.langevin_ddnm or args.method == "ve_ddnm"
    sampler_params.langevin_noise_scale = args.langevin_noise_scale
    sampler_params.pc_corrector_step_rule = args.pc_corrector_step_rule
    if args.pc_snr is not None:
        sampler_params.pc_snr = args.pc_snr
    if args.pc_corrector_denoise_sigma is not None:
        sampler_params.pc_corrector_denoise_sigma = args.pc_corrector_denoise_sigma
    if args.pc_reuse_predictor_layout is not None:
        sampler_params.pc_reuse_predictor_layout = args.pc_reuse_predictor_layout
    if args.method == "ve_ddnm":
        sampler_params.ddnm_pseudoinverse_clip = True
        sampler_params.ddnm_projected_pseudoinverse_clip = True
    if args.ddnm_pseudoinverse_clip is not None:
        sampler_params.ddnm_pseudoinverse_clip = args.ddnm_pseudoinverse_clip
    if args.ddnm_projected_pseudoinverse_clip is not None:
        sampler_params.ddnm_projected_pseudoinverse_clip = (
            args.ddnm_projected_pseudoinverse_clip
        )
    if args.ddnm_corrected_clip is not None:
        sampler_params.ddnm_corrected_clip = args.ddnm_corrected_clip
    if args.clip_initial is not None:
        sampler_params.clip_initial = args.clip_initial
    if args.clip_output is not None:
        sampler_params.clip_output = args.clip_output
    if args.num_steps is not None:
        sampler_params.num_steps = args.num_steps
    if args.inner_steps is not None:
        sampler_params.inner_steps = args.inner_steps
    if args.sigma_min is not None:
        sampler_params.sigma_min = args.sigma_min
    if args.sigma_max is not None:
        sampler_params.sigma_max = args.sigma_max
    if args.rho is not None:
        sampler_params.rho = args.rho
    if args.dps_epsilon is not None:
        sampler_params.dps_epsilon = args.dps_epsilon
    if args.sampling_epsilon is not None:
        sampler_params.sampling_epsilon = args.sampling_epsilon
    if args.zeta is not None:
        sampler_params.zeta = args.zeta
    if args.noise_schedule is not None:
        sampler_params.noise_schedule = args.noise_schedule
    if args.data_consistency_gradient is not None:
        sampler_params.data_consistency_gradient = args.data_consistency_gradient
    if args.adjoint_data_step_schedule is not None:
        sampler_params.adjoint_data_step_schedule = args.adjoint_data_step_schedule
    sampler_params.clip_denoised = args.clip_denoised
    sampler_params.clip_state = args.clip_state
    sampler_params.disable_data_consistency = args.disable_data_consistency
    sampler_params.disable_langevin_noise = args.disable_langevin_noise
    sampler_params.disable_prior_score = args.disable_prior_score
    if args.data_consistency_normalization is not None:
        sampler_params.data_consistency_normalization = (
            args.data_consistency_normalization
        )
    if args.data_consistency_scale is not None:
        sampler_params.data_consistency_scale = args.data_consistency_scale
    if args.adjoint_data_consistency_scale is not None:
        sampler_params.adjoint_data_consistency_scale = (
            args.adjoint_data_consistency_scale
        )
    if args.consume_discarded_measurement_noise is not None:
        sampler_params.consume_discarded_measurement_noise = (
            args.consume_discarded_measurement_noise
        )
    sampler_params.data_consistency_scale_schedule = (
        args.data_consistency_scale_schedule
    )
    sampler_params.data_consistency_scale_power = args.data_consistency_scale_power
    sampler_params.data_consistency_scale_floor = args.data_consistency_scale_floor
    sampler_params.operator_norm = args.operator_norm
    sampler_params.operator_norm_iterations = args.operator_norm_iterations
    sampler_params.operator_norm_tolerance = args.operator_norm_tolerance
    sampler_params.trace_interval = args.trace_interval
    sampler_params.trace_images = args.trace_images
    sampler_params.stop_after_outer_steps = args.stop_after_outer_steps
    if args.prior_mode != "auto":
        sampler_params.prior_mode = (
            "whole_image" if args.prior_mode == "whole-image" else "patch"
        )
    if args.method == "whole_image_diffusion":
        sampler_params.prior_mode = "whole_image"
    elif args.method in ("patch_average", "patch_stitch"):
        sampler_params.prior_mode = "patch"
    if measurement_source == "reconstruction":
        sampler_params.measurement_scale = LIDC_NORMAL_TO_MU_SCALE
        sampler_params.measurement_offset = LIDC_NORMAL_TO_MU_OFFSET
    if args.patch_size is not None:
        sampler_params.patch_size = args.patch_size
    if args.pad_width is not None:
        sampler_params.pad_width = args.pad_width
    if args.patch_assembly is not None:
        sampler_params.patch_assembly = args.patch_assembly
    if args.experiment == "ct_512_60" and args.method in (
        "padis_dps",
        "langevin",
        "predictor_corrector",
        "ve_ddnm",
        "patch_average",
        "patch_stitch",
    ):
        # Memory-only control for the 512 paper row. Patch-prior methods can
        # otherwise materialize more denoiser inputs than local/A100 jobs need.
        if args.patch_batch_size is None:
            sampler_params.patch_batch_size = 1
        if (
            args.method not in ("patch_average", "patch_stitch")
            and args.patch_checkpoint_denoiser is None
        ):
            sampler_params.patch_checkpoint_denoiser = True
    if args.method == "patch_average":
        sampler_params.patch_assembly = "fixed_average"
        sampler_params.fixed_overlap_checkpoint_denoiser = True
        if args.patch_batch_size is None:
            sampler_params.patch_batch_size = 1
        if args.implementation in ("public_repo", "lion_physics"):
            sampler_params.fixed_overlap_layout = "public_overlap"
    elif args.method == "patch_stitch":
        sampler_params.patch_assembly = "fixed_stitch"
        sampler_params.fixed_overlap_checkpoint_denoiser = True
        if args.patch_batch_size is None:
            sampler_params.patch_batch_size = 1
        if args.implementation in ("public_repo", "lion_physics"):
            sampler_params.fixed_overlap_layout = "public_tile"
    if args.patch_overlap is not None:
        sampler_params.patch_overlap = args.patch_overlap
    if args.fixed_overlap_layout is not None:
        sampler_params.fixed_overlap_layout = args.fixed_overlap_layout
    if args.fixed_overlap_checkpoint_denoiser is not None:
        sampler_params.fixed_overlap_checkpoint_denoiser = (
            args.fixed_overlap_checkpoint_denoiser
        )
    if args.patch_checkpoint_denoiser is not None:
        sampler_params.patch_checkpoint_denoiser = args.patch_checkpoint_denoiser
    return sampler_params


def clone_parameters(params: LIONParameter) -> LIONParameter:
    """Clone parameters."""
    copied = LIONParameter()
    for key, value in params.__dict__.items():
        if not key.startswith("_"):
            setattr(copied, key, copy.deepcopy(value))
    return copied
