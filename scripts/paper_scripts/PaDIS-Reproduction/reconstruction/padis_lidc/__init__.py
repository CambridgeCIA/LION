"""Internal components for the PaDIS LIDC reconstruction entry point."""

from padis_lidc.checkpoints import (  # noqa: F401
    PnPDenoiser,
    _checkpoint_geometry,
    _checkpoint_paper_preset,
    checkpoint_model_metadata,
    fallback_metadata,
    load_checkpoint_metadata,
    load_model,
    load_pnp_denoiser,
    project_root,
    resolve_checkpoint_path,
    torch_load,
)
from padis_lidc.cli import build_arg_parser  # noqa: F401
from padis_lidc.data import (  # noqa: F401
    PNGImagePriorDataset,
    build_dataset,
    build_experiment_dataset,
    canonical_experiment_name,
    experiment_class_for_geometry,
    experiment_spec_from_args,
    make_measurement,
    mu_to_lidc_normal,
    validate_public_repo_method,
)
from padis_lidc.experiments import (  # noqa: F401
    ABLATION_VARIANTS,
    DEFAULT_CHECKPOINT,
    DIFFUSION_RECONSTRUCTION_METHODS,
    EXPERIMENT_ALIASES,
    GEOMETRY_CHOICES,
    IMPLEMENTATION_CHOICES,
    LIDC_EXPERIMENTS,
    LIDC_NORMAL_TO_MU_OFFSET,
    LIDC_NORMAL_TO_MU_SCALE,
    LION_EXPERIMENTS_PATH,
    NO_PADIS_PRIOR_METHODS,
    PAPER_CT_EXPERIMENTS,
    PUBLIC_REPO_IMPLEMENTATION_METHODS,
    RECONSTRUCTION_METHOD_CHOICES,
    UNSUPPORTED_PADIS_GEOMETRY_MESSAGE,
    PaperCTExperiment,
)
from padis_lidc.metrics import (  # noqa: F401
    add_ddnm_pseudoinverse_diagnostics,
    add_image_similarity_metrics,
    add_reconstruction_metrics,
    crop_bbox,
    edge_ssim_or_none,
    forward_project_normal_image,
    hu_window,
    image_tensor_from_array,
    mask_bbox,
    masked_mae,
    masked_mse,
    mean_metric,
    min_metric,
    normal_to_hu,
    psnr,
    psnr_from_mse,
    relative_sinogram_residual,
    ssim_on_bbox_or_none,
    ssim_or_none,
)
from padis_lidc.quality import enforce_quality_gates  # noqa: F401
from padis_lidc.sampler import (  # noqa: F401
    build_sampler_params,
    clone_parameters,
    set_run_seed,
)
from padis_lidc.visualization import (  # noqa: F401
    save_preview,
    save_tensor_image,
    save_trace_images,
    save_visual_comparison,
)
