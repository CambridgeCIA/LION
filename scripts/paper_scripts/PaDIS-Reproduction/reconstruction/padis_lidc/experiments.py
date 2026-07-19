"""Experiment definitions and constants for PaDIS LIDC reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import pathlib

import LION.experiments.ct_experiments as ct_experiments
from LION.utils.paths import LION_EXPERIMENTS_PATH  # noqa: F401

DEFAULT_CHECKPOINT = pathlib.Path(
    "Data/experiments/PaDIS/LIDC_256/"
    "padis_lidc_256_reproduction_CSD3/padis_lidc_256.pt"
)

LIDC_EXPERIMENTS = {
    "PaDISFanBeam8CTRecon": ct_experiments.PaDISFanBeam8CTRecon,
    "PaDISFanBeam20CTRecon": ct_experiments.PaDISFanBeam20CTRecon,
    "PaDISFanBeam60CTRecon": ct_experiments.PaDISFanBeam60CTRecon,
    "PaDISFanBeam120LimitedCTRecon": (ct_experiments.PaDISFanBeam120LimitedCTRecon),
    "PaDISFanBeam180CTRecon": ct_experiments.PaDISFanBeam180CTRecon,
    "clinicalCTRecon": ct_experiments.clinicalCTRecon,
    "LowDoseCTRecon": ct_experiments.LowDoseCTRecon,
    "ExtremeLowDoseCTRecon": ct_experiments.ExtremeLowDoseCTRecon,
    "LimitedAngleCTRecon": ct_experiments.LimitedAngleCTRecon,
    "LimitedAngleLowDoseCTRecon": ct_experiments.LimitedAngleLowDoseCTRecon,
    "LimitedAngleExtremeLowDoseCTRecon": (
        ct_experiments.LimitedAngleExtremeLowDoseCTRecon
    ),
    "SparseAngleCTRecon": ct_experiments.SparseAngleCTRecon,
    "SparseAngleLowDoseCTRecon": ct_experiments.SparseAngleLowDoseCTRecon,
    "SparseAngleExtremeLowDoseCTRecon": (
        ct_experiments.SparseAngleExtremeLowDoseCTRecon
    ),
}


@dataclass(frozen=True)
class PaperCTExperiment:
    """Resolved acquisition settings for one study CT experiment."""

    key: str
    views: int
    paper_geometry: str
    lion_experiment: str
    paper_sampler_views: int
    description: str


PAPER_CT_EXPERIMENTS = {
    "ct_8": PaperCTExperiment(
        key="ct_8",
        views=8,
        paper_geometry="parallel",
        lion_experiment="PaDISFanBeam8CTRecon",
        paper_sampler_views=8,
        description="8-view CT experiment from Hu et al.",
    ),
    "ct_20": PaperCTExperiment(
        key="ct_20",
        views=20,
        paper_geometry="parallel",
        lion_experiment="PaDISFanBeam20CTRecon",
        paper_sampler_views=20,
        description="20-view CT experiment from Hu et al.",
    ),
    "ct_60": PaperCTExperiment(
        key="ct_60",
        views=60,
        paper_geometry="parallel",
        lion_experiment="PaDISFanBeam60CTRecon",
        paper_sampler_views=20,
        description="60-view CT experiment from the extra experiments of Hu et al.",
    ),
    "ct_20_limited_angle_120": PaperCTExperiment(
        key="ct_20_limited_angle_120",
        views=20,
        paper_geometry="fan",
        lion_experiment="PaDISFanBeam120LimitedCTRecon",
        paper_sampler_views=20,
        description=(
            "20-view, 120-degree limited-angle fan-beam CT stress experiment."
        ),
    ),
    "ct_512_60": PaperCTExperiment(
        key="ct_512_60",
        views=60,
        paper_geometry="parallel",
        lion_experiment="PaDISFanBeam60CTRecon",
        paper_sampler_views=20,
        description="512x512 60-view CT experiment from Hu et al.",
    ),
}

EXPERIMENT_ALIASES = {
    "8": "ct_8",
    "20": "ct_20",
    "60": "ct_60",
    "180": "ct_20_limited_angle_120",
    "fanbeam_180": "ct_20_limited_angle_120",
    "ct_fan_180": "ct_20_limited_angle_120",
    "ct_fanbeam_180": "ct_20_limited_angle_120",
    "fanbeam_120": "ct_20_limited_angle_120",
    "ct_fan_120": "ct_20_limited_angle_120",
    "512_60": "ct_512_60",
    "PaDISFanBeam8CTRecon": "ct_8",
    "PaDISFanBeam20CTRecon": "ct_20",
    "PaDISFanBeam60CTRecon": "ct_60",
    "PaDISFanBeam120LimitedCTRecon": "ct_20_limited_angle_120",
    "PaDISFanBeam180CTRecon": "ct_20_limited_angle_120",
}

IMPLEMENTATION_CHOICES = (
    "custom",
    "public_repo",
    "paper",
    "lion_physics",
    "lion_quality",
)
GEOMETRY_CHOICES = ("lion", "padis", "padis_parallel", "padis_fanbeam")
RECONSTRUCTION_METHOD_CHOICES = (
    "padis_dps",
    "baseline",
    "cp_tv",
    "admm_tv",
    "pnp_admm",
    "whole_image_diffusion",
    "langevin",
    "predictor_corrector",
    "ve_ddnm",
    "patch_average",
    "patch_stitch",
)
DIFFUSION_RECONSTRUCTION_METHODS = {
    "padis_dps",
    "whole_image_diffusion",
    "langevin",
    "predictor_corrector",
    "ve_ddnm",
    "patch_average",
    "patch_stitch",
}
NO_PADIS_PRIOR_METHODS = {"baseline", "cp_tv", "pnp_admm"}
PUBLIC_REPO_IMPLEMENTATION_METHODS = {
    "padis_dps",
    "langevin",
    "predictor_corrector",
    "ve_ddnm",
    "patch_average",
    "patch_stitch",
}
UNSUPPORTED_PADIS_GEOMETRY_MESSAGE = (
    "PaDIS geometry is intentionally not implemented for LIDC-IDRI. The "
    "processed LIDC slices used by these scripts are saved as 512x512 HU arrays "
    "without the per-scan pixel spacing/orientation needed to resample them into "
    "the PaDIS public-repo coordinate system. The public PaDIS CT operators use "
    "a 40-unit image support and 80-unit detector span, while the LION LIDC CT "
    "setup uses a 300 mm field of view with detector size 900, DSO 575 mm, and "
    "DSD 1050 mm. Treating those as interchangeable would not be a physically "
    "correct detector/object transformation. Use --geometry lion, or provide a "
    "metadata-preserving dataset and a derived physical resampling model before "
    "adding PaDIS geometry."
)

LIDC_NORMAL_TO_MU_SCALE = 2.0 * (1.52 - 0.0012)
LIDC_NORMAL_TO_MU_OFFSET = 0.0012

ABLATION_VARIANTS = {
    "baseline": {},
    "noise_scale_0_25": {"langevin_noise_scale": 0.25},
    "noise_scale_0_10": {"langevin_noise_scale": 0.10},
    "no_data_consistency": {"disable_data_consistency": True},
    "no_langevin_noise": {"disable_langevin_noise": True},
    "no_prior_score": {"disable_prior_score": True},
}
