"""Run PaDIS DPS or Langevin CT reconstruction on LIDC-IDRI splits."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
import os
import pathlib
import random
import warnings

_CACHE_ROOT = pathlib.Path("/tmp") / "lion_matplotlib_cache"
(_CACHE_ROOT / "mpl").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))
os.environ["MPLBACKEND"] = os.environ.get("PADIS_MPLBACKEND", "Agg")

import torch
import numpy as np
import PIL.Image
from tqdm import tqdm

from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import sinogram_add_noise
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
import LION.experiments.ct_experiments as ct_experiments
from LION.classical_algorithms.fdk import fdk
from LION.classical_algorithms.tv_min import tv_min
from LION.models.CNNs.drunet import DRUNet
from LION.models.LIONmodel import LIONModelParameter
from LION.models.diffusion import NCSNpp
from LION.reconstructors import PaDIS
from LION.reconstructors.PnP import PnP
from LION.utils.parameter import LIONParameter
from LION.utils.paths import LION_EXPERIMENTS_PATH

from PaDIS_identifiers import canonical_method


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


class PNGImagePriorDataset(torch.utils.data.Dataset):
    """Image-prior dataset backed by PaDIS-style PNG slices."""

    def __init__(self, image_dir: pathlib.Path, channels: int):
        """Initialize the instance."""
        self.image_dir = pathlib.Path(image_dir).expanduser()
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"PNG image directory not found: {self.image_dir}")
        self.channels = int(channels)
        self.files = sorted(
            path for path in self.image_dir.iterdir() if path.suffix.lower() == ".png"
        )
        if not self.files:
            raise FileNotFoundError(f"No PNG files found in {self.image_dir}")

    def __len__(self):
        """Return the number of available items."""
        return len(self.files)

    def __getitem__(self, index):
        """Return the item at the requested index."""
        image = np.asarray(PIL.Image.open(self.files[index]), dtype=np.float32) / 255.0
        if self.channels == 1:
            if image.ndim == 3:
                image = image[..., 0]
            image = image[None, :, :]
        elif self.channels == 3:
            if image.ndim == 2:
                image = np.repeat(image[..., None], 3, axis=-1)
            image = np.transpose(image, (2, 0, 1))
        else:
            raise ValueError("PNGImagePriorDataset supports 1 or 3 channels.")
        return torch.empty(0), torch.from_numpy(image.copy()).float()


def project_root() -> pathlib.Path:
    """Return the project root."""
    return pathlib.Path(__file__).resolve().parents[3]


def torch_load(path: pathlib.Path, map_location):
    """Load a PyTorch payload from load."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _checkpoint_geometry(checkpoint, *, image_scaling: float) -> Geometry:
    """Handle checkpoint geometry for the PaDIS workflow."""
    if isinstance(checkpoint, dict) and "geometry" in checkpoint:
        return Geometry.init_from_parameter(checkpoint["geometry"])
    return Geometry.default_parameters(image_scaling=image_scaling)


def _checkpoint_paper_preset(checkpoint) -> str | None:
    """Handle checkpoint paper preset for the PaDIS workflow."""
    if not isinstance(checkpoint, dict):
        return None
    training_params = checkpoint.get("training_params")
    if training_params is None:
        return None
    paper_preset = getattr(training_params, "paper_preset", None)
    if isinstance(paper_preset, str) and paper_preset:
        return paper_preset
    prior_mode = getattr(training_params, "prior_mode", None)
    if prior_mode == "whole_image":
        return "padis-paper-whole-ct-256"
    patch_sizes = getattr(training_params, "patch_sizes", None)
    if patch_sizes:
        largest_patch_size = max(int(size) for size in patch_sizes)
        if largest_patch_size == 64:
            return "padis-paper-ct-512"
        if largest_patch_size in (8, 16, 32, 56, 96):
            preset = (
                "padis-paper-ct-256"
                if largest_patch_size == 56
                else f"padis-paper-ct-p{largest_patch_size}"
            )
            if getattr(training_params, "use_position_channels", True) is False:
                preset = f"{preset}-no-position"
            return preset
    return None


def checkpoint_model_metadata(
    checkpoint_path: pathlib.Path,
    *,
    map_location,
    image_scaling: float,
    disable_position_channels: bool,
    checkpoint=None,
):
    """Handle checkpoint model metadata for the PaDIS workflow."""
    json_path = checkpoint_path.with_suffix(".json")
    if json_path.is_file():
        options = LIONParameter()
        options.load(json_path)
        if getattr(options, "model_name", "NCSNpp") != "NCSNpp":
            warnings.warn(
                f"{json_path} says model_name={options.model_name!r}; trying NCSNpp anyway."
            )
        # LIONParameter.load() intentionally reconstructs nested JSON objects as
        # generic LIONParameter instances. NCSNpp now enforces the more specific
        # LIONModelParameter type, so restore that type at this serialization
        # boundary while preserving every saved field.
        model_params = LIONModelParameter(
            **copy.deepcopy(vars(options.model_parameters))
        )
        geometry = Geometry.init_from_parameter(options.geometry)
    else:
        if checkpoint is None:
            checkpoint = torch_load(checkpoint_path, map_location=map_location)
        paper_preset = _checkpoint_paper_preset(checkpoint)
        if paper_preset is None:
            warnings.warn(
                f"No sidecar JSON found at {json_path}; using PaDIS LIDC 256 defaults."
            )
            paper_preset = "padis-paper-ct-256"
        else:
            warnings.warn(
                f"No sidecar JSON found at {json_path}; inferred {paper_preset!r} "
                "from checkpoint metadata."
            )
        model_params = NCSNpp.default_parameters(paper_preset)
        geometry = _checkpoint_geometry(checkpoint, image_scaling=image_scaling)

    if disable_position_channels:
        model_params.input_position_channels = 0
    return model_params, geometry, checkpoint


def resolve_checkpoint_path(path: pathlib.Path) -> pathlib.Path:
    """Resolve checkpoint path."""
    path = path.expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((pathlib.Path.cwd() / path, project_root() / path))

    if path.parts[:1] == ("Data",):
        data_root = os.environ.get("LION_DATA_PATH")
        if data_root is not None:
            candidates.append(
                pathlib.Path(data_root).expanduser() / pathlib.Path(*path.parts[1:])
            )

    if path == DEFAULT_CHECKPOINT:
        candidates.append(
            LION_EXPERIMENTS_PATH
            / "PaDIS/LIDC_256/padis_lidc_256_reproduction_CSD3/padis_lidc_256.pt"
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    tried = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Checkpoint not found. Tried:\n  {tried}")


def load_model(
    checkpoint_path: pathlib.Path,
    device: torch.device,
    use_ema: bool,
    disable_position_channels: bool,
):
    """Load model."""
    checkpoint = torch_load(checkpoint_path, map_location=device)
    model_params, geometry, checkpoint = checkpoint_model_metadata(
        checkpoint_path,
        map_location=device,
        image_scaling=0.5,
        disable_position_channels=disable_position_channels,
        checkpoint=checkpoint,
    )

    model = NCSNpp(model_params, geometry).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    ema_state = (
        checkpoint.get("ema_state_dict") if isinstance(checkpoint, dict) else None
    )
    ema_path = checkpoint_path.with_suffix(".ema.pt")
    if ema_state is None and ema_path.is_file():
        ema_checkpoint = torch_load(ema_path, map_location=device)
        ema_state = ema_checkpoint.get("ema_state_dict")
    if use_ema and ema_state is not None:
        state_dict = dict(state_dict)
        state_dict.update(ema_state)
        print("Loaded EMA weights for PaDIS reconstruction.")
    model.load_state_dict(state_dict)
    model.eval()
    return model, model_params, geometry


def load_checkpoint_metadata(
    checkpoint_path: pathlib.Path,
    *,
    image_scaling: float,
    disable_position_channels: bool,
):
    """Load checkpoint metadata."""
    model_params, geometry, _ = checkpoint_model_metadata(
        checkpoint_path,
        map_location="cpu",
        image_scaling=image_scaling,
        disable_position_channels=disable_position_channels,
    )
    return model_params, geometry


def fallback_metadata(
    *,
    image_scaling: float,
    disable_position_channels: bool,
    paper_preset: str = "padis-paper-ct-256",
):
    """Build fallback metadata."""
    model_params = NCSNpp.default_parameters(paper_preset)
    if disable_position_channels:
        model_params.input_position_channels = 0
    return model_params, Geometry.default_parameters(image_scaling=image_scaling)


class PnPDenoiser(torch.nn.Module):
    """Batch/dataset-normalising wrapper for LION image denoisers used by PnP."""

    def __init__(self, model: torch.nn.Module, noise_level: float | None = None):
        """Initialize the instance."""
        super().__init__()
        self.model = model
        self.noise_level = noise_level

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the forward operation to the requested values."""
        squeeze_batch = image.dim() == 3
        if squeeze_batch:
            image = image.unsqueeze(0)
        normalised = (
            self.model.normalise(image) if hasattr(self.model, "normalise") else image
        )
        model_params = getattr(self.model, "model_parameters", None)
        if bool(getattr(model_params, "use_noise_level", False)):
            noise_level = 0.0 if self.noise_level is None else float(self.noise_level)
            denoised = self.model(normalised, noise_level=noise_level)
        else:
            denoised = self.model(normalised)
        if hasattr(self.model, "unnormalise"):
            denoised = self.model.unnormalise(denoised)
        return denoised.squeeze(0) if squeeze_batch else denoised


def load_pnp_denoiser(
    checkpoint_path: pathlib.Path,
    device: torch.device,
    *,
    noise_level: float | None,
) -> PnPDenoiser:
    """Load pnp denoiser."""
    checkpoint_path = pathlib.Path(checkpoint_path)
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    options = LIONParameter()
    options.load(checkpoint_path.with_suffix(".json"))
    model_name = getattr(options, "model_name", "DRUNet")
    if model_name != "DRUNet":
        raise ValueError(
            f"Only DRUNet PnP denoiser checkpoints are supported here; got {model_name!r}."
        )
    model_parameters = options.model_parameters
    if not isinstance(model_parameters, LIONModelParameter):
        model_parameters = LIONModelParameter(**model_parameters.serialize())
    model = DRUNet(model_parameters).to(device)
    payload = torch_load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise KeyError(f"{checkpoint_path} does not contain a model_state_dict.")
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return PnPDenoiser(model, noise_level=noise_level).to(device).eval()


def build_dataset(args, geometry):
    """Build dataset."""
    task = "image_prior" if args.measurement_source == "normal" else "reconstruction"
    data_params = LIDC_IDRI.default_parameters(geometry=geometry, task=task)
    data_params.device = torch.device("cpu")
    if args.data_folder is not None:
        data_params.folder = args.data_folder
    return LIDC_IDRI(args.split, parameters=data_params, geometry_parameters=geometry)


def canonical_experiment_name(name: str) -> str:
    """Return the canonical experiment name."""
    return EXPERIMENT_ALIASES.get(name, name)


def validate_public_repo_method(implementation: str, method: str) -> None:
    """Validate public repo method."""
    if (
        implementation == "public_repo"
        and method not in PUBLIC_REPO_IMPLEMENTATION_METHODS
    ):
        supported = ", ".join(sorted(PUBLIC_REPO_IMPLEMENTATION_METHODS))
        raise ValueError(
            "--implementation public_repo is only supported for methods with "
            "a public PaDIS inverse-sampler analogue: "
            f"{supported}. Method {method!r} has no runnable public-repo "
            "equivalent."
        )


def experiment_spec_from_args(args) -> PaperCTExperiment | None:
    """Handle experiment spec from args for the PaDIS workflow."""
    canonical = canonical_experiment_name(args.experiment)
    return PAPER_CT_EXPERIMENTS.get(canonical)


def experiment_class_for_geometry(
    spec: PaperCTExperiment,
    geometry_tag: str,
) -> type:
    """Handle experiment class for geometry for the PaDIS workflow."""
    if geometry_tag == "lion":
        return LIDC_EXPERIMENTS[spec.lion_experiment]
    if geometry_tag in ("padis", "padis_parallel", "padis_fanbeam"):
        raise ValueError(UNSUPPORTED_PADIS_GEOMETRY_MESSAGE)
    raise ValueError(f"Unknown geometry tag: {geometry_tag}")


def build_experiment_dataset(args, checkpoint_geometry):
    """Build experiment dataset."""
    image_scaling = float(getattr(checkpoint_geometry, "image_scaling", 1.0))
    spec = experiment_spec_from_args(args)
    if spec is None:
        experiment_cls = LIDC_EXPERIMENTS[args.experiment]
    else:
        experiment_cls = experiment_class_for_geometry(spec, args.geometry)
    experiment = experiment_cls(
        dataset="LIDC-IDRI",
        datafolder=args.data_folder,
        image_scaling=image_scaling,
    )
    if args.split == "validation":
        dataset = experiment.get_validation_dataset()
    elif args.split == "test":
        dataset = experiment.get_testing_dataset()
    else:
        raise ValueError(f"Unsupported split for reconstruction: {args.split}")
    return dataset, experiment.geometry, experiment


def mu_to_lidc_normal(image: torch.Tensor) -> torch.Tensor:
    """Handle mu to lidc normal for the PaDIS workflow."""
    return ((image - LIDC_NORMAL_TO_MU_OFFSET) / LIDC_NORMAL_TO_MU_SCALE).clamp(
        0.0, 1.0
    )


def make_measurement(
    args,
    dataset,
    index,
    reconstructor,
    device,
    *,
    from_experiment,
    experiment_measurement_source,
):
    """Create measurement."""
    sample = dataset[index]
    if from_experiment:
        if experiment_measurement_source == "normal":
            target = sample[1].float().to(device)
            sinogram = reconstructor.op(target)
        else:
            sinogram, target = sample
            sinogram = sinogram.float().to(device)
            target = mu_to_lidc_normal(target.float().to(device))
        return sinogram, target

    if args.measurement_source == "normal":
        target = sample[1].float().to(device)
        sinogram = reconstructor.op(target)
    else:
        sinogram, target = sample
        sinogram = sinogram.float().to(device)
        target = mu_to_lidc_normal(target.float().to(device))

    if args.noise == "low-dose":
        if sinogram.device.type != "cuda":
            raise ValueError("LION low-dose sinogram noise currently requires CUDA.")
        sinogram = sinogram_add_noise(
            sinogram,
            I0=args.noise_i0,
            sigma=args.noise_sigma,
            cross_talk=args.noise_cross_talk,
            enable_gradients=False,
        )
    return sinogram, target


def psnr(recon: torch.Tensor, target: torch.Tensor, data_range: float) -> float:
    """Handle psnr for the PaDIS workflow."""
    mse = torch.mean((recon - target).square()).item()
    if mse == 0:
        return float("inf")
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse))


def psnr_from_mse(mse: float, data_range: float) -> float:
    """Handle psnr from mse for the PaDIS workflow."""
    if mse == 0:
        return float("inf")
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse))


def mean_metric(metrics: list[dict], key: str):
    """Return the mean metric."""
    values = [item[key] for item in metrics if key in item]
    if not values:
        return None
    return sum(values) / len(values)


def min_metric(metrics: list[dict], key: str):
    """Return the minimum metric."""
    values = [item[key] for item in metrics if key in item]
    if not values:
        return None
    return min(values)


def masked_mse(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Handle masked mse for the PaDIS workflow."""
    mask = mask.to(device=recon.device, dtype=torch.bool)
    if not torch.any(mask):
        return None
    return float(torch.mean((recon[mask] - target[mask]).square()).item())


def masked_mae(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Handle masked mae for the PaDIS workflow."""
    mask = mask.to(device=recon.device, dtype=torch.bool)
    if not torch.any(mask):
        return None
    return float(torch.mean(torch.abs(recon[mask] - target[mask])).item())


def ssim_or_none(recon: torch.Tensor, target: torch.Tensor, data_range: float):
    """Calculate SSIM for or none."""
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return None
    recon_np = recon.detach().cpu().squeeze().numpy()
    target_np = target.detach().cpu().squeeze().numpy()
    return float(structural_similarity(target_np, recon_np, data_range=data_range))


def normal_to_hu(image: torch.Tensor) -> torch.Tensor:
    """Convert normalised intensity to to hu."""
    return 3000.0 * image - 1000.0


def hu_window(image: torch.Tensor, *, level: float, width: float) -> torch.Tensor:
    """Handle hu window for the PaDIS workflow."""
    lower = level - width / 2.0
    return ((normal_to_hu(image) - lower) / width).clamp(0.0, 1.0)


def mask_bbox(mask: torch.Tensor, pad: int = 8):
    """Return the mask bbox."""
    mask_2d = mask.detach().cpu().squeeze().bool()
    if not torch.any(mask_2d):
        return None
    rows = torch.where(torch.any(mask_2d, dim=1))[0]
    cols = torch.where(torch.any(mask_2d, dim=0))[0]
    top = max(int(rows[0]) - pad, 0)
    bottom = min(int(rows[-1]) + pad + 1, mask_2d.shape[0])
    left = max(int(cols[0]) - pad, 0)
    right = min(int(cols[-1]) + pad + 1, mask_2d.shape[1])
    return top, bottom, left, right


def crop_bbox(image: torch.Tensor, bbox):
    """Crop bbox."""
    if bbox is None:
        return image
    top, bottom, left, right = bbox
    return image[..., top:bottom, left:right]


def ssim_on_bbox_or_none(
    recon: torch.Tensor,
    target: torch.Tensor,
    data_range: float,
    bbox,
):
    """Calculate SSIM for on bbox or none."""
    return ssim_or_none(crop_bbox(recon, bbox), crop_bbox(target, bbox), data_range)


def edge_ssim_or_none(recon: torch.Tensor, target: torch.Tensor):
    """Handle edge ssim or none for the PaDIS workflow."""
    try:
        from skimage.filters import sobel
        from skimage.metrics import structural_similarity
    except ImportError:
        return None

    recon_np = recon.detach().cpu().squeeze().numpy()
    target_np = target.detach().cpu().squeeze().numpy()
    recon_edges = sobel(recon_np)
    target_edges = sobel(target_np)
    data_range = float(target_edges.max() - target_edges.min())
    if data_range == 0:
        return 1.0 if float(np.max(np.abs(recon_edges - target_edges))) == 0 else 0.0
    return float(
        structural_similarity(target_edges, recon_edges, data_range=data_range)
    )


def add_image_similarity_metrics(
    item: dict,
    *,
    prefix: str,
    image: torch.Tensor,
    reference: torch.Tensor,
    data_range: float,
) -> None:
    """Add image similarity metrics."""
    key = "" if prefix == "" else f"{prefix}_"
    abs_error = torch.abs(image - reference)
    mse = float(torch.mean((image - reference).square()).item())
    item[f"{key}mse"] = mse
    item[f"{key}psnr"] = psnr_from_mse(mse, data_range)
    item[f"{key}mae"] = float(torch.mean(abs_error).item())
    item[f"{key}abs_error_p95"] = float(
        torch.quantile(abs_error.flatten(), 0.95).item()
    )
    item[f"{key}abs_error_p99"] = float(
        torch.quantile(abs_error.flatten(), 0.99).item()
    )
    item[f"{key}mean_delta"] = float((image.mean() - reference.mean()).item())
    ssim_value = ssim_or_none(image, reference, data_range)
    if ssim_value is not None:
        item[f"{key}ssim"] = ssim_value
    edge_ssim_value = edge_ssim_or_none(image, reference)
    if edge_ssim_value is not None:
        item[f"{key}edge_ssim"] = edge_ssim_value


def image_tensor_from_array(array) -> torch.Tensor:
    """Return the image tensor from array."""
    tensor = torch.as_tensor(array).detach().cpu().float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[0] in (1, 3):
            pass
        elif tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        else:
            raise ValueError(
                f"Cannot infer image channel axis for shape {tensor.shape}."
            )
    else:
        raise ValueError(
            f"Expected an image tensor with 2 or 3 dimensions, got {tensor.ndim}."
        )
    return tensor.contiguous()


def load_reference_reconstructions(path: pathlib.Path | None):
    """Load reference reconstructions."""
    if path is None:
        return None
    path = pathlib.Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Reference reconstruction file not found: {path}")

    image_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    if path.suffix.lower() in image_suffixes:
        image = np.asarray(PIL.Image.open(path), dtype=np.float32) / 255.0
        return image_tensor_from_array(image).unsqueeze(0)

    if path.suffix == ".npz":
        payload = np.load(path)
        for key in ("recon", "reconstructions", "images"):
            if key in payload:
                data = payload[key]
                break
        else:
            raise KeyError(
                f"{path} does not contain one of: recon, reconstructions, images."
            )
    else:
        payload = torch_load(path, map_location="cpu")
        if isinstance(payload, dict):
            for key in ("reconstructions", "recon", "images"):
                if key in payload:
                    data = payload[key]
                    break
            else:
                raise KeyError(
                    f"{path} does not contain one of: reconstructions, recon, images."
                )
        else:
            data = payload

    tensor = torch.as_tensor(data).detach().cpu().float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim == 4:
        if tensor.shape[1] in (1, 3):
            pass
        elif tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                f"Cannot infer reference reconstruction channel axis for shape {tensor.shape}."
            )
    elif tensor.ndim == 5 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    else:
        raise ValueError(
            f"Expected reference reconstructions with 3-5 dimensions, got {tensor.ndim}."
        )
    return tensor.contiguous()


def measurement_image_to_sampler_domain(image: torch.Tensor, params) -> torch.Tensor:
    """Handle measurement image to sampler domain for the PaDIS workflow."""
    scale = float(params.measurement_scale)
    if scale == 0:
        raise ValueError("measurement_scale must be non-zero.")
    return (image - float(params.measurement_offset)) / scale


def sampler_payload_with_runtime_scaling(
    params, reconstructor: PaDIS, device: torch.device
) -> dict:
    """Handle sampler payload with runtime scaling for the PaDIS workflow."""
    payload = {
        key: value for key, value in params.__dict__.items() if not key.startswith("_")
    }
    normalization = payload.get("data_consistency_normalization", "none")
    if normalization not in ("operator_norm", "operator_lipschitz"):
        return payload

    operator_norm = payload.get("operator_norm")
    if operator_norm is None:
        cache = getattr(reconstructor, "_operator_norm_cache", {})
        operator_norm = cache.get((device.type, device.index))
        if operator_norm is None:
            operator_norm = next(
                (
                    value
                    for (cache_type, _cache_index), value in cache.items()
                    if cache_type == device.type
                ),
                None,
            )
    if operator_norm is None:
        return payload

    operator_norm = float(operator_norm)
    measurement_operator_norm = abs(float(params.measurement_scale)) * operator_norm
    payload["operator_norm_estimate"] = operator_norm
    payload["measurement_operator_norm"] = measurement_operator_norm
    payload["data_lipschitz"] = measurement_operator_norm**2
    payload["data_lipschitz_objective"] = "sum_squared_residual"
    payload[
        "data_lipschitz_measurement_map"
    ] = "A(measurement_scale*x + measurement_offset)"
    payload["data_lipschitz_offset_included"] = False
    return payload


def fdk_baseline(sinogram: torch.Tensor, reconstructor: PaDIS, params) -> torch.Tensor:
    """Handle fdk baseline for the PaDIS workflow."""
    with torch.no_grad():
        if reconstructor.geometry is None:
            reconstruction = reconstructor.op.inverse(sinogram)
            if bool(getattr(params, "clip_initial", True)):
                reconstruction = reconstruction.clamp(0.0, 1.0)
        else:
            reconstruction = fdk(
                sinogram,
                reconstructor.op,
                clip=True,
                padded=bool(getattr(params, "initial_fdk_padded", True)),
                filter_type=getattr(params, "initial_fdk_filter_type", None),
                frequency_scaling=float(
                    getattr(params, "initial_fdk_frequency_scaling", 1.0)
                ),
                batch_size=int(getattr(params, "initial_fdk_batch_size", 10)),
            )
        reconstruction = measurement_image_to_sampler_domain(reconstruction, params)
        return reconstruction.clamp(0.0, 1.0)


def tv_reconstruction(
    sinogram: torch.Tensor,
    reconstructor: PaDIS,
    params,
    args,
) -> torch.Tensor:
    """Handle tv reconstruction for the PaDIS workflow."""
    with torch.no_grad():
        reconstruction = tv_min(
            sinogram.unsqueeze(0),
            reconstructor.op,
            lam=float(args.tv_lambda),
            num_iterations=int(args.tv_iterations),
            L=args.tv_lipschitz,
            non_negativity=bool(args.tv_non_negativity),
            progress_bar=bool(args.prog_bar),
        )[0]
        reconstruction = measurement_image_to_sampler_domain(reconstruction, params)
        return reconstruction.clamp(0.0, 1.0)


def pnp_reconstruction(
    sinogram: torch.Tensor,
    reconstruction_geometry,
    denoiser: PnPDenoiser,
    params,
    args,
) -> torch.Tensor:
    """Return PnP reconstruction."""
    reconstructor = PnP(reconstruction_geometry, denoiser, algorithm="ADMM")
    with torch.no_grad():
        reconstruction = reconstructor.reconstruct_sample(
            sinogram,
            eta=float(args.pnp_eta),
            max_iter=int(args.pnp_iterations),
            cg_max_iter=int(args.pnp_cg_iterations),
            cg_tol=float(args.pnp_cg_tolerance),
            clip_min=0.0 if bool(args.pnp_clip) else None,
            clip_max=1.0 if bool(args.pnp_clip) else None,
            prog_bar=bool(args.prog_bar),
        )
        reconstruction = measurement_image_to_sampler_domain(reconstruction, params)
        return reconstruction.clamp(0.0, 1.0)


def effective_algorithm(args) -> str:
    """Return the effective algorithm."""
    if args.method in ("langevin", "ve_ddnm"):
        return "langevin"
    if args.method == "predictor_corrector":
        return "pc"
    return args.algorithm


def reconstruction_label(args, sampler_params) -> str:
    """Return the reconstruction label."""
    labels = {
        "baseline": "Baseline FDK",
        "cp_tv": "TV",
        "pnp_admm": "PnP-ADMM",
        "whole_image_diffusion": "Whole-image diffusion",
        "langevin": "Langevin dynamics",
        "predictor_corrector": "Predictor-corrector",
        "ve_ddnm": "VE-DDNM",
        "patch_average": "Patch averaging",
        "patch_stitch": "Patch stitching",
        "padis_dps": "PaDIS",
    }
    if args.method in labels:
        label = labels[args.method]
        if args.method in {"langevin", "predictor_corrector", "ve_ddnm"}:
            prior_prefix = (
                "Whole-image"
                if getattr(sampler_params, "prior_mode", "patch") == "whole_image"
                else "Patch"
            )
            return f"{prior_prefix} {label}"
        return label
    return (
        "Whole-image diffusion"
        if getattr(sampler_params, "prior_mode", "patch") == "whole_image"
        else "PaDIS"
    )


def result_display_label(args, sampler_params) -> str:
    """Handle result display label for the PaDIS workflow."""
    method_names = {
        "baseline": "Baseline FDK",
        "cp_tv": "CP",
        "pnp_admm": "PnP-ADMM",
        "whole_image_diffusion": "Whole image - VE-DPS",
        "padis_dps": "Patch - VE-DPS",
        "langevin": "Langevin",
        "predictor_corrector": "Predictor-corrector",
        "ve_ddnm": "VE-DDNM",
        "patch_average": "Patch averaging",
        "patch_stitch": "Patch stitching",
    }
    label = method_names.get(args.method, args.method)
    if args.method in {"langevin", "predictor_corrector", "ve_ddnm"}:
        prior_prefix = (
            "Whole image"
            if getattr(sampler_params, "prior_mode", "patch") == "whole_image"
            else "Patch"
        )
        return f"{prior_prefix} - {label}"
    return label


def method_settings(args) -> dict:
    """Handle method settings for the PaDIS workflow."""
    if args.method == "baseline":
        return {"baseline": "fdk"}
    if args.method == "cp_tv":
        return {
            "tv_lambda": float(args.tv_lambda),
            "tv_iterations": int(args.tv_iterations),
            "tv_lipschitz": (
                None if args.tv_lipschitz is None else float(args.tv_lipschitz)
            ),
            "tv_non_negativity": bool(args.tv_non_negativity),
        }
    if args.method == "pnp_admm":
        return {
            "pnp_checkpoint": str(args.pnp_checkpoint),
            "pnp_iterations": int(args.pnp_iterations),
            "pnp_eta": float(args.pnp_eta),
            "pnp_cg_iterations": int(args.pnp_cg_iterations),
            "pnp_cg_tolerance": float(args.pnp_cg_tolerance),
            "pnp_noise_level": (
                None if args.pnp_noise_level is None else float(args.pnp_noise_level)
            ),
            "pnp_clip": bool(args.pnp_clip),
        }
    return {}


def forward_project_normal_image(
    image: torch.Tensor,
    reconstructor: PaDIS,
    params,
) -> torch.Tensor:
    """Apply the forward operation to project normal image."""
    measurement_image = float(params.measurement_scale) * image + float(
        params.measurement_offset
    )
    return reconstructor.op(measurement_image)


def relative_sinogram_residual(
    image: torch.Tensor,
    sinogram: torch.Tensor,
    reconstructor: PaDIS,
    params,
) -> float:
    """Calculate the relative sinogram residual."""
    with torch.no_grad():
        predicted = forward_project_normal_image(image, reconstructor, params)
        residual = predicted.to(dtype=sinogram.dtype) - sinogram
        return float(
            torch.linalg.norm(residual)
            .div(torch.linalg.norm(sinogram).clamp_min(1e-12))
            .item()
        )


def add_reconstruction_metrics(
    item: dict,
    *,
    prefix: str,
    recon: torch.Tensor,
    target: torch.Tensor,
    sinogram: torch.Tensor,
    reconstructor: PaDIS,
    params,
    body_mask: torch.Tensor,
    nonair_mask: torch.Tensor,
    body_bbox,
    data_range: float,
) -> None:
    """Add reconstruction metrics."""
    key = "" if prefix == "" else f"{prefix}_"
    abs_error = torch.abs(recon - target)
    mse = float(torch.mean((recon - target).square()).item())
    mae = float(torch.mean(abs_error).item())
    item[f"{key}mse"] = mse
    item[f"{key}mae"] = mae
    item[f"{key}abs_error_p95"] = float(
        torch.quantile(abs_error.flatten(), 0.95).item()
    )
    item[f"{key}abs_error_p99"] = float(
        torch.quantile(abs_error.flatten(), 0.99).item()
    )
    item[f"{key}psnr"] = psnr_from_mse(mse, data_range)
    item[f"{key}min"] = float(recon.detach().amin().cpu())
    item[f"{key}max"] = float(recon.detach().amax().cpu())
    item[f"{key}mean"] = float(recon.detach().mean().cpu())
    item[f"{key}hu_mae"] = float(
        torch.mean(torch.abs(normal_to_hu(recon) - normal_to_hu(target))).item()
    )
    item[f"{key}relative_sinogram_residual"] = relative_sinogram_residual(
        recon, sinogram, reconstructor, params
    )

    for mask_name, mask in (("body", body_mask), ("nonair", nonair_mask)):
        mse_value = masked_mse(recon, target, mask)
        mae_value = masked_mae(normal_to_hu(recon), normal_to_hu(target), mask)
        if mse_value is not None:
            item[f"{key}{mask_name}_mse"] = mse_value
            item[f"{key}{mask_name}_psnr"] = psnr_from_mse(mse_value, data_range)
        if mae_value is not None:
            item[f"{key}{mask_name}_hu_mae"] = mae_value

    ssim_value = ssim_or_none(recon, target, data_range)
    if ssim_value is not None:
        item[f"{key}ssim"] = ssim_value
    edge_ssim_value = edge_ssim_or_none(recon, target)
    if edge_ssim_value is not None:
        item[f"{key}edge_ssim"] = edge_ssim_value
    body_ssim = ssim_on_bbox_or_none(recon, target, data_range, body_bbox)
    if body_ssim is not None:
        item[f"{key}body_bbox_ssim"] = body_ssim

    for window_name, level, width in (
        ("lung", -600.0, 1500.0),
        ("soft_tissue", 40.0, 400.0),
        ("bone", 400.0, 1800.0),
    ):
        recon_window = hu_window(recon, level=level, width=width)
        target_window = hu_window(target, level=level, width=width)
        window_mse = float(torch.mean((recon_window - target_window).square()).item())
        item[f"{key}{window_name}_window_mse"] = window_mse
        item[f"{key}{window_name}_window_psnr"] = psnr_from_mse(window_mse, 1.0)
        window_ssim = ssim_on_bbox_or_none(recon_window, target_window, 1.0, body_bbox)
        if window_ssim is not None:
            item[f"{key}{window_name}_window_body_bbox_ssim"] = window_ssim


def add_ddnm_pseudoinverse_diagnostics(
    item: dict,
    *,
    target: torch.Tensor,
    sinogram: torch.Tensor,
    reconstructor: PaDIS,
    params,
    body_mask: torch.Tensor,
    nonair_mask: torch.Tensor,
    body_bbox,
    data_range: float,
) -> None:
    """Measure the LION pseudoinverse terms used by the DDNM correction."""
    with torch.no_grad():
        measured_pinv = reconstructor._pseudoinverse_reconstruction(
            sinogram,
            params,
        )
        target_sinogram = forward_project_normal_image(target, reconstructor, params)
        projected_target_pinv = reconstructor._pseudoinverse_reconstruction(
            target_sinogram.to(dtype=sinogram.dtype),
            params,
            clip=bool(getattr(params, "ddnm_projected_pseudoinverse_clip", False)),
        )
        perfect_denoiser_corrected = measured_pinv + target - projected_target_pinv
        corrected_clipped = perfect_denoiser_corrected.clamp(0.0, 1.0)

    add_reconstruction_metrics(
        item,
        prefix="ddnm_measured_pseudoinverse",
        recon=measured_pinv,
        target=target,
        sinogram=sinogram,
        reconstructor=reconstructor,
        params=params,
        body_mask=body_mask,
        nonair_mask=nonair_mask,
        body_bbox=body_bbox,
        data_range=data_range,
    )
    add_reconstruction_metrics(
        item,
        prefix="ddnm_projected_target_pseudoinverse",
        recon=projected_target_pinv,
        target=target,
        sinogram=sinogram,
        reconstructor=reconstructor,
        params=params,
        body_mask=body_mask,
        nonair_mask=nonair_mask,
        body_bbox=body_bbox,
        data_range=data_range,
    )
    add_reconstruction_metrics(
        item,
        prefix="ddnm_perfect_denoiser_corrected",
        recon=perfect_denoiser_corrected,
        target=target,
        sinogram=sinogram,
        reconstructor=reconstructor,
        params=params,
        body_mask=body_mask,
        nonair_mask=nonair_mask,
        body_bbox=body_bbox,
        data_range=data_range,
    )
    add_reconstruction_metrics(
        item,
        prefix="ddnm_perfect_denoiser_corrected_clipped",
        recon=corrected_clipped,
        target=target,
        sinogram=sinogram,
        reconstructor=reconstructor,
        params=params,
        body_mask=body_mask,
        nonair_mask=nonair_mask,
        body_bbox=body_bbox,
        data_range=data_range,
    )
    item["ddnm_pseudoinverse_diagnostic"] = {
        "formula": "A^dagger y + x - A^dagger A(x), evaluated with x=target",
        "measured_pseudoinverse_clip": bool(
            getattr(params, "ddnm_pseudoinverse_clip", False)
        ),
        "projected_pseudoinverse_clip": bool(
            getattr(params, "ddnm_projected_pseudoinverse_clip", False)
        ),
        "corrected_min": float(perfect_denoiser_corrected.detach().amin().cpu()),
        "corrected_max": float(perfect_denoiser_corrected.detach().amax().cpu()),
        "corrected_clipped_min": float(corrected_clipped.detach().amin().cpu()),
        "corrected_clipped_max": float(corrected_clipped.detach().amax().cpu()),
    }


def save_preview(
    path: pathlib.Path,
    sinogram,
    fdk_recon,
    recon,
    target,
    *,
    reference=None,
    body_mask,
    error_vmax: float,
    recon_label: str,
) -> None:
    """Save preview."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = 5 if reference is not None else 4
    fig, axes = plt.subplots(2, columns, figsize=(3.25 * columns, 6.2))
    image_kwargs = {"cmap": "gray", "vmin": 0, "vmax": 1}
    axes[0, 0].imshow(sinogram.detach().cpu().squeeze().T, cmap="gray")
    axes[0, 0].set_title("Sinogram")
    axes[0, 1].imshow(fdk_recon.detach().cpu().squeeze(), **image_kwargs)
    axes[0, 1].set_title("FDK")
    axes[0, 2].imshow(recon.detach().cpu().squeeze(), **image_kwargs)
    axes[0, 2].set_title(recon_label)
    axes[0, 3].imshow(target.detach().cpu().squeeze(), **image_kwargs)
    axes[0, 3].set_title("Target")
    if reference is not None:
        axes[0, 4].imshow(reference.detach().cpu().squeeze(), **image_kwargs)
        axes[0, 4].set_title("Public ref")
    axes[1, 0].imshow(body_mask.detach().cpu().squeeze(), cmap="gray")
    axes[1, 0].set_title("Body ROI")
    axes[1, 1].imshow(
        torch.abs(fdk_recon - target).detach().cpu().squeeze(),
        cmap="magma",
        vmin=0,
        vmax=error_vmax,
    )
    axes[1, 1].set_title("|FDK error|")
    axes[1, 2].imshow(
        torch.abs(recon - target).detach().cpu().squeeze(),
        cmap="magma",
        vmin=0,
        vmax=error_vmax,
    )
    axes[1, 2].set_title(f"|{recon_label} error|")
    axes[1, 3].imshow(
        (recon - fdk_recon).detach().cpu().squeeze(),
        cmap="coolwarm",
        vmin=-error_vmax,
        vmax=error_vmax,
    )
    axes[1, 3].set_title(f"{recon_label} - FDK")
    if reference is not None:
        axes[1, 4].imshow(
            torch.abs(recon - reference).detach().cpu().squeeze(),
            cmap="magma",
            vmin=0,
            vmax=error_vmax,
        )
        axes[1, 4].set_title(f"|{recon_label} - ref|")
    for ax in axes.ravel():
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_visual_comparison(
    path: pathlib.Path,
    fdk_recon,
    recon,
    target,
    *,
    reference=None,
    error_vmax: float,
    image_vmax: float,
    recon_label: str,
) -> None:
    """Save visual comparison."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    image_kwargs = {"cmap": "gray", "vmin": 0.0, "vmax": image_vmax}
    error_kwargs = {"cmap": "magma", "vmin": 0.0, "vmax": error_vmax}
    compare_image = reference if reference is not None else target
    compare_label = "Public ref" if reference is not None else "Target"

    panels = [
        ("Target", target.detach().cpu().squeeze(), image_kwargs),
        (compare_label, compare_image.detach().cpu().squeeze(), image_kwargs),
        (recon_label, recon.detach().cpu().squeeze(), image_kwargs),
        ("FDK", fdk_recon.detach().cpu().squeeze(), image_kwargs),
        (
            f"|{recon_label} - Target|",
            torch.abs(recon - target).detach().cpu().squeeze(),
            error_kwargs,
        ),
        (
            f"|{recon_label} - {compare_label}|",
            torch.abs(recon - compare_image).detach().cpu().squeeze(),
            error_kwargs,
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for ax, (title, image, kwargs) in zip(axes.ravel(), panels):
        im = ax.imshow(image, **kwargs)
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=160)
    plt.close(fig)

    individual_folder = path.with_suffix("")
    individual_folder.mkdir(parents=True, exist_ok=True)
    individual_images = {
        "target": (target.detach().cpu().squeeze(), image_kwargs),
        compare_label.lower().replace(" ", "_"): (
            compare_image.detach().cpu().squeeze(),
            image_kwargs,
        ),
        "recon": (recon.detach().cpu().squeeze(), image_kwargs),
        "fdk": (fdk_recon.detach().cpu().squeeze(), image_kwargs),
        "abs_recon_target": (
            torch.abs(recon - target).detach().cpu().squeeze(),
            error_kwargs,
        ),
        "abs_recon_reference": (
            torch.abs(recon - compare_image).detach().cpu().squeeze(),
            error_kwargs,
        ),
    }
    for stem, (image, kwargs) in individual_images.items():
        plt.imsave(individual_folder / f"{stem}.png", image.numpy(), **kwargs)


def save_tensor_image(
    path: pathlib.Path,
    tensor: torch.Tensor,
    *,
    transpose: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Save tensor image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = tensor.detach().cpu().squeeze()
    while image.ndim > 2:
        image = image[0]
    if transpose and image.ndim == 2:
        image = image.T
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, image.numpy(), cmap="gray", vmin=vmin, vmax=vmax)


def save_trace_images(
    output_folder: pathlib.Path,
    sample_index: int,
    snapshots: list[dict],
) -> dict | None:
    """Save trace images."""
    if not snapshots:
        return None

    trace_root = output_folder / "trace_images"
    sample_folder = trace_root / f"sample_{sample_index:04d}"
    sample_folder.mkdir(parents=True, exist_ok=True)
    tensor_path = trace_root / f"sample_{sample_index:04d}.pt"
    torch.save({"index": int(sample_index), "snapshots": snapshots}, tensor_path)

    image_records = []
    for snapshot in snapshots:
        stem = (
            f"step_{int(snapshot['step']):04d}_"
            f"inner_{int(snapshot['inner']):02d}_"
            f"{snapshot['algorithm']}"
        )
        current_path = sample_folder / f"{stem}_current.png"
        denoised_path = sample_folder / f"{stem}_denoised.png"
        projected_path = sample_folder / f"{stem}_projected.png"
        x_next_path = sample_folder / f"{stem}_x_next.png"
        forward_path = sample_folder / f"{stem}_forward_projected.png"
        save_tensor_image(
            current_path,
            snapshot["x"],
            vmin=0.0,
            vmax=1.0,
        )
        save_tensor_image(
            denoised_path,
            snapshot["denoised"],
            vmin=0.0,
            vmax=1.0,
        )
        save_tensor_image(
            projected_path,
            snapshot["projected"],
            vmin=0.0,
            vmax=1.0,
        )
        save_tensor_image(
            x_next_path,
            snapshot["x_next"],
            vmin=0.0,
            vmax=1.0,
        )
        save_tensor_image(
            forward_path,
            snapshot["forward_projected"],
            transpose=True,
        )
        image_records.append(
            {
                "step": int(snapshot["step"]),
                "inner": int(snapshot["inner"]),
                "algorithm": snapshot["algorithm"],
                "sigma": float(snapshot["sigma"]),
                "current_png": str(current_path),
                "denoised_png": str(denoised_path),
                "projected_png": str(projected_path),
                "x_next_png": str(x_next_path),
                "forward_projected_png": str(forward_path),
            }
        )

    return {
        "index": int(sample_index),
        "tensor_path": str(tensor_path),
        "folder": str(sample_folder),
        "images": image_records,
    }


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


def run_reconstruction_variant(
    *,
    args,
    dataset,
    checkpoint_path: pathlib.Path | None,
    geometry,
    reconstruction_geometry,
    experiment,
    model,
    model_params,
    base_params: LIONParameter,
    variant_name: str,
    variant_overrides: dict,
    output_folder: pathlib.Path,
    device: torch.device,
    from_experiment: bool,
    experiment_measurement_source: str,
    reference_reconstructions: torch.Tensor | None,
) -> dict:
    """Run reconstruction variant."""
    set_run_seed(args.seed)
    sampler_params = clone_parameters(base_params)
    for key, value in variant_overrides.items():
        setattr(sampler_params, key, value)

    algorithm = effective_algorithm(args)
    reconstructor = PaDIS(
        reconstruction_geometry,
        model,
        parameters=sampler_params,
        algorithm=algorithm,
    )
    reconstructor.last_trace = []
    reconstructor.last_trace_images = []
    pnp_denoiser = None
    if args.method == "pnp_admm":
        if args.pnp_checkpoint is None:
            raise ValueError("--method pnp_admm requires --pnp-checkpoint.")
        pnp_denoiser = load_pnp_denoiser(
            args.pnp_checkpoint,
            device,
            noise_level=args.pnp_noise_level,
        )
    output_folder.mkdir(parents=True, exist_ok=True)
    recon_label = reconstruction_label(args, sampler_params)
    print(f"Saving {recon_label} {variant_name} reconstructions to {output_folder}")

    stop = min(len(dataset), args.start_index + args.max_samples)
    run_length = stop - args.start_index
    if reference_reconstructions is not None:
        if len(reference_reconstructions) not in (1, run_length):
            raise ValueError(
                "Reference reconstruction count must be 1 for a one-sample run "
                f"or match the run length ({run_length}); got "
                f"{len(reference_reconstructions)}."
            )
        if len(reference_reconstructions) == 1 and run_length != 1:
            raise ValueError(
                "A single reference reconstruction can only be used with "
                "--max-samples 1."
            )
    metrics = []
    reconstructions = []
    fdk_reconstructions = []
    targets = []
    sinograms = []
    traces = []
    trace_images = []
    for output_index, index in enumerate(
        tqdm(
            range(args.start_index, stop),
            desc=f"LIDC {args.split} {variant_name}",
        )
    ):
        sinogram, target = make_measurement(
            args,
            dataset,
            index,
            reconstructor,
            device,
            from_experiment=from_experiment,
            experiment_measurement_source=experiment_measurement_source,
        )
        if bool(getattr(sampler_params, "consume_discarded_measurement_noise", False)):
            _ = torch.randn_like(sinogram)
        fdk_recon = fdk_baseline(sinogram, reconstructor, sampler_params)
        if args.method == "baseline":
            recon = fdk_recon
        elif args.method == "cp_tv":
            recon = tv_reconstruction(sinogram, reconstructor, sampler_params, args)
        elif args.method == "pnp_admm":
            assert pnp_denoiser is not None
            recon = pnp_reconstruction(
                sinogram,
                reconstruction_geometry,
                pnp_denoiser,
                sampler_params,
                args,
            )
        else:
            recon = reconstructor.reconstruct_sample(
                sinogram,
                algorithm=algorithm,
                prog_bar=args.prog_bar,
                generator=None,
            )
        reconstructions.append(recon.detach().cpu())
        fdk_reconstructions.append(fdk_recon.detach().cpu())
        targets.append(target.detach().cpu())
        sinograms.append(sinogram.detach().cpu())
        reference = None
        if reference_reconstructions is not None:
            reference = reference_reconstructions[
                0 if len(reference_reconstructions) == 1 else output_index
            ].to(device=device, dtype=target.dtype)
            if tuple(reference.shape) != tuple(target.shape):
                raise ValueError(
                    f"Reference shape {tuple(reference.shape)} does not match "
                    f"target shape {tuple(target.shape)} for sample {index}."
                )
        body_mask = target > float(args.body_threshold)
        nonair_mask = target > float(args.nonair_threshold)
        body_bbox = mask_bbox(body_mask, pad=int(args.body_bbox_padding))
        item = {
            "index": int(index),
            "target_min": float(target.detach().amin().cpu()),
            "target_max": float(target.detach().amax().cpu()),
            "target_mean": float(target.detach().mean().cpu()),
            "body_threshold": float(args.body_threshold),
            "body_pixel_fraction": float(body_mask.float().mean().detach().cpu()),
            "nonair_threshold": float(args.nonair_threshold),
            "nonair_pixel_fraction": float(nonair_mask.float().mean().detach().cpu()),
            "sinogram_min": float(sinogram.detach().amin().cpu()),
            "sinogram_max": float(sinogram.detach().amax().cpu()),
            "sinogram_mean": float(sinogram.detach().mean().cpu()),
        }
        add_reconstruction_metrics(
            item,
            prefix="",
            recon=recon,
            target=target,
            sinogram=sinogram,
            reconstructor=reconstructor,
            params=sampler_params,
            body_mask=body_mask,
            nonair_mask=nonair_mask,
            body_bbox=body_bbox,
            data_range=args.data_range,
        )
        item["recon_min"] = item.pop("min")
        item["recon_max"] = item.pop("max")
        item["recon_mean"] = item.pop("mean")
        item["recon_hu_mae"] = item.pop("hu_mae")
        item["recon_relative_sinogram_residual"] = item.pop(
            "relative_sinogram_residual"
        )
        add_reconstruction_metrics(
            item,
            prefix="fdk",
            recon=fdk_recon,
            target=target,
            sinogram=sinogram,
            reconstructor=reconstructor,
            params=sampler_params,
            body_mask=body_mask,
            nonair_mask=nonair_mask,
            body_bbox=body_bbox,
            data_range=args.data_range,
        )
        if getattr(args, "diagnose_ddnm_pseudoinverse", False):
            add_ddnm_pseudoinverse_diagnostics(
                item,
                target=target,
                sinogram=sinogram,
                reconstructor=reconstructor,
                params=sampler_params,
                body_mask=body_mask,
                nonair_mask=nonair_mask,
                body_bbox=body_bbox,
                data_range=args.data_range,
            )
        if reference is not None:
            add_image_similarity_metrics(
                item,
                prefix="public_reference",
                image=recon,
                reference=reference,
                data_range=args.data_range,
            )
        if args.trace_interval > 0:
            traces.append({"index": int(index), "trace": reconstructor.last_trace})
        if args.trace_images:
            trace_image_record = save_trace_images(
                output_folder,
                int(index),
                reconstructor.last_trace_images,
            )
            if trace_image_record is not None:
                trace_images.append(trace_image_record)
        metrics.append(item)
        if args.save_previews:
            save_preview(
                output_folder / f"sample_{index:04d}.png",
                sinogram,
                fdk_recon,
                recon,
                target,
                reference=reference,
                body_mask=body_mask,
                error_vmax=float(args.error_vmax),
                recon_label=recon_label,
            )
            save_visual_comparison(
                output_folder / f"sample_{index:04d}_visual_compare.png",
                fdk_recon,
                recon,
                target,
                reference=reference,
                error_vmax=float(args.error_vmax),
                image_vmax=float(args.preview_vmax),
                recon_label=recon_label,
            )

    payload = {
        "checkpoint": "" if checkpoint_path is None else str(checkpoint_path),
        "split": args.split,
        "experiment": args.experiment,
        "implementation": args.implementation,
        "geometry_tag": args.geometry,
        "method": args.method,
        "display_label": result_display_label(args, sampler_params),
        "reconstruction_label": recon_label,
        "algorithm": algorithm,
        "matrix_group": getattr(args, "matrix_group", "main"),
        "ablation": variant_name,
        "ablation_overrides": variant_overrides,
        "measurement_source": experiment_measurement_source,
        "public_reference_reconstructions": (
            str(args.public_reference_reconstructions)
            if args.public_reference_reconstructions is not None
            else None
        ),
        "checkpoint_image_scaling": float(getattr(geometry, "image_scaling", 1.0)),
        "reconstruction_geometry": str(reconstruction_geometry),
        "measurement_scale": float(sampler_params.measurement_scale),
        "measurement_offset": float(sampler_params.measurement_offset),
        "prior_mode": getattr(sampler_params, "prior_mode", "patch"),
        "model_patch_size": int(getattr(model_params, "largest_patch_size", -1)),
        "method_settings": method_settings(args),
        "sampler": sampler_payload_with_runtime_scaling(
            sampler_params, reconstructor, device
        ),
        "metrics": metrics,
    }
    if experiment is not None:
        payload["experiment_name"] = experiment.param.name
    spec = experiment_spec_from_args(args)
    if spec is not None:
        payload["paper_experiment"] = {
            "key": spec.key,
            "views": int(spec.views),
            "paper_geometry": spec.paper_geometry,
            "paper_sampler_views": int(spec.paper_sampler_views),
            "description": spec.description,
        }
    metric_path = output_folder / "metrics.json"
    with open(metric_path, "w") as f:
        json.dump(payload, f, indent=2)
    trace_path = None
    if traces:
        trace_path = output_folder / "trace.json"
        with open(trace_path, "w") as f:
            json.dump(traces, f, indent=2)
    trace_images_path = None
    if trace_images:
        trace_images_path = output_folder / "trace_images.json"
        with open(trace_images_path, "w") as f:
            json.dump(trace_images, f, indent=2)

    tensor_path = output_folder / "reconstructions.pt"
    torch.save(
        {
            "reconstructions": torch.stack(reconstructions),
            "fdk_reconstructions": torch.stack(fdk_reconstructions),
            "targets": torch.stack(targets),
            "sinograms": torch.stack(sinograms),
            "metrics": metrics,
        },
        tensor_path,
    )

    mean_mse = mean_metric(metrics, "mse")
    mean_mae = mean_metric(metrics, "mae")
    max_mae = max(item["mae"] for item in metrics)
    mean_abs_error_p95 = mean_metric(metrics, "abs_error_p95")
    max_abs_error_p95 = max(item["abs_error_p95"] for item in metrics)
    mean_psnr = mean_metric(metrics, "psnr")
    min_psnr = min(item["psnr"] for item in metrics)
    min_fdk_margin = min(item["psnr"] - item["fdk_psnr"] for item in metrics)
    mean_fdk_mse = mean_metric(metrics, "fdk_mse")
    mean_fdk_psnr = mean_metric(metrics, "fdk_psnr")
    mean_body_psnr = mean_metric(metrics, "body_psnr")
    mean_fdk_body_psnr = mean_metric(metrics, "fdk_body_psnr")
    mean_soft_tissue_psnr = mean_metric(metrics, "soft_tissue_window_psnr")
    mean_fdk_soft_tissue_psnr = mean_metric(metrics, "fdk_soft_tissue_window_psnr")
    mean_edge_ssim = mean_metric(metrics, "edge_ssim")
    min_edge_ssim = min_metric(metrics, "edge_ssim")
    mean_reference_ssim = mean_metric(metrics, "public_reference_ssim")
    min_reference_ssim = min_metric(metrics, "public_reference_ssim")
    mean_reference_edge_ssim = mean_metric(metrics, "public_reference_edge_ssim")
    min_reference_edge_ssim = min_metric(metrics, "public_reference_edge_ssim")
    mean_reference_mae = mean_metric(metrics, "public_reference_mae")
    mean_reference_abs_error_p95 = mean_metric(
        metrics, "public_reference_abs_error_p95"
    )
    mean_ddnm_perfect_corrected_psnr = mean_metric(
        metrics, "ddnm_perfect_denoiser_corrected_psnr"
    )
    mean_ddnm_perfect_corrected_clipped_psnr = mean_metric(
        metrics, "ddnm_perfect_denoiser_corrected_clipped_psnr"
    )
    print(f"{variant_name} mean MSE:  {mean_mse:.6g}")
    print(f"{variant_name} mean MAE:  {mean_mae:.6g}")
    print(f"{variant_name} mean p95 abs error: {mean_abs_error_p95:.6g}")
    print(f"{variant_name} mean PSNR: {mean_psnr:.4g} dB")
    print(f"{variant_name} FDK mean MSE:  {mean_fdk_mse:.6g}")
    print(f"{variant_name} FDK mean PSNR: {mean_fdk_psnr:.4g} dB")
    if mean_body_psnr is not None:
        print(f"{variant_name} body PSNR: {mean_body_psnr:.4g} dB")
    if mean_fdk_body_psnr is not None:
        print(f"{variant_name} FDK body PSNR: {mean_fdk_body_psnr:.4g} dB")
    if mean_soft_tissue_psnr is not None:
        print(
            f"{variant_name} soft-tissue window PSNR: "
            f"{mean_soft_tissue_psnr:.4g} dB"
        )
    if mean_fdk_soft_tissue_psnr is not None:
        print(
            f"{variant_name} FDK soft-tissue window PSNR: "
            f"{mean_fdk_soft_tissue_psnr:.4g} dB"
        )
    if "ssim" in metrics[0]:
        mean_ssim = sum(item["ssim"] for item in metrics) / len(metrics)
        min_ssim = min(item["ssim"] for item in metrics)
        print(f"{variant_name} mean SSIM: {mean_ssim:.4g}")
    else:
        mean_ssim = None
        min_ssim = None
    if mean_edge_ssim is not None:
        print(f"{variant_name} mean edge SSIM: {mean_edge_ssim:.4g}")
    if "fdk_ssim" in metrics[0]:
        mean_fdk_ssim = sum(item["fdk_ssim"] for item in metrics) / len(metrics)
        print(f"{variant_name} FDK mean SSIM: {mean_fdk_ssim:.4g}")
    else:
        mean_fdk_ssim = None
    if mean_reference_ssim is not None:
        print(f"{variant_name} public-reference mean SSIM: {mean_reference_ssim:.4g}")
    if mean_reference_edge_ssim is not None:
        print(
            f"{variant_name} public-reference mean edge SSIM: "
            f"{mean_reference_edge_ssim:.4g}"
        )
    if mean_reference_mae is not None:
        print(f"{variant_name} public-reference mean MAE: {mean_reference_mae:.6g}")
    if mean_reference_abs_error_p95 is not None:
        print(
            f"{variant_name} public-reference mean p95 abs error: "
            f"{mean_reference_abs_error_p95:.6g}"
        )
    if mean_ddnm_perfect_corrected_psnr is not None:
        print(
            f"{variant_name} DDNM perfect-denoiser correction PSNR: "
            f"{mean_ddnm_perfect_corrected_psnr:.4g} dB"
        )
    if mean_ddnm_perfect_corrected_clipped_psnr is not None:
        print(
            f"{variant_name} DDNM clipped perfect-denoiser correction PSNR: "
            f"{mean_ddnm_perfect_corrected_clipped_psnr:.4g} dB"
        )
    print(f"Saved metrics to {metric_path}")
    if trace_path is not None:
        print(f"Saved sampler trace to {trace_path}")
    if trace_images_path is not None:
        print(f"Saved trace images to {trace_images_path}")
    print(f"Saved tensors to {tensor_path}")
    return {
        "name": variant_name,
        "folder": str(output_folder),
        "metrics": str(metric_path),
        "trace": str(trace_path) if trace_path is not None else None,
        "trace_images": (
            str(trace_images_path) if trace_images_path is not None else None
        ),
        "tensors": str(tensor_path),
        "mean_mse": mean_mse,
        "mean_mae": mean_mae,
        "max_mae": max_mae,
        "mean_abs_error_p95": mean_abs_error_p95,
        "max_abs_error_p95": max_abs_error_p95,
        "mean_psnr": mean_psnr,
        "min_psnr": min_psnr,
        "min_fdk_margin": min_fdk_margin,
        "mean_fdk_mse": mean_fdk_mse,
        "mean_fdk_psnr": mean_fdk_psnr,
        "mean_ssim": mean_ssim,
        "min_ssim": min_ssim,
        "mean_fdk_ssim": mean_fdk_ssim,
        "mean_edge_ssim": mean_edge_ssim,
        "min_edge_ssim": min_edge_ssim,
        "mean_reference_ssim": mean_reference_ssim,
        "min_reference_ssim": min_reference_ssim,
        "mean_reference_edge_ssim": mean_reference_edge_ssim,
        "min_reference_edge_ssim": min_reference_edge_ssim,
        "mean_reference_mae": mean_reference_mae,
        "mean_reference_abs_error_p95": mean_reference_abs_error_p95,
        "mean_ddnm_perfect_corrected_psnr": mean_ddnm_perfect_corrected_psnr,
        "mean_ddnm_perfect_corrected_clipped_psnr": (
            mean_ddnm_perfect_corrected_clipped_psnr
        ),
        "mean_body_psnr": mean_body_psnr,
        "mean_fdk_body_psnr": mean_fdk_body_psnr,
        "mean_soft_tissue_window_psnr": mean_soft_tissue_psnr,
        "mean_fdk_soft_tissue_window_psnr": mean_fdk_soft_tissue_psnr,
    }


def enforce_quality_gates(args, summaries: list[dict]) -> None:
    """Enforce quality gates."""
    failures = []
    for summary in summaries:
        name = summary["name"]
        mean_psnr = summary["mean_psnr"]
        if args.min_mean_psnr is not None and mean_psnr < args.min_mean_psnr:
            failures.append(
                f"{name}: mean PSNR {mean_psnr:.4g} dB < {args.min_mean_psnr:.4g} dB"
            )

        mean_mae = summary["mean_mae"]
        if args.max_mean_mae is not None and mean_mae > args.max_mean_mae:
            failures.append(
                f"{name}: mean MAE {mean_mae:.4g} > {args.max_mean_mae:.4g}"
            )

        max_mae = summary["max_mae"]
        if args.max_sample_mae is not None and max_mae > args.max_sample_mae:
            failures.append(
                f"{name}: maximum sample MAE {max_mae:.4g} > "
                f"{args.max_sample_mae:.4g}"
            )

        mean_abs_error_p95 = summary["mean_abs_error_p95"]
        if (
            args.max_mean_abs_error_p95 is not None
            and mean_abs_error_p95 > args.max_mean_abs_error_p95
        ):
            failures.append(
                f"{name}: mean p95 abs error {mean_abs_error_p95:.4g} > "
                f"{args.max_mean_abs_error_p95:.4g}"
            )

        max_abs_error_p95 = summary["max_abs_error_p95"]
        if (
            args.max_sample_abs_error_p95 is not None
            and max_abs_error_p95 > args.max_sample_abs_error_p95
        ):
            failures.append(
                f"{name}: maximum sample p95 abs error "
                f"{max_abs_error_p95:.4g} > "
                f"{args.max_sample_abs_error_p95:.4g}"
            )

        mean_ssim = summary.get("mean_ssim")
        if args.min_mean_ssim is not None:
            if mean_ssim is None:
                failures.append(f"{name}: SSIM was not computed")
            elif mean_ssim < args.min_mean_ssim:
                failures.append(
                    f"{name}: mean SSIM {mean_ssim:.4g} < {args.min_mean_ssim:.4g}"
                )

        min_ssim = summary.get("min_ssim")
        if args.min_sample_ssim is not None:
            if min_ssim is None:
                failures.append(f"{name}: sample SSIM was not computed")
            elif min_ssim < args.min_sample_ssim:
                failures.append(
                    f"{name}: minimum sample SSIM {min_ssim:.4g} < "
                    f"{args.min_sample_ssim:.4g}"
                )

        mean_edge_ssim = summary.get("mean_edge_ssim")
        if args.min_mean_edge_ssim is not None:
            if mean_edge_ssim is None:
                failures.append(f"{name}: edge SSIM was not computed")
            elif mean_edge_ssim < args.min_mean_edge_ssim:
                failures.append(
                    f"{name}: mean edge SSIM {mean_edge_ssim:.4g} < "
                    f"{args.min_mean_edge_ssim:.4g}"
                )

        min_edge_ssim = summary.get("min_edge_ssim")
        if args.min_sample_edge_ssim is not None:
            if min_edge_ssim is None:
                failures.append(f"{name}: sample edge SSIM was not computed")
            elif min_edge_ssim < args.min_sample_edge_ssim:
                failures.append(
                    f"{name}: minimum sample edge SSIM {min_edge_ssim:.4g} < "
                    f"{args.min_sample_edge_ssim:.4g}"
                )

        mean_reference_ssim = summary.get("mean_reference_ssim")
        if args.min_mean_reference_ssim is not None:
            if mean_reference_ssim is None:
                failures.append(f"{name}: public-reference SSIM was not computed")
            elif mean_reference_ssim < args.min_mean_reference_ssim:
                failures.append(
                    f"{name}: public-reference mean SSIM "
                    f"{mean_reference_ssim:.4g} < "
                    f"{args.min_mean_reference_ssim:.4g}"
                )

        min_reference_ssim = summary.get("min_reference_ssim")
        if args.min_sample_reference_ssim is not None:
            if min_reference_ssim is None:
                failures.append(
                    f"{name}: public-reference sample SSIM was not computed"
                )
            elif min_reference_ssim < args.min_sample_reference_ssim:
                failures.append(
                    f"{name}: public-reference minimum sample SSIM "
                    f"{min_reference_ssim:.4g} < "
                    f"{args.min_sample_reference_ssim:.4g}"
                )

        mean_reference_edge_ssim = summary.get("mean_reference_edge_ssim")
        if args.min_mean_reference_edge_ssim is not None:
            if mean_reference_edge_ssim is None:
                failures.append(f"{name}: public-reference edge SSIM was not computed")
            elif mean_reference_edge_ssim < args.min_mean_reference_edge_ssim:
                failures.append(
                    f"{name}: public-reference mean edge SSIM "
                    f"{mean_reference_edge_ssim:.4g} < "
                    f"{args.min_mean_reference_edge_ssim:.4g}"
                )

        mean_reference_mae = summary.get("mean_reference_mae")
        if args.max_mean_reference_mae is not None:
            if mean_reference_mae is None:
                failures.append(f"{name}: public-reference MAE was not computed")
            elif mean_reference_mae > args.max_mean_reference_mae:
                failures.append(
                    f"{name}: public-reference mean MAE "
                    f"{mean_reference_mae:.4g} > "
                    f"{args.max_mean_reference_mae:.4g}"
                )

        mean_reference_abs_error_p95 = summary.get("mean_reference_abs_error_p95")
        if args.max_mean_reference_abs_error_p95 is not None:
            if mean_reference_abs_error_p95 is None:
                failures.append(
                    f"{name}: public-reference p95 abs error was not computed"
                )
            elif mean_reference_abs_error_p95 > args.max_mean_reference_abs_error_p95:
                failures.append(
                    f"{name}: public-reference mean p95 abs error "
                    f"{mean_reference_abs_error_p95:.4g} > "
                    f"{args.max_mean_reference_abs_error_p95:.4g}"
                )

        if args.require_better_than_fdk:
            mean_fdk_psnr = summary["mean_fdk_psnr"]
            if mean_psnr <= mean_fdk_psnr:
                failures.append(
                    f"{name}: PaDIS mean PSNR {mean_psnr:.4g} dB <= "
                    f"FDK mean PSNR {mean_fdk_psnr:.4g} dB"
                )
        if args.min_sample_psnr is not None:
            min_psnr = summary["min_psnr"]
            if min_psnr < args.min_sample_psnr:
                failures.append(
                    f"{name}: minimum sample PSNR {min_psnr:.4g} dB < "
                    f"{args.min_sample_psnr:.4g} dB"
                )
        if args.require_each_better_than_fdk and summary["min_fdk_margin"] <= 0:
            failures.append(
                f"{name}: at least one PaDIS sample did not improve over FDK "
                f"(minimum margin {summary['min_fdk_margin']:.4g} dB)"
            )

    if failures:
        message = "Quality gate failed:\n  " + "\n  ".join(failures)
        raise RuntimeError(message)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the PaDIS/LION reconstruction command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    experiment_choices = sorted(
        {
            "none",
            *LIDC_EXPERIMENTS.keys(),
            *PAPER_CT_EXPERIMENTS.keys(),
            *EXPERIMENT_ALIASES.keys(),
        }
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-folder", type=pathlib.Path, default=None)
    parser.add_argument(
        "--image-scaling",
        type=float,
        default=0.5,
        help=(
            "Fallback LIDC image scaling when a method does not need a PaDIS "
            "checkpoint and no checkpoint sidecar JSON is available."
        ),
    )
    parser.add_argument(
        "--output-folder",
        type=pathlib.Path,
        default=LION_EXPERIMENTS_PATH / "PaDIS" / "LIDC_reconstruction",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--experiment",
        choices=experiment_choices,
        default="none",
        help=(
            "Run a LION LIDC CT experiment. Paper aliases include ct_8, "
            "ct_20, ct_60, ct_20_limited_angle_120, and ct_512_60. "
            "image_scaling is read from the PaDIS checkpoint geometry."
        ),
    )
    parser.add_argument(
        "--implementation",
        choices=IMPLEMENTATION_CHOICES,
        default="custom",
        help=(
            "Sampler preset. 'paper' uses the CT schedule described by Hu et al. and "
            "data step; 'public_repo' keeps the PaDIS README reconstruction "
            "mechanics but uses the CT sigma schedule of Hu et al.; "
            "'lion_physics' uses LION CT operators with operator-normalized "
            "least-squares data updates and no public matching constants; "
            "'lion_quality' is the LION-native quality preset; 'custom' uses "
            "the explicit sampler flags."
        ),
    )
    parser.add_argument(
        "--geometry",
        choices=GEOMETRY_CHOICES,
        default="lion",
        help=(
            "CT geometry family for paper CT experiment aliases. Only 'lion' "
            "is currently executable for LIDC-IDRI. PaDIS geometry tags are "
            "accepted only to fail with a physical-correctness explanation."
        ),
    )
    parser.add_argument(
        "--matrix-group",
        default="main",
        help=(
            "Matrix row/group label propagated from "
            "PaDIS_run_reconstruction_matrix.py for verification."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--algorithm",
        choices=("dps_langevin", "langevin", "pc"),
        default="dps_langevin",
    )
    parser.add_argument(
        "--method",
        choices=RECONSTRUCTION_METHOD_CHOICES,
        default="padis_dps",
        help=(
            "Paper-comparison reconstruction method. Diffusion methods reuse "
            "the PaDIS sampler with method-specific prior/algorithm settings; "
            "baseline, CP-TV, and PnP-ADMM use LION-native reconstruction paths."
        ),
    )
    parser.add_argument(
        "--prior-mode",
        choices=("auto", "patch", "whole-image"),
        default="auto",
        help="Use checkpoint prior mode, or override with patch PaDIS / whole-image diffusion.",
    )
    parser.add_argument(
        "--measurement-source",
        choices=("normal", "reconstruction"),
        default="normal",
        help="Manual dataset mode only. Ignored when --experiment is set.",
    )
    parser.add_argument(
        "--public-padis-image-dir",
        type=pathlib.Path,
        default=None,
        help="Use PaDIS-style PNG slices as the image-prior dataset.",
    )
    parser.add_argument(
        "--public-reference-reconstructions",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional public PaDIS reference reconstructions as .npz, .pt, or "
            "a single PNG. When supplied, public-reference similarity metrics "
            "and gates can be used."
        ),
    )
    parser.add_argument("--noise", choices=("none", "low-dose"), default="none")
    parser.add_argument("--noise-i0", type=float, default=3500)
    parser.add_argument("--noise-sigma", type=float, default=5)
    parser.add_argument("--noise-cross-talk", type=float, default=0.05)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=25,
        help="Number of test/validation slices to reconstruct. Default 25 matches the CT evaluation budget of Hu et al.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument(
        "--paper-ct-sampling",
        action="store_true",
        help=(
            "Deprecated alias for --implementation paper. Use the strict "
            "PaDIS paper CT sampler: noise init, 100 outer steps, 10 inner "
            "steps, sigma_max=10."
        ),
    )
    parser.add_argument(
        "--public-padis-ct-sampling",
        action="store_true",
        help=(
            "Deprecated alias for --implementation public_repo. Use the "
            "public PaDIS CT-script compatibility sampler mechanics with the "
            "paper CT sigma schedule."
        ),
    )
    parser.add_argument(
        "--public-repo-sigma-schedule",
        choices=("paper", "readme"),
        default="paper",
        help=(
            "Sigma schedule for --implementation public_repo. Default 'paper' "
            "uses the geometric CT schedule of Hu et al.; 'readme' uses the literal "
            "public README/default EDM schedule for legacy comparisons."
        ),
    )
    parser.add_argument(
        "--public-repo-helper-initialization",
        action="store_true",
        help=(
            "For --implementation public_repo and the public helper methods "
            "predictor_corrector/langevin/ve_ddnm, use the helper functions' "
            "Gaussian initial state instead of the README DPS FDK initial "
            "state. This is for output-level comparisons against "
            "PaDIS_lion_recon --sampler pc|langevin|ddnm."
        ),
    )
    parser.add_argument(
        "--lion-quality-ct-sampling",
        action="store_true",
        help=(
            "Deprecated alias for --implementation lion_quality. "
            "Use the preferred LION-native CT sampler: paper CT schedule and "
            "squared-residual objective with FDK init, Hann 0.9 filtering, "
            "and operator-norm data-consistency scaling."
        ),
    )
    parser.add_argument(
        "--paper-ct-views",
        type=int,
        choices=(8, 20),
        default=20,
        help="Select the PaDIS paper CT sigma_min for --paper-ct-sampling.",
    )
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--inner-steps", type=int, default=None)
    parser.add_argument("--sigma-min", type=float, default=None)
    parser.add_argument("--sigma-max", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument(
        "--noise-schedule",
        choices=("edm", "geometric"),
        default=None,
        help="Sigma schedule. Hu et al. use geometric; the public PaDIS script uses edm/rho.",
    )
    parser.add_argument("--zeta", type=float, default=None)
    parser.add_argument(
        "--initial-reconstruction",
        choices=("noise", "fdk", "inverse"),
        default=None,
        help=(
            "Override sampler initialization. Paper preset defaults to noise; "
            "public/default presets default to FDK."
        ),
    )
    parser.add_argument(
        "--initial-fdk-filter-type",
        choices=("none", "ram-lak", "hann", "hamming", "cosine", "shepp-logan"),
        default=None,
        help="Optional FDK ramp-window filter for FDK initialization.",
    )
    parser.add_argument(
        "--initial-fdk-frequency-scaling",
        type=float,
        default=None,
        help="Cutoff frequency fraction for the initial FDK filter window.",
    )
    parser.set_defaults(initial_fdk_padded=None)
    parser.add_argument(
        "--initial-fdk-padded",
        dest="initial_fdk_padded",
        action="store_true",
        help="Pad projections during initial FDK filtering.",
    )
    parser.add_argument(
        "--no-initial-fdk-padded",
        dest="initial_fdk_padded",
        action="store_false",
        help="Do not pad projections during initial FDK filtering.",
    )
    parser.add_argument("--initial-fdk-batch-size", type=int, default=None)
    parser.add_argument("--dps-epsilon", type=float, default=None)
    parser.add_argument("--sampling-epsilon", type=float, default=None)
    parser.add_argument(
        "--data-consistency-gradient",
        choices=("norm", "least_squares", "paper_squared_residual"),
        default=None,
        help=(
            "DPS measurement gradient. Hu et al. use paper_squared_residual; "
            "LION-physics uses least_squares; the public repo uses norm."
        ),
    )
    parser.add_argument(
        "--adjoint-data-step-schedule",
        choices=("paper", "public_repo"),
        default=None,
        help="Adjoint data-step schedule for Langevin/PC samplers.",
    )
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--pad-width", type=int, default=None)
    parser.add_argument("--patch-batch-size", type=int, default=None)
    parser.set_defaults(patch_checkpoint_denoiser=None)
    parser.add_argument(
        "--patch-checkpoint-denoiser",
        dest="patch_checkpoint_denoiser",
        action="store_true",
        help=(
            "Use activation checkpointing for ordinary PaDIS patch denoiser "
            "batches. This reduces peak memory during DPS data-gradient steps "
            "without changing the CT objective or sigma schedule."
        ),
    )
    parser.add_argument(
        "--no-patch-checkpoint-denoiser",
        dest="patch_checkpoint_denoiser",
        action="store_false",
        help="Disable activation checkpointing for ordinary PaDIS patch denoiser batches.",
    )
    parser.add_argument(
        "--patch-assembly",
        choices=("padis", "fixed_average", "fixed_stitch"),
        default=None,
        help=(
            "Patch score assembly mode. The default PaDIS mode uses shifted "
            "non-overlapping layouts; fixed_average/fixed_stitch implement the "
            "paper comparison patch averaging/stitching forms."
        ),
    )
    parser.add_argument(
        "--patch-overlap",
        type=int,
        default=None,
        help="Overlap in pixels for fixed patch averaging/stitching. Paper default is 8.",
    )
    parser.add_argument(
        "--fixed-overlap-layout",
        choices=("lion_clipped", "public_overlap", "public_tile"),
        default=None,
        help=(
            "Patch start rule for fixed patch averaging/stitching. "
            "public_overlap/public_tile mirror the public PaDIS helper "
            "functions; lion_clipped is the original LION-safe default."
        ),
    )
    parser.set_defaults(fixed_overlap_checkpoint_denoiser=None)
    parser.add_argument(
        "--fixed-overlap-checkpoint-denoiser",
        dest="fixed_overlap_checkpoint_denoiser",
        action="store_true",
        help=(
            "Use activation checkpointing for fixed-overlap patch "
            "averaging/stitching denoiser batches. This preserves gradients "
            "but recomputes model batches during the DPS data-gradient step to "
            "reduce peak memory."
        ),
    )
    parser.add_argument(
        "--no-fixed-overlap-checkpoint-denoiser",
        dest="fixed_overlap_checkpoint_denoiser",
        action="store_false",
        help=(
            "Disable activation checkpointing for fixed-overlap patch "
            "averaging/stitching denoiser batches."
        ),
    )
    parser.add_argument(
        "--langevin-ddnm",
        action="store_true",
        help="Use VE-DDNM correction inside the Langevin sampler.",
    )
    parser.set_defaults(
        ddnm_pseudoinverse_clip=None,
        ddnm_projected_pseudoinverse_clip=None,
        ddnm_corrected_clip=None,
    )
    parser.add_argument(
        "--ddnm-pseudoinverse-clip",
        dest="ddnm_pseudoinverse_clip",
        action="store_true",
        help=(
            "Clip the measured pseudoinverse A^dagger y used by VE-DDNM. "
            "Enabled by default for --method ve_ddnm under LION fan-beam geometry."
        ),
    )
    parser.add_argument(
        "--no-ddnm-pseudoinverse-clip",
        dest="ddnm_pseudoinverse_clip",
        action="store_false",
        help="Disable clipping of the measured VE-DDNM pseudoinverse.",
    )
    parser.add_argument(
        "--ddnm-projected-pseudoinverse-clip",
        dest="ddnm_projected_pseudoinverse_clip",
        action="store_true",
        help=(
            "Clip the A^dagger A(D) pseudoinverse term used by VE-DDNM. "
            "Enabled by default for --method ve_ddnm to keep LION fan-beam "
            "runs finite."
        ),
    )
    parser.add_argument(
        "--no-ddnm-projected-pseudoinverse-clip",
        dest="ddnm_projected_pseudoinverse_clip",
        action="store_false",
        help=(
            "Disable clipping of the A^dagger A(D) VE-DDNM term. This is "
            "closer to the formula of Hu et al. and the public implementation but can be unstable with "
            "LION fan-beam FDK."
        ),
    )
    parser.add_argument(
        "--ddnm-corrected-clip",
        dest="ddnm_corrected_clip",
        action="store_true",
        help=(
            "Clip the corrected VE-DDNM clean estimate "
            "A^dagger y + D - A^dagger A(D) before forming the score. "
            "This is a LION-stability diagnostic, not the formula of Hu et al."
        ),
    )
    parser.add_argument(
        "--no-ddnm-corrected-clip",
        dest="ddnm_corrected_clip",
        action="store_false",
        help="Disable clipping of the corrected VE-DDNM clean estimate.",
    )
    parser.add_argument(
        "--diagnose-ddnm-pseudoinverse",
        action="store_true",
        help=(
            "Record DDNM pseudoinverse diagnostics in metrics.json by applying "
            "A^dagger y + x - A^dagger A(x) to the target image. This is a "
            "debugging aid for checking whether the LION pseudoinverse is "
            "accurate enough for VE-DDNM."
        ),
    )
    parser.add_argument(
        "--ve-ddnm-nfe-layout",
        choices=("paper_1000x1", "public_inner"),
        default=None,
        help=(
            "How VE-DDNM spends its 1000 neural function evaluations. "
            "paper_1000x1 uses 1000 descending sigma levels with one denoise "
            "per level, matching Algorithm A.3 literally. public_inner uses "
            "100 outer sigma levels and 10 inner denoising updates per level, "
            "matching the public helper implementation."
        ),
    )
    parser.add_argument("--langevin-noise-scale", type=float, default=1.0)
    parser.add_argument(
        "--pc-corrector-step-rule",
        choices=("paper_linear", "score_sde_squared"),
        default="paper_linear",
        help=(
            "Predictor-corrector corrector step-size rule. Hu et al. and "
            "public PaDIS code use paper_linear; score_sde_squared is retained "
            "only as a diagnostic score-SDE variant."
        ),
    )
    parser.add_argument(
        "--pc-snr",
        type=float,
        default=None,
        help=(
            "Signal-to-noise ratio used by the predictor-corrector "
            "corrector step. Defaults to the sampler preset value."
        ),
    )
    parser.add_argument(
        "--pc-corrector-denoise-sigma",
        choices=("next", "current"),
        default=None,
        help=(
            "Sigma used for the PC corrector denoising call. Paper mode uses "
            "next; public-repo compatibility uses current to mirror the "
            "published code."
        ),
    )
    parser.set_defaults(pc_reuse_predictor_layout=None)
    parser.add_argument(
        "--pc-reuse-predictor-layout",
        dest="pc_reuse_predictor_layout",
        action="store_true",
        help=(
            "Reuse the predictor patch layout for the PC corrector denoising "
            "call. This mirrors the public PaDIS helper; paper mode leaves it "
            "disabled unless this flag is passed."
        ),
    )
    parser.add_argument(
        "--no-pc-reuse-predictor-layout",
        dest="pc_reuse_predictor_layout",
        action="store_false",
        help="Disable PC predictor-layout reuse for the corrector denoising call.",
    )
    parser.add_argument("--data-range", type=float, default=1.0)
    parser.add_argument(
        "--body-threshold",
        type=float,
        default=0.02,
        help="Target-domain threshold for body/anatomy ROI metrics.",
    )
    parser.add_argument(
        "--nonair-threshold",
        type=float,
        default=1e-4,
        help="Target-domain threshold for non-air ROI metrics.",
    )
    parser.add_argument("--body-bbox-padding", type=int, default=8)
    parser.add_argument("--error-vmax", type=float, default=0.10)
    parser.add_argument(
        "--preview-vmax",
        type=float,
        default=0.75,
        help="Upper display window for fixed-window preview comparison images.",
    )
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument(
        "--no-position-channels",
        action="store_true",
        help="Construct the PaDIS prior without x/y position inputs. The checkpoint must use the same architecture.",
    )
    parser.add_argument("--save-previews", action="store_true")
    parser.add_argument("--prog-bar", action="store_true")
    parser.set_defaults(clip_initial=None, clip_output=None)
    parser.add_argument("--clip-initial", dest="clip_initial", action="store_true")
    parser.add_argument("--no-clip-initial", dest="clip_initial", action="store_false")
    parser.add_argument("--clip-output", dest="clip_output", action="store_true")
    parser.add_argument("--no-clip-output", dest="clip_output", action="store_false")
    parser.add_argument(
        "--clip-denoised",
        action="store_true",
        help="Clamp each clean denoised patch-assembled estimate to the model data range before score/data updates.",
    )
    parser.add_argument(
        "--clip-state",
        action="store_true",
        help="Clamp the noisy sampler state to the model data range after each update. Intended as an ablation.",
    )
    parser.add_argument("--disable-data-consistency", action="store_true")
    parser.add_argument("--disable-langevin-noise", action="store_true")
    parser.add_argument("--disable-prior-score", action="store_true")
    parser.add_argument(
        "--tv-lambda",
        type=float,
        default=0.001,
        help="TV regularisation weight. Hu et al. use 0.001 for CT; fixed-validation also selected 0.001 for the LION TV substitute.",
    )
    parser.add_argument("--tv-iterations", type=int, default=1000)
    parser.add_argument("--tv-lipschitz", type=float, default=None)
    parser.add_argument("--tv-non-negativity", action="store_true")
    parser.add_argument(
        "--pnp-checkpoint",
        type=pathlib.Path,
        default=None,
        help="DRUNet denoiser checkpoint for --method pnp_admm.",
    )
    parser.add_argument("--pnp-iterations", type=int, default=60)
    parser.add_argument("--pnp-eta", type=float, default=3e-5)
    parser.add_argument("--pnp-cg-iterations", type=int, default=50)
    parser.add_argument("--pnp-cg-tolerance", type=float, default=1e-7)
    parser.set_defaults(pnp_clip=True)
    parser.add_argument(
        "--pnp-clip",
        dest="pnp_clip",
        action="store_true",
        help="Clip PnP-ADMM iterates and denoiser outputs to the normalized image support [0, 1].",
    )
    parser.add_argument(
        "--no-pnp-clip",
        dest="pnp_clip",
        action="store_false",
        help="Disable PnP-ADMM iterate clipping.",
    )
    parser.add_argument(
        "--pnp-noise-level",
        type=float,
        default=None,
        help="Optional denoiser noise-level input for DRUNet checkpoints trained with noise channels.",
    )
    parser.add_argument(
        "--data-consistency-normalization",
        choices=("operator_norm", "operator_lipschitz", "none"),
        default=None,
        help=(
            "Optionally scale data-consistency updates by the composed "
            "measurement operator norm or least-squares Lipschitz constant."
        ),
    )
    parser.add_argument(
        "--data-consistency-scale",
        type=float,
        default=None,
        help="Extra multiplier after data-consistency normalisation.",
    )
    parser.add_argument(
        "--adjoint-data-consistency-scale",
        type=float,
        default=None,
        help=(
            "Optional separate multiplier for direct adjoint residual updates "
            "used by Langevin and predictor-corrector. Defaults to "
            "--data-consistency-scale when unset."
        ),
    )
    parser.set_defaults(consume_discarded_measurement_noise=None)
    parser.add_argument(
        "--consume-discarded-measurement-noise",
        dest="consume_discarded_measurement_noise",
        action="store_true",
        help=(
            "Burn the public PaDIS script's zero-noise measurement RNG draw. "
            "This preserves exact public RNG alignment."
        ),
    )
    parser.add_argument(
        "--no-consume-discarded-measurement-noise",
        dest="consume_discarded_measurement_noise",
        action="store_false",
        help=(
            "Skip the public PaDIS script's zero-noise measurement RNG draw. "
            "This keeps the public sampler form but can improve reconstruction quality."
        ),
    )
    parser.add_argument(
        "--data-consistency-scale-schedule",
        choices=("constant", "edm", "inverse_sigma"),
        default="constant",
        help=(
            "Sigma-dependent multiplier for data consistency. 'edm' uses "
            "sigma_data^2/(sigma^2+sigma_data^2); 'inverse_sigma' uses "
            "sigma_min/sigma."
        ),
    )
    parser.add_argument(
        "--data-consistency-scale-power",
        type=float,
        default=1.0,
        help="Power applied to the selected sigma-dependent data-consistency factor.",
    )
    parser.add_argument(
        "--data-consistency-scale-floor",
        type=float,
        default=0.0,
        help="Minimum schedule factor before multiplying by --data-consistency-scale.",
    )
    parser.add_argument(
        "--operator-norm",
        type=float,
        default=None,
        help="Optional precomputed norm of the CT operator.",
    )
    parser.add_argument("--operator-norm-iterations", type=int, default=20)
    parser.add_argument("--operator-norm-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--run-ablations",
        action="store_true",
        help=(
            "Run baseline plus no_data_consistency, no_langevin_noise, "
            "and no_prior_score variants into separate folders."
        ),
    )
    parser.add_argument(
        "--min-mean-psnr",
        type=float,
        default=None,
        help="Fail the run if the mean reconstruction PSNR is below this value.",
    )
    parser.add_argument(
        "--min-mean-ssim",
        type=float,
        default=None,
        help="Fail the run if the mean reconstruction SSIM is below this value.",
    )
    parser.add_argument(
        "--max-mean-mae",
        type=float,
        default=None,
        help="Fail the run if mean normalized MAE to the target is above this value.",
    )
    parser.add_argument(
        "--max-sample-mae",
        type=float,
        default=None,
        help="Fail the run if any normalized MAE to the target is above this value.",
    )
    parser.add_argument(
        "--max-mean-abs-error-p95",
        type=float,
        default=None,
        help="Fail the run if mean target p95 absolute error is above this value.",
    )
    parser.add_argument(
        "--max-sample-abs-error-p95",
        type=float,
        default=None,
        help="Fail the run if any target p95 absolute error is above this value.",
    )
    parser.add_argument(
        "--min-sample-ssim",
        type=float,
        default=None,
        help="Fail the run if any individual reconstruction SSIM is below this value.",
    )
    parser.add_argument(
        "--min-mean-edge-ssim",
        type=float,
        default=None,
        help="Fail the run if the mean Sobel-edge SSIM to the target is below this value.",
    )
    parser.add_argument(
        "--min-sample-edge-ssim",
        type=float,
        default=None,
        help="Fail the run if any Sobel-edge SSIM to the target is below this value.",
    )
    parser.add_argument(
        "--min-mean-reference-ssim",
        type=float,
        default=None,
        help="Fail the run if mean SSIM to --public-reference-reconstructions is below this value.",
    )
    parser.add_argument(
        "--min-sample-reference-ssim",
        type=float,
        default=None,
        help="Fail the run if any SSIM to --public-reference-reconstructions is below this value.",
    )
    parser.add_argument(
        "--min-mean-reference-edge-ssim",
        type=float,
        default=None,
        help="Fail the run if mean Sobel-edge SSIM to the public reference is below this value.",
    )
    parser.add_argument(
        "--max-mean-reference-mae",
        type=float,
        default=None,
        help="Fail the run if mean MAE to --public-reference-reconstructions is above this value.",
    )
    parser.add_argument(
        "--max-mean-reference-abs-error-p95",
        type=float,
        default=None,
        help="Fail the run if mean public-reference p95 absolute error is above this value.",
    )
    parser.add_argument(
        "--min-sample-psnr",
        type=float,
        default=None,
        help="Fail the run if any individual reconstruction PSNR is below this value.",
    )
    parser.add_argument(
        "--require-better-than-fdk",
        action="store_true",
        help="Fail the run if PaDIS does not improve mean PSNR over FDK.",
    )
    parser.add_argument(
        "--require-each-better-than-fdk",
        action="store_true",
        help="Fail the run if any individual PaDIS reconstruction does not improve PSNR over FDK.",
    )
    parser.add_argument(
        "--trace-interval",
        type=int,
        default=0,
        help="Save sampler diagnostics every N outer steps. Set 1 for every step.",
    )
    parser.add_argument(
        "--trace-images",
        action="store_true",
        help=(
            "Save denoised, projected, and forward-projected trace snapshots. "
            "Uses --trace-interval; defaults to every 5 outer steps when no "
            "trace interval is set."
        ),
    )
    parser.add_argument(
        "--stop-after-outer-steps",
        type=int,
        default=None,
        help=(
            "Debugging aid: stop after this many outer sampler steps while "
            "preserving the full configured sigma schedule."
        ),
    )
    return parser


def main() -> None:
    """Run one configured reconstruction method over an LIDC split subset."""
    args = build_arg_parser().parse_args()
    args.method = canonical_method(args.method)
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    legacy_implementations = []
    if args.paper_ct_sampling:
        legacy_implementations.append("paper")
    if args.public_padis_ct_sampling:
        legacy_implementations.append("public_repo")
    if args.lion_quality_ct_sampling:
        legacy_implementations.append("lion_quality")
    if len(legacy_implementations) > 1:
        raise ValueError(
            "--paper-ct-sampling, --public-padis-ct-sampling, and "
            "--lion-quality-ct-sampling are mutually exclusive."
        )
    if legacy_implementations:
        if args.implementation != "custom":
            raise ValueError(
                "Use either --implementation or the deprecated --*-ct-sampling "
                "flags, not both."
            )
        args.implementation = legacy_implementations[0]
    if args.experiment != "none":
        args.experiment = canonical_experiment_name(args.experiment)
    if args.geometry != "lion":
        raise ValueError(UNSUPPORTED_PADIS_GEOMETRY_MESSAGE)
    validate_public_repo_method(args.implementation, args.method)
    if args.trace_interval < 0:
        raise ValueError("--trace-interval must be non-negative.")
    if args.stop_after_outer_steps is not None and args.stop_after_outer_steps <= 0:
        raise ValueError("--stop-after-outer-steps must be positive when set.")
    if args.patch_overlap is not None and args.patch_overlap < 0:
        raise ValueError("--patch-overlap must be non-negative.")
    if args.run_ablations and args.method not in DIFFUSION_RECONSTRUCTION_METHODS:
        raise ValueError("--run-ablations is only supported for diffusion methods.")
    if args.method == "pnp_admm" and args.pnp_checkpoint is None:
        raise ValueError("--method pnp_admm requires --pnp-checkpoint.")
    if args.trace_images and args.trace_interval == 0:
        args.trace_interval = 5

    set_run_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type == "cpu":
        if args.experiment != "none":
            raise RuntimeError(
                "CUDA was requested but is not available. LION CT reconstruction "
                "experiments use CUDA-backed tomography operators; run on a GPU "
                "node or use a non-CT/manual configuration."
            )
        print("CUDA was requested but is not available; using CPU.")

    if args.method in NO_PADIS_PRIOR_METHODS:
        checkpoint_path = None
        metadata_image_scaling = float(args.image_scaling)
        spec = experiment_spec_from_args(args)
        if spec is not None and spec.key == "ct_512_60":
            metadata_image_scaling = 1.0
        paper_preset = (
            "padis-paper-ct-512"
            if spec is not None and spec.key == "ct_512_60"
            else "padis-paper-ct-256"
        )
        if args.checkpoint != DEFAULT_CHECKPOINT:
            try:
                checkpoint_path = resolve_checkpoint_path(args.checkpoint)
            except FileNotFoundError:
                checkpoint_path = None
        if checkpoint_path is None:
            model_params, geometry = fallback_metadata(
                image_scaling=metadata_image_scaling,
                disable_position_channels=args.no_position_channels,
                paper_preset=paper_preset,
            )
        else:
            model_params, geometry = load_checkpoint_metadata(
                checkpoint_path,
                image_scaling=metadata_image_scaling,
                disable_position_channels=args.no_position_channels,
            )
        model = None
    else:
        checkpoint_path = resolve_checkpoint_path(args.checkpoint)
        model, model_params, geometry = load_model(
            checkpoint_path,
            device,
            use_ema=not args.raw_weights,
            disable_position_channels=args.no_position_channels,
        )
    reconstruction_geometry = geometry

    if args.experiment == "none":
        if args.public_padis_image_dir is not None:
            dataset = PNGImagePriorDataset(
                args.public_padis_image_dir,
                channels=int(geometry.image_shape[0]),
            )
        else:
            dataset = build_dataset(args, geometry)
        experiment = None
        from_experiment = False
        experiment_measurement_source = args.measurement_source
    else:
        dataset, reconstruction_geometry, experiment = build_experiment_dataset(
            args, geometry
        )
        from_experiment = True
        experiment_measurement_source = getattr(
            experiment.param, "measurement_source", "reconstruction"
        )
        if args.public_padis_image_dir is not None:
            if experiment_measurement_source != "normal":
                raise ValueError(
                    "--public-padis-image-dir with --experiment is only supported "
                    "for image-prior/normal-domain experiments."
                )
            dataset = PNGImagePriorDataset(
                args.public_padis_image_dir,
                channels=int(reconstruction_geometry.image_shape[0]),
            )
        if args.noise != "none":
            print(
                "--noise is ignored when --experiment is set; using experiment noise."
            )
    if args.start_index >= len(dataset):
        raise ValueError(
            f"--start-index {args.start_index} is outside the {args.split} dataset of length {len(dataset)}."
        )
    reference_reconstructions = load_reference_reconstructions(
        args.public_reference_reconstructions
    )
    if reference_reconstructions is not None:
        print(
            "Loaded public reference reconstructions from "
            f"{args.public_reference_reconstructions} with shape "
            f"{tuple(reference_reconstructions.shape)}."
        )

    sampler_params = build_sampler_params(
        args, model, measurement_source=experiment_measurement_source
    )
    print(
        "PaDIS reconstruction preset: "
        f"method={args.method}, implementation={args.implementation}, "
        f"geometry={args.geometry}, "
        f"experiment={args.experiment}, "
        f"schedule={sampler_params.noise_schedule}, "
        f"sigma_min={sampler_params.sigma_min}, "
        f"sigma_max={sampler_params.sigma_max}"
    )
    if args.run_ablations:
        if (
            args.disable_data_consistency
            or args.disable_langevin_noise
            or args.disable_prior_score
        ):
            print("--disable-* flags are ignored by --run-ablations baseline.")
        sampler_params.disable_data_consistency = False
        sampler_params.disable_langevin_noise = False
        sampler_params.disable_prior_score = False
    if args.experiment != "none":
        run_name = args.experiment
    else:
        run_name = "manual"
    output_root = (
        args.output_folder
        / run_name
        / args.split
        / args.method
        / effective_algorithm(args)
    )

    if args.run_ablations:
        variants = ABLATION_VARIANTS.items()
        run_folder = output_root / "ablations"
    else:
        variants = (("run", {}),)
        run_folder = output_root

    summaries = []
    for variant_name, variant_overrides in variants:
        variant_folder = run_folder / variant_name if args.run_ablations else run_folder
        summaries.append(
            run_reconstruction_variant(
                args=args,
                dataset=dataset,
                checkpoint_path=checkpoint_path,
                geometry=geometry,
                reconstruction_geometry=reconstruction_geometry,
                experiment=experiment,
                model=model,
                model_params=model_params,
                base_params=sampler_params,
                variant_name=variant_name,
                variant_overrides=variant_overrides,
                output_folder=variant_folder,
                device=device,
                from_experiment=from_experiment,
                experiment_measurement_source=experiment_measurement_source,
                reference_reconstructions=reference_reconstructions,
            )
        )

    enforce_quality_gates(args, summaries)

    if args.run_ablations:
        summary_path = run_folder / "ablation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "split": args.split,
                    "experiment": args.experiment,
                    "method": args.method,
                    "algorithm": effective_algorithm(args),
                    "seed": args.seed,
                    "variants": summaries,
                },
                f,
                indent=2,
            )
        print(f"Saved ablation summary to {summary_path}")


if __name__ == "__main__":
    main()
