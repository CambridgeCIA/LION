"""Run PaDIS DPS or Langevin CT reconstruction on LIDC-IDRI splits."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pathlib
import warnings

_CACHE_ROOT = pathlib.Path("/tmp") / "lion_matplotlib_cache"
(_CACHE_ROOT / "mpl").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import torch
from tqdm import tqdm

from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import sinogram_add_noise
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
import LION.experiments.ct_experiments as ct_experiments
from LION.classical_algorithms.fdk import fdk
from LION.models.diffusion import NCSNpp
from LION.reconstructors import PaDIS
from LION.utils.parameter import LIONParameter
from LION.utils.paths import LION_EXPERIMENTS_PATH


DEFAULT_CHECKPOINT = pathlib.Path(
    "Data/experiments/PaDIS/LIDC_256/"
    "padis_lidc_256_reproduction_CSD3/padis_lidc_256.pt"
)

LIDC_EXPERIMENTS = {
    "PaDISFanBeam8CTRecon": ct_experiments.PaDISFanBeam8CTRecon,
    "PaDISFanBeam20CTRecon": ct_experiments.PaDISFanBeam20CTRecon,
    "PaDISFanBeam60CTRecon": ct_experiments.PaDISFanBeam60CTRecon,
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


def project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def torch_load(path: pathlib.Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_checkpoint_path(path: pathlib.Path) -> pathlib.Path:
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
    json_path = checkpoint_path.with_suffix(".json")
    if json_path.is_file():
        options = LIONParameter()
        options.load(json_path)
        if getattr(options, "model_name", "NCSNpp") != "NCSNpp":
            warnings.warn(
                f"{json_path} says model_name={options.model_name!r}; trying NCSNpp anyway."
            )
        model_params = options.model_parameters
        geometry = Geometry.init_from_parameter(options.geometry)
    else:
        warnings.warn(
            f"No sidecar JSON found at {json_path}; using PaDIS LIDC 256 defaults."
        )
        model_params = NCSNpp.default_parameters("padis-paper-ct-256")
        geometry = Geometry.default_parameters(image_scaling=0.5)

    if disable_position_channels:
        model_params.input_position_channels = 0

    model = NCSNpp(model_params, geometry).to(device)
    checkpoint = torch_load(checkpoint_path, map_location=device)
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


def build_dataset(args, geometry):
    task = "image_prior" if args.measurement_source == "normal" else "reconstruction"
    data_params = LIDC_IDRI.default_parameters(geometry=geometry, task=task)
    data_params.device = torch.device("cpu")
    if args.data_folder is not None:
        data_params.folder = args.data_folder
    return LIDC_IDRI(args.split, parameters=data_params, geometry_parameters=geometry)


def build_experiment_dataset(args, checkpoint_geometry):
    image_scaling = float(getattr(checkpoint_geometry, "image_scaling", 1.0))
    experiment_cls = LIDC_EXPERIMENTS[args.experiment]
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
    mse = torch.mean((recon - target).square()).item()
    if mse == 0:
        return float("inf")
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse))


def psnr_from_mse(mse: float, data_range: float) -> float:
    if mse == 0:
        return float("inf")
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse))


def mean_metric(metrics: list[dict], key: str):
    values = [item[key] for item in metrics if key in item]
    if not values:
        return None
    return sum(values) / len(values)


def masked_mse(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    mask = mask.to(device=recon.device, dtype=torch.bool)
    if not torch.any(mask):
        return None
    return float(torch.mean((recon[mask] - target[mask]).square()).item())


def masked_mae(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    mask = mask.to(device=recon.device, dtype=torch.bool)
    if not torch.any(mask):
        return None
    return float(torch.mean(torch.abs(recon[mask] - target[mask])).item())


def ssim_or_none(recon: torch.Tensor, target: torch.Tensor, data_range: float):
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return None
    recon_np = recon.detach().cpu().squeeze().numpy()
    target_np = target.detach().cpu().squeeze().numpy()
    return float(structural_similarity(target_np, recon_np, data_range=data_range))


def normal_to_hu(image: torch.Tensor) -> torch.Tensor:
    return 3000.0 * image - 1000.0


def hu_window(image: torch.Tensor, *, level: float, width: float) -> torch.Tensor:
    lower = level - width / 2.0
    return ((normal_to_hu(image) - lower) / width).clamp(0.0, 1.0)


def mask_bbox(mask: torch.Tensor, pad: int = 8):
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
    return ssim_or_none(crop_bbox(recon, bbox), crop_bbox(target, bbox), data_range)


def measurement_image_to_sampler_domain(image: torch.Tensor, params) -> torch.Tensor:
    scale = float(params.measurement_scale)
    if scale == 0:
        raise ValueError("measurement_scale must be non-zero.")
    return (image - float(params.measurement_offset)) / scale


def fdk_baseline(sinogram: torch.Tensor, reconstructor: PaDIS, params) -> torch.Tensor:
    with torch.no_grad():
        reconstruction = fdk(sinogram, reconstructor.op, clip=True)
        reconstruction = measurement_image_to_sampler_domain(reconstruction, params)
        return reconstruction.clamp(0.0, 1.0)


def forward_project_normal_image(
    image: torch.Tensor,
    reconstructor: PaDIS,
    params,
) -> torch.Tensor:
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
    key = "" if prefix == "" else f"{prefix}_"
    mse = float(torch.mean((recon - target).square()).item())
    item[f"{key}mse"] = mse
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


def save_preview(
    path: pathlib.Path,
    sinogram,
    fdk_recon,
    recon,
    target,
    *,
    body_mask,
    error_vmax: float,
    recon_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
    image_kwargs = {"cmap": "gray", "vmin": 0, "vmax": 1}
    axes[0, 0].imshow(sinogram.detach().cpu().squeeze().T, cmap="gray")
    axes[0, 0].set_title("Sinogram")
    axes[0, 1].imshow(fdk_recon.detach().cpu().squeeze(), **image_kwargs)
    axes[0, 1].set_title("FDK")
    axes[0, 2].imshow(recon.detach().cpu().squeeze(), **image_kwargs)
    axes[0, 2].set_title(recon_label)
    axes[0, 3].imshow(target.detach().cpu().squeeze(), **image_kwargs)
    axes[0, 3].set_title("Target")
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
    for ax in axes.ravel():
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def set_run_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_sampler_params(args, model, *, measurement_source: str) -> LIONParameter:
    if args.paper_ct_sampling:
        sampler_params = PaDIS.paper_ct_parameters(model, views=args.paper_ct_views)
    elif args.public_padis_ct_sampling:
        sampler_params = PaDIS.padis_repo_ct_parameters(model)
    else:
        sampler_params = PaDIS.default_parameters(model)
    sampler_params.num_steps = args.num_steps
    sampler_params.inner_steps = args.inner_steps
    sampler_params.sigma_min = args.sigma_min
    sampler_params.sigma_max = args.sigma_max
    sampler_params.rho = args.rho
    sampler_params.zeta = args.zeta
    sampler_params.initial_reconstruction = args.initial_reconstruction
    if args.paper_ct_sampling:
        paper_params = PaDIS.paper_ct_parameters(model, views=args.paper_ct_views)
        sampler_params.num_steps = paper_params.num_steps
        sampler_params.inner_steps = paper_params.inner_steps
        sampler_params.sigma_min = paper_params.sigma_min
        sampler_params.sigma_max = paper_params.sigma_max
        sampler_params.zeta = paper_params.zeta
        sampler_params.initial_reconstruction = paper_params.initial_reconstruction
        sampler_params.clip_initial = paper_params.clip_initial
        sampler_params.clip_output = paper_params.clip_output
        sampler_params.dps_epsilon = paper_params.dps_epsilon
        sampler_params.sampling_epsilon = paper_params.sampling_epsilon
        sampler_params.data_consistency_gradient = (
            paper_params.data_consistency_gradient
        )
    elif args.public_padis_ct_sampling:
        public_params = PaDIS.padis_repo_ct_parameters(model)
        sampler_params.num_steps = public_params.num_steps
        sampler_params.inner_steps = public_params.inner_steps
        sampler_params.sigma_min = public_params.sigma_min
        sampler_params.sigma_max = public_params.sigma_max
        sampler_params.zeta = public_params.zeta
        sampler_params.initial_reconstruction = public_params.initial_reconstruction
        sampler_params.clip_initial = public_params.clip_initial
        sampler_params.clip_output = public_params.clip_output
        sampler_params.dps_epsilon = public_params.dps_epsilon
        sampler_params.sampling_epsilon = public_params.sampling_epsilon
        sampler_params.data_consistency_gradient = (
            public_params.data_consistency_gradient
        )
    sampler_params.patch_batch_size = args.patch_batch_size
    sampler_params.langevin_ddnm = args.langevin_ddnm
    sampler_params.langevin_noise_scale = args.langevin_noise_scale
    if args.clip_initial is not None:
        sampler_params.clip_initial = args.clip_initial
    if args.clip_output is not None:
        sampler_params.clip_output = args.clip_output
    if args.dps_epsilon is not None:
        sampler_params.dps_epsilon = args.dps_epsilon
    if args.sampling_epsilon is not None:
        sampler_params.sampling_epsilon = args.sampling_epsilon
    if args.data_consistency_gradient is not None:
        sampler_params.data_consistency_gradient = args.data_consistency_gradient
    if args.adjoint_data_step_schedule is not None:
        sampler_params.adjoint_data_step_schedule = args.adjoint_data_step_schedule
    sampler_params.clip_denoised = args.clip_denoised
    sampler_params.clip_state = args.clip_state
    sampler_params.disable_data_consistency = args.disable_data_consistency
    sampler_params.disable_langevin_noise = args.disable_langevin_noise
    sampler_params.disable_prior_score = args.disable_prior_score
    sampler_params.data_consistency_normalization = args.data_consistency_normalization
    sampler_params.data_consistency_scale = args.data_consistency_scale
    sampler_params.data_consistency_scale_schedule = (
        args.data_consistency_scale_schedule
    )
    sampler_params.data_consistency_scale_power = args.data_consistency_scale_power
    sampler_params.data_consistency_scale_floor = args.data_consistency_scale_floor
    sampler_params.operator_norm = args.operator_norm
    sampler_params.operator_norm_iterations = args.operator_norm_iterations
    sampler_params.operator_norm_tolerance = args.operator_norm_tolerance
    sampler_params.trace_interval = args.trace_interval
    if args.prior_mode != "auto":
        sampler_params.prior_mode = (
            "whole_image" if args.prior_mode == "whole-image" else "patch"
        )
    if measurement_source == "reconstruction":
        sampler_params.measurement_scale = LIDC_NORMAL_TO_MU_SCALE
        sampler_params.measurement_offset = LIDC_NORMAL_TO_MU_OFFSET
    if args.patch_size is not None:
        sampler_params.patch_size = args.patch_size
    if args.pad_width is not None:
        sampler_params.pad_width = args.pad_width
    return sampler_params


def clone_parameters(params: LIONParameter) -> LIONParameter:
    copied = LIONParameter()
    for key, value in params.__dict__.items():
        if not key.startswith("_"):
            setattr(copied, key, copy.deepcopy(value))
    return copied


def run_reconstruction_variant(
    *,
    args,
    dataset,
    checkpoint_path: pathlib.Path,
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
) -> dict:
    set_run_seed(args.seed)
    sampler_params = clone_parameters(base_params)
    for key, value in variant_overrides.items():
        setattr(sampler_params, key, value)

    reconstructor = PaDIS(
        reconstruction_geometry,
        model,
        parameters=sampler_params,
        algorithm=args.algorithm,
    )
    output_folder.mkdir(parents=True, exist_ok=True)
    recon_label = (
        "Whole-image diffusion"
        if getattr(sampler_params, "prior_mode", "patch") == "whole_image"
        else "PaDIS"
    )
    print(f"Saving {recon_label} {variant_name} reconstructions to {output_folder}")

    stop = min(len(dataset), args.start_index + args.max_samples)
    metrics = []
    reconstructions = []
    fdk_reconstructions = []
    targets = []
    sinograms = []
    traces = []
    for index in tqdm(
        range(args.start_index, stop),
        desc=f"LIDC {args.split} {variant_name}",
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
        recon = reconstructor.reconstruct_sample(
            sinogram,
            prog_bar=args.prog_bar,
            generator=None,
        )
        fdk_recon = fdk_baseline(sinogram, reconstructor, sampler_params)
        reconstructions.append(recon.detach().cpu())
        fdk_reconstructions.append(fdk_recon.detach().cpu())
        targets.append(target.detach().cpu())
        sinograms.append(sinogram.detach().cpu())
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
        if args.trace_interval > 0:
            traces.append({"index": int(index), "trace": reconstructor.last_trace})
        metrics.append(item)
        if args.save_previews:
            save_preview(
                output_folder / f"sample_{index:04d}.png",
                sinogram,
                fdk_recon,
                recon,
                target,
                body_mask=body_mask,
                error_vmax=float(args.error_vmax),
                recon_label=recon_label,
            )

    payload = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "experiment": args.experiment,
        "algorithm": args.algorithm,
        "ablation": variant_name,
        "ablation_overrides": variant_overrides,
        "measurement_source": experiment_measurement_source,
        "checkpoint_image_scaling": float(getattr(geometry, "image_scaling", 1.0)),
        "reconstruction_geometry": str(reconstruction_geometry),
        "measurement_scale": float(sampler_params.measurement_scale),
        "measurement_offset": float(sampler_params.measurement_offset),
        "prior_mode": getattr(sampler_params, "prior_mode", "patch"),
        "model_patch_size": int(getattr(model_params, "largest_patch_size", -1)),
        "sampler": {
            key: value
            for key, value in sampler_params.__dict__.items()
            if not key.startswith("_")
        },
        "metrics": metrics,
    }
    if experiment is not None:
        payload["experiment_name"] = experiment.param.name
    metric_path = output_folder / "metrics.json"
    with open(metric_path, "w") as f:
        json.dump(payload, f, indent=2)
    trace_path = None
    if traces:
        trace_path = output_folder / "trace.json"
        with open(trace_path, "w") as f:
            json.dump(traces, f, indent=2)

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
    mean_psnr = mean_metric(metrics, "psnr")
    mean_fdk_mse = mean_metric(metrics, "fdk_mse")
    mean_fdk_psnr = mean_metric(metrics, "fdk_psnr")
    mean_body_psnr = mean_metric(metrics, "body_psnr")
    mean_fdk_body_psnr = mean_metric(metrics, "fdk_body_psnr")
    mean_soft_tissue_psnr = mean_metric(metrics, "soft_tissue_window_psnr")
    mean_fdk_soft_tissue_psnr = mean_metric(metrics, "fdk_soft_tissue_window_psnr")
    print(f"{variant_name} mean MSE:  {mean_mse:.6g}")
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
        print(f"{variant_name} mean SSIM: {mean_ssim:.4g}")
    if "fdk_ssim" in metrics[0]:
        mean_fdk_ssim = sum(item["fdk_ssim"] for item in metrics) / len(metrics)
        print(f"{variant_name} FDK mean SSIM: {mean_fdk_ssim:.4g}")
    print(f"Saved metrics to {metric_path}")
    if trace_path is not None:
        print(f"Saved sampler trace to {trace_path}")
    print(f"Saved tensors to {tensor_path}")
    return {
        "name": variant_name,
        "folder": str(output_folder),
        "metrics": str(metric_path),
        "trace": str(trace_path) if trace_path is not None else None,
        "tensors": str(tensor_path),
        "mean_mse": mean_mse,
        "mean_psnr": mean_psnr,
        "mean_fdk_mse": mean_fdk_mse,
        "mean_fdk_psnr": mean_fdk_psnr,
        "mean_body_psnr": mean_body_psnr,
        "mean_fdk_body_psnr": mean_fdk_body_psnr,
        "mean_soft_tissue_window_psnr": mean_soft_tissue_psnr,
        "mean_fdk_soft_tissue_window_psnr": mean_fdk_soft_tissue_psnr,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-folder", type=pathlib.Path, default=None)
    parser.add_argument(
        "--output-folder",
        type=pathlib.Path,
        default=LION_EXPERIMENTS_PATH / "PaDIS" / "LIDC_reconstruction",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--experiment",
        choices=("none", *LIDC_EXPERIMENTS.keys()),
        default="none",
        help="Run a standard LION LIDC CT experiment. image_scaling is read from the PaDIS checkpoint geometry.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--algorithm",
        choices=("dps_langevin", "langevin", "pc"),
        default="dps_langevin",
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
    parser.add_argument("--noise", choices=("none", "low-dose"), default="none")
    parser.add_argument("--noise-i0", type=float, default=3500)
    parser.add_argument("--noise-sigma", type=float, default=5)
    parser.add_argument("--noise-cross-talk", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--paper-ct-sampling",
        action="store_true",
        help="Use the strict PaDIS paper CT sampler: noise init, 100 outer steps, 10 inner steps, sigma_max=10.",
    )
    parser.add_argument(
        "--public-padis-ct-sampling",
        action="store_true",
        help="Use the public PaDIS CT-script compatibility sampler: FDK init and norm-gradient DPS.",
    )
    parser.add_argument(
        "--paper-ct-views",
        type=int,
        choices=(8, 20),
        default=20,
        help="Select the PaDIS paper CT sigma_min for --paper-ct-sampling.",
    )
    parser.add_argument("--num-steps", type=int, default=18)
    parser.add_argument("--inner-steps", type=int, default=10)
    parser.add_argument("--sigma-min", type=float, default=0.005)
    parser.add_argument("--sigma-max", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--zeta", type=float, default=0.3)
    parser.add_argument(
        "--initial-reconstruction",
        choices=("noise", "fdk", "inverse"),
        default="fdk",
        help="'fdk' matches the public PaDIS CT script variant; 'noise' starts from pure diffusion noise.",
    )
    parser.add_argument("--dps-epsilon", type=float, default=None)
    parser.add_argument("--sampling-epsilon", type=float, default=None)
    parser.add_argument(
        "--data-consistency-gradient",
        choices=("norm", "paper_squared_residual"),
        default=None,
        help="DPS measurement gradient. The paper uses paper_squared_residual; the public repo uses norm.",
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
    parser.add_argument(
        "--langevin-ddnm",
        action="store_true",
        help="Use VE-DDNM correction inside the Langevin sampler.",
    )
    parser.add_argument("--langevin-noise-scale", type=float, default=1.0)
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
        "--data-consistency-normalization",
        choices=("operator_norm", "none"),
        default="none",
        help="Optionally scale data-consistency updates by the composed measurement operator norm.",
    )
    parser.add_argument(
        "--data-consistency-scale",
        type=float,
        default=1.0,
        help="Extra multiplier after data-consistency normalisation.",
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
        "--trace-interval",
        type=int,
        default=0,
        help="Save sampler diagnostics every N outer steps. Set 1 for every step.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    if args.paper_ct_sampling and args.public_padis_ct_sampling:
        raise ValueError(
            "--paper-ct-sampling and --public-padis-ct-sampling are mutually exclusive."
        )

    set_run_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type == "cpu":
        print("CUDA was requested but is not available; using CPU.")

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    model, model_params, geometry = load_model(
        checkpoint_path,
        device,
        use_ema=not args.raw_weights,
        disable_position_channels=args.no_position_channels,
    )
    if args.experiment == "none":
        dataset = build_dataset(args, geometry)
        reconstruction_geometry = geometry
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
        if args.noise != "none":
            print(
                "--noise is ignored when --experiment is set; using experiment noise."
            )
    if args.start_index >= len(dataset):
        raise ValueError(
            f"--start-index {args.start_index} is outside the {args.split} dataset of length {len(dataset)}."
        )

    sampler_params = build_sampler_params(
        args, model, measurement_source=experiment_measurement_source
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
    run_name = args.experiment if args.experiment != "none" else "manual"
    output_root = args.output_folder / run_name / args.split / args.algorithm

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
            )
        )

    if args.run_ablations:
        summary_path = run_folder / "ablation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "split": args.split,
                    "experiment": args.experiment,
                    "algorithm": args.algorithm,
                    "seed": args.seed,
                    "variants": summaries,
                },
                f,
                indent=2,
            )
        print(f"Saved ablation summary to {summary_path}")


if __name__ == "__main__":
    main()
