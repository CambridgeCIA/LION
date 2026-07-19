"""Run PaDIS DPS or Langevin CT reconstruction on LIDC-IDRI splits."""

from __future__ import annotations

import json
import os
import pathlib

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

from LION.classical_algorithms.fdk import fdk
from LION.classical_algorithms.tv_min import tv_min
from LION.reconstructors import PaDIS
from LION.reconstructors.PnP import PnP
from LION.utils.parameter import LIONParameter

from PaDIS_identifiers import canonical_method
from padis_lidc import (  # noqa: F401
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
    PNGImagePriorDataset,
    PnPDenoiser,
    PaperCTExperiment,
    _checkpoint_geometry,
    _checkpoint_paper_preset,
    add_ddnm_pseudoinverse_diagnostics,
    add_image_similarity_metrics,
    add_reconstruction_metrics,
    build_arg_parser,
    build_dataset,
    build_experiment_dataset,
    build_sampler_params,
    canonical_experiment_name,
    checkpoint_model_metadata,
    clone_parameters,
    crop_bbox,
    edge_ssim_or_none,
    enforce_quality_gates,
    experiment_class_for_geometry,
    experiment_spec_from_args,
    fallback_metadata,
    forward_project_normal_image,
    hu_window,
    image_tensor_from_array,
    load_checkpoint_metadata,
    load_model,
    load_pnp_denoiser,
    make_measurement,
    mask_bbox,
    masked_mae,
    masked_mse,
    mean_metric,
    min_metric,
    mu_to_lidc_normal,
    normal_to_hu,
    project_root,
    psnr,
    psnr_from_mse,
    relative_sinogram_residual,
    resolve_checkpoint_path,
    save_preview,
    save_tensor_image,
    save_trace_images,
    save_visual_comparison,
    set_run_seed,
    ssim_on_bbox_or_none,
    ssim_or_none,
    torch_load,
    validate_public_repo_method,
)


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
