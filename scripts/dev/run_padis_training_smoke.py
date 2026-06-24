"""Smoke-test PaDIS training paths before launching expensive runs.

This script is intentionally small and destructive-free. It can load the real
LIDC image-prior dataset, construct the real PaDIS NCSN++ model, and run a
small number of optimizer steps with explicit patch sizes. Use CUDA only for
tiny synthetic checks unless the target machine has enough VRAM for the full
training batch.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import resource
import sys
import time

import numpy as np
import torch

LION_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(LION_ROOT) not in sys.path:
    sys.path.insert(0, str(LION_ROOT))

from LION.CTtools.ct_geometry import Geometry  # noqa: E402
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI  # noqa: E402
from LION.losses.PaDIS import PaDISDenoisingLoss  # noqa: E402
from LION.models.diffusion import NCSNpp  # noqa: E402
from LION.optimizers import PaDISSolver  # noqa: E402


def _rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _cuda_summary(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {}
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_total_gib": total_bytes / 1024**3,
        "cuda_free_gib": free_bytes / 1024**3,
        "cuda_allocated_gib": torch.cuda.memory_allocated(device) / 1024**3,
        "cuda_reserved_gib": torch.cuda.memory_reserved(device) / 1024**3,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_from_arg(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def _build_lidc_batch(
    args: argparse.Namespace,
    geometry: Geometry,
) -> tuple[torch.Tensor, dict[str, object]]:
    data_params = LIDC_IDRI.default_parameters(geometry=geometry, task="image_prior")
    data_params.device = torch.device("cpu")
    data_params.max_num_slices_per_patient = int(args.max_slices_per_patient)
    data_params.pcg_slices_nodule = float(args.pcg_slices_nodule)
    if args.data_folder is not None:
        data_params.folder = args.data_folder
    dataset = LIDC_IDRI("train", parameters=data_params, geometry_parameters=geometry)
    if len(dataset) == 0:
        raise ValueError("LIDC training dataset is empty.")

    samples = []
    for index in range(args.batch_size):
        _source, target = dataset[index % len(dataset)]
        samples.append(target.float().cpu())
    batch = torch.stack(samples, dim=0).contiguous()
    metadata = {
        "dataset": "LIDC-IDRI",
        "dataset_len": len(dataset),
        "dataset_folder": str(pathlib.Path(data_params.folder).resolve()),
        "max_slices_per_patient": int(data_params.max_num_slices_per_patient),
        "pcg_slices_nodule": float(data_params.pcg_slices_nodule),
        "batch_source": "real_lidc",
    }
    return batch, metadata


def _build_synthetic_batch(
    args: argparse.Namespace, geometry: Geometry
) -> torch.Tensor:
    channels, height, width = geometry.image_shape
    return torch.linspace(
        0.0,
        1.0,
        args.batch_size * channels * height * width,
        dtype=torch.float32,
    ).reshape(args.batch_size, channels, height, width)


def _patch_mode_from_args(args: argparse.Namespace) -> str:
    mode = args.model_mode
    if args.no_position_channels and not mode.endswith("-no-position"):
        mode = f"{mode}-no-position"
    return mode


def _configure_solver_params(args: argparse.Namespace) -> object:
    solver_params = PaDISSolver.default_parameters(_patch_mode_from_args(args))
    solver_params.patch_sizes = [int(args.patch_size)]
    solver_params.patch_probabilities = [1.0]
    solver_params.patch_batch_multipliers = {int(args.patch_size): 1}
    solver_params.base_patch_batch_size = int(args.batch_size)
    solver_params.microbatch_size = args.microbatch_size
    solver_params.use_ema = not args.no_ema
    solver_params.lr_rampup_kimg = args.lr_rampup_kimg
    solver_params.enforce_data_range = True
    if args.sigma_distribution is not None:
        solver_params.sigma_distribution = args.sigma_distribution
    return solver_params


def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    _set_seed(args.seed)
    device = _device_from_arg(args.device)
    geometry = Geometry.default_parameters(image_scaling=args.image_scaling)
    mode = _patch_mode_from_args(args)

    if args.synthetic_data:
        target = _build_synthetic_batch(args, geometry)
        data_metadata = {"batch_source": "synthetic"}
    else:
        target, data_metadata = _build_lidc_batch(args, geometry)

    if torch.amin(target) < -1e-5 or torch.amax(target) > 1 + 1e-5:
        raise ValueError(
            "Training target is outside [0, 1], which would fail PaDIS training."
        )

    model_params = NCSNpp.default_parameters(mode)
    model_start = time.perf_counter()
    model = NCSNpp(model_params, geometry)
    model_seconds = time.perf_counter() - model_start
    trainable_params = sum(param.numel() for param in model.parameters())

    loss_fn = PaDISDenoisingLoss(
        sigma_min=model_params.sigma_min,
        sigma_max=model_params.sigma_max,
        sigma_distribution=args.sigma_distribution
        or PaDISSolver.default_parameters(mode).sigma_distribution,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    solver = PaDISSolver(
        model,
        optimizer,
        loss_fn,
        geometry=geometry,
        verbose=False,
        device=device,
        solver_params=_configure_solver_params(args),
    )
    solver.seen_patches = int(args.initial_seen_patches)

    target = target.to(device, non_blocking=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    losses = []
    step_seconds = []
    for _step in range(args.steps):
        torch.manual_seed(args.seed + _step)
        start = time.perf_counter()
        loss_value = solver._optimizer_step(target, patch_size=args.patch_size)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_seconds.append(time.perf_counter() - start)
        losses.append(float(loss_value))
        if not np.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite loss: {loss_value}")

    grad_norm = 0.0
    finite_grads = True
    for param in model.parameters():
        if param.grad is None:
            continue
        finite_grads = finite_grads and bool(torch.isfinite(param.grad).all().item())
        grad_norm += float(param.grad.detach().float().square().sum().cpu())
    grad_norm = grad_norm**0.5
    finite_params = all(
        bool(torch.isfinite(param.detach()).all().item())
        for param in model.parameters()
    )
    if not finite_grads:
        raise FloatingPointError("Non-finite gradients after smoke step.")
    if not finite_params:
        raise FloatingPointError("Non-finite parameters after smoke step.")

    summary = {
        "device": str(device),
        "mode": mode,
        "patch_size": int(args.patch_size),
        "batch_size": int(args.batch_size),
        "steps": int(args.steps),
        "trainable_params": int(trainable_params),
        "model_construct_seconds": model_seconds,
        "step_seconds": step_seconds,
        "losses": losses,
        "grad_norm": grad_norm,
        "finite_grads": finite_grads,
        "finite_params": finite_params,
        "seen_patches": int(solver.seen_patches),
        "rss_gib": _rss_gib(),
        "target_min": float(target.min().detach().cpu()),
        "target_max": float(target.max().detach().cpu()),
        **data_metadata,
        **_cuda_summary(device),
    }
    if device.type == "cuda":
        summary["cuda_peak_allocated_gib"] = (
            torch.cuda.max_memory_allocated(device) / 1024**3
        )
        summary["cuda_peak_reserved_gib"] = (
            torch.cuda.max_memory_reserved(device) / 1024**3
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, or auto.")
    parser.add_argument("--model-mode", default="padis-paper-ct-256")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--microbatch-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--image-scaling", type=float, default=0.5)
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--data-folder", type=pathlib.Path, default=None)
    parser.add_argument("--max-slices-per-patient", type=int, default=4)
    parser.add_argument("--pcg-slices-nodule", type=float, default=0.5)
    parser.add_argument("--no-position-channels", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-rampup-kimg", type=float, default=10_000)
    parser.add_argument("--initial-seen-patches", type=int, default=1_000)
    parser.add_argument("--sigma-distribution", type=str, default=None)
    parser.add_argument("--json", type=pathlib.Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_smoke(args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
