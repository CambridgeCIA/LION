"""Side-by-side short-run reproduction check against the PaDIS repo.

This compares the original PaDIS ``Patch_EDMPrecond`` + ``Patch_EDMLoss`` path
with LION's PaDIS training path on the same synthetic image batch. It maps the
paper-sized PaDIS network weights into LION's NCSN++ implementation, including
the q/k/v attention parameter layout difference, then checks losses,
gradients, Adam state, parameter updates, EMA, LR ramp, and patch counters after
each step.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from dataclasses import dataclass

import numpy as np
import torch

LION_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(LION_ROOT) not in sys.path:
    sys.path.insert(0, str(LION_ROOT))

from LION.CTtools.ct_geometry import Geometry  # noqa: E402
from LION.losses.PaDIS import (  # noqa: E402
    PaDISDenoisingLoss,
    sample_image_patch_with_position_channels,
)
from LION.models.diffusion import NCSNpp  # noqa: E402
from LION.optimizers import PaDISSolver  # noqa: E402


@dataclass
class PaDISStepState:
    loss: torch.Tensor
    seen_patches: int
    ema_state: dict[str, torch.Tensor]


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
        raise RuntimeError("CUDA was requested but PyTorch cannot access CUDA.")
    return device


def _rng_devices_for(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    if device.index is not None:
        return [device.index]
    return [torch.cuda.current_device()]


def _import_padis(padis_root: pathlib.Path):
    if not padis_root.is_dir():
        raise FileNotFoundError(f"Missing PaDIS root: {padis_root}")
    root = str(padis_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from training.networks import Patch_EDMPrecond  # noqa: PLC0415
    from training.patch_loss import Patch_EDMLoss  # noqa: PLC0415

    return Patch_EDMPrecond, Patch_EDMLoss


def _build_padis_model(Patch_EDMPrecond, patch_resolution: int, device: torch.device):
    return Patch_EDMPrecond(
        img_resolution=patch_resolution,
        img_channels=3,
        out_channels=1,
        label_dim=0,
        use_fp16=False,
        sigma_data=0.5,
        model_type="SongUNet",
        embedding_type="positional",
        encoder_type="standard",
        decoder_type="standard",
        channel_mult_noise=1,
        resample_filter=[1, 1],
        model_channels=128,
        channel_mult=[1, 2, 2, 2],
        dropout=0.05,
    ).to(device)


def _build_lion_model(device: torch.device):
    geometry = Geometry.default_parameters(image_scaling=0.5)
    model = NCSNpp(NCSNpp.default_parameters("padis-paper-ct-256"), geometry)
    return model.to(device), geometry


def _copy_padis_weights_to_lion(
    padis_model: torch.nn.Module, lion_model: torch.nn.Module
):
    padis_params = list(padis_model.named_parameters())
    lion_params = list(lion_model.named_parameters())
    padis_index = 0
    lion_index = 0
    mapped = []
    with torch.no_grad():
        while padis_index < len(padis_params):
            padis_name, padis_param = padis_params[padis_index]
            if padis_name.endswith(".qkv.weight"):
                qkv_weight = padis_param
                qkv_bias = padis_params[padis_index + 1][1]
                proj_weight = padis_params[padis_index + 2][1]
                proj_bias = padis_params[padis_index + 3][1]
                channels = qkv_bias.numel() // 3
                expected = (
                    "NIN_0.W",
                    "NIN_0.b",
                    "NIN_1.W",
                    "NIN_1.b",
                    "NIN_2.W",
                    "NIN_2.b",
                    "NIN_3.W",
                    "NIN_3.b",
                )
                for offset, suffix in enumerate(expected):
                    lion_name = lion_params[lion_index + offset][0]
                    if not lion_name.endswith(suffix):
                        raise AssertionError(
                            "Unexpected LION attention parameter order: "
                            f"{lion_name} does not end with {suffix}."
                        )
                for part in range(3):
                    weight = qkv_weight[part * channels : (part + 1) * channels]
                    bias = qkv_bias[part * channels : (part + 1) * channels]
                    lion_params[lion_index + 2 * part][1].copy_(weight[:, :, 0, 0].t())
                    lion_params[lion_index + 2 * part + 1][1].copy_(bias)
                lion_params[lion_index + 6][1].copy_(proj_weight[:, :, 0, 0].t())
                lion_params[lion_index + 7][1].copy_(proj_bias)
                mapped.extend(
                    [
                        padis_name,
                        padis_params[padis_index + 1][0],
                        padis_params[padis_index + 2][0],
                        padis_params[padis_index + 3][0],
                    ]
                )
                padis_index += 4
                lion_index += 8
                continue

            lion_name, lion_param = lion_params[lion_index]
            if tuple(padis_param.shape) != tuple(lion_param.shape):
                raise AssertionError(
                    "Parameter shape mismatch while mapping PaDIS to LION: "
                    f"{padis_name} {tuple(padis_param.shape)} vs "
                    f"{lion_name} {tuple(lion_param.shape)}"
                )
            lion_param.copy_(padis_param)
            mapped.append(padis_name)
            padis_index += 1
            lion_index += 1

    if padis_index != len(padis_params) or lion_index != len(lion_params):
        raise AssertionError(
            "Did not consume every parameter while mapping weights: "
            f"PaDIS {padis_index}/{len(padis_params)}, "
            f"LION {lion_index}/{len(lion_params)}."
        )
    return {
        "padis_parameter_tensors": len(padis_params),
        "lion_parameter_tensors": len(lion_params),
        "mapped_padis_tensors": len(mapped),
        "padis_parameters": sum(param.numel() for _, param in padis_params),
        "lion_parameters": sum(param.numel() for _, param in lion_params),
    }


def _attention_reconstruct(
    lion_params: list[tuple[str, torch.Tensor]],
    index: int,
):
    q = lion_params[index][1].t().unsqueeze(-1).unsqueeze(-1)
    qb = lion_params[index + 1][1]
    k = lion_params[index + 2][1].t().unsqueeze(-1).unsqueeze(-1)
    kb = lion_params[index + 3][1]
    v = lion_params[index + 4][1].t().unsqueeze(-1).unsqueeze(-1)
    vb = lion_params[index + 5][1]
    proj = lion_params[index + 6][1].t().unsqueeze(-1).unsqueeze(-1)
    proj_b = lion_params[index + 7][1]
    return {
        "qkv.weight": torch.cat((q, k, v), dim=0),
        "qkv.bias": torch.cat((qb, kb, vb), dim=0),
        "proj.weight": proj,
        "proj.bias": proj_b,
    }


def _lion_tensor_for_padis_param(
    padis_name: str,
    padis_index: int,
    lion_params: list[tuple[str, torch.Tensor]],
    lion_index: int,
) -> tuple[torch.Tensor, int]:
    if padis_name.endswith(".qkv.weight"):
        return (
            _attention_reconstruct(lion_params, lion_index)["qkv.weight"],
            lion_index + 8,
        )
    if padis_name.endswith(".qkv.bias"):
        return (
            _attention_reconstruct(lion_params, lion_index)["qkv.bias"],
            lion_index + 8,
        )
    if padis_name.endswith(".proj.weight"):
        return (
            _attention_reconstruct(lion_params, lion_index)["proj.weight"],
            lion_index + 8,
        )
    if padis_name.endswith(".proj.bias"):
        return (
            _attention_reconstruct(lion_params, lion_index)["proj.bias"],
            lion_index + 8,
        )
    del padis_index
    return lion_params[lion_index][1], lion_index + 1


def _walk_comparable_tensors(
    padis_items: list[tuple[str, torch.Tensor]],
    lion_items: list[tuple[str, torch.Tensor]],
):
    lion_index = 0
    padis_index = 0
    while padis_index < len(padis_items):
        padis_name, padis_tensor = padis_items[padis_index]
        if padis_name.endswith(".qkv.weight"):
            reconstructed = _attention_reconstruct(lion_items, lion_index)
            yield padis_name, padis_tensor, reconstructed["qkv.weight"]
            yield (
                padis_items[padis_index + 1][0],
                padis_items[padis_index + 1][1],
                reconstructed["qkv.bias"],
            )
            yield (
                padis_items[padis_index + 2][0],
                padis_items[padis_index + 2][1],
                reconstructed["proj.weight"],
            )
            yield (
                padis_items[padis_index + 3][0],
                padis_items[padis_index + 3][1],
                reconstructed["proj.bias"],
            )
            padis_index += 4
            lion_index += 8
            continue
        lion_tensor, lion_index = _lion_tensor_for_padis_param(
            padis_name, padis_index, lion_items, lion_index
        )
        yield padis_name, padis_tensor, lion_tensor
        padis_index += 1
    if lion_index != len(lion_items):
        raise AssertionError(f"Unconsumed LION tensors: {lion_index}/{len(lion_items)}")


def _max_abs_for_items(
    padis_items: list[tuple[str, torch.Tensor]],
    lion_items: list[tuple[str, torch.Tensor]],
) -> float:
    return _comparison_summary_for_items(padis_items, lion_items)["max_abs"]


def _abs_and_l2rel_for_items(
    padis_items: list[tuple[str, torch.Tensor]],
    lion_items: list[tuple[str, torch.Tensor]],
) -> tuple[float, float]:
    summary = _comparison_summary_for_items(padis_items, lion_items)
    return summary["max_abs"], summary["l2_relative"]


def _tensor_pair_metrics(
    name: str, padis_tensor: torch.Tensor, lion_tensor: torch.Tensor
) -> dict[str, object]:
    if tuple(padis_tensor.shape) != tuple(lion_tensor.shape):
        raise AssertionError(
            f"Comparable tensor shape mismatch for {name}: "
            f"{tuple(padis_tensor.shape)} vs {tuple(lion_tensor.shape)}"
        )
    if padis_tensor.numel() == 0:
        return {
            "name": name,
            "shape": list(padis_tensor.shape),
            "max_abs": 0.0,
            "l2_relative": 0.0,
            "diff_l2": 0.0,
            "padis_l2": 0.0,
            "padis_max_abs": 0.0,
            "lion_max_abs": 0.0,
        }
    delta = (padis_tensor - lion_tensor).detach().float()
    padis_float = padis_tensor.detach().float()
    lion_float = lion_tensor.detach().float()
    diff_l2 = float(torch.linalg.vector_norm(delta).cpu())
    padis_l2 = float(torch.linalg.vector_norm(padis_float).cpu())
    return {
        "name": name,
        "shape": list(padis_tensor.shape),
        "max_abs": float(torch.amax(torch.abs(delta)).cpu()),
        "l2_relative": diff_l2 / max(padis_l2, 1e-12),
        "diff_l2": diff_l2,
        "padis_l2": padis_l2,
        "padis_max_abs": float(torch.amax(torch.abs(padis_float)).cpu()),
        "lion_max_abs": float(torch.amax(torch.abs(lion_float)).cpu()),
    }


def _single_tensor_summary(
    name: str, padis_tensor: torch.Tensor, lion_tensor: torch.Tensor
) -> dict[str, object]:
    metric = _tensor_pair_metrics(name, padis_tensor, lion_tensor)
    return {
        "max_abs": metric["max_abs"],
        "l2_relative": metric["l2_relative"],
        "tensor": metric,
    }


def _comparison_summary_for_items(
    padis_items: list[tuple[str, torch.Tensor]],
    lion_items: list[tuple[str, torch.Tensor]],
    top_k: int = 0,
) -> dict[str, object]:
    max_abs = 0.0
    diff_sq = 0.0
    ref_sq = 0.0
    per_tensor = []
    for name, padis_tensor, lion_tensor in _walk_comparable_tensors(
        padis_items, lion_items
    ):
        metric = _tensor_pair_metrics(name, padis_tensor, lion_tensor)
        max_abs = max(max_abs, float(metric["max_abs"]))
        diff_sq += float(metric["diff_l2"]) ** 2
        ref_sq += float(metric["padis_l2"]) ** 2
        if top_k > 0:
            per_tensor.append(metric)
    l2_relative = diff_sq**0.5 / max(ref_sq**0.5, 1e-12)
    summary: dict[str, object] = {
        "max_abs": max_abs,
        "l2_relative": l2_relative,
    }
    if top_k > 0:
        summary["top_abs"] = sorted(
            per_tensor, key=lambda item: float(item["max_abs"]), reverse=True
        )[:top_k]
        summary["top_l2_relative"] = sorted(
            per_tensor,
            key=lambda item: float(item["l2_relative"]),
            reverse=True,
        )[:top_k]
    return summary


def _named_parameter_items(module: torch.nn.Module):
    return [(name, param.detach()) for name, param in module.named_parameters()]


def _named_gradient_items(module: torch.nn.Module):
    return [
        (name, torch.zeros_like(param) if param.grad is None else param.grad.detach())
        for name, param in module.named_parameters()
    ]


def _optimizer_state_items(
    optimizer: torch.optim.Optimizer,
    module: torch.nn.Module,
    key: str,
):
    items = []
    for name, param in module.named_parameters():
        state_value = optimizer.state.get(param, {}).get(key)
        if state_value is None:
            state_value = torch.zeros_like(param.detach())
        elif state_value.ndim == 0:
            state_value = state_value.detach().reshape(1).expand_as(param.detach())
        else:
            state_value = state_value.detach()
        items.append((name, state_value))
    return items


def _make_lion_solver(
    lion_model: torch.nn.Module,
    geometry: Geometry,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> PaDISSolver:
    solver_params = PaDISSolver.default_parameters("padis-paper-ct-256")
    solver_params.patch_sizes = list(args.patch_sizes)
    solver_params.patch_probabilities = [1.0 / len(args.patch_sizes)] * len(
        args.patch_sizes
    )
    solver_params.patch_batch_multipliers = {int(size): 1 for size in args.patch_sizes}
    solver_params.pad_width = 0
    solver_params.base_patch_batch_size = int(args.batch_size)
    solver_params.microbatch_size = None
    solver_params.sigma_distribution = "edm_lognormal"
    solver_params.use_ema = True
    solver_params.ema_half_life_patches = float(args.ema_half_life_patches)
    solver_params.ema_rampup_ratio = float(args.ema_rampup_ratio)
    solver_params.lr_rampup_kimg = float(args.lr_rampup_kimg)
    solver_params.enforce_data_range = False
    loss_fn = PaDISDenoisingLoss(sigma_distribution="edm_lognormal")
    solver = PaDISSolver(
        lion_model,
        optimizer,
        loss_fn,
        geometry=geometry,
        verbose=False,
        device=args.resolved_device,
        solver_params=solver_params,
    )
    solver.seen_patches = int(args.initial_seen_patches)
    return solver


def _set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    seen_patches: int,
    lr_rampup_kimg: float,
) -> float:
    lr_scale = min(
        float(seen_patches) / max(float(lr_rampup_kimg) * 1000.0, 1e-8),
        1.0,
    )
    for group in optimizer.param_groups:
        group.setdefault("base_lr", group["lr"])
        group["lr"] = group["base_lr"] * lr_scale
    return float(optimizer.param_groups[0]["lr"])


def _sanitize_gradients(module: torch.nn.Module) -> None:
    for param in module.parameters():
        if param.grad is not None:
            torch.nan_to_num(
                param.grad, nan=0.0, posinf=1e5, neginf=-1e5, out=param.grad
            )


def _update_padis_ema(
    ema_state: dict[str, torch.Tensor],
    module: torch.nn.Module,
    seen_patches: int,
    batch_count: int,
    args: argparse.Namespace,
) -> None:
    half_life = float(args.ema_half_life_patches)
    half_life = min(half_life, float(seen_patches) * float(args.ema_rampup_ratio))
    beta = 0.5 ** (float(batch_count) / max(half_life, 1e-8))
    with torch.no_grad():
        for name, param in module.named_parameters():
            ema_state[name].mul_(beta).add_(param.detach(), alpha=1.0 - beta)


def _forward_diagnostic(
    padis_model: torch.nn.Module,
    lion_model: torch.nn.Module,
    padis_loss_fn,
    lion_loss_fn: PaDISDenoisingLoss,
    images: torch.Tensor,
    patch_size: int,
    step_seed: int,
    device: torch.device,
) -> dict[str, object]:
    devices = _rng_devices_for(device)
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(step_seed)
        padis_patch, padis_position = padis_loss_fn.pachify(images, patch_size)
        padis_sigma = (
            torch.randn([images.shape[0], 1, 1, 1], device=images.device)
            * padis_loss_fn.P_std
            + padis_loss_fn.P_mean
        ).exp()
        padis_noise = torch.randn_like(padis_patch)

        torch.manual_seed(step_seed)
        lion_patch, lion_position = sample_image_patch_with_position_channels(
            images, patch_size
        )
        lion_sigma = lion_loss_fn.sample_sigma(
            int(images.shape[0]), images.device
        ).reshape(-1, 1, 1, 1)
        lion_noise = torch.randn_like(lion_patch)

        sigma = padis_sigma
        noisy = padis_patch + sigma * padis_noise
        sigma_data = torch.as_tensor(
            padis_loss_fn.sigma_data,
            device=images.device,
            dtype=images.dtype,
        )
        c_skip = sigma_data.square() / (sigma.square() + sigma_data.square())
        c_out = sigma * sigma_data / (sigma.square() + sigma_data.square()).sqrt()
        c_in = 1 / (sigma_data.square() + sigma.square()).sqrt()
        c_noise = sigma.flatten().log() / 4
        model_input = torch.cat((c_in * noisy, padis_position), dim=1)

        model_seed = step_seed + 500_000
        torch.manual_seed(model_seed)
        padis_raw = padis_model.model(model_input, c_noise, class_labels=None)
        torch.manual_seed(model_seed)
        lion_raw = lion_model(model_input, c_noise)

        padis_denoised = c_skip * noisy + c_out * padis_raw
        lion_denoised = c_skip * noisy + c_out * lion_raw
        weight = (sigma.square() + sigma_data.square()) / (sigma * sigma_data).square()
        padis_loss = weight * (padis_denoised - padis_patch).square()
        lion_loss = weight * (lion_denoised - lion_patch).square()
        padis_scalar_loss = padis_loss.flatten(1).sum(dim=1).mean()
        lion_scalar_loss = lion_loss.flatten(1).sum(dim=1).mean()

        torch.manual_seed(model_seed)
        padis_wrapper_denoised = padis_model(
            noisy, sigma, x_pos=padis_position, class_labels=None
        )

    return {
        "patch": _single_tensor_summary("patch", padis_patch, lion_patch),
        "position": _single_tensor_summary("position", padis_position, lion_position),
        "sigma": _single_tensor_summary("sigma", padis_sigma, lion_sigma),
        "noise": _single_tensor_summary("noise", padis_noise, lion_noise),
        "inner_output": _single_tensor_summary("inner_output", padis_raw, lion_raw),
        "denoised": _single_tensor_summary("denoised", padis_denoised, lion_denoised),
        "loss_tensor": _single_tensor_summary("loss_tensor", padis_loss, lion_loss),
        "scalar_loss": _single_tensor_summary(
            "scalar_loss",
            padis_scalar_loss.reshape(1),
            lion_scalar_loss.reshape(1),
        ),
        "padis_wrapper_vs_manual": _single_tensor_summary(
            "padis_wrapper_vs_manual",
            padis_wrapper_denoised,
            padis_denoised,
        ),
    }


def _padis_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    images: torch.Tensor,
    patch_size: int,
    seen_patches: int,
    ema_state: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> PaDISStepState:
    optimizer.zero_grad()
    loss_tensor = loss_fn(
        net=model,
        images=images,
        patch_size=int(patch_size),
        resolution=int(images.shape[-1]),
        labels=None,
        augment_pipe=None,
    )
    loss = loss_tensor.flatten(1).sum(dim=1).mean()
    loss.backward()
    _set_optimizer_lr(optimizer, seen_patches, args.lr_rampup_kimg)
    _sanitize_gradients(model)
    optimizer.step()
    _update_padis_ema(ema_state, model, seen_patches, int(images.shape[0]), args)
    return PaDISStepState(
        loss=loss.detach(),
        seen_patches=seen_patches + int(images.shape[0]),
        ema_state=ema_state,
    )


def _compare_ema(
    padis_ema: dict[str, torch.Tensor],
    lion_ema: dict[str, torch.Tensor],
    padis_model: torch.nn.Module,
    lion_model: torch.nn.Module,
) -> float:
    padis_items = [
        (name, padis_ema[name].detach())
        for name, _param in padis_model.named_parameters()
    ]
    lion_items = [
        (name, lion_ema[name].detach())
        for name, _param in lion_model.named_parameters()
    ]
    return _max_abs_for_items(padis_items, lion_items)


def _compare_step(
    step_index: int,
    patch_size: int,
    padis_loss: torch.Tensor,
    lion_loss: float,
    padis_model: torch.nn.Module,
    lion_model: torch.nn.Module,
    padis_optimizer: torch.optim.Optimizer,
    lion_optimizer: torch.optim.Optimizer,
    padis_ema: dict[str, torch.Tensor],
    lion_ema: dict[str, torch.Tensor],
    absolute_tolerance: float,
    relative_tolerance: float,
    top_k: int,
    forward_summary: dict[str, object] | None,
) -> dict[str, object]:
    loss_abs = abs(float(padis_loss.detach().cpu()) - float(lion_loss))
    loss_relative = loss_abs / max(abs(float(padis_loss.detach().cpu())), 1e-12)
    grad_summary = _comparison_summary_for_items(
        _named_gradient_items(padis_model),
        _named_gradient_items(lion_model),
        top_k=top_k,
    )
    param_summary = _comparison_summary_for_items(
        _named_parameter_items(padis_model),
        _named_parameter_items(lion_model),
        top_k=top_k,
    )
    exp_avg_summary = _comparison_summary_for_items(
        _optimizer_state_items(padis_optimizer, padis_model, "exp_avg"),
        _optimizer_state_items(lion_optimizer, lion_model, "exp_avg"),
        top_k=top_k,
    )
    exp_avg_sq_summary = _comparison_summary_for_items(
        _optimizer_state_items(padis_optimizer, padis_model, "exp_avg_sq"),
        _optimizer_state_items(lion_optimizer, lion_model, "exp_avg_sq"),
        top_k=top_k,
    )
    ema_items = [
        (name, padis_ema[name].detach())
        for name, _param in padis_model.named_parameters()
    ]
    lion_ema_items = [
        (name, lion_ema[name].detach())
        for name, _param in lion_model.named_parameters()
    ]
    ema_summary = _comparison_summary_for_items(ema_items, lion_ema_items, top_k=top_k)
    grad_max_abs = float(grad_summary["max_abs"])
    grad_l2_relative = float(grad_summary["l2_relative"])
    param_max_abs = float(param_summary["max_abs"])
    param_l2_relative = float(param_summary["l2_relative"])
    exp_avg_max_abs = float(exp_avg_summary["max_abs"])
    exp_avg_l2_relative = float(exp_avg_summary["l2_relative"])
    exp_avg_sq_max_abs = float(exp_avg_sq_summary["max_abs"])
    exp_avg_sq_l2_relative = float(exp_avg_sq_summary["l2_relative"])
    ema_max_abs = float(ema_summary["max_abs"])
    ema_l2_relative = float(ema_summary["l2_relative"])
    max_abs = max(
        loss_abs,
        grad_max_abs,
        param_max_abs,
        exp_avg_max_abs,
        exp_avg_sq_max_abs,
        ema_max_abs,
    )
    max_l2_relative = max(
        loss_relative,
        grad_l2_relative,
        param_l2_relative,
        exp_avg_l2_relative,
        exp_avg_sq_l2_relative,
        ema_l2_relative,
    )
    if max_abs > absolute_tolerance and max_l2_relative > relative_tolerance:
        raise AssertionError(
            f"Step {step_index} patch {patch_size} mismatch: "
            f"max_abs={max_abs}, max_l2_relative={max_l2_relative}"
        )
    summary: dict[str, object] = {
        "step": int(step_index),
        "patch_size": int(patch_size),
        "loss_abs": loss_abs,
        "loss_relative": loss_relative,
        "grad_max_abs": grad_max_abs,
        "grad_l2_relative": grad_l2_relative,
        "param_max_abs": param_max_abs,
        "param_l2_relative": param_l2_relative,
        "optimizer_exp_avg_max_abs": exp_avg_max_abs,
        "optimizer_exp_avg_l2_relative": exp_avg_l2_relative,
        "optimizer_exp_avg_sq_max_abs": exp_avg_sq_max_abs,
        "optimizer_exp_avg_sq_l2_relative": exp_avg_sq_l2_relative,
        "ema_max_abs": ema_max_abs,
        "ema_l2_relative": ema_l2_relative,
        "max_abs": max_abs,
        "max_l2_relative": max_l2_relative,
    }
    if top_k > 0:
        summary["top_offenders"] = {
            "grad": grad_summary,
            "param": param_summary,
            "optimizer_exp_avg": exp_avg_summary,
            "optimizer_exp_avg_sq": exp_avg_sq_summary,
            "ema": ema_summary,
        }
    if forward_summary is not None:
        summary["forward_diagnostic"] = forward_summary
    return summary


def run_check(args: argparse.Namespace) -> dict[str, object]:
    args.resolved_device = _device_from_arg(args.device)
    top_k = int(getattr(args, "top_k", 0))
    run_forward_diagnostics = not bool(getattr(args, "no_forward_diagnostics", False))
    _set_seed(args.seed)
    Patch_EDMPrecond, Patch_EDMLoss = _import_padis(args.padis_root)

    padis_model = _build_padis_model(
        Patch_EDMPrecond, int(args.model_patch_resolution), args.resolved_device
    )
    lion_model, geometry = _build_lion_model(args.resolved_device)
    mapping_summary = _copy_padis_weights_to_lion(padis_model, lion_model)
    initial_param_max_abs = _max_abs_for_items(
        _named_parameter_items(padis_model), _named_parameter_items(lion_model)
    )
    if initial_param_max_abs > args.tolerance:
        raise AssertionError(f"Initial parameter mismatch: {initial_param_max_abs}")

    padis_optimizer = torch.optim.Adam(padis_model.parameters(), lr=float(args.lr))
    lion_optimizer = torch.optim.Adam(lion_model.parameters(), lr=float(args.lr))
    padis_loss_fn = Patch_EDMLoss()
    lion_solver = _make_lion_solver(lion_model, geometry, lion_optimizer, args)
    padis_ema = {
        name: param.detach().clone()
        for name, param in padis_model.named_parameters()
        if param.requires_grad
    }
    padis_seen = int(args.initial_seen_patches)

    images = torch.linspace(
        0.0,
        1.0,
        int(args.batch_size) * int(args.image_size) * int(args.image_size),
        device=args.resolved_device,
        dtype=torch.float32,
    ).reshape(int(args.batch_size), 1, int(args.image_size), int(args.image_size))

    padis_model.train()
    lion_model.train()
    step_summaries = []
    for step_index in range(int(args.steps)):
        patch_size = int(args.patch_sizes[step_index % len(args.patch_sizes)])
        step_seed = int(args.seed) + 10_000 + step_index
        forward_summary = None
        if run_forward_diagnostics:
            forward_summary = _forward_diagnostic(
                padis_model,
                lion_model,
                padis_loss_fn,
                lion_solver.loss_fn,
                images,
                patch_size,
                step_seed,
                args.resolved_device,
            )
        torch.manual_seed(step_seed)
        padis_state = _padis_training_step(
            padis_model,
            padis_optimizer,
            padis_loss_fn,
            images,
            patch_size,
            padis_seen,
            padis_ema,
            args,
        )
        padis_seen = padis_state.seen_patches
        torch.manual_seed(step_seed)
        lion_loss = lion_solver._optimizer_step(images, patch_size=patch_size)
        if padis_seen != lion_solver.seen_patches:
            raise AssertionError(
                f"Seen patch mismatch: PaDIS {padis_seen}, "
                f"LION {lion_solver.seen_patches}"
            )
        step_summaries.append(
            _compare_step(
                step_index,
                patch_size,
                padis_state.loss,
                lion_loss,
                padis_model,
                lion_model,
                padis_optimizer,
                lion_optimizer,
                padis_state.ema_state,
                lion_solver.ema_state,
                args.tolerance,
                args.relative_tolerance,
                top_k,
                forward_summary,
            )
        )

    return {
        "device": str(args.resolved_device),
        "padis_root": str(args.padis_root),
        "seed": int(args.seed),
        "tolerance": float(args.tolerance),
        "relative_tolerance": float(args.relative_tolerance),
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "steps": int(args.steps),
        "patch_sizes": [int(size) for size in args.patch_sizes],
        "mapping": mapping_summary,
        "initial_param_max_abs": initial_param_max_abs,
        "final_seen_patches": int(lion_solver.seen_patches),
        "steps_summary": step_summaries,
        "max_abs": max(item["max_abs"] for item in step_summaries),
        "max_l2_relative": max(item["max_l2_relative"] for item in step_summaries),
        "passed": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--padis-root",
        type=pathlib.Path,
        default=LION_ROOT.parent / "PaDIS",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of seeds to run as a matrix.",
    )
    parser.add_argument("--tolerance", type=float, default=2e-2)
    parser.add_argument("--relative-tolerance", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--model-patch-resolution", type=int, default=56)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=(16, 32, 56))
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-rampup-kimg", type=float, default=0.02)
    parser.add_argument("--initial-seen-patches", type=int, default=1000)
    parser.add_argument("--ema-half-life-patches", type=float, default=100.0)
    parser.add_argument("--ema-rampup-ratio", type=float, default=0.05)
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of named offending tensors to report per category.",
    )
    parser.add_argument(
        "--no-forward-diagnostics",
        action="store_true",
        help="Skip pre-update patch/sigma/noise/output diagnostics.",
    )
    parser.add_argument("--json", type=pathlib.Path, default=None)
    return parser


def _run_seed_matrix(args: argparse.Namespace) -> dict[str, object]:
    if args.seeds is None:
        return run_check(args)

    runs = []
    for seed in args.seeds:
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = int(seed)
        seed_args.seeds = None
        runs.append(run_check(seed_args))

    return {
        "device": runs[0]["device"] if runs else str(_device_from_arg(args.device)),
        "padis_root": str(args.padis_root),
        "seeds": [int(seed) for seed in args.seeds],
        "runs": runs,
        "max_abs": max(float(run["max_abs"]) for run in runs),
        "max_l2_relative": max(float(run["max_l2_relative"]) for run in runs),
        "passed": all(bool(run["passed"]) for run in runs),
    }


def main() -> None:
    args = build_parser().parse_args()
    args.padis_root = args.padis_root.resolve()
    summary = _run_seed_matrix(args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
