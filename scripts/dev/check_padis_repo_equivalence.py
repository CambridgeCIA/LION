"""Golden-trace checks between LION PaDIS and the local PaDIS checkout.

This script intentionally does not copy PaDIS implementation code into LION.
It reads the relevant oracle functions from the sibling PaDIS repository at
runtime, patches only hard-coded device allocation so the check can run on CPU,
and compares controlled synthetic traces.
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import random
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

LION_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(LION_ROOT) not in sys.path:
    sys.path.insert(0, str(LION_ROOT))

from LION.CTtools.ct_geometry import Geometry  # noqa: E402
from LION.models.LIONmodel import LIONmodel, LIONModelParameter  # noqa: E402
from LION.operators import Operator  # noqa: E402
from LION.losses.PaDIS import (  # noqa: E402
    PaDISDenoisingLoss,
    build_position_grid,
    sample_image_patch_with_position_channels,
    zero_pad_images,
)
from LION.optimizers import PaDISSolver  # noqa: E402
from LION.reconstructors import PaDIS  # noqa: E402
from LION.reconstructors.PaDIS import _PatchLayout  # noqa: E402
from LION.utils.parameter import LIONParameter  # noqa: E402


class ScaledIdentityOp(Operator):
    def __init__(self, scale: float = 1.0):
        super().__init__(device=torch.device("cpu"))
        self.scale = float(scale)

    @property
    def domain_shape(self) -> tuple[int, ...]:
        return (1, 8, 8)

    @property
    def range_shape(self) -> tuple[int, ...]:
        return (1, 8, 8)

    def __call__(self, x: torch.Tensor, out=None) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return self.scale * y

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y / self.scale

    # Public PaDIS oracle API.
    def A(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def AT(self, y: torch.Tensor) -> torch.Tensor:
        return self.adjoint(y)

    def Adagger(self, y: torch.Tensor) -> torch.Tensor:
        return self.inverse(y)


class CompatibleDenoiser(nn.Module):
    """Deterministic denoiser with PaDIS-repo and LION call signatures."""

    def __init__(self, pad_width: int = 4, patch_size: int = 4):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))
        self.model_parameters = LIONParameter()
        self.model_parameters.pad_width = pad_width
        self.model_parameters.largest_patch_size = patch_size
        self.model_parameters.input_position_channels = 2
        self.model_parameters.prior_mode = "patch"

    def _desired(
        self,
        image: torch.Tensor,
        sigma: torch.Tensor,
        position: torch.Tensor | None,
    ) -> torch.Tensor:
        sigma = torch.as_tensor(sigma, device=image.device, dtype=image.dtype)
        if sigma.ndim == 0:
            sigma = sigma.expand(image.shape[0])
        sigma = sigma.reshape(image.shape[0], 1, 1, 1)
        position_term = torch.zeros_like(image)
        if position is not None:
            position_term = 0.13 * position[:, 0:1] - 0.07 * position[:, 1:2]
        return image + position_term + 0.05 * sigma

    def forward(self, x: torch.Tensor, sigma_or_c_noise: torch.Tensor, *args, **kwargs):
        if args:
            position = args[0]
            return self._desired(x, sigma_or_c_noise, position)
        if "x_pos" in kwargs:
            return self._desired(x, sigma_or_c_noise, kwargs["x_pos"])

        position = x[:, 1:3]
        image = x[:, 0:1]
        sigma = torch.exp(4.0 * sigma_or_c_noise).to(device=x.device, dtype=x.dtype)
        sigma_view = sigma.reshape(x.shape[0], 1, 1, 1)
        sigma_data = torch.as_tensor(0.5, device=x.device, dtype=x.dtype)
        c_skip = sigma_data.square() / (sigma_view.square() + sigma_data.square())
        c_out = (
            sigma_view * sigma_data / (sigma_view.square() + sigma_data.square()).sqrt()
        )
        c_in = 1.0 / (sigma_data.square() + sigma_view.square()).sqrt()
        raw_image = image / c_in
        desired = self._desired(raw_image, sigma, position)
        return (desired - c_skip * raw_image) / c_out


class TrainableCompatibleDenoiser(LIONmodel):
    """Tiny trainable denoiser for optimizer-step equivalence checks."""

    def __init__(self, pad_width: int = 0, patch_size: int = 8):
        params = LIONModelParameter()
        params.pad_width = pad_width
        params.largest_patch_size = patch_size
        params.input_position_channels = 2
        params.prior_mode = "patch"
        super().__init__(params, geometry=None)
        self.gain = nn.Parameter(torch.tensor(0.85))
        self.bias = nn.Parameter(torch.tensor(0.03))

    @staticmethod
    def default_parameters(mode="ct") -> LIONModelParameter:
        del mode
        params = LIONModelParameter()
        params.pad_width = 0
        params.largest_patch_size = 8
        params.input_position_channels = 2
        params.prior_mode = "patch"
        return params

    def _desired(
        self,
        image: torch.Tensor,
        sigma: torch.Tensor,
        position: torch.Tensor | None,
    ) -> torch.Tensor:
        sigma = torch.as_tensor(sigma, device=image.device, dtype=image.dtype)
        if sigma.ndim == 0:
            sigma = sigma.expand(image.shape[0])
        sigma = sigma.reshape(image.shape[0], 1, 1, 1)
        position_term = torch.zeros_like(image)
        if position is not None:
            position_term = 0.11 * position[:, 0:1] - 0.05 * position[:, 1:2]
        return self.gain * image + self.bias + position_term + 0.04 * sigma

    def forward(self, x: torch.Tensor, sigma_or_c_noise: torch.Tensor, *args, **kwargs):
        if args:
            position = args[0]
            return self._desired(x, sigma_or_c_noise, position)
        if "x_pos" in kwargs:
            return self._desired(x, sigma_or_c_noise, kwargs["x_pos"])

        position = x[:, 1:3]
        image = x[:, 0:1]
        sigma = torch.exp(4.0 * sigma_or_c_noise).to(device=x.device, dtype=x.dtype)
        sigma_view = sigma.reshape(x.shape[0], 1, 1, 1)
        sigma_data = torch.as_tensor(0.5, device=x.device, dtype=x.dtype)
        c_skip = sigma_data.square() / (sigma_view.square() + sigma_data.square())
        c_out = (
            sigma_view * sigma_data / (sigma_view.square() + sigma_data.square()).sqrt()
        )
        c_in = 1.0 / (sigma_data.square() + sigma_view.square()).sqrt()
        raw_image = image / c_in
        desired = self._desired(raw_image, sigma, position)
        return (desired - c_skip * raw_image) / c_out


def _extract_functions(
    source_path: pathlib.Path, names: set[str], device: torch.device
):
    source = source_path.read_text()
    tree = ast.parse(source)
    chunks = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            segment = ast.get_source_segment(source, node)
            if segment is None:
                raise RuntimeError(f"Could not extract {node.name} from {source_path}.")
            chunks.append(segment)
    missing = names.difference(
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    )
    if missing:
        raise RuntimeError(f"Missing oracle functions: {sorted(missing)}")

    oracle_source = "\n\n".join(chunks).replace("torch.device('cuda')", "device")
    namespace = {
        "device": device,
        "np": np,
        "random": random,
        "torch": torch,
    }
    exec(compile(oracle_source, str(source_path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def _extract_classes(source_path: pathlib.Path, names: set[str], device: torch.device):
    source = source_path.read_text()
    tree = ast.parse(source)
    chunks = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in names:
            segment = ast.get_source_segment(source, node)
            if segment is None:
                raise RuntimeError(f"Could not extract {node.name} from {source_path}.")
            segment = segment.replace("@persistence.persistent_class\n", "")
            chunks.append(segment)
    missing = names.difference(
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    )
    if missing:
        raise RuntimeError(f"Missing oracle classes: {sorted(missing)}")

    namespace = {
        "device": device,
        "np": np,
        "torch": torch,
    }
    exec(compile("\n\n".join(chunks), str(source_path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def _build_params(model: CompatibleDenoiser) -> LIONParameter:
    params = PaDIS.padis_repo_ct_parameters(model)
    params.num_steps = 2
    params.inner_steps = 2
    params.sigma_min = 0.01
    params.sigma_max = 0.02
    params.pad_width = 4
    params.patch_size = 4
    params.patch_batch_size = 3
    params.initial_reconstruction = "inverse"
    params.clip_initial = False
    params.clip_output = False
    params.sigma_data = 0.5
    return params


def _build_paper_params(model: CompatibleDenoiser) -> LIONParameter:
    params = PaDIS.paper_ct_parameters(model, views=20)
    params.num_steps = 2
    params.inner_steps = 2
    params.sigma_min = 0.01
    params.sigma_max = 0.02
    params.pad_width = 4
    params.patch_size = 4
    params.patch_batch_size = 3
    params.sigma_data = 0.5
    return params


def _fixed_indices(image_size: int, patch_size: int, offset: tuple[int, int]):
    patches = image_size // patch_size + 1
    spaced = np.linspace(0, (patches - 1) * patch_size, patches, dtype=int)
    row_offset, col_offset = offset
    return [
        (
            int(row_start) + row_offset,
            int(row_start) + row_offset + patch_size,
            int(col_start) + col_offset,
            int(col_start) + col_offset + patch_size,
        )
        for row_start in spaced.tolist()
        for col_start in spaced.tolist()
    ]


def _compare_patch_assembly(
    oracle_functions,
    reconstructor: PaDIS,
    params: LIONParameter,
    tolerance: float,
) -> dict[str, float]:
    denoised_from_patches = oracle_functions["denoisedFromPatches"]
    device = next(reconstructor.model.parameters()).device
    dtype = torch.float32
    central = torch.linspace(0.05, 0.95, 64, device=device, dtype=dtype).reshape(
        1, 1, 8, 8
    )
    padded = F.pad(central, (4, 4, 4, 4), mode="constant", value=0.0)
    sigma = torch.tensor(0.02, device=device, dtype=dtype)
    indices = _fixed_indices(8, 4, offset=(1, 2))
    latents_pos = reconstructor._position_grid(padded)
    layout = _PatchLayout(indices=indices, image_height=8, image_width=8)

    oracle = denoised_from_patches(
        reconstructor.model,
        padded,
        sigma,
        latents_pos,
        None,
        [list(index) for index in indices],
        t_goal=0,
    )
    lion = reconstructor._denoise_patches(padded, sigma.reshape(1), layout, params)
    max_abs = float(torch.max(torch.abs(oracle - lion)).detach().cpu())
    if max_abs > tolerance:
        raise AssertionError(f"Patch assembly mismatch: max_abs={max_abs}")
    return {"patch_assembly_max_abs": max_abs}


def _compare_dps_update(
    oracle_functions,
    reconstructor: PaDIS,
    params: LIONParameter,
    tolerance: float,
) -> dict[str, float]:
    denoised_from_patches = oracle_functions["denoisedFromPatches"]
    measurement_cond_fn = oracle_functions["measurement_cond_fn"]
    device = next(reconstructor.model.parameters()).device
    dtype = torch.float32
    central = torch.linspace(0.1, 0.9, 64, device=device, dtype=dtype).reshape(
        1, 1, 8, 8
    )
    padded = F.pad(central, (4, 4, 4, 4), mode="constant", value=0.0)
    x_oracle = padded.squeeze(0).clone().detach().requires_grad_(True)
    x_lion = padded.clone().detach().requires_grad_(True)
    sigma = torch.tensor(0.02, device=device, dtype=dtype)
    indices = _fixed_indices(8, 4, offset=(2, 1))
    latents_pos = reconstructor._position_grid(padded)
    layout = _PatchLayout(indices=indices, image_height=8, image_width=8)
    measurement = torch.linspace(0.2, 0.8, 64, device=device, dtype=dtype).reshape(
        1, 8, 8
    )
    noise = torch.linspace(-0.2, 0.2, x_oracle.numel(), device=device, dtype=dtype)
    noise = noise.reshape_as(x_oracle)

    oracle_d = denoised_from_patches(
        reconstructor.model,
        x_oracle.unsqueeze(0),
        sigma,
        latents_pos,
        None,
        [list(index) for index in indices],
        t_goal=0,
    ).squeeze(0)
    lion_d = reconstructor._denoise_patches(x_lion, sigma.reshape(1), layout, params)
    oracle_grad = measurement_cond_fn(
        measurement,
        x_oracle,
        oracle_d,
        reconstructor.op,
        pad=int(params.pad_width),
        w=8,
    )
    previous_params = getattr(reconstructor, "_active_params", None)
    reconstructor._active_params = params
    try:
        (
            lion_grad,
            lion_raw_grad,
            _residual,
            _normalizer,
            lion_scale,
            step,
        ) = reconstructor._dps_data_gradient(
            measurement, x_lion, lion_d, params, sigma=sigma
        )
    finally:
        reconstructor._active_params = previous_params

    alpha = float(params.dps_epsilon) * sigma.square()
    oracle_score = (oracle_d - x_oracle) / sigma.square()
    lion_score = (lion_d - x_lion) / sigma.square()
    oracle_scaled_grad = oracle_grad * float(lion_scale)
    oracle_next = (
        x_oracle
        - float(params.zeta) * oracle_scaled_grad
        + alpha / 2.0 * oracle_score
        + torch.sqrt(alpha) * noise
    )
    lion_next = (
        x_lion
        - step * lion_grad
        + alpha / 2.0 * lion_score
        + torch.sqrt(alpha) * noise.unsqueeze(0)
    )

    raw_gradient_max_abs = float(
        torch.max(torch.abs(oracle_grad - lion_raw_grad.squeeze(0))).detach().cpu()
    )
    scaled_gradient_max_abs = float(
        torch.max(torch.abs(oracle_scaled_grad - lion_grad.squeeze(0))).detach().cpu()
    )
    update_max_abs = float(
        torch.max(torch.abs(oracle_next - lion_next.squeeze(0))).detach().cpu()
    )
    if (
        raw_gradient_max_abs > tolerance
        or scaled_gradient_max_abs > tolerance
        or update_max_abs > tolerance
    ):
        raise AssertionError(
            "DPS update mismatch: "
            f"raw_gradient_max_abs={raw_gradient_max_abs}, "
            f"scaled_gradient_max_abs={scaled_gradient_max_abs}, "
            f"update_max_abs={update_max_abs}"
        )
    return {
        "public_repo_dps_raw_gradient_max_abs": raw_gradient_max_abs,
        "public_repo_dps_scaled_gradient_max_abs": scaled_gradient_max_abs,
        "public_repo_dps_data_consistency_scale": float(lion_scale),
        "public_repo_dps_update_max_abs": update_max_abs,
    }


def _compare_paper_dps_update(
    reconstructor: PaDIS,
    params: LIONParameter,
    tolerance: float,
) -> dict[str, float]:
    device = next(reconstructor.model.parameters()).device
    dtype = torch.float32
    central = torch.linspace(0.1, 0.9, 64, device=device, dtype=dtype).reshape(
        1, 1, 8, 8
    )
    x = F.pad(central, (4, 4, 4, 4), mode="constant", value=0.0)
    x = x.clone().detach().requires_grad_(True)
    sigma = torch.tensor(0.02, device=device, dtype=dtype)
    layout = _PatchLayout(
        indices=_fixed_indices(8, 4, offset=(2, 1)),
        image_height=8,
        image_width=8,
    )
    measurement = torch.linspace(0.2, 0.8, 64, device=device, dtype=dtype).reshape(
        1, 8, 8
    )
    noise = torch.linspace(-0.2, 0.2, x.numel(), device=device, dtype=dtype)
    noise = noise.reshape_as(x)

    previous_params = getattr(reconstructor, "_active_params", None)
    reconstructor._active_params = params
    try:
        denoised = reconstructor._denoise_patches(x, sigma.reshape(1), layout, params)
        predicted = reconstructor._forward_project(
            reconstructor._crop(denoised, params).squeeze(0)
        )
        expected_residual = measurement - predicted.to(dtype=measurement.dtype)
        expected_norm = torch.linalg.norm(expected_residual).clamp_min(1e-12)
        expected_raw_gradient = torch.autograd.grad(
            outputs=expected_residual.square().sum(), inputs=x, retain_graph=True
        )[0]
        (
            gradient,
            raw_gradient,
            residual,
            _normalizer,
            _scale,
            step,
        ) = reconstructor._dps_data_gradient(
            measurement, x, denoised, params, sigma=sigma
        )
    finally:
        reconstructor._active_params = previous_params

    expected_step = float(params.zeta) / float(expected_norm.detach().cpu())
    alpha = float(params.dps_epsilon) * sigma.square()
    score = (denoised - x) / sigma.square()
    expected_next = (
        x
        - expected_step * expected_raw_gradient
        + alpha / 2.0 * score
        + torch.sqrt(alpha) * noise
    )
    lion_next = x - step * gradient + alpha / 2.0 * score + torch.sqrt(alpha) * noise

    raw_gradient_max_abs = float(
        torch.max(torch.abs(expected_raw_gradient - raw_gradient)).detach().cpu()
    )
    residual_max_abs = float(
        torch.max(torch.abs(expected_residual - residual)).detach().cpu()
    )
    step_abs = abs(float(step) - expected_step)
    update_max_abs = float(
        torch.max(torch.abs(expected_next - lion_next)).detach().cpu()
    )
    if (
        raw_gradient_max_abs > tolerance
        or residual_max_abs > tolerance
        or step_abs > tolerance
        or update_max_abs > tolerance
    ):
        raise AssertionError(
            "Paper DPS update mismatch: "
            f"raw_gradient_max_abs={raw_gradient_max_abs}, "
            f"residual_max_abs={residual_max_abs}, "
            f"step_abs={step_abs}, update_max_abs={update_max_abs}"
        )
    return {
        "paper_dps_raw_gradient_max_abs": raw_gradient_max_abs,
        "paper_dps_residual_max_abs": residual_max_abs,
        "paper_dps_step_abs": step_abs,
        "paper_dps_update_max_abs": update_max_abs,
    }


def _compare_public_training_path(
    oracle_classes,
    model: CompatibleDenoiser,
    tolerance: float,
) -> dict[str, float]:
    oracle_loss = oracle_classes["Patch_EDMLoss"]()
    device = next(model.parameters()).device
    dtype = torch.float32
    images = torch.linspace(0.0, 1.0, 2 * 8 * 8, device=device, dtype=dtype).reshape(
        2, 1, 8, 8
    )

    torch.manual_seed(310)
    oracle_patch, oracle_pos = oracle_loss.pachify(images, patch_size=4)
    torch.manual_seed(310)
    lion_patch, lion_pos = sample_image_patch_with_position_channels(images, 4)
    patch_max_abs = float(torch.max(torch.abs(oracle_patch - lion_patch)).cpu())
    position_max_abs = float(torch.max(torch.abs(oracle_pos - lion_pos)).cpu())

    clean_patch = torch.linspace(
        0.0, 1.0, 2 * 4 * 4, device=device, dtype=dtype
    ).reshape(2, 1, 4, 4)
    position_patch = build_position_grid(2, 4, 4, device=device, dtype=dtype)
    torch.manual_seed(311)
    oracle_loss_tensor = oracle_loss(model, clean_patch, patch_size=4, resolution=4)
    oracle_loss_scalar = oracle_loss_tensor.flatten(1).sum(dim=1).mean()
    torch.manual_seed(311)
    lion_loss_scalar = PaDISDenoisingLoss(sigma_distribution="edm_lognormal")(
        model, clean_patch, position_patch
    )
    loss_abs = float(torch.abs(oracle_loss_scalar - lion_loss_scalar).cpu())

    if (
        patch_max_abs > tolerance
        or position_max_abs > tolerance
        or loss_abs > tolerance
    ):
        raise AssertionError(
            "Public training path mismatch: "
            f"patch_max_abs={patch_max_abs}, "
            f"position_max_abs={position_max_abs}, loss_abs={loss_abs}"
        )
    return {
        "public_training_patch_max_abs": patch_max_abs,
        "public_training_position_max_abs": position_max_abs,
        "public_training_loss_abs": loss_abs,
    }


def _compare_paper_training_path(
    model: CompatibleDenoiser,
    tolerance: float,
) -> dict[str, float | bool]:
    device = next(model.parameters()).device
    dtype = torch.float32
    images = torch.linspace(0.0, 1.0, 2 * 8 * 8, device=device, dtype=dtype).reshape(
        2, 1, 8, 8
    )
    padded = zero_pad_images(images, 4)
    patch_size = 4
    batch_size, _channels, height, width = padded.shape

    torch.manual_seed(410)
    top = torch.randint(0, height - patch_size + 1, (batch_size,), device=device)
    left = torch.randint(0, width - patch_size + 1, (batch_size,), device=device)
    rows = top[:, None] + torch.arange(patch_size, device=device)[None, :]
    cols = left[:, None] + torch.arange(patch_size, device=device)[None, :]
    batch = torch.arange(batch_size, device=device)[:, None, None]
    expected_patch = padded.permute(1, 0, 2, 3)[
        :, batch, rows[:, :, None], cols[:, None, :]
    ].permute(1, 0, 2, 3)
    expected_x = (cols.to(dtype) / (width - 1) - 0.5) * 2.0
    expected_y = (rows.to(dtype) / (height - 1) - 0.5) * 2.0
    expected_x = expected_x[:, None, None, :].expand(-1, 1, patch_size, -1)
    expected_y = expected_y[:, None, :, None].expand(-1, 1, -1, patch_size)
    expected_pos = torch.cat((expected_x, expected_y), dim=1)

    torch.manual_seed(410)
    lion_patch, lion_pos = sample_image_patch_with_position_channels(padded, patch_size)
    patch_max_abs = float(torch.max(torch.abs(expected_patch - lion_patch)).cpu())
    position_max_abs = float(torch.max(torch.abs(expected_pos - lion_pos)).cpu())

    bounded_loss = PaDISDenoisingLoss(
        sigma_min=0.002,
        sigma_max=40.0,
        sigma_distribution="edm_lognormal_truncated",
    )
    torch.manual_seed(411)
    sigma = bounded_loss.sample_sigma(4096, device)
    sigma_in_bounds = bool(
        (sigma >= 0.002).all().item() and (sigma <= 40.0).all().item()
    )

    clean_patch = torch.linspace(
        0.0, 1.0, 2 * 4 * 4, device=device, dtype=dtype
    ).reshape(2, 1, 4, 4)
    position_patch = build_position_grid(2, 4, 4, device=device, dtype=dtype)
    torch.manual_seed(412)
    loss_a = bounded_loss(model, clean_patch, position_patch)
    torch.manual_seed(412)
    loss_b = bounded_loss(model, clean_patch, position_patch)
    torch.manual_seed(413)
    loss_c = bounded_loss(model, clean_patch, position_patch)
    replay_abs = float(torch.abs(loss_a - loss_b).cpu())
    different_seed_abs = float(torch.abs(loss_a - loss_c).cpu())

    if (
        patch_max_abs > tolerance
        or position_max_abs > tolerance
        or not sigma_in_bounds
        or replay_abs > tolerance
    ):
        raise AssertionError(
            "Paper training path mismatch: "
            f"patch_max_abs={patch_max_abs}, "
            f"position_max_abs={position_max_abs}, "
            f"sigma_in_bounds={sigma_in_bounds}, replay_abs={replay_abs}"
        )
    return {
        "paper_training_patch_max_abs": patch_max_abs,
        "paper_training_position_max_abs": position_max_abs,
        "paper_training_sigma_min": float(sigma.min().cpu()),
        "paper_training_sigma_max": float(sigma.max().cpu()),
        "paper_training_sigma_in_bounds": sigma_in_bounds,
        "paper_training_seed_replay_abs": replay_abs,
        "paper_training_different_seed_abs": different_seed_abs,
    }


def _check_seeding(
    prefix: str,
    oracle_functions,
    reconstructor: PaDIS,
    tolerance: float,
) -> dict[str, float | bool]:
    get_indices = oracle_functions["getIndices"]
    random.seed(123)
    oracle_indices_a = get_indices([0, 4, 8], 3, 4, 4)
    random.seed(123)
    oracle_indices_b = get_indices([0, 4, 8], 3, 4, 4)
    oracle_seed_replay = oracle_indices_a == oracle_indices_b

    measurement = torch.linspace(0.1, 0.9, 64, dtype=torch.float32).reshape(1, 8, 8)
    generator_a = torch.Generator(device="cpu").manual_seed(999)
    generator_b = torch.Generator(device="cpu").manual_seed(999)
    out_a = reconstructor.reconstruct_sample(
        measurement, prog_bar=False, generator=generator_a
    )
    out_b = reconstructor.reconstruct_sample(
        measurement, prog_bar=False, generator=generator_b
    )
    replay_max_abs = float(torch.max(torch.abs(out_a - out_b)).detach().cpu())

    generator_c = torch.Generator(device="cpu").manual_seed(1000)
    out_c = reconstructor.reconstruct_sample(
        measurement, prog_bar=False, generator=generator_c
    )
    different_seed_max_abs = float(torch.max(torch.abs(out_a - out_c)).detach().cpu())

    if not oracle_seed_replay:
        raise AssertionError("PaDIS oracle Python-random replay was not deterministic.")
    if replay_max_abs > tolerance:
        raise AssertionError(
            f"LION {prefix} seeded replay mismatch: max_abs={replay_max_abs}"
        )

    return {
        f"{prefix}_oracle_seed_replay": oracle_seed_replay,
        f"{prefix}_lion_seed_replay_max_abs": replay_max_abs,
        f"{prefix}_lion_different_seed_max_abs": different_seed_max_abs,
    }


def _patch_assembly_trace(
    oracle_functions,
    reconstructor: PaDIS,
    params: LIONParameter,
    source: str,
) -> dict[str, torch.Tensor]:
    device = next(reconstructor.model.parameters()).device
    dtype = torch.float32
    central = torch.linspace(0.05, 0.95, 64, device=device, dtype=dtype).reshape(
        1, 1, 8, 8
    )
    padded = F.pad(central, (4, 4, 4, 4), mode="constant", value=0.0)
    sigma = torch.tensor(0.02, device=device, dtype=dtype)
    indices = _fixed_indices(8, 4, offset=(1, 2))
    layout = _PatchLayout(indices=indices, image_height=8, image_width=8)
    if source == "oracle":
        denoised_from_patches = oracle_functions["denoisedFromPatches"]
        latents_pos = reconstructor._position_grid(padded)
        denoised = denoised_from_patches(
            reconstructor.model,
            padded,
            sigma,
            latents_pos,
            None,
            [list(index) for index in indices],
            t_goal=0,
        )
    elif source == "lion":
        denoised = reconstructor._denoise_patches(
            padded, sigma.reshape(1), layout, params
        )
    else:
        raise ValueError(f"Unknown trace source: {source}")
    return {"public_patch_denoised": denoised}


def _public_dps_trace(
    oracle_functions,
    reconstructor: PaDIS,
    params: LIONParameter,
    source: str,
) -> dict[str, torch.Tensor]:
    device = next(reconstructor.model.parameters()).device
    dtype = torch.float32
    central = torch.linspace(0.1, 0.9, 64, device=device, dtype=dtype).reshape(
        1, 1, 8, 8
    )
    padded = F.pad(central, (4, 4, 4, 4), mode="constant", value=0.0)
    sigma = torch.tensor(0.02, device=device, dtype=dtype)
    indices = _fixed_indices(8, 4, offset=(2, 1))
    layout = _PatchLayout(indices=indices, image_height=8, image_width=8)
    measurement = torch.linspace(0.2, 0.8, 64, device=device, dtype=dtype).reshape(
        1, 8, 8
    )

    if source == "oracle":
        x = padded.squeeze(0).clone().detach().requires_grad_(True)
        noise = torch.linspace(-0.2, 0.2, x.numel(), device=device, dtype=dtype)
        noise = noise.reshape_as(x)
        denoised_from_patches = oracle_functions["denoisedFromPatches"]
        measurement_cond_fn = oracle_functions["measurement_cond_fn"]
        latents_pos = reconstructor._position_grid(x.unsqueeze(0))
        denoised = denoised_from_patches(
            reconstructor.model,
            x.unsqueeze(0),
            sigma,
            latents_pos,
            None,
            [list(index) for index in indices],
            t_goal=0,
        ).squeeze(0)
        raw_gradient = measurement_cond_fn(
            measurement,
            x,
            denoised,
            reconstructor.op,
            pad=int(params.pad_width),
            w=8,
        )
        gradient = raw_gradient * float(params.data_consistency_scale)
        score = (denoised - x) / sigma.square()
        update = (
            x
            - float(params.zeta) * gradient
            + float(params.dps_epsilon) * sigma.square() / 2.0 * score
            + torch.sqrt(float(params.dps_epsilon) * sigma.square()) * noise
        )
    elif source == "lion":
        x = padded.clone().detach().requires_grad_(True)
        noise = torch.linspace(-0.2, 0.2, x.numel(), device=device, dtype=dtype)
        noise = noise.reshape_as(x)
        denoised = reconstructor._denoise_patches(x, sigma.reshape(1), layout, params)
        previous_params = getattr(reconstructor, "_active_params", None)
        reconstructor._active_params = params
        try:
            (
                gradient,
                raw_gradient,
                _residual,
                _normalizer,
                _scale,
                step,
            ) = reconstructor._dps_data_gradient(
                measurement, x, denoised, params, sigma=sigma
            )
        finally:
            reconstructor._active_params = previous_params
        score = (denoised - x) / sigma.square()
        alpha = float(params.dps_epsilon) * sigma.square()
        update = x - step * gradient + alpha / 2.0 * score + torch.sqrt(alpha) * noise
        gradient = gradient.squeeze(0)
        raw_gradient = raw_gradient.squeeze(0)
        update = update.squeeze(0)
    else:
        raise ValueError(f"Unknown trace source: {source}")
    return {
        "public_dps_raw_gradient": raw_gradient,
        "public_dps_gradient": gradient,
        "public_dps_update": update,
    }


def _paper_dps_trace(
    reconstructor: PaDIS,
    params: LIONParameter,
    source: str,
) -> dict[str, torch.Tensor]:
    device = next(reconstructor.model.parameters()).device
    dtype = torch.float32
    central = torch.linspace(0.1, 0.9, 64, device=device, dtype=dtype).reshape(
        1, 1, 8, 8
    )
    x = F.pad(central, (4, 4, 4, 4), mode="constant", value=0.0)
    x = x.clone().detach().requires_grad_(True)
    sigma = torch.tensor(0.02, device=device, dtype=dtype)
    layout = _PatchLayout(
        indices=_fixed_indices(8, 4, offset=(2, 1)),
        image_height=8,
        image_width=8,
    )
    measurement = torch.linspace(0.2, 0.8, 64, device=device, dtype=dtype).reshape(
        1, 8, 8
    )
    noise = torch.linspace(-0.2, 0.2, x.numel(), device=device, dtype=dtype)
    noise = noise.reshape_as(x)

    previous_params = getattr(reconstructor, "_active_params", None)
    reconstructor._active_params = params
    try:
        denoised = reconstructor._denoise_patches(x, sigma.reshape(1), layout, params)
        if source == "paper":
            predicted = reconstructor._forward_project(
                reconstructor._crop(denoised, params).squeeze(0)
            )
            residual = measurement - predicted.to(dtype=measurement.dtype)
            normalizer = torch.linalg.norm(residual).clamp_min(1e-12)
            gradient = torch.autograd.grad(
                outputs=residual.square().sum(), inputs=x, retain_graph=True
            )[0]
            step = float(params.zeta) / float(normalizer.detach().cpu())
        elif source == "lion":
            (
                gradient,
                _raw,
                residual,
                _normalizer,
                _scale,
                step,
            ) = reconstructor._dps_data_gradient(
                measurement, x, denoised, params, sigma=sigma
            )
        else:
            raise ValueError(f"Unknown trace source: {source}")
    finally:
        reconstructor._active_params = previous_params

    alpha = float(params.dps_epsilon) * sigma.square()
    score = (denoised - x) / sigma.square()
    update = x - step * gradient + alpha / 2.0 * score + torch.sqrt(alpha) * noise
    return {
        "paper_dps_gradient": gradient,
        "paper_dps_residual": residual,
        "paper_dps_update": update,
    }


def _public_training_trace(
    oracle_classes,
    model: CompatibleDenoiser,
    source: str,
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    dtype = torch.float32
    images = torch.linspace(0.0, 1.0, 2 * 8 * 8, device=device, dtype=dtype).reshape(
        2, 1, 8, 8
    )
    clean_patch = torch.linspace(
        0.0, 1.0, 2 * 4 * 4, device=device, dtype=dtype
    ).reshape(2, 1, 4, 4)
    position_patch = build_position_grid(2, 4, 4, device=device, dtype=dtype)

    if source == "oracle":
        oracle_loss = oracle_classes["Patch_EDMLoss"]()
        torch.manual_seed(310)
        patch, position = oracle_loss.pachify(images, patch_size=4)
        torch.manual_seed(311)
        loss_tensor = oracle_loss(model, clean_patch, patch_size=4, resolution=4)
        loss = loss_tensor.flatten(1).sum(dim=1).mean()
    elif source == "lion":
        torch.manual_seed(310)
        patch, position = sample_image_patch_with_position_channels(images, 4)
        torch.manual_seed(311)
        loss = PaDISDenoisingLoss(sigma_distribution="edm_lognormal")(
            model, clean_patch, position_patch
        )
    else:
        raise ValueError(f"Unknown trace source: {source}")

    return {
        "public_training_patch": patch,
        "public_training_position": position,
        "public_training_loss": loss.reshape(1),
    }


def _paper_training_trace(
    model: CompatibleDenoiser,
    source: str,
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    dtype = torch.float32
    images = torch.linspace(0.0, 1.0, 2 * 8 * 8, device=device, dtype=dtype).reshape(
        2, 1, 8, 8
    )
    padded = zero_pad_images(images, 4)
    patch_size = 4
    batch_size, _channels, height, width = padded.shape

    if source == "paper":
        torch.manual_seed(410)
        top = torch.randint(0, height - patch_size + 1, (batch_size,), device=device)
        left = torch.randint(0, width - patch_size + 1, (batch_size,), device=device)
        rows = top[:, None] + torch.arange(patch_size, device=device)[None, :]
        cols = left[:, None] + torch.arange(patch_size, device=device)[None, :]
        batch = torch.arange(batch_size, device=device)[:, None, None]
        patch = padded.permute(1, 0, 2, 3)[
            :, batch, rows[:, :, None], cols[:, None, :]
        ].permute(1, 0, 2, 3)
        expected_x = (cols.to(dtype) / (width - 1) - 0.5) * 2.0
        expected_y = (rows.to(dtype) / (height - 1) - 0.5) * 2.0
        expected_x = expected_x[:, None, None, :].expand(-1, 1, patch_size, -1)
        expected_y = expected_y[:, None, :, None].expand(-1, 1, -1, patch_size)
        position = torch.cat((expected_x, expected_y), dim=1)
    elif source == "lion":
        torch.manual_seed(410)
        patch, position = sample_image_patch_with_position_channels(padded, patch_size)
    else:
        raise ValueError(f"Unknown trace source: {source}")

    bounded_loss = PaDISDenoisingLoss(
        sigma_min=0.002,
        sigma_max=40.0,
        sigma_distribution="edm_lognormal_truncated",
    )
    clean_patch = torch.linspace(
        0.0, 1.0, 2 * 4 * 4, device=device, dtype=dtype
    ).reshape(2, 1, 4, 4)
    position_patch = build_position_grid(2, 4, 4, device=device, dtype=dtype)

    torch.manual_seed(411)
    sigma = bounded_loss.sample_sigma(32, device)

    if source == "paper":
        torch.manual_seed(412)
        loss_sigma = bounded_loss.sample_sigma(clean_patch.shape[0], device)
        sigma_view = loss_sigma.view(-1, 1, 1, 1)
        noisy_patch = clean_patch + sigma_view * torch.randn_like(clean_patch)
        sigma_data = torch.as_tensor(
            bounded_loss.sigma_data, device=device, dtype=clean_patch.dtype
        )
        c_skip = sigma_data.square() / (sigma_view.square() + sigma_data.square())
        c_out = (
            sigma_view * sigma_data / (sigma_view.square() + sigma_data.square()).sqrt()
        )
        c_in = 1 / (sigma_data.square() + sigma_view.square()).sqrt()
        c_noise = loss_sigma.log() / 4
        model_input = torch.cat((c_in * noisy_patch, position_patch), dim=1)
        model_output = model(model_input, c_noise)
        denoised = c_skip * noisy_patch + c_out * model_output
        weight = (sigma_view.square() + sigma_data.square()) / (
            sigma_view * sigma_data
        ).square()
        loss = weight.view(-1, 1, 1, 1) * (denoised - clean_patch).square()
        loss = loss.flatten(1).sum(dim=1).mean()
    elif source == "lion":
        torch.manual_seed(412)
        loss = bounded_loss(model, clean_patch, position_patch)
    else:
        raise ValueError(f"Unknown trace source: {source}")

    return {
        "paper_training_patch": patch,
        "paper_training_position": position,
        "paper_training_sigma_sample": sigma,
        "paper_training_loss": loss.reshape(1),
    }


def _training_step_params(
    *,
    pad_width: int,
    sigma_distribution: str,
) -> LIONParameter:
    params = PaDISSolver.default_parameters("padis-paper-ct-p8")
    params.patch_sizes = [8]
    params.patch_probabilities = [1.0]
    params.patch_batch_multipliers = {8: 1}
    params.pad_width = pad_width
    params.largest_patch_size = 8
    params.sigma_distribution = sigma_distribution
    params.use_position_channels = True
    params.use_ema = True
    params.ema_half_life_patches = 100
    params.ema_rampup_ratio = 0.05
    params.lr_rampup_kimg = 0.02
    params.enforce_data_range = True
    return params


def _training_target(device: torch.device) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, 2 * 16 * 16, device=device).reshape(2, 1, 16, 16)


def _new_training_model(
    device: torch.device, *, pad_width: int
) -> TrainableCompatibleDenoiser:
    model = TrainableCompatibleDenoiser(pad_width=pad_width, patch_size=8)
    return model.to(device=device, dtype=torch.float32)


def _named_parameter_payload(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        f"param_{name}": param.detach().clone()
        for name, param in model.named_parameters()
    }


def _named_gradient_payload(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        f"grad_{name}": param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def _optimizer_state_payload(
    optimizer: torch.optim.Optimizer,
) -> dict[str, torch.Tensor]:
    payload = {}
    for group_index, group in enumerate(optimizer.param_groups):
        for param_index, param in enumerate(group["params"]):
            state = optimizer.state.get(param, {})
            for key in ("step", "exp_avg", "exp_avg_sq"):
                value = state.get(key)
                if value is None:
                    continue
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(float(value), device=param.device)
                payload[
                    f"optimizer_g{group_index}_p{param_index}_{key}"
                ] = value.detach().clone()
    return payload


def _ema_beta(
    seen_patches: int,
    batch_patch_count: int,
    params: LIONParameter,
) -> float:
    half_life = float(params.ema_half_life_patches)
    if params.ema_rampup_ratio is not None:
        half_life = min(half_life, float(seen_patches) * float(params.ema_rampup_ratio))
    return 0.5 ** (float(batch_patch_count) / max(half_life, 1e-8))


def _set_training_lr(
    optimizer: torch.optim.Optimizer,
    params: LIONParameter,
    seen_patches: int,
) -> float:
    if params.lr_rampup_kimg is None:
        return float(optimizer.param_groups[0]["lr"])
    lr_scale = min(
        float(seen_patches) / max(float(params.lr_rampup_kimg) * 1000, 1e-8),
        1.0,
    )
    for group in optimizer.param_groups:
        group.setdefault("base_lr", group["lr"])
        group["lr"] = group["base_lr"] * lr_scale
    return float(optimizer.param_groups[0]["lr"])


def _sanitize_gradients(model: nn.Module) -> None:
    for param in model.parameters():
        if param.grad is not None:
            torch.nan_to_num(
                param.grad, nan=0.0, posinf=1e5, neginf=-1e5, out=param.grad
            )


def _training_step_payload(
    prefix: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    ema_state: dict[str, torch.Tensor],
    seen_patches: int,
) -> dict[str, torch.Tensor]:
    payload = {
        f"{prefix}_loss": loss.detach().reshape(1).clone(),
        f"{prefix}_seen_patches": torch.tensor([seen_patches], dtype=torch.float32),
        f"{prefix}_lr": torch.tensor(
            [optimizer.param_groups[0]["lr"]], dtype=torch.float32
        ),
    }
    for key, value in _named_parameter_payload(model).items():
        payload[f"{prefix}_{key}"] = value
    for key, value in _named_gradient_payload(model).items():
        payload[f"{prefix}_{key}"] = value
    for key, value in _optimizer_state_payload(optimizer).items():
        payload[f"{prefix}_{key}"] = value
    for name, value in ema_state.items():
        payload[f"{prefix}_ema_{name}"] = value.detach().clone()
    return payload


def _public_repo_training_step_trace(
    oracle_classes,
    device: torch.device,
    source: str,
) -> dict[str, torch.Tensor]:
    params = _training_step_params(pad_width=0, sigma_distribution="edm_lognormal")
    target = _training_target(device)
    initial_seen = 10
    base_lr = 1e-3
    seed = 710

    model = _new_training_model(device, pad_width=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999))
    model.train()

    if source == "oracle":
        ema_state = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        oracle_loss = oracle_classes["Patch_EDMLoss"]()
        torch.manual_seed(seed)
        optimizer.zero_grad(set_to_none=True)
        loss_tensor = oracle_loss(
            model, target, patch_size=8, resolution=16, labels=None, augment_pipe=None
        )
        loss = loss_tensor.flatten(1).sum(dim=1).mean()
        loss.backward()
        _set_training_lr(optimizer, params, initial_seen)
        _sanitize_gradients(model)
        optimizer.step()
        beta = _ema_beta(initial_seen, int(target.shape[0]), params)
        with torch.no_grad():
            for name, param in model.named_parameters():
                ema_state[name].mul_(beta).add_(param.detach(), alpha=1.0 - beta)
        seen_patches = initial_seen + int(target.shape[0])
    elif source == "lion":
        geometry = Geometry.default_parameters(image_scaling=0.5)
        loss_fn = PaDISDenoisingLoss(sigma_distribution="edm_lognormal")
        solver = PaDISSolver(
            model,
            optimizer,
            loss_fn,
            geometry=geometry,
            verbose=False,
            device=device,
            solver_params=params,
        )
        solver.seen_patches = initial_seen
        torch.manual_seed(seed)
        loss = torch.tensor(
            solver._optimizer_step(target, patch_size=8),
            device=device,
            dtype=torch.float32,
        )
        ema_state = solver.ema_state
        seen_patches = solver.seen_patches
    else:
        raise ValueError(f"Unknown trace source: {source}")

    return _training_step_payload(
        "public_training_step", model, optimizer, loss, ema_state, seen_patches
    )


def _sample_paper_patch_reference(
    target: torch.Tensor,
    pad_width: int,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    padded = zero_pad_images(target, pad_width)
    batch_size, _channels, height, width = padded.shape
    top = torch.randint(0, height - patch_size + 1, (batch_size,), device=target.device)
    left = torch.randint(0, width - patch_size + 1, (batch_size,), device=target.device)
    rows = top[:, None] + torch.arange(patch_size, device=target.device)[None, :]
    cols = left[:, None] + torch.arange(patch_size, device=target.device)[None, :]
    batch = torch.arange(batch_size, device=target.device)[:, None, None]
    patch = padded.permute(1, 0, 2, 3)[
        :, batch, rows[:, :, None], cols[:, None, :]
    ].permute(1, 0, 2, 3)
    x_pos = (cols.to(target.dtype) / (width - 1) - 0.5) * 2.0
    y_pos = (rows.to(target.dtype) / (height - 1) - 0.5) * 2.0
    x_pos = x_pos[:, None, None, :].expand(-1, 1, patch_size, -1)
    y_pos = y_pos[:, None, :, None].expand(-1, 1, -1, patch_size)
    return patch, torch.cat((x_pos, y_pos), dim=1)


def _paper_training_loss_reference(
    model: nn.Module,
    clean_patch: torch.Tensor,
    position_patch: torch.Tensor,
) -> torch.Tensor:
    loss_config = PaDISDenoisingLoss(
        sigma_min=0.002,
        sigma_max=40.0,
        sigma_distribution="edm_lognormal_truncated",
    )
    sigma = loss_config.sample_sigma(clean_patch.shape[0], clean_patch.device)
    sigma_view = sigma.reshape(clean_patch.shape[0], 1, 1, 1)
    noisy_patch = clean_patch + sigma_view * torch.randn_like(clean_patch)
    sigma_data = torch.as_tensor(
        loss_config.sigma_data, device=clean_patch.device, dtype=clean_patch.dtype
    )
    c_skip = sigma_data.square() / (sigma_view.square() + sigma_data.square())
    c_out = sigma_view * sigma_data / (sigma_view.square() + sigma_data.square()).sqrt()
    c_in = 1 / (sigma_data.square() + sigma_view.square()).sqrt()
    c_noise = sigma.log() / 4
    model_input = torch.cat((c_in * noisy_patch, position_patch), dim=1)
    model_output = model(model_input, c_noise)
    denoised = c_skip * noisy_patch + c_out * model_output
    weight = (sigma_view.square() + sigma_data.square()) / (
        sigma_view * sigma_data
    ).square()
    loss = weight * (denoised - clean_patch).square()
    return loss.flatten(1).sum(dim=1).mean()


def _paper_training_step_trace(
    device: torch.device,
    source: str,
) -> dict[str, torch.Tensor]:
    params = _training_step_params(
        pad_width=4, sigma_distribution="edm_lognormal_truncated"
    )
    target = _training_target(device)
    initial_seen = 10
    base_lr = 1e-3
    seed = 810

    model = _new_training_model(device, pad_width=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999))
    model.train()

    if source == "paper":
        ema_state = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        torch.manual_seed(seed)
        optimizer.zero_grad(set_to_none=True)
        patch, position = _sample_paper_patch_reference(target, 4, 8)
        loss = _paper_training_loss_reference(model, patch, position)
        loss.backward()
        _set_training_lr(optimizer, params, initial_seen)
        _sanitize_gradients(model)
        optimizer.step()
        beta = _ema_beta(initial_seen, int(target.shape[0]), params)
        with torch.no_grad():
            for name, param in model.named_parameters():
                ema_state[name].mul_(beta).add_(param.detach(), alpha=1.0 - beta)
        seen_patches = initial_seen + int(target.shape[0])
    elif source == "lion":
        geometry = Geometry.default_parameters(image_scaling=0.5)
        loss_fn = PaDISDenoisingLoss(
            sigma_min=0.002,
            sigma_max=40.0,
            sigma_distribution="edm_lognormal_truncated",
        )
        solver = PaDISSolver(
            model,
            optimizer,
            loss_fn,
            geometry=geometry,
            verbose=False,
            device=device,
            solver_params=params,
        )
        solver.seen_patches = initial_seen
        torch.manual_seed(seed)
        loss = torch.tensor(
            solver._optimizer_step(target, patch_size=8),
            device=device,
            dtype=torch.float32,
        )
        ema_state = solver.ema_state
        seen_patches = solver.seen_patches
    else:
        raise ValueError(f"Unknown trace source: {source}")

    return _training_step_payload(
        "paper_training_step", model, optimizer, loss, ema_state, seen_patches
    )


def _make_reference_payload(
    oracle_functions,
    oracle_classes,
    model: CompatibleDenoiser,
    public_reconstructor: PaDIS,
    paper_reconstructor: PaDIS,
    public_params: LIONParameter,
    paper_params: LIONParameter,
) -> dict[str, torch.Tensor]:
    payload = {}
    payload.update(
        _patch_assembly_trace(
            oracle_functions, public_reconstructor, public_params, source="oracle"
        )
    )
    payload.update(
        _public_dps_trace(
            oracle_functions, public_reconstructor, public_params, source="oracle"
        )
    )
    payload.update(_paper_dps_trace(paper_reconstructor, paper_params, source="paper"))
    payload.update(_public_training_trace(oracle_classes, model, source="oracle"))
    payload.update(_paper_training_trace(model, source="paper"))
    device = next(model.parameters()).device
    payload.update(_public_repo_training_step_trace(oracle_classes, device, "oracle"))
    payload.update(_paper_training_step_trace(device, "paper"))
    return payload


def _make_candidate_payload(
    oracle_functions,
    oracle_classes,
    model: CompatibleDenoiser,
    public_reconstructor: PaDIS,
    paper_reconstructor: PaDIS,
    public_params: LIONParameter,
    paper_params: LIONParameter,
) -> dict[str, torch.Tensor]:
    payload = {}
    payload.update(
        _patch_assembly_trace(
            oracle_functions, public_reconstructor, public_params, source="lion"
        )
    )
    payload.update(
        _public_dps_trace(
            oracle_functions, public_reconstructor, public_params, source="lion"
        )
    )
    payload.update(_paper_dps_trace(paper_reconstructor, paper_params, source="lion"))
    payload.update(_public_training_trace(oracle_classes, model, source="lion"))
    payload.update(_paper_training_trace(model, source="lion"))
    device = next(model.parameters()).device
    payload.update(_public_repo_training_step_trace(oracle_classes, device, "lion"))
    payload.update(_paper_training_step_trace(device, "lion"))
    return payload


def _to_cpu_payload(payload: dict[str, object]) -> dict[str, object]:
    output = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.detach().cpu()
        elif isinstance(value, dict):
            output[key] = _to_cpu_payload(value)
        elif isinstance(value, (bool, int, float, str)):
            output[key] = value
        else:
            raise TypeError(
                f"Unsupported golden payload value for {key}: {type(value)}"
            )
    return output


def _load_golden(path: pathlib.Path) -> dict[str, object]:
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        artifact = torch.load(path, map_location="cpu")
    if not isinstance(artifact, dict):
        raise TypeError(f"Golden artifact {path} did not contain a dict.")
    if "payload" in artifact:
        payload = artifact["payload"]
    else:
        payload = artifact
    if not isinstance(payload, dict):
        raise TypeError(f"Golden artifact {path} did not contain a payload dict.")
    return payload


def _compare_golden_payloads(
    golden: dict[str, object],
    candidate: dict[str, object],
    tolerance: float,
) -> dict[str, object]:
    golden = _to_cpu_payload(golden)
    candidate = _to_cpu_payload(candidate)
    missing = sorted(set(golden).difference(candidate))
    extra = sorted(set(candidate).difference(golden))
    if missing or extra:
        raise AssertionError(f"Golden keys mismatch: missing={missing}, extra={extra}")

    summary: dict[str, object] = {"golden_all_passed": True}
    for key in sorted(golden):
        expected = golden[key]
        actual = candidate[key]
        if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
            if tuple(expected.shape) != tuple(actual.shape):
                raise AssertionError(
                    f"Golden shape mismatch for {key}: "
                    f"expected={tuple(expected.shape)}, actual={tuple(actual.shape)}"
                )
            if expected.numel() == 0:
                max_abs = 0.0
            else:
                max_abs = float(torch.max(torch.abs(expected - actual)).cpu())
            summary[f"golden_{key}_max_abs"] = max_abs
            if max_abs > tolerance:
                summary["golden_all_passed"] = False
                raise AssertionError(
                    f"Golden value mismatch for {key}: max_abs={max_abs}"
                )
        elif isinstance(expected, bool) and isinstance(actual, bool):
            if expected != actual:
                summary["golden_all_passed"] = False
                raise AssertionError(
                    f"Golden bool mismatch for {key}: "
                    f"expected={expected}, actual={actual}"
                )
        elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            abs_diff = abs(float(expected) - float(actual))
            summary[f"golden_{key}_abs"] = abs_diff
            if abs_diff > tolerance:
                summary["golden_all_passed"] = False
                raise AssertionError(
                    f"Golden scalar mismatch for {key}: abs_diff={abs_diff}"
                )
        elif isinstance(expected, str) and isinstance(actual, str):
            if expected != actual:
                summary["golden_all_passed"] = False
                raise AssertionError(
                    f"Golden string mismatch for {key}: "
                    f"expected={expected}, actual={actual}"
                )
        else:
            raise TypeError(
                f"Golden type mismatch for {key}: "
                f"expected={type(expected)}, actual={type(actual)}"
            )
    return summary


def run_check(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    denoise_path = args.padis_root / "denoise_padding.py"
    inverse_path = args.padis_root / "inverse_nodist.py"
    patch_loss_path = args.padis_root / "training" / "patch_loss.py"
    if (
        not denoise_path.is_file()
        or not inverse_path.is_file()
        or not patch_loss_path.is_file()
    ):
        raise FileNotFoundError(
            f"Could not find PaDIS oracle files under {args.padis_root}."
        )
    oracle_functions = {}
    oracle_functions.update(
        _extract_functions(
            denoise_path, {"getIndices", "denoisedFromPatches"}, device=device
        )
    )
    oracle_functions.update(
        _extract_functions(inverse_path, {"measurement_cond_fn"}, device=device)
    )
    oracle_classes = _extract_classes(patch_loss_path, {"Patch_EDMLoss"}, device=device)

    model = CompatibleDenoiser().to(device=device, dtype=torch.float32)
    public_params = _build_params(model)
    paper_params = _build_paper_params(model)
    op = ScaledIdentityOp(scale=1.7)
    public_reconstructor = PaDIS(op, model, public_params, algorithm="dps_langevin")
    paper_reconstructor = PaDIS(op, model, paper_params, algorithm="dps_langevin")
    public_reconstructor.model.eval()
    paper_reconstructor.model.eval()

    summary: dict[str, object] = {
        "device": str(device),
        "padis_root": str(args.padis_root),
        "seed": int(args.seed),
        "tolerance": float(args.tolerance),
    }
    summary.update(
        _compare_patch_assembly(
            oracle_functions, public_reconstructor, public_params, args.tolerance
        )
    )
    summary.update(
        _compare_dps_update(
            oracle_functions, public_reconstructor, public_params, args.tolerance
        )
    )
    summary.update(
        _compare_paper_dps_update(paper_reconstructor, paper_params, args.tolerance)
    )
    summary.update(_compare_public_training_path(oracle_classes, model, args.tolerance))
    summary.update(_compare_paper_training_path(model, args.tolerance))
    summary.update(
        _check_seeding(
            "public_repo_mode", oracle_functions, public_reconstructor, args.tolerance
        )
    )
    summary.update(
        _check_seeding(
            "paper_mode", oracle_functions, paper_reconstructor, args.tolerance
        )
    )
    if args.write_golden is not None or args.golden is not None:
        reference_payload = _make_reference_payload(
            oracle_functions,
            oracle_classes,
            model,
            public_reconstructor,
            paper_reconstructor,
            public_params,
            paper_params,
        )
        candidate_payload = _make_candidate_payload(
            oracle_functions,
            oracle_classes,
            model,
            public_reconstructor,
            paper_reconstructor,
            public_params,
            paper_params,
        )
        if args.write_golden is not None:
            args.write_golden.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "metadata": {
                        "device": str(device),
                        "padis_root": str(args.padis_root),
                        "seed": int(args.seed),
                        "tolerance": float(args.tolerance),
                    },
                    "payload": _to_cpu_payload(reference_payload),
                },
                args.write_golden,
            )
            summary["golden_written"] = str(args.write_golden)
        if args.golden is not None:
            golden_payload = _load_golden(args.golden)
            summary.update(
                _compare_golden_payloads(
                    golden_payload, candidate_payload, args.tolerance
                )
            )
            summary["golden_compared"] = str(args.golden)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_padis_root = pathlib.Path(__file__).resolve().parents[3] / "PaDIS"
    parser.add_argument("--padis-root", type=pathlib.Path, default=default_padis_root)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--write-golden", type=pathlib.Path, default=None)
    parser.add_argument("--golden", type=pathlib.Path, default=None)
    parser.add_argument("--json", type=pathlib.Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.padis_root = args.padis_root.resolve()
    if args.write_golden is not None:
        args.write_golden = args.write_golden.resolve()
    if args.golden is not None:
        args.golden = args.golden.resolve()
    summary = run_check(args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
