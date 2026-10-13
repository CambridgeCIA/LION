"""Shared PCM acquisition simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from LION.data_loaders.pcm.data_primitives import PCMSamplingParameter
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.noise import PCMNoiseSimulation
from LION.pcm_tools.measure import measure_split_hadamard_transform
from LION.pcm_tools.noise_generate import add_noise, generate_noise
from LION.pcm_tools.pcm_sampling import sample_indices


@dataclass(frozen=True, kw_only=True)
class PCMAcquisition:
    """One PCM acquisition already prepared for reconstruction."""

    op: PhotocurrentMapOp
    measurement: torch.Tensor
    sampled_indices: np.ndarray
    seed: int | None = None
    i0: torch.Tensor | None = None


def _sample_split_measurements(
    target_image: torch.Tensor, op: PhotocurrentMapOp
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return clean split measurements in normalized-operator units."""
    vectors = op.vectorize(target_image)
    y_plus, y_minus = measure_split_hadamard_transform(vectors)

    indices = op.sp.indices
    y_plus = y_plus.index_select(-1, indices)
    y_minus = y_minus.index_select(-1, indices)

    factor = op.normalization_factor
    return y_plus * factor, y_minus * factor


@torch.no_grad()
def simulate_pcm_acquisition(
    target_image: torch.Tensor,
    *,
    j_order: int,
    sampled_indices: np.ndarray,
    noise: PCMNoiseSimulation | None,
    rng: np.random.Generator,
    device: torch.device | str,
    seed: int | None = None,
) -> PCMAcquisition:
    """Simulate a PCM acquisition in the canonical reconstruction convention.

    The input image is already in model image units. The forward operator is normalized. ``dark_sigma`` is
    already in normalized measurement units and is not rescaled here.
    """
    target_image = target_image.to(device)
    op = PhotocurrentMapOp(
        j_order=j_order, s_indices=sampled_indices, normalized=True, device=device
    )

    y_plus_clean, y_minus_clean = _sample_split_measurements(target_image, op)

    if noise is None:
        return PCMAcquisition(
            op=op,
            measurement=y_plus_clean - y_minus_clean,
            sampled_indices=sampled_indices,
            seed=seed,
            i0=None,
        )

    total_photocurrent = target_image.sum(dim=(-2, -1))
    # total_photocurrent = (y_plus_clean - y_minus_clean).sum(dim=-1)  # TODO: Remove this!
    i_bias = total_photocurrent * op.normalization_factor / noise.signal_to_bias_ratio

    flat_plus = y_plus_clean.reshape(-1, y_plus_clean.shape[-1])
    flat_minus = y_minus_clean.reshape(-1, y_minus_clean.shape[-1])
    flat_i_bias = i_bias.reshape(-1)

    noisy_measurements: list[torch.Tensor] = []
    for y_plus_i, y_minus_i, i_bias_i in zip(
        flat_plus, flat_minus, flat_i_bias, strict=True
    ):
        repeated_measurements: list[torch.Tensor] = []
        for _ in range(noise.n_repeat_acquisitions):
            e_dark, e_bias, e_rel = generate_noise(
                num_measurements=2 * len(sampled_indices),
                i_bias=float(i_bias_i.item()),
                noise_params=noise.noise_params,
                rng=rng,
            )
            y_plus_noisy, y_minus_noisy, _ = add_noise(
                y_clean_plus=y_plus_i,
                y_clean_minus=y_minus_i,
                e_abs_trace=torch.as_tensor(
                    e_dark + e_bias,
                    dtype=target_image.dtype,
                    device=target_image.device,
                ),
                e_rel=torch.as_tensor(
                    e_rel, dtype=target_image.dtype, device=target_image.device
                ),
                rng=rng,
            )
            repeated_measurements.append(y_plus_noisy - y_minus_noisy)
        noisy_measurements.append(torch.stack(repeated_measurements).mean(dim=0))

    measurement = torch.stack(noisy_measurements).reshape_as(y_plus_clean)

    return PCMAcquisition(
        op=op,
        measurement=measurement,
        sampled_indices=sampled_indices,
        seed=seed,
        i0=i_bias,
    )


@torch.no_grad()
def sample_and_simulate_pcm_acquisition(
    target_image: torch.Tensor,
    *,
    j_order: int,
    sampling: PCMSamplingParameter,
    sampling_ratio: float,
    coarse_j: int,
    noise: PCMNoiseSimulation | None,
    rng: np.random.Generator,
    device: torch.device | str,
    seed: int | None = None,
    noise_rng: np.random.Generator | None = None,
) -> PCMAcquisition:
    """Choose a sampling pattern set and simulate the corresponding PCM acquisition."""
    sampled_indices = sample_indices(
        j_order=j_order,
        sampling_ratio=sampling_ratio,
        coarse_j=coarse_j,
        randomising_scheme=sampling.randomising_scheme,
        rng=rng,
    )
    return simulate_pcm_acquisition(
        target_image,
        j_order=j_order,
        sampled_indices=sampled_indices,
        noise=noise,
        rng=rng if noise_rng is None else noise_rng,
        device=device,
        seed=seed,
    )


def acquisition_from_prepared_measurement(
    measurement: torch.Tensor,
    *,
    j_order: int,
    sampled_indices: np.ndarray,
    device: torch.device | str,
    seed: int | None = None,
) -> PCMAcquisition:
    """Wrap a measurement already prepared for the canonical reconstruction model."""
    op = PhotocurrentMapOp(
        j_order=j_order, s_indices=sampled_indices, normalized=True, device=device
    )
    return PCMAcquisition(
        op=op,
        measurement=measurement.to(device),
        sampled_indices=sampled_indices,
        seed=seed,
        i0=None,
    )
