from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.pcm_config import SAPMPCMConfig
from LION.pcm.pcm_sapm import PCMForwardData, make_pcm_forward_data


@torch.no_grad()
def split_pcm_measurements(
    *,
    source_indices: np.ndarray,
    source_measurement: torch.Tensor,
    p_target: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """Split acquired PCM measurements into disjoint input and held-out target subsets."""
    n_measurements = len(source_indices)
    n_input = int((1.0 - p_target) * n_measurements)

    permutation = rng.permutation(n_measurements)
    input_positions = np.sort(permutation[:n_input])
    target_positions = np.sort(permutation[n_input:])

    input_indices = source_indices[input_positions]
    target_indices = source_indices[target_positions]

    input_position_tensor = torch.as_tensor(
        input_positions, dtype=torch.long, device=source_measurement.device
    )
    target_position_tensor = torch.as_tensor(
        target_positions, dtype=torch.long, device=source_measurement.device
    )

    input_measurement = source_measurement.index_select(-1, input_position_tensor)
    target_measurement = source_measurement.index_select(-1, target_position_tensor)

    return input_indices, target_indices, input_measurement, target_measurement


@dataclass(frozen=True, kw_only=True)
class PCMSSDUSplit:
    """One PCM SSDU split."""

    forward_data: PCMForwardData
    target_op: PhotocurrentMapOp
    target: torch.Tensor
    input_indices: np.ndarray
    input_measurement: torch.Tensor


def make_pcm_ssdu_split(
    *,
    config: SAPMPCMConfig,
    source_indices: np.ndarray,
    source_measurement: torch.Tensor,
    p_target: float,
    rng: np.random.Generator,
    key: int,
    sampling_ratio: float,
    coarse_j: int,
    device: torch.device | str,
) -> PCMSSDUSplit:
    """Create one PCM SSDU split and its reconstruction input."""
    j_order = config.data.j_order
    assert j_order is not None

    (
        input_indices,
        target_indices,
        input_measurement,
        target_measurement,
    ) = split_pcm_measurements(
        source_indices=source_indices,
        source_measurement=source_measurement,
        p_target=p_target,
        rng=rng,
    )

    input_op = PhotocurrentMapOp(
        j_order=j_order, s_indices=input_indices, normalized=True, device=device
    )
    target_op = PhotocurrentMapOp(
        j_order=j_order, s_indices=target_indices, normalized=True, device=device
    )

    return PCMSSDUSplit(
        forward_data=make_pcm_forward_data(
            config=config,
            op=input_op,
            measurement=input_measurement,
            key=key,
            sampling_ratio=sampling_ratio,
            coarse_j=coarse_j,
            device=device,
        ),
        target_op=target_op,
        target=target_measurement,
        input_indices=input_indices,
        input_measurement=input_measurement,
    )
