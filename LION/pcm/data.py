"""Data loading and preparation for PCM experiments."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from LION.data_loaders.pcm.data_loader import GrayscaleImage2D, Measurement1D
from LION.data_loaders.pcm.data_utils import read_pcm_measurement_data_from_txt
from LION.experiments.pcm_experiments import PCMExperimentParameter
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp


def minmax_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor to the ``[0, 1]`` interval.

    Args:
        tensor: Input tensor.

    Returns:
        Min-max normalised tensor.
    """
    min_val = tensor.min()
    max_val = tensor.max()
    denom = max_val - min_val
    if torch.isclose(
        denom, torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
    ):
        raise ValueError("Cannot min-max normalize a constant tensor.")
    return (tensor - min_val) / denom


def load_ground_truth_current_map(config: PCMExperimentParameter) -> np.ndarray | None:
    """Load the ground truth current map."""
    assert config.this_sample is not None
    if config.this_sample.ground_truth_filename is None:
        return None

    ground_truth_path = config.data.root / config.this_sample.ground_truth_filename

    ground_truth_np: np.ndarray
    if ground_truth_path.suffix == ".npy":
        ground_truth_np = np.load(ground_truth_path)
    elif ground_truth_path.suffix == ".map":
        ground_truth_np = np.loadtxt(ground_truth_path, dtype=np.float64)
    elif ground_truth_path.suffix == ".png":
        ground_truth_np = np.array(Image.open(ground_truth_path).convert("L")) / 255.0
    else:
        raise ValueError(f"Unsupported data file format: {ground_truth_path.suffix}")

    scaling = config.scaling
    ground_truth_np = scaling.data_sign * ground_truth_np
    if scaling.divide_by is not None:
        ground_truth_np = ground_truth_np / scaling.divide_by

    if scaling.requires_scaling:
        assert scaling.r_high is not None
        assert scaling.r_low is not None
        scale = scaling.r_high - scaling.r_low
        ground_truth_np = scale * ground_truth_np + scaling.r_low

    return ground_truth_np  # TODO: Should return torch tensor instead?


def prepare_data(
    config: PCMExperimentParameter,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[GrayscaleImage2D, Measurement1D | None]:
    """Load PCM data and prepare the ground truth current map and pattern measurement vector."""
    ground_truth_np = load_ground_truth_current_map(config)
    ground_truth = None
    if ground_truth_np is not None:
        ground_truth = torch.as_tensor(ground_truth_np, device=device, dtype=dtype)

    assert config.this_sample is not None
    raw_data_path = config.data.root / config.this_sample.data_filename

    if raw_data_path.suffix == ".npy":
        raw_data_np = np.load(raw_data_path)
    elif raw_data_path.suffix == ".txt":
        measurements_df = read_pcm_measurement_data_from_txt(raw_data_path)
        raw_data_np = measurements_df.to_numpy(dtype=np.float64)
    elif raw_data_path.suffix == ".map":
        raw_data_np = np.loadtxt(raw_data_path, dtype=np.float64)
    elif raw_data_path.suffix == ".png":
        raw_data_np = np.array(Image.open(raw_data_path).convert("L")) / 255.0
    else:
        raise ValueError(
            f"Unsupported format: {raw_data_path.suffix}. Path: {raw_data_path}"
        )

    j_order = config.data.j_order
    assert j_order is not None

    if config.data.data_type.startswith("image"):
        measurement_vector = None

    elif config.data.data_type == "hadamard_measurement_vector":
        measurement_vector = torch.tensor(raw_data_np, dtype=dtype, device=device)
        if ground_truth is None:
            with torch.no_grad():
                pcm_op_full = PhotocurrentMapOp(j_order=j_order, device=device)
                ground_truth = pcm_op_full.pseudo_inv(measurement_vector)
            print(
                "Reconstructed ground truth image shape from Hadamard measurement vector: "
                f"{ground_truth.shape}"
            )
            print(
                f"Min and max of ground truth image: {ground_truth.min().item()}, "
                f"{ground_truth.max().item()}"
            )

    elif config.data.data_type.startswith("original_measurement_data"):
        num_measurements = raw_data_np.shape[0]
        if num_measurements % 2 != 0:
            raise ValueError("Number of measurements should be even.")

        sign_vector = np.round(np.sign(raw_data_np[:, 0]))
        sign_vector[:2] = [1.0, -1.0]
        measurement_np = (raw_data_np[:, 1] * sign_vector).reshape(-1, 2).sum(axis=1)
        measurement_vector = torch.tensor(measurement_np, dtype=dtype, device=device)
        measurement_vector = config.scaling.data_sign * measurement_vector

        # Prepare the real measurement for the same image scaling and operator
        # convention used by reconstruction.
        #
        # If x_scaled = (x - r_low) / (r_high - r_low), then
        #
        #     A x_scaled = (A x - r_low A 1) / (r_high - r_low).
        #
        # The real differential measurement is in the unnormalised Hadamard
        # convention, so divide it by 2**J when using the normalised operator.
        if config.uses_normalized_op:
            measurement_vector = measurement_vector / (1 << j_order)

        scaling = config.scaling
        if scaling.r_low is not None and scaling.r_high is not None:
            pcm_op_full = PhotocurrentMapOp(
                j_order=j_order, normalized=config.uses_normalized_op, device=device
            )
            ones_response = pcm_op_full(
                torch.ones((1 << j_order, 1 << j_order), dtype=dtype, device=device)
            )
            measurement_vector = (
                measurement_vector - scaling.r_low * ones_response
            ) / max(scaling.r_high - scaling.r_low, scaling.scale_eps)

        if ground_truth is None:
            with torch.no_grad():
                pcm_op_full = PhotocurrentMapOp(j_order=j_order, device=device)
                ground_truth = pcm_op_full.pseudo_inv(measurement_vector)
    else:
        raise ValueError(f"Unknown data_type '{config.data.data_type}'.")

    assert isinstance(ground_truth, torch.Tensor), f"Got {type(ground_truth)}"
    return ground_truth, measurement_vector
