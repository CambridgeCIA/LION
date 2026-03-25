from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.config import DataConfig
from LION.pcm.types import GrayscaleImage2D, Measurement1D


def minmax_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor to the ``[0, 1]`` interval.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Min-max normalised tensor.

    Raises
    ------
    ValueError
        If the tensor is constant.
    """
    min_val = tensor.min()
    max_val = tensor.max()
    denom = max_val - min_val
    if torch.isclose(
        denom, torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
    ):
        raise ValueError("Cannot min-max normalize a constant tensor.")
    return (tensor - min_val) / denom


def assert_data_exists(config: DataConfig) -> Path:
    """Validate the configured input path and return it.

    Parameters
    ----------
    config : DataConfig
        Input data configuration.

    Returns
    -------
    Path
        Path to the configured ``.npy`` file.
    """
    if not config.data_dir.exists():
        raise FileNotFoundError(f"Data directory {config.data_dir} does not exist.")

    data_path = config.data_dir / config.data_filename
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data {config.data_filename} not found in {config.data_dir}."
        )
    return data_path


def prepare_data(
    config: DataConfig,
    device: torch.device,
) -> tuple[GrayscaleImage2D, Measurement1D | None]:
    """Load PCM data and prepare the ground truth image and measurement vector.

    Parameters
    ----------
    config : DataConfig
        Input data configuration.
    device : torch.device
        Torch device used for the returned tensors.

    Returns
    -------
    tuple[GrayscaleImage2D, Measurement1D | None]
        Ground truth image and optional full measurement vector.
    """
    data_path = assert_data_exists(config)
    print(f"Loading data file: {data_path.name}")
    print(f"The type of raw data is: {config.data_type}")

    raw_data: np.ndarray = np.load(data_path)
    print(f"Type of image data: {config.data_type}, dtype: {raw_data.dtype}")
    print(f"Raw data shape: {raw_data.shape}")
    print(f"J_order: {config.j_order}")

    if config.data_type == "image":
        ground_truth_image = torch.tensor(raw_data, dtype=torch.float32, device=device)
        if config.inverse_sign:
            ground_truth_image = -ground_truth_image

        j_data = int(np.log2(ground_truth_image.shape[0]))
        if j_data != config.j_order:
            raise ValueError(
                f"Data J ({j_data}) does not match expected J_order ({config.j_order})."
            )

        measurement_vector = None
        print(f"Ground truth image shape: {ground_truth_image.shape}")

    elif config.data_type == "hadamard_measurement_vector":
        j_data = int(np.log2(raw_data.shape[0]) / 2)
        if j_data != config.j_order:
            raise ValueError(
                f"Data J ({j_data}) does not match expected J_order ({config.j_order})."
            )

        measurement_vector = torch.tensor(raw_data, dtype=torch.float32, device=device)
        if config.inverse_sign:
            measurement_vector = -measurement_vector

        index_of_max = torch.argmax(measurement_vector).item()
        index_of_min = torch.argmin(measurement_vector).item()
        print(
            f"Max value in measurement vector: {measurement_vector[index_of_max].item()} at index {index_of_max}"
        )
        print(
            f"Min value in measurement vector: {measurement_vector[index_of_min].item()} at index {index_of_min}"
        )

        pcm_op_full = PhotocurrentMapOp(J=config.j_order, device=device)
        with torch.no_grad():
            ground_truth_image = pcm_op_full.pseudo_inv(measurement_vector)
        print(
            "Reconstructed ground truth image shape from Hadamard measurement vector: "
            f"{ground_truth_image.shape}"
        )

    elif config.data_type == "original_measurement_data":
        num_measurements = raw_data.shape[0]
        if num_measurements % 2 != 0:
            raise ValueError("Number of measurements should be even.")

        working_raw = raw_data.copy()
        if config.inverse_sign:
            working_raw[:, 1] = -working_raw[:, 1]

        index_of_max_raw = int(np.argmax(working_raw[:, 1]))
        index_of_min_raw = int(np.argmin(working_raw[:, 1]))
        min_raw_value = working_raw[index_of_min_raw, 1]
        max_raw_value = working_raw[index_of_max_raw, 1]
        print(
            f"Max value in original measurement data: {max_raw_value} at index {index_of_max_raw}"
        )
        print(
            f"Min value in original measurement data: {min_raw_value} at index {index_of_min_raw}"
        )

        sign_vector = np.round(np.sign(working_raw[:, 0]))
        sign_vector[:2] = [1.0, -1.0]

        measurement_vector = torch.tensor(
            (working_raw[:, 1] * sign_vector).reshape(-1, 2).sum(axis=1),
            dtype=torch.float32,
            device=device,
        )

        pcm_op_full = PhotocurrentMapOp(J=config.j_order, device=device)
        print(
            f"pcm_op_full domain shape: {pcm_op_full.domain_shape}, range shape: {pcm_op_full.range_shape}"
        )
        with torch.no_grad():
            ground_truth_image = pcm_op_full.pseudo_inv(measurement_vector)
        print(
            "Reconstructed ground truth image shape from original measurement data: "
            f"{ground_truth_image.shape}"
        )

    else:
        raise ValueError(f"Unknown data_type '{config.data_type}'.")

    if config.tests_scale_ground_truth:
        ground_truth_image = minmax_normalize(ground_truth_image)
        print(
            "Normalized ground truth image to [0, 1]. "
            f"Min: {ground_truth_image.min().item()}, Max: {ground_truth_image.max().item()}"
        )
        measurement_vector = None

    return ground_truth_image, measurement_vector
