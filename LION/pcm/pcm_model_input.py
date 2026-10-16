from __future__ import annotations

import torch

from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.pcm_config import SAPMPCMConfig
from LION.pcm.reconstruct import cg_reconstruction, spgl1_reconstruction
from LION.train.training_components import ModelInput


def make_pcm_image(
    image_type: str,
    *,
    config: SAPMPCMConfig,
    op: PhotocurrentMapOp,
    measurement: torch.Tensor,
    zero_filled: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    """Create one configured PCM image."""
    image = None
    if image_type == "zeros":
        image = torch.zeros_like(zero_filled)
    elif image_type == "zero_filled":
        image = zero_filled
    elif image_type == "adjoint":
        image = op.adjoint(measurement)
    elif image_type == "cg":
        assert config.model.cg is not None
        image = cg_reconstruction(
            op, measurement, config.model.cg, zero_filled=zero_filled
        )
    elif image_type == "spgl1":
        assert config.spgl1 is not None
        image = spgl1_reconstruction(
            op, measurement, config.spgl1, device=torch.device(device)
        )
    else:
        raise ValueError(f"Unknown PCM image type: {image_type}")
    return image


def make_pcm_model_input(
    *,
    config: SAPMPCMConfig,
    op: PhotocurrentMapOp,
    measurement: torch.Tensor,
    zero_filled: torch.Tensor,
    device: torch.device | str,
) -> ModelInput:
    """Create the configured PCM input and initial images."""
    input_image = make_pcm_image(
        config.model.input_image,
        config=config,
        op=op,
        measurement=measurement,
        zero_filled=zero_filled,
        device=device,
    )

    if config.model.initial_image is None:
        initial_image = None
    elif config.model.initial_image == config.model.input_image:
        initial_image = input_image
    else:
        initial_image = make_pcm_image(
            config.model.initial_image,
            config=config,
            op=op,
            measurement=measurement,
            zero_filled=zero_filled,
            device=device,
        )

    return ModelInput(input_image=input_image, initial_image=initial_image)
