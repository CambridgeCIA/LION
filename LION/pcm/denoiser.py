from __future__ import annotations

from functools import partial
from typing import Callable

import deepinv
import torch
from tqdm import tqdm as std_tqdm

from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm.config import PnPConfig
from LION.pcm.types import GrayscaleImage2D, Measurement1D

ReconFn = Callable[
    [PhotocurrentMapOp, Measurement1D, GrayscaleImage2D], GrayscaleImage2D
]

tqdm = partial(std_tqdm, dynamic_ncols=True)


def build_denoiser(config: PnPConfig, device: torch.device) -> torch.nn.Module:
    """Build the configured deepinv denoiser.

    Parameters
    ----------
    config : PnPConfig
        PnP configuration.
    device : torch.device
        Device on which the denoiser should run.

    Returns
    -------
    torch.nn.Module
        Loaded denoiser in evaluation mode.
    """
    if config.denoiser_name == "drunet":
        with torch.device("cpu"):
            denoiser = deepinv.models.DRUNet(
                pretrained="download",
                in_channels=1,
                out_channels=1,
                device=device,
            )
    elif config.denoiser_name == "gs_drunet":
        with torch.device("cpu"):
            denoiser = deepinv.models.GSDRUNet(
                pretrained="download",
                in_channels=1,
                out_channels=1,
                device=device,
            )
    else:
        raise ValueError(f"Unknown denoiser_name '{config.denoiser_name}'.")

    denoiser = denoiser.to(device=device)
    denoiser.eval()
    return denoiser
