from __future__ import annotations
import torch
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from ts_algorithms import fdk as ts_fdk
from ts_algorithms.fbp import ram_lak
import tomosipo as ts

from LION.exceptions.exceptions import NoDataException


def make_fdk_filter(
    width: int,
    *,
    filter_type: str | None = None,
    frequency_scaling: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    if filter_type in (None, "none"):
        return None

    filter_name = filter_type.lower()
    frequency_scaling = float(frequency_scaling)
    if frequency_scaling <= 0:
        raise ValueError("frequency_scaling must be positive.")

    base = ram_lak(int(width))
    spectrum = torch.fft.rfft(base)
    bins = torch.arange(spectrum.shape[-1], dtype=torch.float32)
    cutoff = min(frequency_scaling, 1.0) * float(spectrum.shape[-1] - 1)
    mask = bins <= cutoff
    window = torch.zeros_like(bins)

    if filter_name in ("ram-lak", "ram_lak", "ramp"):
        window[mask] = 1.0
    elif filter_name == "hann":
        window[mask] = 0.5 + 0.5 * torch.cos(torch.pi * bins[mask] / cutoff)
    elif filter_name == "hamming":
        window[mask] = 0.54 + 0.46 * torch.cos(torch.pi * bins[mask] / cutoff)
    elif filter_name == "cosine":
        window[mask] = torch.cos(torch.pi * bins[mask] / (2.0 * cutoff))
    elif filter_name in ("shepp-logan", "shepp_logan"):
        window[mask] = torch.sinc(bins[mask] / (2.0 * cutoff))
    else:
        raise ValueError(
            "filter_type must be one of none, ram-lak, hann, hamming, "
            "cosine, or shepp-logan."
        )

    filt = torch.fft.irfft(spectrum * window.to(spectrum.dtype), n=int(width)).float()
    if device is not None:
        filt = filt.to(device)
    return filt


def fdk(
    sino: torch.Tensor,
    op: ts.Operator.Operator | Geometry,
    clip=True,
    *,
    padded: bool = True,
    filter: torch.Tensor | None = None,
    filter_type: str | None = None,
    frequency_scaling: float = 1.0,
    batch_size: int = 10,
) -> torch.Tensor:
    if sino.dim() == 4:
        B, _, _, _ = sino.shape
        remove_batch = False
    elif sino.dim() == 3:
        B = 1
        sino = sino.unsqueeze(0)
        remove_batch = True
    if B == 0:
        raise NoDataException("Given 0 batches, no data to operate on!")
    if isinstance(op, Geometry):
        op = make_operator(op)
    recon = sino.new_zeros(B, *op.domain_shape)
    if filter is None and filter_type not in (None, "none"):
        filter_width = sino.shape[-1] * (2 if padded else 1)
        filter = make_fdk_filter(
            filter_width,
            filter_type=filter_type,
            frequency_scaling=frequency_scaling,
            device=sino.device,
        )
    # ts fdk doesn't support mini-batches so we apply it one at a time to each batch
    for i in range(B):
        sub_recon = ts_fdk(
            op,
            sino[i],
            padded=padded,
            filter=filter,
            batch_size=int(batch_size),
        )
        if clip:
            sub_recon = torch.clip(sub_recon, min=0)
        recon[i] = sub_recon
    if remove_batch:
        recon = recon.squeeze(0)
    return recon
