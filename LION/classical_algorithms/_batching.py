"""Apply single-sample tomosipo/ts_algorithms reconstructions to batches."""

from __future__ import annotations

from typing import Callable

import torch

from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from LION.exceptions.exceptions import NoDataException
from LION.operators.Operator import Operator


def batched_reconstruction(
    recon_fn: Callable[[Operator, torch.Tensor, int], torch.Tensor],
    sino: torch.Tensor,
    op: Operator | Geometry,
) -> torch.Tensor:
    """Reconstruct every sample of ``sino`` with ``recon_fn(op, sino_i, i)``.

    ``sino`` is (B, C, angles, detectors) or a single (C, angles, detectors)
    sample, in which case the result has no batch axis either. ``op`` may be a
    Geometry, in which case the CT operator is built from it.
    """
    if sino.dim() == 4:
        remove_batch = False
    elif sino.dim() == 3:
        sino = sino.unsqueeze(0)
        remove_batch = True
    else:
        raise ValueError(
            "Expected a sinogram of shape (B, C, angles, detectors) or "
            f"(C, angles, detectors), got shape {tuple(sino.shape)}"
        )
    if sino.shape[0] == 0:
        raise NoDataException("Given 0 batches, no data to operate on!")
    if isinstance(op, Geometry):
        op = make_operator(op)

    # stacked rather than written into a preallocated tensor, so results that
    # carry an autograd graph (e.g. fdk(..., differentiable=True)) stay clean
    recon = torch.stack([recon_fn(op, sino[i], i) for i in range(sino.shape[0])])
    if remove_batch:
        recon = recon.squeeze(0)
    return recon


def per_sample(value: torch.Tensor | None, op: Operator, i: int):
    """Pick sample ``i`` of a batched per-sample tensor (e.g. an initial guess).

    Tensors with one more dimension than the operator domain are treated as
    batched; anything else is shared by all samples.
    """
    if isinstance(value, torch.Tensor) and value.dim() == len(op.domain_shape) + 1:
        return value[i]
    return value
