from __future__ import annotations

import numpy as np
import torch

from LION.classical_algorithms._batching import batched_reconstruction
from LION.CTtools.ct_geometry import Geometry
from LION.operators.Operator import Operator
from LION.utils.tomosipo_import import import_tomosipo


def fdk(
    sino: torch.Tensor,
    op: Operator | Geometry,
    clip: bool = True,
    differentiable: bool = False,
) -> torch.Tensor:
    """FDK (cone beam) or FBP (parallel beam) reconstruction.

    Args:
        sino: Sinograms, batched (B, C, angles, detectors) or a single
            (C, angles, detectors) sample.
        op: The CT operator, or a Geometry to build it from.
        clip: Set negative values of the reconstruction to zero.
        differentiable: If True, the reconstruction supports autograd with
            respect to ``sino`` (the backprojection goes through tomosipo's
            autograd wrapper), e.g. for FDK inside a network trained end to end.
            Slightly slower and uses more memory than the default.

    Returns:
        The reconstructions, with the same batch layout as ``sino``.
    """

    def recon_fn(op, sino_i, i):
        sub_recon = _fdk_sample(op, sino_i, differentiable)
        if clip:
            sub_recon = torch.clip(sub_recon, min=0)
        return sub_recon

    # ts_algorithms works on one sample at a time
    return batched_reconstruction(recon_fn, sino, op)


def _is_cone_beam(op) -> bool:
    geometry = import_tomosipo("tomosipo.geometry", "fdk")
    return isinstance(
        op.astra_compat_pg, (geometry.ConeGeometry, geometry.ConeVectorGeometry)
    )


def _fdk_sample(op, y: torch.Tensor, differentiable: bool) -> torch.Tensor:
    """FDK (cone beam) or FBP (parallel beam) of one (C, angles, detectors) sinogram."""
    ts_algorithms = import_tomosipo("ts_algorithms", "fdk")
    cone = _is_cone_beam(op)
    if not differentiable:
        return ts_algorithms.fdk(op, y) if cone else ts_algorithms.fbp(op, y)

    # Same steps as ts_algorithms.fdk / fbp, with an autograd-aware backprojection.
    # Weighting and filtering are torch operations; they are called with
    # overwrite_y=False so that no tensor in the autograd graph is overwritten.
    if cone:
        # the ts_algorithms.fdk submodule (the package exports a function of that name)
        fdk_module = import_tomosipo("ts_algorithms.fdk", "fdk")
        y = fdk_module.fdk_weigh_projections(op, y, overwrite_y=False)
    y = import_tomosipo("ts_algorithms.fbp", "fdk").filter_sino(y, overwrite_y=False)
    to_autograd = import_tomosipo("tomosipo.torch_support", "fdk").to_autograd
    rec = to_autograd(op.T)(y)
    # scaling from ts_algorithms.fbp, so that fdk(A(x)) ~ x
    vg, pg = op.astra_compat_vg, op.astra_compat_pg
    pixel_height = pg.det_size[0] / pg.det_shape[0]
    voxel_volume = np.prod(np.array(vg.size) / np.array(vg.shape))
    return rec * ((np.pi / pg.num_angles) * pixel_height / voxel_volume)
