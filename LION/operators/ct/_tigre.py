# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""The TIGRE backend of ``CTProjectionOp``: LION's geometry as TIGRE's, its
projection and backprojection in LION's layout, their autograd functions, and
the filtering of FDK.

LION's layout (tomosipo's): an image is ``(z, y, x)``, the same as TIGRE's; a
sinogram is ``(v, angles, u)``, TIGRE's is ``(angles, v, u)``. The angles and
the detector direction are converted so that the same LION ``Geometry`` gives
the same sinogram with either backend (``ANGLE_SIGN``, ``ANGLE_OFFSET``,
``FLIP_U``).

CUDA tensors stay on their GPU (TIGRE >= 3.3.0); tensors on the CPU go through
NumPy, on all GPUs.
"""

from __future__ import annotations

import numpy as np
import torch

from LION.utils.tigre_import import import_tigre

#: LION's angle ``a`` is TIGRE's ``ANGLE_SIGN * a + ANGLE_OFFSET``. At angle 0,
#: tomosipo (astra) has its source on -y and TIGRE on +x; both rotate
#: counter-clockwise around z.
ANGLE_SIGN = 1.0
ANGLE_OFFSET = -np.pi / 2
#: Whether TIGRE's detector columns (u) run the other way from tomosipo's.
FLIP_U = False


def tigre_geometry(geometry):
    """TIGRE's geometry and angles (radians) for a LION CT ``Geometry``.

    Fan beam is TIGRE's cone beam with one detector row; parallel beam is
    TIGRE's parallel beam (whose source distances only scale FBP's filter and
    are set equal)."""
    tigre_geometry_module = import_tigre("tigre.utilities.geometry", "CTProjectionOp")
    geo = tigre_geometry_module.Geometry()
    fan = geometry.mode == "fan"
    geo.mode = "cone" if fan else "parallel"
    geo.nVoxel = np.array(geometry.image_shape, dtype=np.int64)
    geo.sVoxel = np.array(geometry.image_size, dtype=float)
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.nDetector = np.array(geometry.detector_shape, dtype=np.int64)
    geo.sDetector = np.array(geometry.detector_size, dtype=float)
    geo.dDetector = geo.sDetector / geo.nDetector
    if fan:
        geo.DSO = float(geometry.dso)
        geo.DSD = float(geometry.dsd)
    else:
        geo.DSO = geo.DSD = 1.0
    geo.offOrigin = np.array(geometry.image_pos, dtype=float)
    geo.offDetector = np.array([0.0, 0.0])
    geo.rotDetector = np.array([0.0, 0.0, 0.0])
    geo.COR = 0.0
    geo.accuracy = 0.5
    angles = ANGLE_SIGN * np.asarray(geometry.angles, dtype=float) + ANGLE_OFFSET
    return geo, angles


def _apply_per_sample(fn, x: torch.Tensor, sample_dims: int) -> torch.Tensor:
    """``fn`` on each sample of ``x`` (its last ``sample_dims`` dimensions)."""
    if x.dim() == sample_dims:
        return fn(x)
    lead = x.shape[:-sample_dims]
    flat = x.reshape(-1, *x.shape[-sample_dims:])
    out = torch.stack([fn(sample) for sample in flat])
    return out.reshape(*lead, *out.shape[1:])


def _call(tigre_fn, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """``tigre_fn(data, *args)`` for a float32 tensor: on its GPU, or through
    NumPy for a tensor on the CPU."""
    data = data.detach().to(torch.float32).contiguous()
    if data.is_cuda:
        return tigre_fn(data, *args, **kwargs)
    return torch.from_numpy(tigre_fn(data.numpy(), *args, **kwargs))


def project(op, x: torch.Tensor) -> torch.Tensor:
    """The projections, ``(v, angles, u)``, of the image(s) ``x``,
    ``(..., z, y, x)``."""
    tigre = import_tigre("tigre", "CTProjectionOp")

    def one(image):
        sino = _call(tigre.Ax, image, op._tigre_geo, op._tigre_angles)
        sino = sino.permute(1, 0, 2)  # (angles, v, u) -> (v, angles, u)
        if FLIP_U:
            sino = sino.flip(-1)
        return sino.contiguous()

    return _apply_per_sample(one, x, 3)


def backproject(op, y: torch.Tensor, backprojection_type: str = "matched"):
    """The backprojection, ``(z, y, x)``, of the projections ``y``,
    ``(..., v, angles, u)``. ``backprojection_type``: TIGRE's weights,
    "matched" (the adjoint of the projection) or "FDK"."""
    tigre = import_tigre("tigre", "CTProjectionOp")

    def one(sino):
        if FLIP_U:
            sino = sino.flip(-1)
        sino = sino.permute(1, 0, 2).contiguous()  # (v, angles, u) -> (angles, v, u)
        return _call(
            tigre.Atb, sino, op._tigre_geo, op._tigre_angles, backprojection_type
        )

    return _apply_per_sample(one, y, 3)


class Project(torch.autograd.Function):
    """The projection, with the matched backprojection as its gradient."""

    @staticmethod
    def forward(ctx, x, op):
        """The projections of ``x`` (see ``project``)."""
        ctx.op = op
        return project(op, x)

    @staticmethod
    def backward(ctx, grad):
        """The matched backprojection of ``grad``."""
        return backproject(ctx.op, grad), None


class Backproject(torch.autograd.Function):
    """The backprojection (TIGRE's ``backprojection_type`` weights), with the
    projection as its gradient (exact for "matched")."""

    @staticmethod
    def forward(ctx, y, op, backprojection_type="matched"):
        """The backprojection of ``y`` (see ``backproject``)."""
        ctx.op = op
        return backproject(op, y, backprojection_type)

    @staticmethod
    def backward(ctx, grad):
        """The projections of ``grad``."""
        return project(ctx.op, grad), None, None


def _next_pow2_exponent(n: int) -> int:
    i = 1
    while 2**i <= n:
        i += 1
    return i


def ramp_filter(n_detectors: int) -> np.ndarray:
    """TIGRE's Ram-Lak filter in Fourier space (``tigre.utilities.filtering``),
    for rows of ``n_detectors`` padded to its length."""
    length = max(64, 2 ** _next_pow2_exponent(2 * n_detectors))
    nn = np.arange(-length / 2, length / 2)
    kernel = np.zeros(nn.shape, dtype=np.float32)
    kernel[int(length / 2)] = 1 / 4
    odd = nn % 2 == 1
    kernel[odd] = -1 / (np.pi * nn[odd]) ** 2
    filt = (np.abs(np.fft.fft(kernel)) * 2)[: length // 2 + 1]
    filt = np.hstack((filt, filt[1:-1][::-1]))
    return filt.astype(np.float32)


def filter_projections(y: torch.Tensor, n_angles: int, scale: float) -> torch.Tensor:
    """The rows of ``y`` (detector columns last) filtered with TIGRE's
    Ram-Lak filter (zero-padded), times ``scale * (2 pi / n_angles)``."""
    n_u = y.shape[-1]
    filt = torch.from_numpy(ramp_filter(n_u)).to(y.device)
    length = filt.shape[0]
    padding = (length - n_u) // 2
    padded = torch.nn.functional.pad(y, (padding, length - n_u - padding))
    filtered = torch.fft.ifft(torch.fft.fft(padded, dim=-1) * filt, dim=-1).real
    return filtered[..., padding : padding + n_u] * (scale * 2 * np.pi / n_angles)


def fdk_sample(op, y: torch.Tensor, differentiable: bool) -> torch.Tensor:
    """TIGRE's FDK (fan beam) or FBP (parallel beam) of one ``(v, angles, u)``
    sinogram, as ``tigre.algorithms.fdk`` / ``fbp`` without Parker weights:
    cosine weights (fan beam), TIGRE's Ram-Lak filter, then the
    backprojection, with FDK's weights for a fan beam. In torch, on the
    sinogram's device; ``differentiable``: through ``Backproject``."""
    geometry = op.geometry
    n_v, n_angles, n_u = y.shape
    dv, du = np.asarray(geometry.pixel_size, dtype=float)
    fan = geometry.mode == "fan"
    if fan:
        dsd, dso = float(geometry.dsd), float(geometry.dso)
        u = (torch.arange(n_u, device=y.device) - n_u / 2 + 0.5) * du
        v = (torch.arange(n_v, device=y.device) - n_v / 2 + 0.5) * dv
        weight = dsd / torch.sqrt(dsd**2 + v[:, None] ** 2 + u[None, :] ** 2)
        y = y * weight[:, None, :].to(y.dtype)
        scale = (dsd / dso) / (4 * du)
        backprojection_type = "FDK"
    else:
        scale = 1 / (4 * du)
        backprojection_type = "matched"
    y = filter_projections(y, n_angles, scale).to(torch.float32)
    if differentiable:
        return Backproject.apply(y, op, backprojection_type)
    return backproject(op, y, backprojection_type)
