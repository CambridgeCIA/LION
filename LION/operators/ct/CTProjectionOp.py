# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Thanh Trung (Troy) Vu, Ander Biguri
# =============================================================================
"""Operator class for computed tomographic projection."""

from __future__ import annotations

from typing import Literal

import torch

from LION.operators.ct import _tigre
from LION.operators.ct.backend import resolve_ct_backend
from LION.operators.ct.ct_geometry import Geometry, check_geometry
from LION.operators.Operator import Operator
from LION.utils.common_primitives import DeviceLikeType
from LION.utils.tomosipo_import import import_tomosipo


class CTProjectionOp(Operator):
    """The CT projection operator of ``geometry`` and its adjoint, the
    backprojection, on ``device`` (LION's if None).

    ``backend``: the library that computes them, "tomosipo" or "tigre"; None:
    the geometry's (``geometry.backend``, whose ``"default"`` is the one set
    with ``LION.operators.ct.set_ct_backend``). The backend used is written in
    the geometry, so records say which one made a result. Both give the same
    layout: images ``(1, H, W)``, sinograms ``(1, angles, detectors)``.

    Its default reconstruction is FDK; ``autograd()`` gives autograd versions
    of it and its adjoint."""

    def __init__(
        self,
        geometry: Geometry,
        backend: Literal["tomosipo", "tigre"] | None = None,
        device: DeviceLikeType | None = None,
    ):
        super().__init__(device=device)
        if not isinstance(geometry, Geometry):
            raise ValueError(
                "Input geometry is not of class LION.operators.ct.ct_geometry.Geometry"
            )
        check_geometry(geometry)
        backend = resolve_ct_backend(geometry.backend if backend is None else backend)
        geometry.backend = backend
        self.geometry = geometry
        self.backend = backend
        if backend == "tomosipo":
            self._ts = _tomosipo_operator(geometry)
        else:
            self._tigre_geo, self._tigre_angles = _tigre.tigre_geometry(geometry)

    def __call__(self, x: torch.Tensor, out=None) -> torch.Tensor:
        """The projections (sinogram) of the image ``x``. ``out``: where to
        write them (for tomosipo's ``to_autograd``)."""
        if self.backend == "tomosipo":
            return self._ts._fp(x, out=out)
        sino = _tigre.project(self, x)
        if out is None:
            return sino
        return out.copy_(sino)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The projections (sinogram) of the image ``x``. Call the operator,
        ``op(x)``, rather than this method (as for torch modules)."""
        if self.backend == "tomosipo":
            return self._ts._fp(x)
        return _tigre.project(self, x)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """The backprojection of the projections ``y``."""
        if self.backend == "tomosipo":
            return self._ts._bp(y)
        return _tigre.backproject(self, y)

    def autograd(self):
        """Autograd versions of the operator and its adjoint, on batches:
        tomosipo's ``to_autograd`` wrappers, or (TIGRE) functions whose
        gradients are the matched backprojection and the projection."""
        if self.backend == "tomosipo":
            ts = import_tomosipo("tomosipo.torch_support", "CTProjectionOp.autograd")
            return (
                ts.to_autograd(self._ts, num_extra_dims=1),
                ts.to_autograd(self._ts.T, num_extra_dims=1),
            )
        return (
            lambda x: _tigre.Project.apply(x, self),
            lambda y: _tigre.Backproject.apply(y, self),
        )

    def default_reconstruction_fn(self):
        """``fdk``: FDK (fan or cone beam) or FBP (parallel beam), clipped at 0."""
        from LION.operators.ct.fdk import fdk

        return fdk

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying tomosipo operator (with
        the tomosipo backend)."""
        # nn.Module first (registered parameters, buffers, submodules)
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # Look _ts up directly: if it is not set yet (during __init__, copy or
        # unpickling), self._ts would call __getattr__ again and recurse forever.
        ts_op = self.__dict__.get("_ts")
        if ts_op is None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(ts_op, name)

    @property
    def domain_shape(self) -> tuple[int, ...]:
        """Return the shape of the volume domain."""
        return tuple(int(n) for n in self.geometry.image_shape)

    @property
    def range_shape(self) -> tuple[int, ...]:
        """Return the shape of the projection range."""
        n_v, n_u = (int(n) for n in self.geometry.detector_shape)
        return (n_v, len(self.geometry.angles), n_u)


def _tomosipo_operator(geometry: Geometry):
    """The tomosipo operator of a LION CT ``geometry``."""
    ts = import_tomosipo("tomosipo", "CTProjectionOp")
    vg = ts.volume(
        shape=geometry.image_shape,
        size=geometry.image_size,
        pos=geometry.image_pos,
    )
    if geometry.mode == "fan":
        pg = ts.cone(
            angles=geometry.angles,
            shape=geometry.detector_shape,
            size=geometry.detector_size,
            src_orig_dist=geometry.dso,
            src_det_dist=geometry.dsd,
        )
    else:
        pg = ts.parallel(
            angles=geometry.angles,
            shape=geometry.detector_shape,
            size=geometry.detector_size,
        )
    return ts.operator(vg, pg)
