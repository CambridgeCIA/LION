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

from LION.operators.ct.ct_geometry import Geometry, check_geometry
from LION.operators.Operator import Operator
from LION.utils.common_primitives import DeviceLikeType
from LION.utils.tomosipo_import import import_tomosipo


class CTProjectionOp(Operator):
    """The CT projection operator of ``geometry`` (tomosipo) and its adjoint,
    the backprojection, on ``device`` (LION's if None). ``backend``: "tomosipo"
    (the only one). Its default reconstruction is FDK; ``autograd()`` gives
    tomosipo's autograd versions of it and its adjoint."""

    def __init__(
        self,
        geometry: Geometry,
        backend: Literal["tomosipo"] = "tomosipo",
        device: DeviceLikeType | None = None,
    ):
        super().__init__(device=device)
        if not isinstance(geometry, Geometry):
            raise ValueError(
                "Input geometry is not of class LION.operators.ct.ct_geometry.Geometry"
            )
        check_geometry(geometry)
        self.geometry = geometry
        self.backend = backend
        if backend == "tomosipo":
            # Wrap the tomosipo operator
            ts = import_tomosipo("tomosipo", "CTProjectionOp")

            if geometry.mode == "fan":
                vg = ts.volume(
                    shape=geometry.image_shape,
                    size=geometry.image_size,
                    pos=geometry.image_pos,
                )
                pg = ts.cone(
                    angles=geometry.angles,
                    shape=geometry.detector_shape,
                    size=geometry.detector_size,
                    src_orig_dist=geometry.dso,
                    src_det_dist=geometry.dsd,
                )
            elif geometry.mode == "parallel":
                vg = ts.volume(
                    shape=geometry.image_shape,
                    size=geometry.image_size,
                    pos=geometry.image_pos,
                )
                pg = ts.parallel(
                    angles=geometry.angles,
                    shape=geometry.detector_shape,
                    size=geometry.detector_size,
                )
            else:
                raise ValueError(
                    "Geometry mode not understood, has to be 'fan' or 'parallel'"
                )
            self._ts = ts.operator(vg, pg)
        else:
            raise ValueError(
                f"Backend {backend} not supported. "
                "Currently only 'tomosipo' is supported."
            )

    def __call__(self, x: torch.Tensor, out=None) -> torch.Tensor:
        """The projections (sinogram) of the image ``x``. ``out``: where to
        write them (for tomosipo's ``to_autograd``)."""
        return self._ts._fp(x, out=out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The projections (sinogram) of the image ``x``. Call the operator,
        ``op(x)``, rather than this method (as for torch modules)."""
        return self._ts._fp(x)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """The backprojection of the projections ``y``."""
        return self._ts._bp(y)

    def autograd(self):
        """tomosipo's autograd wrappers of the (tomosipo) operator and its
        adjoint, on batches."""
        ts = import_tomosipo("tomosipo.torch_support", "CTProjectionOp.autograd")
        return (
            ts.to_autograd(self._ts, num_extra_dims=1),
            ts.to_autograd(self._ts.T, num_extra_dims=1),
        )

    def default_reconstruction_fn(self):
        """``fdk``: FDK (fan or cone beam) or FBP (parallel beam), clipped at 0."""
        from LION.operators.ct.fdk import fdk

        return fdk

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying tomosipo operator."""
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
        return self._ts.domain_shape

    @property
    def range_shape(self) -> tuple[int, ...]:
        """Return the shape of the projection range."""
        return self._ts.range_shape
