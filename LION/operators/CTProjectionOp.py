"""Operator class for computed tomographic projection."""

from __future__ import annotations

from typing import Literal

import torch

from LION.CTtools.ct_geometry import Geometry
from LION.operators.Operator import Operator
from LION.utils.common_primitives import DeviceLikeType
from LION.utils.tomosipo_import import import_tomosipo


class CTProjectionOp(Operator):
    """Tomographic projection operator and its adjoint using tomosipo.

    Args:
        geometry: The geometry of the CT scan.
        backend: The backend to use for the projection operator.
            Currently only 'tomosipo' is supported.
        device: Device for computations. If None, use default device.
    """

    def __init__(
        self,
        geometry: Geometry,
        backend: Literal["tomosipo"] = "tomosipo",
        device: DeviceLikeType | None = None,
    ):
        """Initialize the CTProjectionOp."""
        super().__init__(device=device)
        if backend == "tomosipo":
            # Wrap the tomosipo operator
            ts = import_tomosipo("tomosipo", "CTProjectionOp")

            if not isinstance(geometry, Geometry):
                raise ValueError(
                    "Input geometry is not of class LION.CTtools.ct_geometry.Geometry"
                )
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
        """Apply the forward projection.

        Args:
            x: The input volume dataset to which the forward projection is applied.
            out: Optional output holder to store the output projections. Needed for ``to_autograd`` functionality.

        Returns:
            The projection dataset on which the volume has been forward projected.
        """
        return self._ts._fp(x, out=out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the forward projection.

        .. note::
            Prefer calling the instance of the operator as ``operator(x)`` over directly calling this method.
            See PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        return self._ts._fp(x)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the backprojection.

        Args:
            y: The input projection dataset to which the backprojection is applied.

        Returns:
            The volume dataset on which the projection dataset has been backprojected.
        """
        return self._ts._bp(y)

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
