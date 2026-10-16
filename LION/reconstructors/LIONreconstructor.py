"""Base class for a Reconstructor in the LION framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic

import torch

from LION.operators.Operator import Operator


class LIONReconstructor(ABC):
    """Base class for a reconstructor in the LION framework."""

    def __init__(self, operator):
        """Initialize the reconstructor.
        This assumes a trained model.

        Parameters
        ----------
        operator : Geometry or Operator
            The forward operator representing the imaging system.
            If a Geometry is provided, the corresponding CT operator will be created.
        """
        if isinstance(operator, Operator):
            self.geometry = None
            self.op = operator
            self.op_autograd = None  # TODO: Add support for autograd-wrapped operators that are not CT operators
        else:
            # Import CT utils when needed to avoid import errors when
            #   not intending to use CT and CT dependencies are not installed
            from LION.CTtools.ct_geometry import Geometry
            from LION.models.LIONmodel import to_autograd

            if isinstance(operator, Geometry):
                from LION.CTtools.ct_utils import make_operator

                self.geometry = operator
                self.op = make_operator(self.geometry)
            else:
                raise ValueError(
                    "Input operator is neither of class LION.operators.operator.Operator nor LION.CTtools.ct_geometry.Geometry"
                )
            self.op_autograd = to_autograd(self.op)

    def reconstruct(self, sino: torch.Tensor, **kwargs) -> torch.Tensor:
        """Reconstruct the sinogram using the model and geometry.

        :param sino: Sinogram tensor.
        :param noise_level: Noise level for denoising.
        """
        # call reconstruct_sample for each batch if sino is 4D
        if sino.dim() == 4:
            recons = torch.zeros(
                sino.size(0),
                *self.op.domain_shape,
                dtype=sino.dtype,
                device=sino.device,
            )
            for i, s in enumerate(sino):
                recons[i] = self.reconstruct_sample(s, **kwargs)
            return recons

        return self.reconstruct_sample(sino, **kwargs)

    @abstractmethod
    def reconstruct_sample(self, measurement: torch.Tensor, **kwargs) -> torch.Tensor:
        """Reconstruct a single sample from the measurement."""
