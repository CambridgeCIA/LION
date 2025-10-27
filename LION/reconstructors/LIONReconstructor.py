"""Base class for LION Reconstructors."""

from abc import ABC, ABCMeta, abstractmethod
from typing import Union

import torch

from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from LION.operators import Operator
from LION.models.LIONmodel import to_autograd


class LIONReconstructor(ABC):
    def __init__(self, physics: Union[Geometry, Operator]):
        """
        Base class for a Reconstructor in the LION framework.
        This assumes a trained model.

        Parameters
        ----------
        physics : Geometry or Operator
            The forward operator or information required to create a forward operator.
            If a Geometry is provided, the corresponding CT operator will be created.
        """
        __metaclass__ = ABCMeta

        if isinstance(physics, Operator):
            self.geometry = None
            self.op = physics
        elif isinstance(physics, Geometry):
            self.geometry = physics
            self.op = make_operator(self.geometry)
        else:
            raise ValueError(
                "Input operator is neither of class LION.operators.operator.Operator "
                "nor LION.CTtools.ct_geometry.Geometry"
            )
        self.op_autograd = to_autograd(self.op)

    def reconstruct(self, measurement: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Reconstruct the image using the model and operator.

        Parameters
        ----------
        measurement : torch.Tensor
            The measurement tensor (e.g., sinogram).
        noise_level : float, optional
            Some denoisers (e.g., DRUNet) require the user to estimate the noise level before denoising.
        """
        # call reconstruct_sample for each batch if measurement tensor is 4D
        if measurement.dim() == 4:
            recons = torch.zeros(
                measurement.size(0),
                *self.op.domain_shape,
                dtype=measurement.dtype,
                device=measurement.device
            )
            for i, s in enumerate(measurement):
                recons[i] = self.reconstruct_sample(s, **kwargs)
            return recons

        return self.reconstruct_sample(measurement, **kwargs)

    @abstractmethod
    def reconstruct_sample(self, sino, **kwargs):
        pass
