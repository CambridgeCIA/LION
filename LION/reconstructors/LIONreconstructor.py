"""Base class for LION Reconstructors."""

from typing import Union

# Import CT utils
from LION.CTtools.ct_utils import make_operator
from abc import ABC, ABCMeta, abstractmethod
import torch

from LION.CTtools.ct_geometry import Geometry
from LION.operators.operator import Operator
from LION.models.LIONmodel import to_autograd


class LIONReconstructor(ABC):
    def __init__(self, operator: Union[Geometry, Operator]):
        """
        Base class for a Reconstructor in the LION framework.
        This assumes a trained model.

        Parameters
        ----------
        operator : Geometry or Operator
            The forward operator representing the imaging system.
            If a Geometry is provided, the corresponding CT operator will be created.
        """
        __metaclass__ = ABCMeta

        if isinstance(operator, Operator):
            self.geometry = None
            self.op = operator
        elif isinstance(operator, Geometry):
            self.geometry = operator
            self.op = make_operator(self.geometry)
        else:
            raise ValueError(
                "Input operator is neither of class LION.operators.operator.Operator nor LION.CTtools.ct_geometry.Geometry"
            )
        self.op_autograd = to_autograd(self.op)

    def reconstruct(self, sino, **kwargs):
        """
        Reconstruct the sinogram using the model and geometry.

        :param sino: Sinogram tensor.
        :param noise_level: Noise level for denoising.
        """
        # call reconstruct_sample for each batch if sino is 4D
        if sino.dim() == 4:
            recons = torch.zeros(
                sino.size(0),
                *self.op.domain.shape,
                dtype=sino.dtype,
                device=sino.device
            )
            for i, s in enumerate(sino):
                recons[i] = self.reconstruct_sample(s, **kwargs)
            return recons

        return self.reconstruct_sample(sino, **kwargs)

    @abstractmethod
    def reconstruct_sample(self, sino, **kwargs):
        pass
