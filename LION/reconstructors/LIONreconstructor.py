# Import CT utils
from LION.CTtools.ct_utils import make_operator
from abc import ABC, ABCMeta, abstractmethod
import torch

# Base class for a Reconstructor in the LION framework.
# This assumes a trained model


class LIONReconstructor(ABC):
    def __init__(self, geometry):
        __metaclass__ = ABCMeta

        self.geometry = geometry
        self.op = make_operator(self.geometry)

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
