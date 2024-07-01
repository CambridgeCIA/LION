from typing import Optional
from LION.utils.parameter import LIONParameter
from LION.models.CNNs.MSDNets.MS_D2 import MSD_Net
from ts_algorithms import fdk
import torch
import LION.CTtools.ct_geometry as ct

class FBPMSD_Net(MSD_Net):
    def __init__(self, geometry_parameters: ct.Geometry, model_parameters: Optional[LIONParameter]=None):
        super().__init__(model_parameters)
        self.geo = geometry_parameters

        self._make_operator()
    
    def forward(self, sinogram):
        image = sinogram.new_zeros(sinogram.shape[0], 1, *self.geo.image_shape[1:])
        for i in range(sinogram.shape[0]):
            aux = fdk(self.op, sinogram[i, 0])
            aux = torch.clip(aux, min=0)
            image[i] = aux
        
        return super().forward(image)