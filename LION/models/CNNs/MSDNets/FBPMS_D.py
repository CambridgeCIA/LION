from typing import Optional
from LION.utils.parameter import LIONParameter
from LION.models.CNNs.MSDNets.MS_D2 import MSD_Net
from LION.classical_algorithms.fdk import fdk
import torch
import LION.CTtools.ct_geometry as ct

class FBPMSD_Net(MSD_Net):
    def __init__(self, geometry_parameters: ct.Geometry, model_parameters: Optional[LIONParameter]=None):
        super().__init__(model_parameters)
        self.geo = geometry_parameters
        self._make_operator()
    
    def forward(self, sinogram):    
        return super().forward(fdk(sinogram, self.op))