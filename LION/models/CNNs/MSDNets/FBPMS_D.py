from typing import Optional
from LION.models.CNNs.MS_D import MS_D
from LION.models.LIONmodel import ModelInputType
from LION.utils.parameter import LIONParameter
from LION.models.CNNs.MSDNets.MS_D2 import MSD_Net, MSDParams
from LION.classical_algorithms.fdk import fdk
import torch
import LION.CTtools.ct_geometry as ct

class FBPMSD_Net(MSD_Net):
    def __init__(self, geometry_parameters: ct.Geometry, model_parameters: Optional[MSDParams]=None):
        super().__init__(model_parameters)
        self.model_parameters.model_input_type=ModelInputType.SINOGRAM # bad should probably make new params class
        self.geo = geometry_parameters
        self._make_operator()
    
    def forward(self, sinogram):    
        return super().forward(fdk(sinogram, self.op))
    
class OGFBPMSD_Net(MS_D):
    def __init__(self, geometry_parameters: ct.Geometry, model_parameters: Optional[LIONParameter]=None):
        super().__init__(model_parameters)
        self.model_parameters.model_input_type=ModelInputType.SINOGRAM # bad should probably make new params class
        self.geo = geometry_parameters
        self._make_operator()
    
    def forward(self, sinogram):    
        return super().forward(fdk(sinogram, self.op))