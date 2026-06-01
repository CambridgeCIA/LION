import torch
import torch.nn as nn
from LION.models.LIONmodel import LIONmodel, ModelInputType, ModelParams


class PaDISDenoiser(LIONmodel):
    def __init__(
        self,
        geometry_parameters: ct.Geometry,
        model_parameters: Optional[ACRParams] = None,
    ):
        super().__init__(model_parameters, geometry_parameters)
