from __future__ import annotations

from LION.classical_algorithms.fdk import fdk
from LION.CTtools.ct_geometry import Geometry
from LION.models.CNNs.MSDNet import MSDNet
from LION.models.LIONmodel import LIONmodel


class FBPMSDNet(LIONmodel):
    def __init__(
        self, model_parameters: LIONParameter | None, geometry_parameters: Geometry
    ):
        assert (
            geometry_parameters is not None
        ), "Geometry parameters are required for FBPMSDNet"
        super().__init__(model_parameters, geometry_parameters)
        self.net = MSDNet(model_parameters)

    @staticmethod
    def default_parameters(self):
        params = MSDNet.default_parameters()
        params.model_input_type = ModelInputType.SINOGRAM
        return params

    def forward(self, x):
        return self.net(fdk(x, self.op))
