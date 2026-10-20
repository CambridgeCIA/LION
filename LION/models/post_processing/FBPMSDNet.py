from __future__ import annotations

from LION.classical_algorithms.fdk import fdk
from LION.CTtools.ct_geometry import Geometry
from LION.models.CNNs.MSDNet import MSDNet
from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType


class FBPMSDNet(LIONmodel):
    def __init__(
        self, model_parameters: LIONModelParameter | None, geometry_parameters: Geometry
    ):
        assert (
            geometry_parameters is not None
        ), "Geometry parameters are required for FBPMSDNet"
        super().__init__(model_parameters, geometry_parameters)
        self.net = MSDNet(self.model_parameters)
        self._make_operator()

    @staticmethod
    def default_parameters():
        params = MSDNet.default_parameters()
        params.model_input_type = ModelInputType.SINOGRAM
        return params

    def forward(self, x):
        return self.net(fdk(x, self.op))
