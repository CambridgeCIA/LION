from LION.CTtools.ct_geometry import Geometry
from LION.classical_algorithms.fdk import fdk
from LION.models.CNNs.MSDNet import MSD_Params, MSDNet
from LION.models.LIONmodel import LIONmodel


class FBPMSDNet(LIONmodel):
    def __init__(
        self, model_parameters: MSD_Params | None, geometry_parameters: Geometry
    ):
        assert (
            geometry_parameters is not None
        ), "Geometry parameters are required for FBPMSDNet"
        super().__init__(model_parameters, geometry_parameters)
        self.net = MSDNet(model_parameters)

    def forward(self, x):
        return self.net(fdk(x, self.op))
