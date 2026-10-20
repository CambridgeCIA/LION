"""Models that could not be constructed (or run) with their default parameters."""

import pytest
import torch

from LION.CTtools.ct_geometry import Geometry
from LION.models.CNNs.MSDNet import MSDNet
from LION.models.CNNs.UNets.ResUNet3D import ResUNet3D
from LION.models.learned_regularizer.ACR import ACR
from LION.models.LIONmodel import LIONmodel, ModelInputType
from LION.models.post_processing.FBPMSDNet import FBPMSDNet

pytestmark = pytest.mark.filterwarnings("ignore")


def small_msd_params():
    params = MSDNet.default_parameters()
    params.depth = 4
    params.dilations = [1, 2, 3, 4]
    return params


def test_msdnet_defaults():
    params = MSDNet.default_parameters()
    assert params.model_input_type is ModelInputType.IMAGE
    model = MSDNet(small_msd_params())
    assert model(torch.rand(2, 1, 16, 16)).shape == (2, 1, 16, 16)


def test_msdnet_accepts_activation_module():
    params = small_msd_params()
    params.activation = torch.nn.LeakyReLU()
    MSDNet(params)


def test_msdnet_rejects_unknown_activation():
    params = small_msd_params()
    params.activation = "NotAnActivation"
    with pytest.raises(ValueError):
        MSDNet(params)


def test_fbpmsdnet_defaults(monkeypatch):
    # building the CT operator needs tomosipo; it is not what is tested here
    monkeypatch.setattr(LIONmodel, "_make_operator", lambda self: None)
    params = FBPMSDNet.default_parameters()
    assert params.model_input_type is ModelInputType.SINOGRAM
    params.depth, params.dilations = 2, [1, 2]
    model = FBPMSDNet(params, Geometry.default_parameters())
    assert isinstance(model.net, MSDNet)


def test_resunet3d():
    params = ResUNet3D.default_parameters()
    model = ResUNet3D(params)
    out = model(torch.rand(2, 1, 16, 16, 16))
    assert out.shape == (2, 1, 16, 16, 16)
    assert (out >= 0).all() and (out <= 1).all()  # sigmoid output


def test_acr_without_geometry():
    model = ACR()
    x = torch.rand(3, 1, 16, 16)
    assert model(x).shape == (3, 1)
    # the parametrized weights are positive after initialisation
    assert (model.ICNN_layer_0.blue.weight >= 0).all()
    assert model.ICNN_layer_0.blue.weight.abs().max() <= 1e-3


def test_acr_is_convex_along_a_line():
    torch.manual_seed(0)
    model = ACR()
    a, b = torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 16)
    t = 0.3
    lhs = model(t * a + (1 - t) * b)
    rhs = t * model(a) + (1 - t) * model(b)
    assert lhs.item() <= rhs.item() + 1e-6


def test_msd_pytorch_defaults():
    pytest.importorskip("msd_pytorch")
    from LION.models.CNNs.MSD_pytorch import MSD_pytorch

    assert MSD_pytorch.default_parameters().model_input_type is ModelInputType.IMAGE
    assert MSD_pytorch.default_parameters("segmentation").c_in == 1
