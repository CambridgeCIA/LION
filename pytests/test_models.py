"""Models: construction with default parameters, save/load round trips, UNet,
ItNet and the gradient-step denoisers. All on CPU, without a CT operator."""

import numpy as np
import pytest
import torch

import LION.models.iterative_unrolled.ItNet as itnet_module
from LION.CTtools.ct_geometry import Geometry
from LION.models.CNNs.dncnn import DnCNN
from LION.models.CNNs.drunet import DRUNet
from LION.models.CNNs.MSDNet import MSDNet
from LION.models.CNNs.UNets.ResUNet3D import ResUNet3D
from LION.models.CNNs.UNets.Unet import UNet
from LION.models.iterative_unrolled.ItNet import ItNet
from LION.models.learned_regularizer.ACR import ACR
from LION.models.learned_regularizer.AR import AR
from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType
from LION.models.post_processing.FBPConvNet import FBPConvNet
from LION.models.post_processing.FBPConvNetImage import FBPConvNetImage
from LION.models.post_processing.FBPMSDNet import FBPMSDNet
from LION.utils.parameter import LIONParameter
from pytests.helpers import ImageMatrixOp, small_gsd, small_gsdrunet

# Saving a model without geometry warns about it; that is expected for DnCNN.
pytestmark = pytest.mark.filterwarnings("ignore:Expected 'geometry' parameter")

# Constructing some of the older models warns (deprecations etc.); not tested here.
ignore_warnings = pytest.mark.filterwarnings("ignore")


@pytest.fixture
def no_ct_operator(monkeypatch):
    """Building the CT operator needs tomosipo; it is not what is tested here."""
    monkeypatch.setattr(LIONmodel, "_make_operator", lambda self: None)


# --------------------------------------------------------------------------
# Construction with default parameters


def small_msd_params():
    params = MSDNet.default_parameters()
    params.depth = 4
    params.dilations = [1, 2, 3, 4]
    return params


@ignore_warnings
def test_msdnet_defaults():
    params = MSDNet.default_parameters()
    assert params.model_input_type is ModelInputType.IMAGE
    model = MSDNet(small_msd_params())
    assert model(torch.rand(2, 1, 16, 16)).shape == (2, 1, 16, 16)


@ignore_warnings
def test_msdnet_accepts_activation_module():
    params = small_msd_params()
    params.activation = torch.nn.LeakyReLU()
    MSDNet(params)


@ignore_warnings
def test_msdnet_rejects_unknown_activation():
    params = small_msd_params()
    params.activation = "NotAnActivation"
    with pytest.raises(ValueError):
        MSDNet(params)


@ignore_warnings
def test_fbpmsdnet_defaults(no_ct_operator):
    params = FBPMSDNet.default_parameters()
    assert params.model_input_type is ModelInputType.SINOGRAM
    params.depth, params.dilations = 2, [1, 2]
    model = FBPMSDNet(params, Geometry.default_parameters())
    assert isinstance(model.net, MSDNet)


@ignore_warnings
def test_resunet3d():
    params = ResUNet3D.default_parameters()
    model = ResUNet3D(params)
    out = model(torch.rand(2, 1, 16, 16, 16))
    assert out.shape == (2, 1, 16, 16, 16)
    assert (out >= 0).all() and (out <= 1).all()  # sigmoid output


@ignore_warnings
def test_acr_without_geometry():
    model = ACR()
    x = torch.rand(3, 1, 16, 16)
    assert model(x).shape == (3, 1)
    # the parametrized weights are positive after initialisation
    assert (model.ICNN_layer_0.blue.weight >= 0).all()
    assert model.ICNN_layer_0.blue.weight.abs().max() <= 1e-3


@ignore_warnings
def test_acr_is_convex_along_a_line():
    torch.manual_seed(0)
    model = ACR()
    a, b = torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 16)
    t = 0.3
    lhs = model(t * a + (1 - t) * b)
    rhs = t * model(a) + (1 - t) * model(b)
    assert lhs.item() <= rhs.item() + 1e-6


@ignore_warnings
def test_msd_pytorch_defaults():
    pytest.importorskip("msd_pytorch")
    from LION.models.CNNs.MSD_pytorch import MSD_pytorch

    assert MSD_pytorch.default_parameters().model_input_type is ModelInputType.IMAGE
    assert MSD_pytorch.default_parameters("segmentation").c_in == 1


# --------------------------------------------------------------------------
# Save/load round trips
#
# Regression tests for two bugs that together broke loading saved models:
# - load()/load_checkpoint() passed the loaded parameters to the model as a plain
#   LIONParameter (that is how nested parameters come back from JSON), which the
#   LIONModelParameter type check in LIONmodel.__init__ rejects.
# - LIONModelParameter.__init__ applied its defaults *after* the keyword
#   arguments, so values passed in (e.g. model_input_type) were overwritten.
# DnCNN is used because it needs no CT geometry.


def _save(model, path):
    model.save(path, dataset=LIONParameter(name="test"), training=LIONParameter())


def _json_normalised(params):
    # JSON has no tuple type, so tuples come back as lists.
    return {k: list(v) if isinstance(v, tuple) else v for k, v in vars(params).items()}


def test_model_parameter_defaults():
    params = LIONModelParameter()
    assert params.model_input_type is None
    assert params.normalisator is None


def test_model_parameter_kwargs_are_not_overwritten():
    params = LIONModelParameter(model_input_type=ModelInputType.IMAGE, depth=3)
    assert params.model_input_type == ModelInputType.IMAGE
    assert params.depth == 3


def test_model_parameter_file_round_trip(tmp_path):
    params = LIONModelParameter()
    params.model_input_type = ModelInputType.IMAGE
    params.save(tmp_path / "params.json")

    loaded = LIONModelParameter()
    loaded.load(tmp_path / "params.json")
    assert loaded.model_input_type == ModelInputType.IMAGE


def test_load_round_trip(tmp_path):
    torch.manual_seed(0)
    model = DnCNN()
    _save(model, tmp_path / "model")

    loaded, options, _ = DnCNN.load(tmp_path / "model", supress_warnings=True)

    assert isinstance(loaded, DnCNN)
    assert isinstance(loaded.model_parameters, LIONModelParameter)
    assert loaded.get_input_type() is ModelInputType.IMAGE
    assert _json_normalised(loaded.model_parameters) == _json_normalised(
        model.model_parameters
    )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key].cpu(), value.cpu())


def test_load_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = DnCNN()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
    model.save_checkpoint(
        tmp_path / "checkpoint",
        epoch=3,
        loss=0.5,
        optimizer=optimiser,
        training_param=LIONParameter(),
        dataset=LIONParameter(name="test"),
    )

    loaded, _, data = DnCNN.load_checkpoint(tmp_path / "checkpoint")

    assert isinstance(loaded.model_parameters, LIONModelParameter)
    assert loaded.get_input_type() is ModelInputType.IMAGE
    assert data["epoch"] == 3
    assert data["loss"] == 0.5
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key].cpu(), value.cpu())


@ignore_warnings
def test_save_without_optional_metadata(tmp_path):
    # dataset/training/geometry are optional (save() only warns)
    DnCNN().save(tmp_path / "model")
    loaded, _, _ = DnCNN.load(tmp_path / "model", supress_warnings=True)
    assert isinstance(loaded, DnCNN)


def test_save_does_not_change_model_parameters(tmp_path):
    params = DnCNN.default_parameters()
    params.extra = LIONParameter(depth=2)
    model = DnCNN(params)
    _save(model, tmp_path / "model")
    assert isinstance(model.model_parameters.extra, LIONParameter)
    assert model.model_parameters.extra.depth == 2


def small_geometry(n=16):
    geometry = Geometry.default_parameters()
    geometry.image_shape = [1, n, n]
    return geometry


def small_fbpmsdnet_params():
    params = FBPMSDNet.default_parameters()
    params.depth, params.dilations = 2, [1, 2]
    return params


@ignore_warnings
@pytest.mark.parametrize(
    "make_model",
    [
        lambda: AR(AR.default_parameters(), small_geometry()),
        lambda: ACR(small_geometry()),
        lambda: FBPMSDNet(small_fbpmsdnet_params(), small_geometry()),
    ],
    ids=["AR", "ACR", "FBPMSDNet"],
)
def test_models_with_geometry_load(monkeypatch, tmp_path, make_model):
    """load() passes the saved geometry as `geometry=`: every model must call
    its geometry argument that."""
    # a stand-in CT operator (identity on 16x16 images), so no tomosipo is needed
    op = ImageMatrixOp(torch.eye(16 * 16), (1, 16, 16))
    monkeypatch.setattr(
        LIONmodel, "_make_operator", lambda self: setattr(self, "op", op)
    )
    model = make_model()
    _save(model, tmp_path / "model")
    loaded, _, _ = type(model).load(tmp_path / "model", supress_warnings=True)
    assert list(loaded.geometry.image_shape) == [1, 16, 16]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key].cpu(), value.cpu())


@pytest.mark.parametrize("shape", [(512, 512), (362, 362), (100, 60)])
def test_ar_any_image_size(shape):
    geometry = Geometry.default_parameters()
    geometry.image_shape = [1, *shape]
    model = AR(AR.default_parameters(), geometry)
    assert model(torch.rand(2, 1, *shape)).shape == (2, 1)
    if shape == (512, 512):  # same layer as before for multiples of 16
        assert model.fc[0].in_features == 128 * 32 * 32


def test_fbpconvnet_image_takes_images():
    assert FBPConvNetImage.default_parameters().model_input_type is ModelInputType.IMAGE
    assert FBPConvNet.default_parameters().model_input_type is ModelInputType.SINOGRAM


def test_drunet_noise_level_with_several_channels():
    params = DRUNet.default_parameters()
    params.in_channels = params.out_channels = 3
    params.use_noise_level = True
    params.int_channels, params.n_blocks = 8, 1
    out = DRUNet(params)(torch.rand(2, 3, 16, 16), noise_level=0.1)
    assert out.shape == (2, 3, 16, 16)


@pytest.mark.parametrize("model_class", [DRUNet, DnCNN])
def test_activation_name_is_case_insensitive(model_class):
    params = model_class.default_parameters()
    params.act = "Leaky_ReLU"
    model_class(params)


# --------------------------------------------------------------------------
# LPD / cLPD with a matrix "CT" operator (tomosipo stubbed, CPU)

LPD_MODULES = [
    "LION.models.iterative_unrolled.LPD",
    "LION.models.iterative_unrolled.cLPD",
]


@pytest.fixture
def lpd_module(monkeypatch, request):
    """Import LPD or cLPD with tomosipo/ts_algorithms stubbed; the operator is
    a fixed matrix from 8x8 images to 6x5 "sinograms"."""
    import importlib
    import sys
    import types

    stubs = {
        "tomosipo": types.ModuleType("tomosipo"),
        "tomosipo.torch_support": types.ModuleType("tomosipo.torch_support"),
        "ts_algorithms": types.ModuleType("ts_algorithms"),
    }
    stubs["tomosipo.torch_support"].to_autograd = lambda op, **kw: op
    stubs["ts_algorithms"].fdk = None  # replaced in make_lpd
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    name = request.param
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    yield module
    sys.modules.pop(name, None)  # don't leave the stubbed import behind


def lpd_geometry():
    geometry = Geometry.default_parameters()
    geometry.image_shape = [1, 8, 8]
    geometry.angles = np.linspace(0, np.pi, 6, endpoint=False)
    geometry.detector_shape = [1, 5]
    return geometry


def make_lpd(module, monkeypatch, calls=None, **params):
    torch.manual_seed(0)
    M = torch.randn(30, 64) / 8

    def A(x):
        if calls is not None:
            calls.append(x.detach().clone())
        return (x.reshape(x.shape[0], -1) @ M.T).reshape(x.shape[0], 1, 6, 5)

    def AT(y):
        return (y.reshape(y.shape[0], -1) @ M).reshape(y.shape[0], 1, 8, 8)

    def make_operator(self):
        self.op, self.A, self.AT = None, A, AT

    monkeypatch.setattr(LIONmodel, "_make_operator", make_operator)
    monkeypatch.setattr(module, "fdk", lambda op, sino: torch.zeros(8, 8))
    model_class = module.LPD if hasattr(module, "LPD") else module.cLPD
    p = model_class.default_parameters()
    p.n_iters, p.step_size = 2, 0.5
    for k, v in params.items():
        setattr(p, k, v)
    return model_class(lpd_geometry(), p)


@ignore_warnings
@pytest.mark.parametrize("lpd_module", LPD_MODULES, indirect=True)
def test_lpd_channels_follow_the_parameters(lpd_module, monkeypatch):
    sino = torch.rand(2, 1, 6, 5)
    assert make_lpd(lpd_module, monkeypatch)(sino).shape == (2, 1, 8, 8)
    custom = make_lpd(
        lpd_module,
        monkeypatch,
        data_channels=[6, 16, 16, 4],  # 4 dual channels (+ A f and g)
        reg_channels=[4, 16, 16, 3],  # 3 primal channels (+ A^T h)
    )
    assert custom(sino).shape == (2, 1, 8, 8)
    with pytest.raises(ValueError, match="data_channels"):
        make_lpd(lpd_module, monkeypatch, data_channels=[7, 16, 16, 4])


@ignore_warnings
@pytest.mark.parametrize("lpd_module", LPD_MODULES, indirect=True)
@pytest.mark.parametrize("channel, saved_before", [(1, False), (0, True)])
def test_lpd_dual_step_projects_the_chosen_primal_channel(
    lpd_module, monkeypatch, channel, saved_before
):
    """Adler & Oktem project primal channel 1 in the dual step; parameter files
    from before the option existed keep LION's old channel 0."""
    calls = []
    model = make_lpd(lpd_module, monkeypatch, calls)
    if saved_before:
        del model.model_parameters.dual_step_primal_channel
    else:
        assert model.model_parameters.dual_step_primal_channel == channel

    class ChannelIndex(torch.nn.Module):  # primal update: channel k gets k + 1
        def forward(self, x):
            n = model.model_parameters.reg_channels[-1]
            k = torch.arange(1.0, n + 1).view(1, n, 1, 1)
            return k * torch.ones(x.shape[0], n, *x.shape[2:])

    setattr(model, "0_primal", ChannelIndex())
    model(torch.rand(1, 1, 6, 5))
    # FDK start is 0, so after one primal step channel c is c + 1 (+ the update)
    assert calls[1].mean().item() == pytest.approx(channel + 1)


@ignore_warnings
@pytest.mark.parametrize("lpd_module", LPD_MODULES, indirect=True)
def test_lpd_learned_steps_positive_and_old_checkpoints_load(lpd_module, monkeypatch):
    sino = torch.rand(1, 1, 6, 5)
    model = make_lpd(lpd_module, monkeypatch, learned_step=True, step_positive=True)
    assert not hasattr(model, "lambda_primal")
    with torch.no_grad():
        out_positive = model(sino)
        for step in model.lambda_dual:
            step.neg_()  # a negative learned step is used as its absolute value
        torch.testing.assert_close(model(sino), out_positive)

    # checkpoints from before lambda_primal was removed
    state = model.state_dict()
    state.update({f"lambda_primal.{i}": torch.ones(1) for i in range(2)})
    model.load_state_dict(state, strict=True)


# --------------------------------------------------------------------------
# UNet: conv-norm-relu twice; old single-norm checkpoints still load


def _norms(block):
    return [
        name for name, m in block.named_children() if isinstance(m, torch.nn.GroupNorm)
    ]


def small_unet_params(**overrides):
    params = UNet.default_parameters()
    params.baseDim = 4
    params.noGrp = 2
    for k, v in overrides.items():
        setattr(params, k, v)
    return params


def test_default_blocks_have_two_norms():
    model = UNet(small_unet_params())
    for block in (model.encoder1, model.base, model.decoder1):
        assert len(_norms(block)) == 2
    assert model(torch.rand(2, 1, 32, 32)).shape == (2, 1, 32, 32)


def test_parameters_without_the_option_keep_old_architecture():
    params = small_unet_params()
    del params.norm_after_second_conv  # as in parameter files saved before it existed
    model = UNet(params)
    assert _norms(model.encoder1) == ["enc1gn1"]


def test_old_state_dict_loads_strictly():
    old = small_unet_params(norm_after_second_conv=False)
    state = UNet(old).state_dict()
    assert not any("gn2" in k for k in state)
    UNet(old).load_state_dict(state, strict=True)


# --------------------------------------------------------------------------
# ItNet step sizes and differentiable data step


def itnet_params(step_size, learned_step):
    p = ItNet.default_parameters()
    p.n_iters = 3
    p.step_size = step_size
    p.learned_step = learned_step
    p.Unet_params.baseDim = 4
    p.Unet_params.noGrp = 2
    return p


@pytest.mark.parametrize("learned_step", [True, False])
@pytest.mark.parametrize(
    "step_size, expected",
    [(0.5, [0.5, 0.5, 0.5]), ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]), (None, [1, 1, 1])],
)
def test_step_size_per_iteration(no_ct_operator, step_size, expected, learned_step):
    model = ItNet(Geometry.default_parameters(), itnet_params(step_size, learned_step))
    steps = [float(model.step_size[i].detach()) for i in range(3)]
    np.testing.assert_allclose(steps, expected, rtol=1e-6)


def test_wrong_number_of_steps(no_ct_operator):
    with pytest.raises(ValueError):
        ItNet(Geometry.default_parameters(), itnet_params([0.1, 0.2], True))


def test_fixed_steps_move_with_the_model(no_ct_operator):
    model = ItNet(Geometry.default_parameters(), itnet_params(0.5, False))
    assert "step_size" in dict(model.named_buffers())
    model.to(torch.float64)
    assert model.step_size.dtype == torch.float64


def test_data_step_is_differentiable(no_ct_operator, monkeypatch):
    """The ItNet update must use the autograd operator and a differentiable FDK,
    so gradients reach the data-consistency term (checked with torch stand-ins:
    the CT operator is a fixed linear map, FDK its adjoint)."""
    torch.manual_seed(0)
    geometry = Geometry.default_parameters()
    model = ItNet(geometry, itnet_params(0.5, True))
    H, W = geometry.image_shape[1:]
    M = torch.randn(10, H * W) / H

    def A(x):  # (B, 1, H, W) -> (B, 1, 1, 10), differentiable
        return (x.reshape(x.shape[0], -1) @ M.T).reshape(x.shape[0], 1, 1, 10)

    calls = []

    def fake_fdk(sino, op, clip=True, differentiable=False):
        calls.append((clip, differentiable))
        return (sino.reshape(sino.shape[0], -1) @ M).reshape(sino.shape[0], 1, H, W)

    def not_differentiable(x):
        raise AssertionError("the data step must use self.A, not self.op")

    monkeypatch.setattr(itnet_module, "fdk", fake_fdk)
    model.A, model.op = A, not_differentiable
    model.Unet_0 = model.Unet_1 = model.Unet_2 = torch.nn.Identity()

    sino = torch.randn(2, 1, 1, 10, requires_grad=True)
    (grad,) = torch.autograd.grad(model(sino).sum(), sino)
    assert grad.abs().sum() > 0
    # initial FDK, then one differentiable, unclipped FDK per iteration
    assert calls == [(False, False)] + [(False, True)] * 3


# --------------------------------------------------------------------------
# Gradient-step denoisers need autograd in forward(); they must work when
# called under torch.no_grad() (validation, testing, PnP loops).


@pytest.mark.parametrize("make_model", [small_gsd, small_gsdrunet])
def test_same_output_with_and_without_no_grad(make_model):
    torch.manual_seed(0)
    model = make_model().eval()
    x = torch.rand(2, 1, 16, 16)
    expected = model(x)
    with torch.no_grad():
        out = model(x)
    torch.testing.assert_close(out, expected.detach())
    assert not out.requires_grad  # nothing leaks out of no_grad


@pytest.mark.parametrize("make_model", [small_gsd, small_gsdrunet])
def test_training_still_backpropagates_to_weights(make_model):
    model = make_model().train()
    loss = model(torch.rand(2, 1, 16, 16)).pow(2).mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
