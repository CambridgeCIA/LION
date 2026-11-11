# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Models: construction with default parameters, save/load round trips, UNet,
ItNet and the gradient-step denoisers. All on CPU, without a CT operator."""

import os
import warnings

import numpy as np
import importlib

import pytest
import torch

import LION.models.iterative_unrolled.ItNet as itnet_module
from LION.operators.ct.ct_geometry import Geometry
from LION.models.CNNs.dncnn import DnCNN
from LION.models.CNNs.drunet import DRUNet
from LION.models.iterative_unrolled.ItNet import ItNet
from LION.models.learned_regularizer.ACR import ACR
from LION.models.learned_regularizer.AR import AR
from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType
from LION.utils import spec
from LION.utils.parameter import LIONParameter
from LION.models.PnP.gradient_step_denoiser import GSD
from pytests.helpers import (
    CTStandIn,
    ImageMatrixOp,
    replace_fdk,
    small_gsd,
    small_gsd_dncnn,
)

# Saving a model without geometry warns about it; that is expected for DnCNN.

# Constructing some of the older models warns (deprecations etc.); not tested here.
ignore_warnings = pytest.mark.filterwarnings("ignore")


@pytest.fixture
def no_ct_operator(monkeypatch):
    """Building the CT operator needs tomosipo; it is not what is tested here."""
    monkeypatch.setattr(LIONmodel, "_make_operator", lambda self, autograd=False: None)


# --------------------------------------------------------------------------
# Construction with default parameters


@ignore_warnings
def test_acr_is_convex_along_a_line():
    torch.manual_seed(0)
    model = ACR()
    a, b = torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 16)
    t = 0.3
    lhs = model(t * a + (1 - t) * b)
    rhs = t * model(a) + (1 - t) * model(b)
    assert lhs.item() <= rhs.item() + 1e-6


# --------------------------------------------------------------------------
# Saved and loaded: the normalisation, nested parameters, checkpoints


def _save(model, path):
    model.save(path, dataset=LIONParameter(name="test"), training=LIONParameter())


def test_the_normalisation_is_saved_and_loaded(tmp_path):
    """A model reconstructs with the normalisation it was trained with, so its
    file keeps it (a "dataset" one by its range: the dataset isn't needed)."""
    for normalisation, expected in [
        (dict(normalisation_type="custom", data_range=(0.0, 10.0)), [0.0, 10.0]),
        (dict(normalisation_type="sample"), None),
        (
            dict(
                normalisation_type="dataset",
                dataset=[(None, torch.tensor([-1.0, 3.0])), (None, torch.tensor(2.0))],
            ),
            [-1.0, 3.0],
        ),
    ]:
        model = DnCNN()
        model.set_normalisation(**normalisation)
        _save(model, tmp_path / "model")
        loaded, _, _ = DnCNN.load(tmp_path / "model", supress_warnings=True)
        assert loaded.normalisator.record() == {
            "type": normalisation["normalisation_type"],
            "range": expected,
        }
        x = 4 * torch.rand(1, 1, 8, 8)
        torch.testing.assert_close(loaded.normalise(x), model.normalise(x))

    # It is in the model's parameters (so its spec), and made with it.
    model = DnCNN()
    model.set_normalisation("custom", data_range=(0.0, 4.0))
    assert vars(model.model_parameters.normalisation) == {
        "type": "custom",
        "range": [0.0, 4.0],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # not an "unknown parameter"
        made = LIONmodel.instantiate(spec.to_spec(model))
    assert made.normalisator.record() == {"type": "custom", "range": [0.0, 4.0]}
    # parameters without it (files saved before): no normalisation
    assert DnCNN(DnCNN.default_parameters()).normalisator is None


def test_gsd_takes_a_noise_level_if_its_denoiser_does():
    from pytests.helpers import small_gsd_noise_channel

    torch.manual_seed(0)
    model = small_gsd_noise_channel().eval()
    assert model.takes_noise_level()
    x = torch.rand(2, 1, 8, 8)
    low, high = model(x, noise_level=0.01), model(x, noise_level=0.5)
    assert low.shape == x.shape
    assert not torch.allclose(low, high)

    assert not small_gsd_dncnn().takes_noise_level()
    params = GSD.default_parameters()
    assert not GSD(params).takes_noise_level()
    params.backbone_params.use_noise_level = True
    params.backbone_params.in_channels = 1
    assert GSD(params).takes_noise_level()  # DRUNet's own option


class Middle(LIONmodel):
    """A model that builds another LIONmodel from parameters in its own."""

    def __init__(self, model_parameters=None):
        super().__init__(model_parameters)
        self.inner = DnCNN(self.model_parameters.inner_params)

    @staticmethod
    def default_parameters():
        params = LIONModelParameter(model_input_type=ModelInputType.IMAGE)
        params.inner_params = DnCNN.default_parameters()
        params.inner_params.blocks = 3
        params.inner_params.int_channels = 4
        return params

    def forward(self, x):
        return self.inner(x)


class Outer(LIONmodel):
    """Two levels of nesting: Outer -> Middle -> DnCNN."""

    def __init__(self, model_parameters=None):
        super().__init__(model_parameters)
        self.middle = Middle(self.model_parameters.middle_params)
        self.scale = torch.nn.Parameter(torch.tensor(2.0))

    @staticmethod
    def default_parameters():
        params = LIONModelParameter(model_input_type=ModelInputType.IMAGE)
        params.middle_params = Middle.default_parameters()
        return params

    def forward(self, x):
        return self.scale * self.middle(x)


def test_models_inside_models_load(tmp_path):
    """Any LIONmodel that builds other LIONmodels from nested parameters, at
    any depth, loads: each model converts its own parameters."""
    for cls in [Middle, Outer]:
        torch.manual_seed(0)
        model = cls().eval()
        _save(model, tmp_path / "model")
        loaded, _, _ = cls.load(tmp_path / "model", supress_warnings=True)
        loaded = loaded.cpu().eval()
        x = torch.rand(1, 1, 16, 16)
        torch.testing.assert_close(loaded(x), model(x))
        inner = loaded.inner if cls is Middle else loaded.middle.inner
        assert isinstance(inner.model_parameters, LIONModelParameter)
        assert inner.get_input_type() is ModelInputType.IMAGE

    # GSD keeps its backbone's parameters inside its own: loaded from JSON they
    # are plain LIONParameters, which the backbone must accept.
    from LION.models.PnP.gradient_step_denoiser import GSD

    torch.manual_seed(0)
    model = small_gsd().eval()
    _save(model, tmp_path / "model")
    loaded, _, _ = GSD.load(tmp_path / "model", supress_warnings=True)
    loaded = loaded.cpu().eval()  # load() puts the model on the default device
    backbone = loaded._backbone_model.model_parameters
    assert isinstance(backbone, LIONModelParameter)
    assert backbone.model_input_type is ModelInputType.IMAGE
    x = torch.rand(1, 1, 16, 16)
    torch.testing.assert_close(loaded(x), model(x))

    from LION.models.PnP.gradient_step_denoiser import GSD

    torch.manual_seed(0)
    model = small_gsd_dncnn().eval()
    assert model.model_parameters.backbone_params is None
    _save(model, tmp_path / "model")
    loaded, _, _ = GSD.load(tmp_path / "model", supress_warnings=True)
    loaded = loaded.cpu().eval()
    assert type(loaded._backbone_model).__name__ == "DnCNN"
    x = torch.rand(1, 1, 16, 16)
    torch.testing.assert_close(loaded(x), model(x))


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


def small_geometry(n=16):
    operator_params = Geometry.default_parameters()
    operator_params.image_shape = [1, n, n]
    return operator_params


@ignore_warnings
@pytest.mark.parametrize(
    "make_model",
    [
        lambda: AR(small_geometry(), AR.default_parameters()),
        lambda: ACR(small_geometry()),
    ],
    ids=["AR", "ACR"],
)
def test_models_with_operator_params_load(monkeypatch, tmp_path, make_model):
    """load() passes the saved operator parameters as `operator_params=`:
    every model must call its argument that."""
    # a stand-in CT operator (identity on 16x16 images), so no tomosipo is needed
    op = ImageMatrixOp(torch.eye(16 * 16), (1, 16, 16))
    monkeypatch.setattr(
        LIONmodel,
        "_make_operator",
        lambda self, autograd=False: setattr(self, "op", op),
    )
    model = make_model()
    _save(model, tmp_path / "model")
    loaded, _, _ = type(model).load(tmp_path / "model", supress_warnings=True)
    assert list(loaded.operator_params.image_shape) == [1, 16, 16]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key].cpu(), value.cpu())


@ignore_warnings
def test_files_with_the_old_name_of_the_operator_params_load(monkeypatch, tmp_path):
    """Files saved before November 2026 call operator_params `geometry`, in
    the model's spec and at the top of the file: they load."""
    import json

    op = ImageMatrixOp(torch.eye(16 * 16), (1, 16, 16))
    monkeypatch.setattr(
        LIONmodel,
        "_make_operator",
        lambda self, autograd=False: setattr(self, "op", op),
    )
    model = AR(small_geometry(), AR.default_parameters())
    _save(model, tmp_path / "model")
    path = tmp_path / "model.json"
    saved = json.loads(path.read_text())
    saved["model"]["geometry"] = saved["model"].pop("operator_params")
    saved["geometry"] = saved["model"]["geometry"]
    path.write_text(json.dumps(saved))
    loaded, options, _ = AR.load(tmp_path / "model", supress_warnings=True)
    assert list(loaded.operator_params.image_shape) == [1, 16, 16]
    assert list(options.operator_params.image_shape) == [1, 16, 16]

    from LION.utils.parameter import with_current_names

    old = {
        "_target_": "LION.data_loaders.LIDC_IDRI.LIDC_IDRI",
        "geometry_parameters": {"mode": "fan"},
        "parameters": {"geometry": {"mode": "fan"}, "task": "reconstruction"},
        "datasets": [{"geometry_params": None}],
    }
    assert with_current_names(old) == {
        "_target_": "LION.data_loaders.LIDC_IDRI.LIDC_IDRI",
        "operator_params": {"mode": "fan"},
        "parameters": {"operator_params": {"mode": "fan"}, "task": "reconstruction"},
        "datasets": [{"operator_params": None}],
    }
    current = {"operator_params": 1, "geometry": 2}  # a current name wins
    assert with_current_names(current) == current


def test_drunet_takes_one_noise_level_per_image():
    params = DRUNet.default_parameters()
    params.use_noise_level = True
    params.int_channels = 4
    model = DRUNet(params)
    x = torch.rand(2, 1, 16, 16)
    torch.manual_seed(0)
    both = model(x, noise_level=torch.tensor([0.01, 0.5]))
    torch.testing.assert_close(both[:1], model(x[:1], noise_level=0.01))
    torch.testing.assert_close(both[1:], model(x[1:], noise_level=0.5))
    with pytest.raises(ValueError, match="one per image"):
        model(x, noise_level=torch.tensor([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError, match="non-negative"):
        model(x, noise_level=-0.1)


@pytest.mark.skipif("CI" in os.environ, reason="downloads 130 MB of weights")
def test_pretrained_drunet_denoises():
    # Downloads the authors' weights (130 MB) once; skipped without a network
    # and in CI (GitHub sets CI)
    try:
        model = DRUNet.pretrained(verbose=0).cpu().eval()
    except OSError as error:  # URLError is an OSError
        pytest.skip(f"cannot download the weights: {error}")
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij"
    )
    clean = ((xx**2 + yy**2 < 0.6).float() * 0.6 + 0.1)[None, None]
    torch.manual_seed(0)
    noisy = clean + 25 / 255 * torch.randn_like(clean)
    with torch.no_grad():
        denoised = model(noisy, noise_level=25 / 255)
    # 20 dB noisy; the authors' weights give ~38 dB, random ones far less
    assert ((denoised - clean) ** 2).mean() < ((noisy - clean) ** 2).mean() / 30


# --------------------------------------------------------------------------
# cFBPConvNet (FBPConvNet with neural ODE blocks)


def small_cfbpconvnet_params(**overrides):
    from LION.models.post_processing.cFBPConvNet import cFBPConvNet

    params = cFBPConvNet.default_parameters()
    c = 4
    for name, mult in [("down_1", 1), ("down_2", 2), ("down_3", 4), ("down_4", 8)]:
        n = len(getattr(params, f"{name}_channels"))
        setattr(params, f"{name}_channels", [c * mult] * n)
    for name, mult in [("up_1", 8), ("up_2", 4), ("up_3", 2), ("up_4", 1)]:
        n = len(getattr(params, f"{name}_channels"))
        setattr(params, f"{name}_channels", [c * mult] * n)
    params.latent_channels = [c * 16] * 3
    params.last_block = [c, 1, 1]
    for k, v in overrides.items():
        setattr(params, k, v)
    return params


@ignore_warnings
def test_cfbpconvnet_runs_on_images(tmp_path):
    from LION.models.post_processing.cFBPConvNet import cFBPConvNet

    defaults = cFBPConvNet.default_parameters()
    assert defaults.model_input_type is ModelInputType.IMAGE
    assert (defaults.ode_adjoint, defaults.do_second_order) == (False, False)

    torch.manual_seed(0)
    model = cFBPConvNet(model_parameters=small_cfbpconvnet_params())
    x = torch.rand(2, 1, 32, 32, requires_grad=True)
    out = model(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    model.save(tmp_path / "model", dataset=LIONParameter(), training=LIONParameter())
    loaded, _, _ = cFBPConvNet.load(tmp_path / "model", supress_warnings=True)
    loaded = loaded.cpu()  # load() puts the model on the default device (a GPU, if any)
    loaded.eval(), model.eval()
    torch.testing.assert_close(loaded(x.detach()), model(x.detach()))


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


def test_data_step_is_differentiable(no_ct_operator, monkeypatch):
    """The ItNet update must use the autograd operator and a differentiable FDK,
    so gradients reach the data-consistency term (checked with torch stand-ins:
    the CT operator is a fixed linear map, FDK its adjoint)."""
    torch.manual_seed(0)
    operator_params = Geometry.default_parameters()
    model = ItNet(operator_params, itnet_params(0.5, True))
    H, W = operator_params.image_shape[1:]
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
# LPD / cLPD with a matrix "CT" operator (tomosipo stubbed, CPU)

LPD_MODULES = [
    "LION.models.iterative_unrolled.LPD",
    "LION.models.iterative_unrolled.cLPD",
]


@pytest.fixture
def lpd_module(import_without_tomosipo, request):
    """LPD or cLPD, imported with tomosipo/ts_algorithms stubbed (see conftest)."""
    return import_without_tomosipo(request.param)


def lpd_geometry():
    operator_params = Geometry.default_parameters()
    operator_params.image_shape = [1, 8, 8]
    operator_params.angles = np.linspace(0, np.pi, 6, endpoint=False)
    operator_params.detector_shape = [1, 5]
    return operator_params


def make_lpd(module, monkeypatch, calls=None, **params):
    torch.manual_seed(0)
    M = torch.randn(30, 64) / 8

    def A(x):
        if calls is not None:
            calls.append(x.detach().clone())
        return (x.reshape(x.shape[0], -1) @ M.T).reshape(x.shape[0], 1, 6, 5)

    def AT(y):
        return (y.reshape(y.shape[0], -1) @ M).reshape(y.shape[0], 1, 8, 8)

    def make_operator(self, autograd=False):
        self.op, self.A, self.AT = CTStandIn((1, 8, 8), (1, 6, 5)), A, AT

    monkeypatch.setattr(LIONmodel, "_make_operator", make_operator)
    # the CT operator's default reconstruction (FDK): a stand-in
    replace_fdk(monkeypatch, lambda sino, op: torch.zeros(1, 8, 8))
    model_class = module.LPD if hasattr(module, "LPD") else module.cLPD
    p = model_class.default_parameters()
    p.n_iters, p.step_size = 2, 0.5
    for k, v in params.items():
        setattr(p, k, v)
    return model_class(lpd_geometry(), p)


def constant_reconstruction(sino, op, value=0.0):
    """An initial reconstruction for tests: a constant 8x8 image per sample."""
    return torch.full((sino.shape[0], 1, 8, 8), value)


@ignore_warnings
@pytest.mark.parametrize("lpd_module", LPD_MODULES, indirect=True)
def test_lpd_starts_from_its_initial_reconstruction(lpd_module, monkeypatch):
    import functools

    sino = torch.rand(2, 1, 6, 5)
    model = make_lpd(lpd_module, monkeypatch)  # FDK stand-in: zeros
    default = model(sino)
    model.set_initial_reconstruction(
        functools.partial(constant_reconstruction, value=0.0)
    )
    torch.testing.assert_close(model(sino), default)
    model.set_initial_reconstruction(
        functools.partial(constant_reconstruction, value=1.0)
    )
    assert not torch.allclose(model(sino), default)

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


# --------------------------------------------------------------------------
# Gradient-step denoisers need autograd in forward(); they must work when
# called under torch.no_grad() (validation, testing, PnP loops).


@pytest.mark.parametrize("make_model", [small_gsd, small_gsd_dncnn])
def test_same_output_with_and_without_no_grad(make_model):
    torch.manual_seed(0)
    model = make_model().eval()
    x = torch.rand(2, 1, 16, 16)
    expected = model(x)
    with torch.no_grad():
        out = model(x)
    torch.testing.assert_close(out, expected.detach())
    assert not out.requires_grad  # nothing leaks out of no_grad

    model = make_model().train()
    loss = model(torch.rand(2, 1, 16, 16)).pow(2).mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------
# Learned FBPs (DeepFBP, FusionFBP), with a matrix "CT" operator on CPU


def test_models_differentiating_through_an_operator_need_its_autograd(monkeypatch):
    """An operator without autograd() can't be used by a model whose gradients
    go through it (it says so when the model is made); a model that only uses
    the operator otherwise (e.g. AR, for its shape) works with it."""
    from LION.models.iterative_unrolled.LPD import LPD
    from LION.models.learned_regularizer.AR import AR
    from LION.operators import Operator
    from pytests.helpers import ImageMatrixOp

    class NoAutograd(ImageMatrixOp):
        autograd = Operator.autograd

    op = NoAutograd(torch.eye(64), (1, 8, 8), range_shape=(1, 8, 8))
    with pytest.raises(NotImplementedError, match="has no autograd"):
        op.autograd()
    monkeypatch.setattr(Geometry, "make_operator", lambda self: op)
    with pytest.raises(NotImplementedError, match="LPD's gradients go through"):
        LPD(Geometry.default_parameters())
    model = AR(Geometry.default_parameters(), AR.default_parameters())
    assert model.op is op and not hasattr(model, "A")


def _model_classes():
    import inspect
    import pkgutil

    import LION.models

    for info in pkgutil.walk_packages(LION.models.__path__, "LION.models."):
        module = importlib.import_module(info.name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(cls, LIONmodel)
                and cls is not LIONmodel
                and cls.__module__ == info.name
            ):
                yield cls


@pytest.mark.parametrize("cls", sorted(set(_model_classes()), key=str), ids=str)
def test_model_arguments_operator_first(cls):
    """A model that uses an operator takes (operator_params, model_parameters
    =None); in every model, model_parameters is optional."""
    import inspect

    arguments = list(inspect.signature(cls.__init__).parameters.values())[1:]
    names = [a.name for a in arguments]
    assert arguments[names.index("model_parameters")].default is None
    if "operator_params" in names:
        assert names[:2] == ["operator_params", "model_parameters"]


def test_msdnet_gradients_are_those_of_the_network():
    """MSDNet computes its gradients layer by layer (to keep memory linear in
    depth): they are the gradients of the plain network, and its padding is
    the symmetric reflection msd_pytorch uses."""
    import torch.nn.functional as F

    from LION.models.CNNs.MSDNet import MSDNet, _conv, _symmetric_pad

    x = torch.randn(1, 1, 5, 7)
    padded = np.pad(x.numpy(), ((0, 0), (0, 0), (3, 3), (3, 3)), mode="symmetric")
    assert np.allclose(_symmetric_pad(x, 3).numpy(), padded)

    p = MSDNet.default_parameters()
    p.depth, p.width, p.in_channels, p.dilations = 5, 2, 2, [1, 2, 3]
    torch.manual_seed(0)
    model = MSDNet(p).double()
    with torch.no_grad():  # nonzero, so every gradient is checked
        for parameter in model.parameters():
            parameter.normal_()

    def plain(x):
        layers = [x]
        for i, (weight, dilation) in enumerate(zip(model.weights, model.dilations)):
            bias = model.bias[i * 2 : (i + 1) * 2]
            layers.append(F.relu(_conv(torch.cat(layers, 1), weight, bias, dilation)))
        return model.final_layer(torch.cat(layers, 1))

    x = torch.randn(2, 2, 9, 11, dtype=torch.double, requires_grad=True)
    output, expected = model(x), plain(x)
    assert torch.allclose(output, expected)
    g = torch.randn_like(output)
    inputs = [x, model.bias, *model.weights]
    for a, b in zip(
        torch.autograd.grad(output, inputs, g), torch.autograd.grad(expected, inputs, g)
    ):
        assert torch.allclose(a, b)
