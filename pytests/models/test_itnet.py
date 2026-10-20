"""ItNet step size handling (no CT operator needed)."""

import numpy as np
import pytest
import torch

from LION.CTtools.ct_geometry import Geometry
from LION.models.iterative_unrolled.ItNet import ItNet
from LION.models.LIONmodel import LIONmodel


@pytest.fixture(autouse=True)
def no_operator(monkeypatch):
    monkeypatch.setattr(LIONmodel, "_make_operator", lambda self: None)


def params(step_size, learned_step):
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
def test_step_size_per_iteration(step_size, expected, learned_step):
    model = ItNet(Geometry.default_parameters(), params(step_size, learned_step))
    steps = [float(model.step_size[i].detach()) for i in range(3)]
    np.testing.assert_allclose(steps, expected, rtol=1e-6)


def test_wrong_number_of_steps():
    with pytest.raises(ValueError):
        ItNet(Geometry.default_parameters(), params([0.1, 0.2], True))


def test_fixed_steps_move_with_the_model():
    model = ItNet(Geometry.default_parameters(), params(0.5, False))
    assert "step_size" in dict(model.named_buffers())
    model.to(torch.float64)
    assert model.step_size.dtype == torch.float64


def test_data_step_is_differentiable(monkeypatch):
    """The ItNet update must use the autograd operator and a differentiable FDK,
    so gradients reach the data-consistency term (checked with torch stand-ins:
    the CT operator is a fixed linear map, FDK its adjoint)."""
    import LION.models.iterative_unrolled.ItNet as itnet_module

    torch.manual_seed(0)
    geometry = Geometry.default_parameters()
    model = ItNet(geometry, params(0.5, True))
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
