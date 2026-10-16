import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import torch
from LION.classical_algorithms.conjugate_gradient import CGProxParameter
from LION.experiments.config.other_config import NormParameter
from LION.operators.Operator import Operator
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm_tools.pcm_prox import closed_form_pcm_prox
from LION.reconstructors.GSPnP import GSDenoiser, GSPnP, PCMDeepInvPhysics


@pytest.fixture(
    autouse=True,
    params=[
        pytest.param("cpu", id="cpu"),
        pytest.param("default", marks=pytest.mark.cuda, id="default"),
    ],
)
def device_mode(request):
    if request.param == "cpu":
        with torch.device("cpu"):
            yield
    else:
        yield


class ScaleNetwork(torch.nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor, sigma=None) -> torch.Tensor:
        return self.scale * x


class BatchChannelGSDenoiser(GSDenoiser):
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1)

    def postprocess(self, nabla_g: torch.Tensor) -> torch.Tensor:
        return nabla_g.squeeze(1)


class IdentityOp(Operator):
    def __call__(self, x: torch.Tensor, out=None) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return y

    @property
    def domain_shape(self) -> tuple[int, ...]:
        return (4, 5)

    @property
    def range_shape(self) -> tuple[int, ...]:
        return (4, 5)


def test_gs_denoiser_grad_matches_analytic_gradient():
    torch.manual_seed(0)

    x = torch.randn(3, 4, 5, dtype=torch.float64)
    denoiser = BatchChannelGSDenoiser(ScaleNetwork(0.5))

    gradient, _ = denoiser.grad(x, sigma=0.05, set_reset_grad=True)

    expected = 0.25 * x

    torch.testing.assert_close(gradient, expected)


def test_gs_denoiser_forward_matches_gradient_step():
    torch.manual_seed(1)

    x = torch.randn(3, 4, 5, dtype=torch.float64)
    denoiser = BatchChannelGSDenoiser(ScaleNetwork(0.5))

    output = denoiser(x, sigma=0.05, set_reset_grad=True)

    expected = 0.75 * x

    torch.testing.assert_close(output, expected)


def test_gs_denoiser_preserves_batch_image_shape():
    x = torch.randn(3, 4, 5)
    denoiser = BatchChannelGSDenoiser(ScaleNetwork(0.5))

    output = denoiser(x, sigma=0.05, set_reset_grad=True)

    assert output.shape == x.shape


def test_data_fidelity_normalizes_over_measurement_dims_only():
    batch_size = 3
    height = 4
    width = 5

    op = IdentityOp()
    denoiser = GSDenoiser(ScaleNetwork(1.0))
    pnp = GSPnP(op, denoiser, measurement_dims=(-2, -1))

    config = NormParameter(ord=2, normalize=True)
    f = pnp.prep_data_fidelity(config, pnp.measurement_dims)

    recon = torch.zeros(batch_size, height, width)
    measurement = torch.ones_like(recon)

    actual = f(recon, measurement)
    expected = 0.5 * batch_size

    assert actual == expected


def test_cg_data_fidelity_prox_supports_batch():
    torch.manual_seed(2)

    batch_size = 3
    height = 4
    width = 5
    tau = 0.7

    op = IdentityOp()
    denoiser = GSDenoiser(ScaleNetwork(1.0))
    pnp = GSPnP(op, denoiser, measurement_dims=(-2, -1))

    config = CGProxParameter(iters=100, eps=1e-30, rel_tol=1e-12, normalize=True)

    measurement = torch.randn(batch_size, height, width, dtype=torch.float64)
    z = torch.randn_like(measurement)

    prox = pnp.prep_data_fidelity_prox(config, measurement, pnp.measurement_dims)
    actual = prox(z, tau)

    measurement_size = height * width
    factor = 1.0 / measurement_size
    eta = 1.0 / tau
    expected = (factor * measurement + eta * z) / (factor + eta)

    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)


def test_pcm_deepinv_physics_supports_batch():
    torch.manual_seed(3)

    batch_size = 3
    gamma = 0.7

    op = PhotocurrentMapOp(j_order=2, s_indices=[0, 1, 3, 7, 10])
    physics = PCMDeepInvPhysics(op)

    x = torch.randn(batch_size, *op.domain_shape, dtype=torch.float64)
    y = physics.A(x)

    assert y.shape == (batch_size, *op.range_shape)
    assert physics.A_adjoint(y).shape == x.shape

    z = torch.randn_like(x)

    actual = physics.prox_l2(z, y, gamma)
    expected = closed_form_pcm_prox(op, rho=1.0 / gamma, y=y, z=z)

    torch.testing.assert_close(actual, expected)
