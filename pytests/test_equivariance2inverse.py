"""Equivariance2Inverse on CPU without tomosipo: the paper's Gaussian
post-log noise, its estimate from the data, and lambda.

The CT operator is the identity (images and sinograms have the same shape).
"""

import types

import pytest
import torch

import LION.optimizers.LIONsolver as lion_solver
from LION.CTtools.ct_geometry import Geometry
from LION.utils.parameter import LIONParameter
from pytests.helpers import small_dncnn


@pytest.fixture(autouse=True)
def no_ct(monkeypatch):
    monkeypatch.setattr(lion_solver, "make_operator", lambda geometry: None)


# the authors' measured point response of 2DeteCT's detector 1 (dataset_consts.py)
PRF_2DETECT = [
    0.00553719,
    0.01118421,
    0.02565282,
    0.0636673,
    0.15764087,
    0.47263522,
    0.15764087,
    0.0636673,
    0.02565282,
    0.01118421,
    0.00553719,
]


def smooth_sinograms(n=4, angles=90, detectors=200):
    t = torch.linspace(0, 1, detectors)
    a = torch.linspace(0, 3.14, angles)[:, None]
    sino = 3 * torch.exp(-((t - 0.5 - 0.2 * torch.sin(a)) ** 2) / 0.02)
    sino = sino * (t > 0.1) * (t < 0.9)  # air at both ends, as clipped data
    return sino.expand(n, 1, angles, detectors).clone()


@pytest.mark.parametrize("kernel", [None, PRF_2DETECT], ids=["iid", "blurred"])
def test_e2i_noise_std_estimate(kernel):
    from LION.optimizers.Equivariance2InverseSolver import estimate_noise_std

    torch.manual_seed(0)
    clean = smooth_sinograms()
    noise = 0.05 * torch.randn_like(clean)
    if kernel is not None:
        k = torch.tensor(kernel).view(1, 1, -1)
        noise = torch.nn.functional.conv1d(
            torch.nn.functional.pad(noise.reshape(-1, 1, 200), (5, 5), mode="reflect"),
            k,
        ).reshape(clean.shape)
    noisy = torch.clamp(clean + noise, min=0) * (clean > 0)  # air clipped to 0
    assert estimate_noise_std(noisy, kernel) == pytest.approx(0.05, rel=0.05)


def make_e2i_solver(monkeypatch, params=None):
    import LION.optimizers.Equivariance2InverseSolver as e2i

    identity = types.SimpleNamespace(
        to_autograd=lambda op, num_extra_dims: (lambda x: x)
    )
    monkeypatch.setattr(e2i, "import_tomosipo", lambda *args: identity)
    model = small_dncnn()
    if params is not None:
        params.recon_fn = lambda sino, op: sino
    else:
        params = e2i.Equivariance2InverseSolver.default_parameters()
        params.recon_fn = lambda sino, op: sino
    return e2i.Equivariance2InverseSolver(
        model,
        torch.optim.SGD(model.parameters(), lr=1e-2),
        torch.nn.MSELoss(),
        solver_params=params,
        geometry=Geometry.default_parameters(),
        device=torch.device("cpu"),
        verbose=False,
    )


def test_e2i_estimates_noise_std_once(monkeypatch):
    from LION.optimizers.Equivariance2InverseSolver import estimate_noise_std

    solver = make_e2i_solver(monkeypatch)
    assert solver.solver_params.noise_std is None
    torch.manual_seed(0)
    sinos = torch.rand(2, 1, 16, 16)
    solver.mini_batch_step(sinos, None).backward()
    assert solver.solver_params.noise_std == pytest.approx(estimate_noise_std(sinos))
    first = solver.solver_params.noise_std
    solver.mini_batch_step(torch.rand(2, 1, 16, 16), None)
    assert solver.solver_params.noise_std == first  # kept, e.g. in saved params


@pytest.mark.parametrize("kernel", [None, PRF_2DETECT], ids=["iid", "blurred"])
def test_e2i_noise(monkeypatch, kernel):
    import LION.optimizers.Equivariance2InverseSolver as e2i

    params = e2i.Equivariance2InverseSolver.default_parameters()
    params.noise_std = 0.1
    params.noise_kernel = kernel
    solver = make_e2i_solver(monkeypatch, params)
    torch.manual_seed(0)
    noise = solver._noise(torch.zeros(8, 1, 64, 256))
    expected = 0.1 * (1.0 if kernel is None else torch.tensor(kernel).norm().item())
    assert noise.std().item() == pytest.approx(expected, rel=0.03)
    assert noise.mean().item() == pytest.approx(0.0, abs=0.01)


def test_e2i_equivariance_strength(monkeypatch):
    """With equivariance_strength = 0 only the data consistency term is left."""
    import LION.optimizers.Equivariance2InverseSolver as e2i

    params = e2i.Equivariance2InverseSolver.default_parameters()
    params.noise_std = 0.1
    params.equivariance_strength = 0.0
    solver = make_e2i_solver(monkeypatch, params)
    sinos = torch.rand(2, 1, 16, 16)
    torch.manual_seed(3)
    loss = solver.mini_batch_step(sinos, None)
    torch.manual_seed(3)
    j = torch.randint(0, 16, (1,)).item()  # the held-out angle, drawn first
    masked = sinos.clone()
    masked[:, :, j] = 0
    with torch.no_grad():
        expected = (
            (solver.model(masked * 16 / 15)[:, :, j] - sinos[:, :, j]) ** 2
        ).mean()
    torch.testing.assert_close(loss.detach(), expected)


def test_e2i_old_parameters(monkeypatch):
    """Parameters saved with the old noise model: warned about, and the new
    ones get their defaults."""
    params = LIONParameter(
        I0=500, sigma=50**0.5, sigma_blur=0.8, noise_model="relative"
    )
    with pytest.warns(UserWarning, match="no longer uses"):
        solver = make_e2i_solver(monkeypatch, params)
    assert solver.solver_params.noise_std is None
    assert solver.solver_params.equivariance_strength == 1.0


def test_e2i_does_not_clamp_the_rotated_image(monkeypatch):
    """As in the paper and the authors' code, the rotated reconstruction is
    not clamped at 0 (a model output of -1 stays -1 inside the image)."""
    import LION.optimizers.Equivariance2InverseSolver as e2i

    def equivariance_term(strength):
        params = e2i.Equivariance2InverseSolver.default_parameters()
        params.noise_std = 0.0
        params.equivariance_strength = strength
        solver = make_e2i_solver(monkeypatch, params)
        solver.model.forward = lambda x: -torch.ones_like(x)
        torch.manual_seed(0)
        return solver.mini_batch_step(torch.rand(1, 1, 16, 16), None).item()

    term = equivariance_term(1.0) - equivariance_term(0.0)
    # clamped, the rotated image would be 0 everywhere and the term exactly 1;
    # unclamped, only the corners rotated in from outside (zeros) differ
    assert term < 0.5
