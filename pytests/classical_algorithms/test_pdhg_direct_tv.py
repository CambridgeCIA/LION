"""Tests for the direct scalar TV solver used by PCM."""

import pytest
import torch
from LION.classical_algorithms.pdhg_direct import gradient, tv_pdhg


class IdentityOp:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def adjoint(self, measurement: torch.Tensor) -> torch.Tensor:
        return measurement


def test_scalar_tv_callback_stride_does_not_change_result() -> None:
    measurement = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8) / 63
    params = dict(
        measurement=measurement,
        initial_image=torch.zeros_like(measurement),
        forward_op=IdentityOp(),
        lambda_reg=0.1,
        max_iterations=17,
        primal_stepsize=0.3,
        dual_stepsize=0.3,
    )
    iterations: list[int] = []
    with torch.no_grad():
        baseline = tv_pdhg(**params)
        logged = tv_pdhg(
            **params,
            iteration_callback=lambda iteration, image: iterations.append(iteration),
            callback_every=4,
        )
    assert iterations == [4, 8, 12, 16, 17]
    torch.testing.assert_close(logged, baseline)


def test_scalar_tv_objective_improves_for_identity_measurements() -> None:
    torch.manual_seed(1)
    measurement = torch.rand(1, 1, 8, 8)
    initial = torch.zeros_like(measurement)
    lam = 0.1

    def objective(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x - measurement).square().sum() + lam * gradient(x).abs().sum()

    with torch.no_grad():
        reconstruction = tv_pdhg(
            measurement=measurement,
            initial_image=initial,
            forward_op=IdentityOp(),
            lambda_reg=lam,
            max_iterations=256,
            primal_stepsize=0.3,
            dual_stepsize=0.3,
        )
    assert bool(torch.isfinite(reconstruction).all())
    assert float(objective(reconstruction)) < float(objective(initial))


def test_tv_rejects_nonpositive_callback_period() -> None:
    image = torch.zeros(1, 1, 4, 4)
    with pytest.raises(ValueError, match="callback_every"):
        tv_pdhg(
            measurement=image,
            initial_image=image,
            forward_op=IdentityOp(),
            lambda_reg=0.1,
            max_iterations=1,
            primal_stepsize=0.3,
            dual_stepsize=0.3,
            callback_every=0,
        )


def test_tv_with_subsampled_orthonormal_hadamard() -> None:
    h2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]]) / (2.0**0.5)
    hadamard = h2
    for _ in range(5):
        hadamard = torch.kron(hadamard, h2)
    sampled = hadamard[:20, :]

    class SampledHadamardOp:
        def __call__(self, image: torch.Tensor) -> torch.Tensor:
            return (sampled @ image.reshape(-1)).reshape(1, 1, -1)

        def adjoint(self, measurement: torch.Tensor) -> torch.Tensor:
            return (sampled.T @ measurement.reshape(-1)).reshape(1, 1, 8, 8)

    torch.manual_seed(2)
    target = torch.rand(1, 1, 8, 8)
    op = SampledHadamardOp()
    measured = op(target)
    x0 = torch.zeros_like(target)
    lam = 0.03

    def objective(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (op(x) - measured).square().sum() + lam * gradient(x).abs().sum()

    with torch.no_grad():
        recon = tv_pdhg(
            measurement=measured,
            initial_image=x0,
            forward_op=op,
            lambda_reg=lam,
            max_iterations=256,
            primal_stepsize=0.30,
            dual_stepsize=0.30,
        )
    assert bool(torch.isfinite(recon).all())
    assert float(objective(recon)) < float(objective(x0))
