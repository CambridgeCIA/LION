import numpy as np
import pytest
import torch
from LION.classical_algorithms.conjugate_gradient import conjugate_gradient
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.pcm_tools.pcm_prox import closed_form_pcm_prox


def explicit_matrix(op: PhotocurrentMapOp) -> torch.Tensor:
    """Construct the matrix representing the operator."""

    n = op.num_pixels
    eye = torch.eye(n, dtype=torch.float64)

    columns = []
    for i in range(n):
        image = op.unvectorize(eye[:, i])
        columns.append(op(image))

    return torch.stack(columns, dim=1)


def test_closed_form_pcm_prox_matches_dense_solve():
    j_order = 2
    sampled_indices = np.array([0, 1, 3, 7, 10])
    rho = 0.7

    op = PhotocurrentMapOp(j_order=j_order, s_indices=sampled_indices)

    torch.manual_seed(0)

    batch_size = 3
    z = torch.randn(batch_size, *op.domain_shape, dtype=torch.float64)
    y = torch.randn(batch_size, *op.range_shape, dtype=torch.float64)

    x_closed = closed_form_pcm_prox(op=op, rho=rho, y=y, z=z)

    A = explicit_matrix(op)
    z_vec = op.vectorize(z)

    lhs = A.T @ A + rho * torch.eye(op.num_pixels, dtype=torch.float64)
    rhs = y @ A + rho * z_vec
    # rhs = (A.T @ y.T).T + rho * z_vec

    x_reference = torch.linalg.solve(lhs, rhs.T).T
    x_reference = op.unvectorize(x_reference)

    torch.testing.assert_close(x_closed, x_reference, rtol=1e-10, atol=1e-10)


def test_closed_form_pcm_prox_satisfies_optimality_condition():
    j_order = 2
    sampled_indices = np.array([0, 2, 5, 8, 12])
    rho = 0.7

    op = PhotocurrentMapOp(j_order=j_order, s_indices=sampled_indices)

    torch.manual_seed(1)

    batch_size = 3
    z = torch.randn(batch_size, *op.domain_shape, dtype=torch.float64)
    y = torch.randn(batch_size, *op.range_shape, dtype=torch.float64)

    x = closed_form_pcm_prox(op=op, rho=rho, y=y, z=z)

    gradient = op.adjoint(op(x) - y) + rho * (x - z)

    torch.testing.assert_close(
        gradient, torch.zeros_like(gradient), rtol=1e-10, atol=1e-10
    )


def test_closed_form_pcm_prox_returns_z_for_consistent_data():
    j_order = 2
    sampled_indices = np.array([0, 1, 4, 6, 9])
    rho = 0.7

    op = PhotocurrentMapOp(j_order=j_order, s_indices=sampled_indices)

    torch.manual_seed(2)

    batch_size = 3
    z = torch.randn(batch_size, *op.domain_shape, dtype=torch.float64)
    y = op(z)

    x = closed_form_pcm_prox(op=op, rho=rho, y=y, z=z)

    torch.testing.assert_close(x, z, rtol=1e-10, atol=1e-10)


def test_closed_form_pcm_prox_has_zero_autograd_gradient():
    j_order = 2
    sampled_indices = np.array([0, 1, 3, 7, 10])
    rho = 0.7

    op = PhotocurrentMapOp(j_order=j_order, s_indices=sampled_indices)

    torch.manual_seed(0)

    batch_size = 3
    z = torch.randn(batch_size, *op.domain_shape, dtype=torch.float64)
    y = torch.randn(batch_size, *op.range_shape, dtype=torch.float64)

    x_closed = closed_form_pcm_prox(op=op, rho=rho, y=y, z=z)

    # Make x an independent variable for the objective.
    x = x_closed.detach().clone().requires_grad_(True)

    objective = 0.5 * torch.sum((y - op(x)) ** 2) + 0.5 * rho * torch.sum((x - z) ** 2)

    (gradient,) = torch.autograd.grad(objective, x)

    torch.testing.assert_close(
        gradient, torch.zeros_like(gradient), atol=1e-10, rtol=1e-10
    )

    analytical_gradient = op.adjoint(op(x) - y) + rho * (x - z)

    torch.testing.assert_close(gradient, analytical_gradient, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("rho", [0.01, 0.1, 1.0, 10.0])
def test_closed_form_pcm_prox_matches_conjugate_gradient(rho):
    torch.manual_seed(0)

    op = PhotocurrentMapOp(j_order=2, s_indices=np.array([0, 1, 4, 7, 10, 13]))

    batch_size = 3
    z = torch.randn(batch_size, *op.domain_shape, dtype=torch.float64)
    y = torch.randn(batch_size, *op.range_shape, dtype=torch.float64)

    x_closed = closed_form_pcm_prox(op, rho=rho, y=y, z=z)

    rhs = op.adjoint(y) + rho * z

    x_cg = conjugate_gradient(
        lambda x: op.adjoint(op(x)) + rho * x,
        rhs,
        x0=torch.zeros_like(z),
        max_iter=100,
        rel_tol=1e-12,
    )

    torch.testing.assert_close(x_closed, x_cg, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("rho", [1e-2, 1e-1, 1.0, 10.0])
def test_closed_form_pcm_prox_is_finite(rho: float):
    torch.manual_seed(0)

    op = PhotocurrentMapOp(j_order=2, s_indices=np.array([0, 1, 4, 7, 10, 13]))
    batch_size = 3
    y = torch.randn(batch_size, *op.range_shape, dtype=torch.float64)
    z = torch.randn(batch_size, *op.domain_shape, dtype=torch.float64)
    x = closed_form_pcm_prox(op, rho=rho, y=y, z=z)

    assert torch.isfinite(x).all()
