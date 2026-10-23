import torch
from LION.classical_algorithms.conjugate_gradient import conjugate_gradient


def test_conjugate_gradient_matches_direct_solve():
    torch.manual_seed(0)

    n = 20
    B = torch.randn(n, n, dtype=torch.float64)
    A = B.T @ B + 0.5 * torch.eye(n, dtype=torch.float64)

    b = torch.randn(n, dtype=torch.float64)
    x0 = torch.zeros_like(b)

    x_cg = conjugate_gradient(
        lambda x: A @ x, b, x0=x0, max_iter=100, rel_tol=1e-12, dim=None
    )

    x_reference = torch.linalg.solve(A, b)

    torch.testing.assert_close(x_cg, x_reference, rtol=1e-8, atol=1e-10)


def test_conjugate_gradient_has_small_residual():
    torch.manual_seed(1)

    n = 20
    B = torch.randn(n, n, dtype=torch.float64)
    A = B.T @ B + torch.eye(n, dtype=torch.float64)

    b = torch.randn(n, dtype=torch.float64)
    x0 = torch.zeros_like(b)

    x = conjugate_gradient(
        lambda x: A @ x, b, x0=x0, max_iter=100, rel_tol=1e-12, dim=None
    )

    residual = A @ x - b

    assert torch.linalg.vector_norm(residual) < 1e-8


def test_conjugate_gradient_solution_has_zero_autograd_gradient():
    torch.manual_seed(2)

    n = 20
    B = torch.randn(n, n, dtype=torch.float64)
    A = B.T @ B + torch.eye(n, dtype=torch.float64)

    b = torch.randn(n, dtype=torch.float64)
    x0 = torch.zeros_like(b)

    x_cg = conjugate_gradient(
        lambda x: A @ x, b, x0=x0, max_iter=100, rel_tol=1e-12, dim=None
    )

    x = x_cg.detach().clone().requires_grad_(True)

    objective = 0.5 * x @ (A @ x) - b @ x

    (gradient,) = torch.autograd.grad(objective, x)

    torch.testing.assert_close(
        gradient, torch.zeros_like(gradient), rtol=0.0, atol=1e-8
    )


def test_conjugate_gradient_identity():
    torch.manual_seed(3)

    b = torch.randn(10, dtype=torch.float64)
    x0 = torch.zeros_like(b)

    x = conjugate_gradient(lambda x: x, b, x0=x0, max_iter=10, rel_tol=1e-12, dim=None)

    torch.testing.assert_close(x, b)


def test_conjugate_gradient_returns_exact_initial_solution():
    A = torch.eye(5, dtype=torch.float64)  # Identity matrix
    rhs = torch.randn(5, dtype=torch.float64)
    x0 = rhs.clone()

    x = conjugate_gradient(
        lambda x: A @ x, rhs, x0=x0, max_iter=10, rel_tol=1e-10, dim=None
    )

    torch.testing.assert_close(x, x0)


def test_conjugate_gradient_2d_matches_direct_solve():
    torch.manual_seed(4)

    height, width = 4, 5
    n = height * width
    B = torch.randn(n, n, dtype=torch.float64)
    A = B.T @ B + 0.5 * torch.eye(n, dtype=torch.float64)

    rhs = torch.randn(height, width, dtype=torch.float64)
    x0 = torch.zeros_like(rhs)

    def matmul(x: torch.Tensor) -> torch.Tensor:
        return (A @ x.reshape(-1)).reshape(height, width)

    x_cg = conjugate_gradient(matmul, rhs, x0=x0, max_iter=100, rel_tol=1e-12)

    x_reference = torch.linalg.solve(A, rhs.reshape(-1)).reshape(height, width)

    torch.testing.assert_close(x_cg, x_reference, rtol=1e-8, atol=1e-10)


def test_batched_vector_conjugate_gradient_matches_direct_solve():
    torch.manual_seed(5)

    batch_size = 4
    n = 12
    B = torch.randn(batch_size, n, n, dtype=torch.float64)
    A = B.transpose(-1, -2) @ B + 0.5 * torch.eye(n, dtype=torch.float64).unsqueeze(0)

    rhs = torch.randn(batch_size, n, dtype=torch.float64)
    x0 = torch.zeros_like(rhs)

    def matmul(x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bij,bj->bi", A, x)

    x_cg = conjugate_gradient(
        matmul, rhs, x0=x0, max_iter=100, eps=1e-30, rel_tol=1e-12, dim=(-1,)
    )

    x_reference = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)

    torch.testing.assert_close(x_cg, x_reference, rtol=1e-8, atol=1e-10)


def test_batched_image_conjugate_gradient_matches_direct_solve():
    torch.manual_seed(6)

    batch_size = 3
    height, width = 3, 4
    n = height * width
    B = torch.randn(batch_size, n, n, dtype=torch.float64)
    A = B.transpose(-1, -2) @ B + 0.5 * torch.eye(n, dtype=torch.float64).unsqueeze(0)

    rhs = torch.randn(batch_size, height, width, dtype=torch.float64)
    x0 = torch.zeros_like(rhs)

    def matmul(x: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(batch_size, n)
        return torch.einsum("bij,bj->bi", A, x_flat).reshape(batch_size, height, width)

    x_cg = conjugate_gradient(
        matmul, rhs, x0=x0, max_iter=100, eps=1e-30, rel_tol=1e-12
    )

    rhs_flat = rhs.reshape(batch_size, n)
    x_reference = torch.linalg.solve(A, rhs_flat.unsqueeze(-1)).squeeze(-1)
    x_reference = x_reference.reshape(batch_size, height, width)

    torch.testing.assert_close(x_cg, x_reference, rtol=1e-8, atol=1e-10)


def test_4d_conjugate_gradient_uses_last_two_dimensions():
    torch.manual_seed(7)

    batch_size = 2
    channels = 3
    height, width = 4, 5

    weights = torch.rand(batch_size, channels, 1, 1, dtype=torch.float64) + 0.5
    rhs = torch.randn(batch_size, channels, height, width, dtype=torch.float64)
    x0 = torch.zeros_like(rhs)

    x_cg = conjugate_gradient(
        lambda x: weights * x, rhs, x0=x0, max_iter=10, rel_tol=1e-12
    )

    x_reference = rhs / weights

    torch.testing.assert_close(x_cg, x_reference, rtol=1e-10, atol=1e-12)
