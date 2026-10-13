import torch


def estimate_operator_norm_squared(
    forward_op, adjoint_op, shape, num_iters=50, device="cuda"
):
    """Estimate the squared spectral norm of a linear operator.

    Parameters
    ----------
    forward_op : callable
        Function applying the forward operator A.
    adjoint_op : callable
        Function applying the adjoint operator A^*.
    shape : tuple[int, ...]
        Shape of an image-domain tensor.
    num_iters : int, optional
        Number of power iterations.
    device : str, optional
        Device used for the computation.

    Returns
    -------
    float
        Estimate of ||A||_2^2.
    """
    x = torch.randn(shape, device=device)
    x = x / torch.linalg.vector_norm(x)

    for _ in range(num_iters):
        x = adjoint_op(forward_op(x))
        norm_x = torch.linalg.vector_norm(x)

        if norm_x == 0:
            return 0.0

        x = x / norm_x

    Ax = forward_op(x)
    return float(torch.linalg.vector_norm(Ax) ** 2)
