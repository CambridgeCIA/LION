import doctest

import torch


def dx_forward(u):
    """
    Computes the forward difference in the x direction.

    >>> u = torch.tensor(
    >>>     [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32) # 2D
    >>> print(u.shape)
    torch.Size([3, 3])
    >>> dx_forward(u)
    tensor([[3., 3., 3.],
            [3., 3., 3.],
            [0., 0., 0.]])

    >>> u = torch.tensor(
    >>>     [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32) # 3D
    >>> print(u.shape)
    torch.Size([1, 3, 3])
    >>> dx_forward(u)
    tensor([[[3., 3., 3.],
             [3., 3., 3.],
             [0., 0., 0.]]])

    >>> u = torch.tensor(
    >>>     [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=torch.float32) # 4D
    >>> print(u.shape)
    torch.Size([1, 1, 3, 3])
    >>> dx_forward(u)
    tensor([[[[3., 3., 3.],
              [3., 3., 3.],
              [0., 0., 0.]]]])
    """
    diff_x = torch.zeros_like(u)
    # Handle the middle rows (1 <= i < N1 - 1)
    diff_x[..., :-1, :] = u[..., 1:, :] - u[..., :-1, :]
    return diff_x


def dy_forward(u):
    """
    Computes the forward difference in the y direction.

    >>> u = torch.tensor(
    >>>     [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    >>> print(u.shape)
    torch.Size([3, 3])
    >>> dy_forward(u)
    tensor([[1., 1., 0.],
            [1., 1., 0.],
            [1., 1., 0.]])

    >>> u = torch.tensor(
    >>>     [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32)
    >>> print(u.shape)
    torch.Size([1, 3, 3])
    >>> dy_forward(u)
    tensor([[[1., 1., 0.],
             [1., 1., 0.],
             [1., 1., 0.]]])

    >>> u = torch.tensor(
    >>>     [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=torch.float32)
    >>> print(u.shape)
    torch.Size([1, 1, 3, 3])
    >>> dy_forward(u)
    tensor([[[[1., 1., 0.],
              [1., 1., 0.],
              [1., 1., 0.]]]])
    """
    diff_y = torch.zeros_like(u)
    # Handle the middle columns (1 <= j < N2 - 1)
    diff_y[..., :, :-1] = u[..., :, 1:] - u[..., :, :-1]
    return diff_y


def dtime_forward(u):
    diff_time = torch.zeros_like(u)
    diff_time[..., :-1, :, :, :, :] = u[..., 1:, :, :, :, :] - u[..., :-1, :, :, :, :]
    return diff_time


def dx_backward(u):
    """
    Computes the backward difference in the x direction.

    >>> u = torch.tensor(
    >>>     [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32) # 2D
    >>> print(u.shape)
    torch.Size([3, 3])
    >>> dx_backward(u)
    tensor([[ 1.,  2.,  3.],
            [ 3.,  3.,  3.],
            [-4., -5., -6.]])

    >>> u = torch.tensor(
    >>>     [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32) # 3D
    >>> print(u.shape)
    torch.Size([1, 3, 3])
    >>> dx_backward(u)
    tensor([[[ 1.,  2.,  3.],
             [ 3.,  3.,  3.],
             [-4., -5., -6.]]])

    >>> u = torch.tensor(
    >>>     [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=torch.float32) # 4D
    >>> print(u.shape)
    torch.Size([1, 1, 3, 3])
    >>> dx_backward(u)
    tensor([[[[ 1.,  2.,  3.],
              [ 3.,  3.,  3.],
              [-4., -5., -6.]]]])
    """
    diff_x = torch.zeros_like(u)
    # Handle the first row (i == 0)
    diff_x[..., 0, :] = u[..., 0, :]
    # Handle the middle rows (1 <= i < N1 - 1)
    diff_x[..., 1:-1, :] = u[..., 1:-1, :] - u[..., :-2, :]
    # Handle the last row (i == N1 - 1)
    diff_x[..., -1, :] = -u[..., -2, :]
    return diff_x


def dy_backward(u):
    """
    Computes the backward difference in the y direction.

    >>> u = torch.tensor(
    >>>     [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32) # 2D
    >>> print(u.shape)
    torch.Size([3, 3])
    >>> dy_backward(u)
    tensor([[ 1.,  1., -2.],
            [ 4.,  1., -5.],
            [ 7.,  1., -8.]])

    >>> u = torch.tensor(
    >>>     [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32) # 3D
    >>> print(u.shape)
    torch.Size([1, 3, 3])
    >>> dy_backward(u)
    tensor([[[ 1.,  1., -2.],
             [ 4.,  1., -5.],
             [ 7.,  1., -8.]]])

    >>> u = torch.tensor(
    >>>     [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=torch.float32) # 4D
    >>> print(u.shape)
    torch.Size([1, 1, 3, 3])
    >>> dy_backward(u)
    tensor([[[[ 1.,  1., -2.],
              [ 4.,  1., -5.],
              [ 7.,  1., -8.]]]])
    """
    diff_y = torch.zeros_like(u)
    # Handle the first column (j == 0)
    diff_y[..., :, 0] = u[..., :, 0]
    # Handle the middle columns (1 <= j < N2 - 1)
    diff_y[..., :, 1:] = u[..., :, 1:] - u[..., :, :-1]
    # Handle the last column (j == N2 - 1)
    diff_y[..., :, -1] = -u[..., :, -2]
    return diff_y


def dtime_backward(u):
    diff_time = torch.zeros_like(u)
    diff_time[..., 0, :, :, :, :] = u[..., 0, :, :, :, :]
    diff_time[..., 1:, :, :, :, :] = u[..., 1:, :, :, :, :] - u[..., :-1, :, :, :, :]
    diff_time[..., -1, :, :, :, :] = -u[..., -2, :, :, :, :]
    return diff_time


def test_adjoint_property():
    """
    Test the adjoint property for forward and backward difference operators.

    >>> adjoint_x, adjoint_y = test_adjoint_property()
    >>> assert abs(adjoint_x) < 1e-6, f"Adjoint property failed: x={adjoint_x}"
    >>> assert abs(adjoint_y) < 1e-6, f"Adjoint property failed: y={adjoint_y}"
    """
    N1, N2 = 4, 4  # Example dimensions
    u = torch.rand((N1, N2))
    v = torch.rand((N1, N2))

    fwd_x_u = dx_forward(u)
    bwd_x_v = dx_backward(v)
    adjoint_x = torch.sum(fwd_x_u * v) + torch.sum(u * bwd_x_v)

    fwd_y_u = dy_forward(u)
    bwd_y_v = dy_backward(v)
    adjoint_y = torch.sum(fwd_y_u * v) + torch.sum(u * bwd_y_v)

    return adjoint_x.item(), adjoint_y.item()


doctest.testmod()
test_adjoint_property()

if __name__ == "__main__":
    import doctest

    doctest.testmod()
    print("Adjoint property test results:", test_adjoint_property())
