"""Non-CT operators (photocurrent map, wavelets, Walsh-Hadamard, composites)
and operator norms. The CT operator is tested in test_ct.py."""

import pytest
import torch
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from LION.operators.CompositeOp import CompositeOp
from LION.operators.Operator import Operator
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp, Subsampler
from LION.operators.WalshHadamard2D import WalshHadamard2D
from LION.operators.Wavelet2D import Wavelet2D
from LION.utils.math import power_method
from spyrit.core.torch import fwht, ifwht

from pytests.helper import dotproduct_adjointness_test

J = 4  # 16x16 images
N = 1 << J


def pcm_op(delta=0.25):
    """Photocurrent mapping operator keeping a fraction delta of the measurements."""
    subsampler = Subsampler(n=N * N, delta=delta, coarseJ=J - 1)
    return PhotocurrentMapOp(J=J, subsampler=subsampler)


# --------------------------------------------------------------------------
# Photocurrent mapping


def test_pcm_autograd_op_forward_and_backward():
    operator = pcm_op()

    torch.manual_seed(0)
    input_tensor = torch.randn(*operator.domain_shape, requires_grad=True)

    output_tensor = operator(input_tensor)

    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.requires_grad
    assert input_tensor.grad is None

    output_tensor.mean().backward()

    assert input_tensor.grad is not None
    assert isinstance(input_tensor.grad, torch.Tensor)
    assert input_tensor.grad.shape == input_tensor.shape
    assert torch.isfinite(input_tensor.grad).all()


def test_pcm_autograd_matches_adjoint():
    operator = pcm_op()

    x = torch.randn(*operator.domain_shape, requires_grad=True)
    y = torch.randn(*operator.range_shape)

    loss = torch.sum(operator(x) * y)
    loss.backward()

    torch.testing.assert_close(x.grad, operator.adjoint(y), rtol=1e-5, atol=1e-5)


def test_pcm_op_adjointness():
    """Test photocurrent mapping operator adjoint property."""
    operator = pcm_op()

    assert operator.domain_shape == (16, 16)
    assert operator.range_shape == (64,)

    test_image = torch.rand(*operator.domain_shape)
    test_measurement = torch.rand(*operator.range_shape)
    dotproduct_adjointness_test(operator, test_image, test_measurement)

    # Test pseudoinverse runs without error
    assert operator.pseudo_inv(test_measurement) is not None


# --------------------------------------------------------------------------
# Wavelets, Walsh-Hadamard, composites


def test_wavelet_db4_adjointness():
    image_shape = (N, N)
    operator = Wavelet2D(image_shape, wavelet_name="db4")

    x = torch.rand(*image_shape)

    # infer coefficient shape
    c0 = operator(x * 0.0)
    w = torch.rand_like(c0)

    dotproduct_adjointness_test(operator, x, w)


def test_wht_adjointness():
    """Test WHT operator adjoint property."""
    n = 256
    x = torch.rand(n)
    y = torch.rand(n)

    class WhtOp(Operator):
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return fwht(x, dim=0)

        def adjoint(self, y: torch.Tensor) -> torch.Tensor:
            return fwht(y, dim=0)

    operator = WhtOp()
    dotproduct_adjointness_test(operator, x, y)

    torch.testing.assert_close(ifwht(operator(y)), y)


def test_composite_op_adjointness():
    """Test adjoint property of A = Phi Psi^{-1}."""
    wavelet = Wavelet2D((N, N), wavelet_name="db4")  # Psi
    phi = pcm_op()  # Phi
    operator = CompositeOp(wavelet, phi, device=torch.get_default_device())

    # Domain: wavelet coefficients (length = number of coefficients)
    x = torch.rand(wavelet.size)

    # Codomain: measurements (infer length by one forward pass)
    y0 = operator(x * 0.0)
    y = torch.rand_like(y0)

    dotproduct_adjointness_test(operator, x, y)


# --------------------------------------------------------------------------
# Operator norms (power method)


def test_matrix_op_operator_norm_torch():
    """Test operator norm computation on a small matrix."""
    rng = torch.Generator().manual_seed(0)
    A = torch.rand((5, 3), dtype=torch.float32, generator=rng)

    # Operator 2-norm = largest singular value
    op_norm_expected = torch.linalg.svdvals(A)[0]
    torch.testing.assert_close(
        torch.linalg.norm(A, 2), op_norm_expected, atol=1e-6, rtol=1e-6
    )

    class MatrixOperatorTorch(Operator):
        def __init__(self, matrix: torch.Tensor):
            self.matrix = matrix

        def __call__(self, x: torch.Tensor, out=None) -> torch.Tensor:
            return self.forward(x)

        def forward(self, x: torch.Tensor, out=None) -> torch.Tensor:
            return self.matrix @ x

        def adjoint(self, y: torch.Tensor, out=None) -> torch.Tensor:
            return self.matrix.T @ y

        @property
        def domain_shape(self):
            return (self.matrix.shape[1],)

        @property
        def range_shape(self):
            return (self.matrix.shape[0],)

    op_norm_computed = power_method(MatrixOperatorTorch(A))
    torch.testing.assert_close(op_norm_computed, op_norm_expected, atol=1e-6, rtol=1e-6)


@pytest.mark.tomosipo
def test_ct_operator_norm_torch():
    """Test with CT operator using default geometry."""
    ct_op = make_operator(Geometry.default_parameters())
    ct_op_norm = power_method(ct_op)
    # TODO: This is just the actual computed value, we need to verify it is correct.
    ct_op_norm_true = 285.5718078613281
    torch.testing.assert_close(ct_op_norm.item(), ct_op_norm_true, atol=1e-2, rtol=1e-2)


def test_pcm_operator_norm_torch():
    """Photocurrent mapping with undersampling has norm N, whatever delta is."""
    for delta in (1.0 / 4, 1.0 / 16):
        pcm_op_norm = power_method(pcm_op(delta))
        torch.testing.assert_close(pcm_op_norm.item(), float(N), atol=1e-2, rtol=1e-2)


def test_wht_operator_norm_torch():
    """Test with Walsh-Hadamard Transform operator."""
    wht_op = WalshHadamard2D(height=N, width=N)

    wht_op_norm = power_method(wht_op)
    rng = torch.Generator().manual_seed(0)
    x = torch.rand((N, N), dtype=torch.float32, generator=rng)
    y = wht_op(x)
    ratio = torch.norm(y) / torch.norm(x)
    # Since our operator is not normalized,  ||WHT x|| / ||x|| = ||WHT||.
    torch.testing.assert_close(ratio.item(), wht_op_norm.item(), atol=1e-4, rtol=1e-4)

    # WHT has norm sqrt(height*width) = sqrt(N*N) = N
    torch.testing.assert_close(wht_op_norm.item(), float(N), atol=1e-2, rtol=1e-2)


def test_wavelet_operator_norm_torch():
    """Orthonormal wavelets have norm 1."""
    wavelet_op = Wavelet2D((N, N), wavelet_name="db4")
    wavelet_op_norm = power_method(wavelet_op)
    torch.testing.assert_close(wavelet_op_norm.item(), 1.0, atol=1e-4, rtol=1e-4)


def test_power_method_is_reproducible_and_scale_free():
    """Seeded start (same estimate every call); relative tolerance (the number
    of iterations does not depend on the scale of the operator)."""
    from pytests.helpers import ImageMatrixOp

    torch.manual_seed(0)
    A = torch.randn(30, 20)

    class Counting(ImageMatrixOp):
        calls = 0

        def __call__(self, x, out=None):
            type(self).calls += 1
            return super().__call__(x)

    def iterations(scale):
        Counting.calls = 0
        norm = power_method(Counting(scale * A, (20,)), tol=1e-5, maxiter=1000)
        return norm, Counting.calls

    (n1, c1), (n2, c2) = iterations(1.0), iterations(1000.0)
    torch.testing.assert_close(n2 / 1000, n1, rtol=1e-4, atol=0)
    assert c1 == c2 < 2000  # converged, in the same number of steps
    torch.testing.assert_close(iterations(1.0)[0], n1, rtol=0, atol=0)  # same start
