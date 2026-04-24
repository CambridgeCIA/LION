"""Tests for computing the operator norm of operators."""

import pytest
import torch
from LION.operators.Operator import Operator
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.operators.WalshHadamard2D import WalshHadamard2D
from LION.operators.Wavelet2D import Wavelet2D
from LION.pcm.experiment import sample_indices
from LION.utils.math import power_method


def test_matrix_op_operator_norm_torch():
    """Test operator norm computation on a small matrix."""
    # Example matrix
    rng = torch.Generator().manual_seed(0)
    A = torch.rand((5, 3), dtype=torch.float32, generator=rng)

    # Singular values (Sigma in SVD A = U Σ V^T)
    singular_values = torch.linalg.svdvals(A)

    # Operator 2-norm (largest singular value)
    op_2_norm: torch.Tensor = torch.linalg.norm(A, 2)
    # # or equivalently:
    op_norm_expected: torch.Tensor = singular_values[0]

    torch.testing.assert_close(op_2_norm, op_norm_expected, atol=1e-6, rtol=1e-6)

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

    matrix_op = MatrixOperatorTorch(A)
    op_norm_computed = power_method(matrix_op)
    torch.testing.assert_close(op_norm_computed, op_norm_expected, atol=1e-6, rtol=1e-6)


@pytest.mark.tomosipo  # Add argument `-m "not tomosipo"` to the `pytest` command line to skip this test
def test_ct_operator_norm_torch():
    """Test with CT operator using default geometry."""
    from LION.CTtools.ct_geometry import Geometry
    from LION.CTtools.ct_utils import make_operator

    geometry = Geometry.default_parameters()
    ct_op = make_operator(geometry)

    ct_op_norm = power_method(ct_op)
    # TODO: This is just the actual computed value, we need to verify it is correct.
    ct_op_norm_true = 285.5718078613281
    torch.testing.assert_close(ct_op_norm.item(), ct_op_norm_true, atol=1e-2, rtol=1e-2)


def test_pcm_operator_norm_torch():
    """Test with photocurrent mapping operator with undersampling."""
    J = 4
    N = 1 << J  # 16x16 image
    sampled_indices = sample_indices(
        j_order=J,
        sampling_ratio=0.25,  # keep only 1/4 of measurements
        coarse_j=J - 1,
        randomising_scheme="uniform",
        seed=0,
    )
    pcm_op = PhotocurrentMapOp(J=J, sampled_indices=sampled_indices)

    pcm_op_norm = power_method(pcm_op)
    pcm_op_norm_true = float(N)

    torch.testing.assert_close(
        pcm_op_norm.item(), pcm_op_norm_true, atol=1e-2, rtol=1e-2
    )

    J = 4
    N = 1 << J  # 16x16 image
    sampled_indices = sample_indices(
        j_order=J,
        sampling_ratio=1.0 / 16,  # keep only 1/16 of measurements
        coarse_j=J - 2,
        randomising_scheme="uniform",
        seed=0,
    )
    pcm_op = PhotocurrentMapOp(J=J, sampled_indices=sampled_indices)

    pcm_op_norm = power_method(pcm_op)
    pcm_op_norm_true = float(N)

    torch.testing.assert_close(
        pcm_op_norm.item(), pcm_op_norm_true, atol=1e-2, rtol=1e-2
    )


def test_wht_operator_norm_torch():
    """Test with Walsh-Hadamard Transform operator."""
    J = 4
    N = 1 << J  # 16x16 image
    wht_op = WalshHadamard2D(height=N, width=N)

    wht_op_norm = power_method(wht_op)
    rng = torch.Generator().manual_seed(0)
    x = torch.rand((N, N), dtype=torch.float32, generator=rng)
    y = wht_op(x)
    ratio = torch.norm(y) / torch.norm(x)
    # Since our operator is not normalized,  ||WHT x|| / ||x|| = ||WHT||.
    torch.testing.assert_close(ratio.item(), wht_op_norm.item(), atol=1e-4, rtol=1e-4)

    # WHT has norm sqrt(height*width) = sqrt(N*N) = N
    wht_op_norm_true = float(N)
    torch.testing.assert_close(
        wht_op_norm.item(), wht_op_norm_true, atol=1e-2, rtol=1e-2
    )


def test_wavelet_operator_norm_torch():
    """Test with Daubechies 4 wavelet transform operator."""
    image_shape = (16, 16)
    wavelet_op = Wavelet2D(image_shape, wavelet_name="db4")

    wavelet_op_norm = power_method(wavelet_op)
    wavelet_op_norm_true = 1.0  # orthonormal wavelet

    torch.testing.assert_close(
        wavelet_op_norm.item(), wavelet_op_norm_true, atol=1e-4, rtol=1e-4
    )
