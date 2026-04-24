import torch
from LION.operators.CompositeOp import CompositeOp
from LION.operators.multilevel_sample import multilevel_sample
from LION.operators.PhotocurrentMapOp import PhotocurrentMapOp
from LION.operators.Wavelet2D import Wavelet2D
from pytests.helper import dotproduct_adjointness_test


def test_composite_op_adjointness():
    """Test adjoint property of A = Phi Psi^{-1}."""
    J = 4  # 16x16 images
    H = W = 1 << J
    subtract_from_J = 1
    coarseJ = J - subtract_from_J
    delta = 1.0 / 4

    # Wavelet transform Psi
    wavelet = Wavelet2D((H, W), wavelet_name="db4")

    # Photocurrent mapping operator Phi
    sampled_indices = multilevel_sample(
        J=J, num_samples=int(delta * H * W), coarse_J=coarseJ, alpha=1.0
    )
    phi = PhotocurrentMapOp(J=J, sampled_indices=sampled_indices)

    # Composite operator A = Phi Psi^{-1}
    operator = CompositeOp(wavelet, phi, device=torch.get_default_device())

    # Domain: wavelet coefficients (length = number of coefficients)
    n_w = wavelet.size
    x = torch.rand(n_w)

    # Codomain: measurements (infer length by one forward pass)
    y0 = operator(x * 0.0)  # same shape as A(x)
    y = torch.rand_like(y0)

    dotproduct_adjointness_test(operator, x, y)
