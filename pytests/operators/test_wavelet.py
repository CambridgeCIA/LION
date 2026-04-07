import torch

from LION.operators.Wavelet2D import Wavelet2D
from pytests.helper import dotproduct_adjointness_test


def test_wavelet_db4_adjointness():
    J = 4
    H = W = 1 << J  # 16x16 image
    image_shape = (H, W)

    operator = Wavelet2D(image_shape, wavelet_name="db4")

    x = torch.rand(*image_shape)

    # infer coefficient shape
    c0 = operator(x * 0.0)
    w = torch.rand_like(c0)

    dotproduct_adjointness_test(operator, x, w)
