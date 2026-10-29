"""Plain-torch Gaussian blur and rotation (LION.utils.image_transforms),
replacing torchvision's."""

import pytest
import torch


@pytest.mark.parametrize("shape", [(2, 1, 16, 16), (3, 15, 17)])
def test_rotate(shape):
    from LION.utils.image_transforms import rotate

    torch.manual_seed(0)
    x = torch.rand(*shape)
    torch.testing.assert_close(rotate(x, 0.0), x)
    # 90 degrees counter-clockwise, for square images exactly a rot90
    if shape[-1] == shape[-2]:
        torch.testing.assert_close(rotate(x, 90.0), torch.rot90(x, 1, dims=(-2, -1)))
    disk = torch.zeros(1, 1, 32, 32)
    yy, xx = torch.meshgrid(
        torch.arange(32.0) - 15.5, torch.arange(32.0) - 15.5, indexing="ij"
    )
    disk[..., (xx**2 + yy**2) < 100] = 1.0
    torch.testing.assert_close(rotate(disk, 37.0, "bilinear"), disk, atol=0.6, rtol=0)
    with pytest.raises(ValueError):
        rotate(x, 10.0, "bicubic")


@pytest.mark.parametrize("angle", [30.0, 137.5, -45.0])
def test_transforms_match_torchvision(angle):
    """Same results as torchvision, which LION used before."""
    TF = pytest.importorskip("torchvision.transforms.functional")
    from LION.utils.image_transforms import gaussian_blur, rotate

    torch.manual_seed(0)
    x = torch.rand(2, 1, 20, 23)
    bilinear = TF.InterpolationMode.BILINEAR
    torch.testing.assert_close(
        rotate(x, angle, "bilinear"), TF.rotate(x, angle, interpolation=bilinear)
    )
    # nearest: equal up to rounding at exact ties between two pixels
    same = rotate(x, angle) == TF.rotate(x, angle)
    assert same.float().mean() > 0.99
    # ct_utils's detector cross-talk, and Noisier2Inverse's 2D blur
    torch.testing.assert_close(
        gaussian_blur(x, 3, 0.3015, dims=(-1,)),
        TF.gaussian_blur(x, kernel_size=[3, 1], sigma=[0.3015, 1.0]),
    )
    torch.testing.assert_close(
        gaussian_blur(x, 7, 1.2),
        TF.gaussian_blur(x, kernel_size=[7, 7], sigma=[1.2, 1.2]),
    )
