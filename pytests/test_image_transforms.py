# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
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


def test_random_transforms():
    from LION.utils.image_transforms import random_crop

    torch.manual_seed(0)
    x = torch.arange(2 * 64.0).reshape(2, 1, 8, 8).repeat(8, 1, 1, 1)
    crops = random_crop(x, (3, 4))
    assert crops.shape == (16, 1, 3, 4)
    for crop, image in zip(crops, x):  # a block of its own image
        top, left = divmod(int(crop[0, 0, 0] - image[0, 0, 0]), 8)
        torch.testing.assert_close(crop, image[..., top : top + 3, left : left + 4])
    assert len({float(c[0, 0, 0]) % 64 for c in crops}) > 1
    with pytest.raises(ValueError, match="doesn't fit"):
        random_crop(x, (9, 4))

    from LION.utils.image_transforms import random_erasing

    torch.manual_seed(0)
    x = torch.ones(200, 1, 16, 16)
    erased = random_erasing(x)
    assert torch.equal(x, torch.ones_like(x))  # a new tensor
    changed = (erased == 0).flatten(1).any(1)
    assert 60 < int(changed.sum()) < 140  # p = 0.5
    for image in erased[changed]:
        rows, cols = torch.nonzero(image[0] == 0, as_tuple=True)
        h, w = int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1)
        assert h * w == len(rows)  # one rectangle
        assert 0.02 * 256 * 0.5 <= h * w <= 0.33 * 256 * 1.5

    from LION.utils.image_transforms import random_flip_rotate

    torch.manual_seed(0)
    image = torch.arange(16.0).reshape(1, 4, 4)
    symmetries = [torch.rot90(image, k, dims=(-2, -1)) for k in range(4)]
    symmetries += [torch.flip(s, dims=(-1,)) for s in symmetries]
    out = random_flip_rotate(image.expand(64, 1, 4, 4))
    found = {
        next(i for i, s in enumerate(symmetries) if torch.equal(o, s)) for o in out
    }
    assert found == set(range(8))

    rectangle = torch.arange(12.0).reshape(1, 1, 3, 4).expand(32, 1, 3, 4)
    out = random_flip_rotate(rectangle)
    assert out.shape == rectangle.shape
    kept = {
        "same": rectangle[0],
        "180": torch.rot90(rectangle[0], 2, dims=(-2, -1)),
        "flip": torch.flip(rectangle[0], dims=(-1,)),
        "180+flip": torch.flip(torch.rot90(rectangle[0], 2, dims=(-2, -1)), dims=(-1,)),
    }
    assert {k for o in out for k, v in kept.items() if torch.equal(o, v)} == set(kept)
