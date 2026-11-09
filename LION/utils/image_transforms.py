# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""Image transforms in plain torch.

``gaussian_blur`` and ``rotate`` give the same results as
``torchvision.transforms.functional``'s (checked in pytests/test_utils.py),
which LION used before, without the torchvision dependency. Both work on any
number of leading dimensions, on any device, and are differentiable.

``random_crop``, ``random_flip_rotate`` and ``random_erasing`` are
``GaussianDenoiserSolver``'s augmentations (crop and erasing: kornia's
``RandomCrop`` and ``RandomErasing`` before, with the same defaults), on
batches (B, C, H, W). They draw from
torch's random number generator on the CPU, so ``LION.utils.rng.seeded``
makes them the same every time.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F


def gaussian_kernel1d(
    kernel_size: int, sigma: float, dtype=None, device=None
) -> torch.Tensor:
    """Normalised 1D Gaussian kernel of odd size ``kernel_size`` and standard
    deviation ``sigma`` (in pixels), sampled at the pixel centres."""
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd and positive, got {kernel_size}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    half = (kernel_size - 1) / 2
    x = torch.linspace(-half, half, kernel_size, dtype=dtype, device=device)
    pdf = torch.exp(-0.5 * (x / sigma) ** 2)
    return pdf / pdf.sum()


def gaussian_blur(
    x: torch.Tensor,
    kernel_size: int | Sequence[int],
    sigma: float | Sequence[float],
    dims: Sequence[int] = (-2, -1),
) -> torch.Tensor:
    """Gaussian blur of ``x`` along ``dims``, with reflected borders: a tensor
    of any shape, e.g. images (B, C, H, W) or sinograms (B, C, angles,
    detectors). Returns the blurred tensor, with the shape and dtype of ``x``.

    ``kernel_size``: the odd kernel size, one for all ``dims`` or one each;
    ``sigma``: the standard deviation in pixels, likewise; ``dims``: the
    dimensions to blur along (the last two, an image's, by default; ``(-1,)``
    blurs along the last only)."""
    dims = list(dims)
    sizes = [kernel_size] * len(dims) if isinstance(kernel_size, int) else kernel_size
    sigmas = [sigma] * len(dims) if isinstance(sigma, (int, float)) else sigma
    if len(sizes) != len(dims) or len(sigmas) != len(dims):
        raise ValueError("kernel_size and sigma need one value per dim (or one)")
    for dim, size, s in zip(dims, sizes, sigmas):
        if size == 1:
            continue  # the kernel is [1]
        kernel = gaussian_kernel1d(size, s, dtype=x.dtype, device=x.device)
        x = _convolve_along(x, kernel, dim)
    return x


def _convolve_along(x: torch.Tensor, kernel: torch.Tensor, dim: int) -> torch.Tensor:
    """Convolve every 1D line of x along dim with kernel, reflected borders."""
    moved = x.movedim(dim, -1)
    shape = moved.shape
    lines = moved.reshape(-1, 1, shape[-1])
    pad = kernel.numel() // 2
    lines = F.pad(lines, (pad, pad), mode="reflect")
    lines = F.conv1d(lines, kernel.flip(0).view(1, 1, -1))
    return lines.reshape(shape).movedim(-1, dim)


def rotate(
    x: torch.Tensor, angle: float, interpolation: str = "nearest"
) -> torch.Tensor:
    """Images ``x`` (..., H, W) rotated counter-clockwise by ``angle`` degrees
    about their centre, with the shape and dtype of ``x``: points that come
    from outside the image are 0. ``interpolation``: "nearest" or "bilinear".
    """
    if interpolation not in ("nearest", "bilinear"):
        raise ValueError(
            f'interpolation must be "nearest" or "bilinear", got {interpolation!r}'
        )
    shape = x.shape
    h, w = shape[-2:]
    images = x.reshape(-1, 1, h, w)
    dtype = x.dtype if x.is_floating_point() else torch.float32

    # Inverse map: for each output pixel, where it comes from in the input.
    theta = math.radians(angle)
    c, s = math.cos(theta), math.sin(theta)
    # pixel centres, relative to the image centre, in pixels
    xs = torch.linspace(-w * 0.5 + 0.5, w * 0.5 - 0.5, w, dtype=dtype, device=x.device)
    ys = torch.linspace(-h * 0.5 + 0.5, h * 0.5 - 0.5, h, dtype=dtype, device=x.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    # rotation of the coordinates by -angle (y points down), in grid_sample's
    # coordinates ([-1, 1] across the image)
    src_x = (c * gx - s * gy) / (0.5 * w)
    src_y = (s * gx + c * gy) / (0.5 * h)
    grid = torch.stack((src_x, src_y), dim=-1).expand(images.shape[0], h, w, 2)

    out = F.grid_sample(
        images.to(dtype),
        grid,
        mode=interpolation,
        padding_mode="zeros",
        align_corners=False,
    )
    return out.to(x.dtype).reshape(shape)


def random_crop(x: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """A ``size`` (h, w) crop of each image of the batch ``x`` (B, C, H, W),
    at a random place in each image."""
    h, w = size
    height, width = x.shape[-2:]
    if h > height or w > width:
        raise ValueError(f"a {h}x{w} crop doesn't fit in {height}x{width} images")
    tops = torch.randint(0, height - h + 1, (x.shape[0],)).tolist()
    lefts = torch.randint(0, width - w + 1, (x.shape[0],)).tolist()
    return torch.stack(
        [x[i, ..., t : t + h, l : l + w] for i, (t, l) in enumerate(zip(tops, lefts))]
    )


def random_erasing(
    x: torch.Tensor,
    p: float = 0.5,
    scale: Sequence[float] = (0.02, 0.33),
    ratio: Sequence[float] = (0.3, 3.3),
    value: float = 0.0,
) -> torch.Tensor:
    """Each image of the batch ``x`` (B, C, H, W), with probability ``p``, with
    a random rectangle set to ``value``: its area a fraction of the image's
    drawn uniformly from ``scale``, its aspect ratio (height / width) drawn
    log-uniformly from ``ratio`` (up to 10 draws for one that fits, as in
    torchvision's ``RandomErasing``). Returns a new tensor."""
    out = x.clone()
    height, width = x.shape[-2:]
    log_ratio = torch.log(torch.tensor(ratio, dtype=torch.float64))
    for i in range(x.shape[0]):
        if torch.rand(1).item() >= p:
            continue
        for _ in range(10):
            area = height * width * torch.empty(1).uniform_(*scale).item()
            aspect = math.exp(torch.empty(1).uniform_(*log_ratio.tolist()).item())
            h = int(round(math.sqrt(area * aspect)))
            w = int(round(math.sqrt(area / aspect)))
            if 0 < h < height and 0 < w < width:
                top = torch.randint(0, height - h + 1, (1,)).item()
                left = torch.randint(0, width - w + 1, (1,)).item()
                out[i, ..., top : top + h, left : left + w] = value
                break
    return out


def random_flip_rotate(x: torch.Tensor) -> torch.Tensor:
    """Each image of the batch ``x`` (B, C, H, W) flipped and rotated by a
    random one of the 8 symmetries of the square (rotations by multiples of
    90 degrees, with or without a flip), as denoisers are trained (e.g.
    KAIR's ``augment_img``). Images that aren't square get one of the 4
    symmetries that keep their shape."""
    square = x.shape[-1] == x.shape[-2]
    modes = torch.randint(0, 8 if square else 4, (x.shape[0],)).tolist()
    out = []
    for image, mode in zip(x, modes):
        if not square:  # 0, 180 degrees, each with or without a flip
            mode = (mode % 2) * 2 + (mode // 2) * 4
        image = torch.rot90(image, mode % 4, dims=(-2, -1))
        if mode >= 4:
            image = torch.flip(image, dims=(-1,))
        out.append(image)
    return torch.stack(out)
