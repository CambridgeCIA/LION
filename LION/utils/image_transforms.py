"""Gaussian blur and rotation of images in plain torch.

They give the same results as ``torchvision.transforms.functional``'s
``gaussian_blur`` and ``rotate`` (checked in pytests/test_utils.py), which LION
used before, without the torchvision dependency. Both work on any number of
leading dimensions, on any device, and are differentiable.
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
    """Gaussian blur of ``x`` along ``dims``, with reflected borders.

    Args:
        x: Tensor of any shape, e.g. images (B, C, H, W) or sinograms
            (B, C, angles, detectors).
        kernel_size: Odd kernel size, one for all dims or one per dim.
        sigma: Standard deviation in pixels, one for all dims or one per dim.
        dims: The dimensions to blur along. Default: the last two (an image).
            ``dims=(-1,)`` blurs along the last dimension only.

    Returns:
        The blurred tensor, with the shape and dtype of ``x``.
    """
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
    """Rotate images counter-clockwise by ``angle`` degrees about their centre.

    The output has the size of the input; points that come from outside the
    image are 0.

    Args:
        x: Images (..., H, W), e.g. (B, C, H, W).
        angle: Rotation angle in degrees, counter-clockwise.
        interpolation: "nearest" or "bilinear".

    Returns:
        The rotated images, with the shape and dtype of ``x``.
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
