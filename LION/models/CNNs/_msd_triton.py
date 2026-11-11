# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri; after msd_pytorch's CUDA kernels (Allard A.
#           Hendriksen, GPL-3, https://github.com/ahendriksen/msd_pytorch)
# =============================================================================
"""MSDNet's GPU kernels, in Triton (which comes with PyTorch on Linux with
CUDA; nothing to compile): the convolutions of one hidden layer, read straight
from the network's buffer with the symmetric-reflection padding computed in
the indices, as ``msd_pytorch``'s CUDA kernels do. No padded copy and no
workspace, and each layer's convolution, bias and ReLU are one kernel.

- ``forward``: a layer's ``width`` outputs, from all the channels before it;
- ``grad_weight``: the gradient of a layer's weights;
- the gradient of a layer's input is a transposed convolution (``torch``),
  then the reflection folded back (``fold_reflection``).

``available(x)``: whether these kernels can run on ``x`` (a CUDA tensor and
Triton installed). ``MSDNet`` uses them then, and plain torch otherwise."""

from __future__ import annotations

import torch

try:  # comes with PyTorch's CUDA builds on Linux
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on the installation
    triton = None

#: Pixels per Triton program.
BLOCK = 1024


def available(x: torch.Tensor) -> bool:
    """Whether the Triton kernels can run on ``x``."""
    return triton is not None and x.is_cuda and x.dtype == torch.float32


if triton is not None:

    @triton.jit
    def _reflect(i, n):
        # symmetric reflection: -1 is 0, n is n - 1 (as msd_pytorch's kernels)
        i = tl.where(i < 0, -i - 1, i)
        i = tl.where(i >= n, 2 * n - i - 1, i)
        return tl.minimum(tl.maximum(i, 0), n - 1)

    @triton.jit
    def _forward_kernel(
        buffer,  # (B, C_total, H, W), contiguous
        weight,  # (width, n_in, 3, 3), contiguous
        bias,  # (width,)
        n_in,
        width,
        C_total,
        H,
        W,
        dilation,
        BLOCK: tl.constexpr,
    ):
        b_k = tl.program_id(0)
        b = (b_k // width).to(tl.int64)  # offsets can pass 2**31
        k = b_k % width
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < H * W
        h = offs // W
        w = offs % W
        # the nine reflected neighbours, the same for every channel
        h0 = _reflect(h - dilation, H) * W
        h1 = h * W
        h2 = _reflect(h + dilation, H) * W
        w0 = _reflect(w - dilation, W)
        w2 = _reflect(w + dilation, W)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        image = buffer + b * C_total * H * W
        kernel = weight + k * n_in * 9
        for c in range(0, n_in):
            x = image + c * H * W
            f = kernel + c * 9
            acc += tl.load(f + 0) * tl.load(x + h0 + w0, mask=mask, other=0.0)
            acc += tl.load(f + 1) * tl.load(x + h0 + w, mask=mask, other=0.0)
            acc += tl.load(f + 2) * tl.load(x + h0 + w2, mask=mask, other=0.0)
            acc += tl.load(f + 3) * tl.load(x + h1 + w0, mask=mask, other=0.0)
            acc += tl.load(f + 4) * tl.load(x + h1 + w, mask=mask, other=0.0)
            acc += tl.load(f + 5) * tl.load(x + h1 + w2, mask=mask, other=0.0)
            acc += tl.load(f + 6) * tl.load(x + h2 + w0, mask=mask, other=0.0)
            acc += tl.load(f + 7) * tl.load(x + h2 + w, mask=mask, other=0.0)
            acc += tl.load(f + 8) * tl.load(x + h2 + w2, mask=mask, other=0.0)
        acc = tl.maximum(acc + tl.load(bias + k), 0.0)
        tl.store(image + (n_in + k) * H * W + offs, acc, mask=mask)

    @triton.jit
    def _grad_weight_kernel(
        buffer,  # (B, C_total, H, W): the layer's input is channels [0, n_in)
        grad,  # (B, width, H, W): the gradient of the layer's convolution
        grad_weight,  # (width, n_in, 3, 3), zeros, accumulated into
        n_in,
        width,
        C_total,
        H,
        W,
        n_pixels,  # B * H * W
        dilation,
        BLOCK: tl.constexpr,
    ):
        c = tl.program_id(0)
        offs = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_pixels
        b = offs // (H * W)
        pixel = offs % (H * W)
        h = pixel // W
        w = pixel % W
        h0 = _reflect(h - dilation, H) * W
        h1 = h * W
        h2 = _reflect(h + dilation, H) * W
        w0 = _reflect(w - dilation, W)
        w2 = _reflect(w + dilation, W)
        x = buffer + (b * C_total + c) * H * W
        x0 = tl.load(x + h0 + w0, mask=mask, other=0.0)
        x1 = tl.load(x + h0 + w, mask=mask, other=0.0)
        x2 = tl.load(x + h0 + w2, mask=mask, other=0.0)
        x3 = tl.load(x + h1 + w0, mask=mask, other=0.0)
        x4 = tl.load(x + h1 + w, mask=mask, other=0.0)
        x5 = tl.load(x + h1 + w2, mask=mask, other=0.0)
        x6 = tl.load(x + h2 + w0, mask=mask, other=0.0)
        x7 = tl.load(x + h2 + w, mask=mask, other=0.0)
        x8 = tl.load(x + h2 + w2, mask=mask, other=0.0)
        for k in range(0, width):
            g = tl.load(grad + (b * width + k) * H * W + pixel, mask=mask, other=0.0)
            out = grad_weight + (k * n_in + c) * 9
            tl.atomic_add(out + 0, tl.sum(g * x0, axis=0))
            tl.atomic_add(out + 1, tl.sum(g * x1, axis=0))
            tl.atomic_add(out + 2, tl.sum(g * x2, axis=0))
            tl.atomic_add(out + 3, tl.sum(g * x3, axis=0))
            tl.atomic_add(out + 4, tl.sum(g * x4, axis=0))
            tl.atomic_add(out + 5, tl.sum(g * x5, axis=0))
            tl.atomic_add(out + 6, tl.sum(g * x6, axis=0))
            tl.atomic_add(out + 7, tl.sum(g * x7, axis=0))
            tl.atomic_add(out + 8, tl.sum(g * x8, axis=0))


def forward(buffer, n_in, weight, bias, dilation):
    """Write the layer of ``weight`` (width, n_in, 3, 3) and ``bias``
    (width,), relu(conv(channels [0, n_in)) + bias), into the channels
    [n_in, n_in + width) of ``buffer`` (B, C_total, H, W)."""
    B, C_total, H, W = buffer.shape
    width = weight.shape[0]
    grid = (B * width, triton.cdiv(H * W, BLOCK))
    _forward_kernel[grid](
        buffer,
        weight.contiguous(),
        bias.contiguous(),
        n_in,
        width,
        C_total,
        H,
        W,
        dilation,
        BLOCK=BLOCK,
    )


def grad_weight(buffer, n_in, grad, dilation, weight_shape):
    """The gradient of a layer's weights (``weight_shape``), from its input
    (channels [0, n_in) of ``buffer``) and the gradient ``grad`` (B, width, H,
    W) of its convolution's output."""
    B, C_total, H, W = buffer.shape
    width = grad.shape[1]
    result = torch.zeros(weight_shape, device=buffer.device, dtype=torch.float32)
    grid = (n_in, triton.cdiv(B * H * W, BLOCK))
    _grad_weight_kernel[grid](
        buffer,
        grad.contiguous(),
        result,
        n_in,
        width,
        C_total,
        H,
        W,
        B * H * W,
        dilation,
        BLOCK=BLOCK,
    )
    return result


def fold_reflection(padded: torch.Tensor, pad: int) -> torch.Tensor:
    """The gradient of an image from that of its padded copy (``padded``,
    ``pad`` pixels more on each side of H and W, padded by symmetric
    reflection): each padded pixel's gradient added to the pixel it copies."""
    for dim in (2, 3):
        n = padded.shape[dim] - 2 * pad
        inner = padded.narrow(dim, pad, n).clone()
        # padded index pad - 1 - j copies pixel j, index pad + n + j pixel n-1-j
        inner.narrow(dim, 0, pad).add_(padded.narrow(dim, 0, pad).flip(dim))
        inner.narrow(dim, n - pad, pad).add_(padded.narrow(dim, pad + n, pad).flip(dim))
        padded = inner
    return padded


def grad_input(grad, weight, dilation):
    """The gradient of a layer's input (B, n_in, H, W), from the gradient
    ``grad`` (B, width, H, W) of its convolution's output: the transposed
    convolution onto the padded input, then the padding folded back."""
    padded = torch.nn.functional.conv_transpose2d(grad, weight, dilation=dilation)
    return fold_reflection(padded, dilation)
