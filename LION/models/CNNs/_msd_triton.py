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
- ``grad_weight``: the gradient of a layer's weights, each program summing
  its share of the pixels before adding it;
- ``add_grad_input``: the gradient of a layer's input (the transposed
  convolution, the reflection in its indices), added to the gradients' buffer
  in place.

``available(x)``: whether these kernels can run on ``x`` (a CUDA tensor and
Triton installed). ``MSDNet`` uses them then, and plain torch otherwise."""

from __future__ import annotations

import warnings

import torch

try:  # comes with PyTorch's CUDA builds on Linux
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on the installation
    triton = None

#: Pixels per Triton program.
BLOCK = 1024
#: Programs per input channel for the weights' gradient: each sums its share
#: of the pixels, then adds it (few atomic additions).
PROGRAMS_PER_CHANNEL = 64


_working = None  # whether Triton can launch kernels here (checked once)


def available(x: torch.Tensor) -> bool:
    """Whether the Triton kernels can run on ``x``: a float32 CUDA tensor, and
    Triton working. The first time, Triton compiles a helper with CUDA's
    headers; without them (``cuda.h``: conda's ``cuda-cudart-dev``), a warning
    says so, and MSDNet runs in plain torch (much slower)."""
    global _working
    if triton is None or not x.is_cuda or x.dtype != torch.float32:
        return False
    if _working is None:
        try:
            triton.runtime.driver.active.get_current_device()
            _working = True
        except Exception as error:  # e.g. cuda.h missing: a CalledProcessError
            _working = False
            warnings.warn(
                f"Triton can't run here ({type(error).__name__}: {error}), so "
                "MSDNet runs in plain torch, much slower on a GPU. Triton needs "
                "CUDA's headers: install conda's cuda-cudart-dev (env_base.yml).",
                stacklevel=3,
            )
    return _working


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
        PROGRAMS,  # programs per channel, each summing a share of the pixels
        BLOCK: tl.constexpr,
    ):
        # each program sums its pixels (every PROGRAMS-th block) before adding
        # them to the gradient: PROGRAMS atomic additions per weight
        c = tl.program_id(0)
        first = tl.program_id(1).to(tl.int64) * BLOCK
        for k in range(0, width):
            a0 = tl.zeros((BLOCK,), dtype=tl.float32)
            a1 = tl.zeros((BLOCK,), dtype=tl.float32)
            a2 = tl.zeros((BLOCK,), dtype=tl.float32)
            a3 = tl.zeros((BLOCK,), dtype=tl.float32)
            a4 = tl.zeros((BLOCK,), dtype=tl.float32)
            a5 = tl.zeros((BLOCK,), dtype=tl.float32)
            a6 = tl.zeros((BLOCK,), dtype=tl.float32)
            a7 = tl.zeros((BLOCK,), dtype=tl.float32)
            a8 = tl.zeros((BLOCK,), dtype=tl.float32)
            for start in range(first, n_pixels, PROGRAMS * BLOCK):
                offs = start + tl.arange(0, BLOCK)
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
                g = tl.load(
                    grad + (b * width + k) * H * W + pixel, mask=mask, other=0.0
                )
                a0 += g * tl.load(x + h0 + w0, mask=mask, other=0.0)
                a1 += g * tl.load(x + h0 + w, mask=mask, other=0.0)
                a2 += g * tl.load(x + h0 + w2, mask=mask, other=0.0)
                a3 += g * tl.load(x + h1 + w0, mask=mask, other=0.0)
                a4 += g * tl.load(x + h1 + w, mask=mask, other=0.0)
                a5 += g * tl.load(x + h1 + w2, mask=mask, other=0.0)
                a6 += g * tl.load(x + h2 + w0, mask=mask, other=0.0)
                a7 += g * tl.load(x + h2 + w, mask=mask, other=0.0)
                a8 += g * tl.load(x + h2 + w2, mask=mask, other=0.0)
            out = grad_weight + (k * n_in + c) * 9
            tl.atomic_add(out + 0, tl.sum(a0, axis=0))
            tl.atomic_add(out + 1, tl.sum(a1, axis=0))
            tl.atomic_add(out + 2, tl.sum(a2, axis=0))
            tl.atomic_add(out + 3, tl.sum(a3, axis=0))
            tl.atomic_add(out + 4, tl.sum(a4, axis=0))
            tl.atomic_add(out + 5, tl.sum(a5, axis=0))
            tl.atomic_add(out + 6, tl.sum(a6, axis=0))
            tl.atomic_add(out + 7, tl.sum(a7, axis=0))
            tl.atomic_add(out + 8, tl.sum(a8, axis=0))

    @triton.jit
    def _gather(g, r1, m1, r2, m2, c1, n1, c2, n2, mask):
        # the gradient at the (up to) 2 x 2 output pixels that read this pixel
        # through one tap: rows r1, r2 (valid where m1, m2), columns c1, c2
        v = tl.load(g + r1 + c1, mask=mask & m1 & n1, other=0.0)
        v += tl.load(g + r1 + c2, mask=mask & m1 & n2, other=0.0)
        v += tl.load(g + r2 + c1, mask=mask & m2 & n1, other=0.0)
        v += tl.load(g + r2 + c2, mask=mask & m2 & n2, other=0.0)
        return v

    @triton.jit
    def _grad_input_kernel(
        grad,  # (B, width, H, W): the gradient of the layer's convolution
        weight,  # (width, n_in, 3, 3)
        gradients,  # (B, C_total, H, W): channels [0, n_in) are added to
        n_in,
        width,
        C_total,
        H,
        W,
        dilation,
        BLOCK: tl.constexpr,
    ):
        # The input pixel h is read by output pixel h' through the tap of shift
        # s (-d, 0, d) where reflect(h' + s) == h: h' = h - s, and near the
        # edges also a reflected one (h' = d - 1 - h for s = -d, h' = 2H - 1 -
        # d - h for s = d). The same along W. The gradient gathered for each
        # tap is the same for every input channel: gathered once, then each
        # channel's gradient is 9 multiplications.
        b = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < H * W
        h = offs // W
        w = offs % W
        d = dilation
        never = h < 0
        always = h >= 0
        # rows: tap 0 (s = -d), tap 1 (s = 0), tap 2 (s = d)
        r0a = (h + d) * W
        m0a = h + d < H
        r0b = (d - 1 - h) * W
        m0b = (h < d) & (d - 1 - h < H)
        r1 = h * W
        r2a = (h - d) * W
        m2a = h >= d
        r2b = (2 * H - 1 - d - h) * W
        m2b = (h >= H - d) & (2 * H - 1 - d - h >= 0)
        # columns
        c0a = w + d
        n0a = w + d < W
        c0b = d - 1 - w
        n0b = (w < d) & (d - 1 - w < W)
        c2a = w - d
        n2a = w >= d
        c2b = 2 * W - 1 - d - w
        n2b = (w >= W - d) & (2 * W - 1 - d - w >= 0)
        image = gradients + b * C_total * H * W + offs
        for k in range(0, width):
            g = grad + (b * width + k) * H * W
            g0 = _gather(g, r0a, m0a, r0b, m0b, c0a, n0a, c0b, n0b, mask)
            g1 = _gather(g, r0a, m0a, r0b, m0b, w, always, w, never, mask)
            g2 = _gather(g, r0a, m0a, r0b, m0b, c2a, n2a, c2b, n2b, mask)
            g3 = _gather(g, r1, always, r1, never, c0a, n0a, c0b, n0b, mask)
            g4 = _gather(g, r1, always, r1, never, w, always, w, never, mask)
            g5 = _gather(g, r1, always, r1, never, c2a, n2a, c2b, n2b, mask)
            g6 = _gather(g, r2a, m2a, r2b, m2b, c0a, n0a, c0b, n0b, mask)
            g7 = _gather(g, r2a, m2a, r2b, m2b, w, always, w, never, mask)
            g8 = _gather(g, r2a, m2a, r2b, m2b, c2a, n2a, c2b, n2b, mask)
            kernel = weight + k * n_in * 9
            for c in range(0, n_in):
                f = kernel + c * 9
                acc = tl.load(f + 0) * g0
                acc += tl.load(f + 1) * g1
                acc += tl.load(f + 2) * g2
                acc += tl.load(f + 3) * g3
                acc += tl.load(f + 4) * g4
                acc += tl.load(f + 5) * g5
                acc += tl.load(f + 6) * g6
                acc += tl.load(f + 7) * g7
                acc += tl.load(f + 8) * g8
                out = image + c * H * W
                tl.store(out, tl.load(out, mask=mask, other=0.0) + acc, mask=mask)


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
    programs = min(triton.cdiv(B * H * W, BLOCK), PROGRAMS_PER_CHANNEL)
    _grad_weight_kernel[(n_in, programs)](
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
        programs,
        BLOCK=BLOCK,
    )
    return result


def add_grad_input(gradients, n_in, grad, weight, dilation):
    """Add the gradient of a layer's input to channels [0, n_in) of
    ``gradients`` (B, C_total, H, W), from the gradient ``grad`` (B, width, H,
    W) of its convolution's output (the transposed convolution, with the
    reflection, read straight from ``grad``)."""
    B, C_total, H, W = gradients.shape
    width = grad.shape[1]
    _grad_input_kernel[(B, triton.cdiv(H * W, BLOCK))](
        grad.contiguous(),
        weight.contiguous(),
        gradients,
        n_in,
        width,
        C_total,
        H,
        W,
        dilation,
        BLOCK=BLOCK,
    )
