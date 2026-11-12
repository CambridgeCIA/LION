# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Erich Kobler (the TDV regulariser, https://github.com/VLOGroup/tdv,
#           MIT licence, below), Zakhar Shumaylov, Ander Biguri
# Modifications: the variational network with a CT data term (Zakhar
#                Shumaylov, 2024); one file, LION's operators and parameters
# =============================================================================
#
# The TDV regulariser (``_Conv``, ``_ConvScale``, ``_StudentT``,
# ``_MicroBlock``, ``_MacroBlock``, ``_TDVRegulariser``) is adapted from
# https://github.com/VLOGroup/tdv, under the MIT licence:
#
# Copyright (c) 2018 Vision, Learning and Optimization Group
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================
"""Total Deep Variation (Kobler, Effland, Kunisch and Pock, 2020): a learned
regulariser, and the variational network that reconstructs with it."""

from __future__ import annotations

from math import sqrt
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType
from LION.utils.citation import citation
from LION.utils.math import power_method
from LION.utils.parameter import OperatorParameters


def _symmetric_index(n: int, pad: int, device) -> torch.Tensor:
    """The indices that pad a dimension of size ``n`` by ``pad`` on each side
    by symmetric reflection (index -1 is 0, -2 is 1)."""
    index = torch.arange(-pad, n + pad, device=device)
    index = torch.where(index < 0, -index - 1, index)
    return torch.where(index >= n, 2 * n - index - 1, index)


def _pad(x: torch.Tensor, pad: int, padding: str) -> torch.Tensor:
    """``x`` (B, C, H, W) padded by ``pad`` pixels on each side: "symmetric"
    (the edge repeated) or "reflect" (torch's, the edge not repeated)."""
    if padding == "reflect":
        return F.pad(x, (pad, pad, pad, pad), mode="reflect")
    for dim in (2, 3):
        x = x.index_select(dim, _symmetric_index(x.shape[dim], pad, x.device))
    return x


def _pad_transpose(x: torch.Tensor, pad: int, padding: str) -> torch.Tensor:
    """The transpose of ``_pad``: "symmetric" adds the padded pixels back to
    those they copy; "reflect" crops them (an approximation of its
    transpose)."""
    if padding == "reflect":
        return x[:, :, pad:-pad, pad:-pad]
    for dim in (2, 3):
        n = x.shape[dim] - 2 * pad
        shape = list(x.shape)
        shape[dim] = n
        x = x.new_zeros(shape).index_add_(dim, _symmetric_index(n, pad, x.device), x)
    return x


class _Conv(nn.Module):
    """A convolution without bias, padded to keep the size (``padding``), with
    its transpose (``backward``). The weights start normal, std 1 / sqrt(fan
    in); ``zero_mean`` and ``bound_norm`` project them once, as they start
    (each filter's mean 0 and its norm 1)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        zero_mean=False,
        bound_norm=False,
        padding="symmetric",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.normal_(self.weight, 0.0, sqrt(1 / (in_channels * kernel_size**2)))
        with torch.no_grad():
            if zero_mean:
                self.weight -= self.weight.mean(dim=(1, 2, 3), keepdim=True)
            if bound_norm:
                self.weight /= (
                    self.weight.pow(2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                )

    def get_weight(self):
        """The convolution's kernel."""
        return self.weight

    def forward(self, x):
        """The convolution of ``x``."""
        weight = self.get_weight()
        pad = weight.shape[-1] // 2
        if pad > 0:
            x = _pad(x, pad, self.padding)
        return F.conv2d(x, weight, stride=self.stride)

    def backward(self, x, output_shape=None):
        """The transpose of the convolution, applied to ``x``; of the size
        ``output_shape`` (B, C, H, W) for strided ones."""
        weight = self.get_weight()
        output_padding = 0
        if output_shape is not None:
            output_padding = (
                output_shape[2] - ((x.shape[2] - 1) * self.stride + 1),
                output_shape[3] - ((x.shape[3] - 1) * self.stride + 1),
            )
        x = F.conv_transpose2d(
            x, weight, stride=self.stride, output_padding=output_padding
        )
        pad = weight.shape[-1] // 2
        if pad > 0:
            x = _pad_transpose(x, pad, self.padding)
        return x


class _ConvScale(_Conv):
    """A strided (downsampling) convolution, its kernel blurred by a binomial
    filter; ``transpose`` makes it the upsampling (its transpose)."""

    def __init__(self, in_channels, out_channels, padding, transpose=False):
        super().__init__(in_channels, out_channels, 3, stride=2, padding=padding)
        self.transpose = transpose
        blur = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        blur = torch.outer(blur, blur)
        self.register_buffer("blur", (blur / blur.sum())[None, None])

    def get_weight(self):
        """The kernel, blurred: 7x7."""
        weight = self.weight.reshape(-1, 1, self.kernel_size, self.kernel_size)
        weight = F.conv2d(weight, self.blur, padding=4)
        return weight.reshape(
            self.out_channels,
            self.in_channels,
            self.kernel_size + 2 * self.stride,
            self.kernel_size + 2 * self.stride,
        )

    def forward(self, x, output_shape=None):
        """Downsamples ``x`` (upsamples, to ``output_shape``, if
        ``transpose``)."""
        if self.transpose:
            return super().backward(x, output_shape)
        return super().forward(x)

    def backward(self, x, output_shape=None):
        """The transpose of ``forward``."""
        if self.transpose:
            return super().forward(x)
        return super().backward(x, output_shape)


class _StudentTFunction(torch.autograd.Function):
    """log(1 + x²) / 2 and its derivative x / (1 + x²), both
    differentiable."""

    @staticmethod
    def forward(ctx, x):
        """The function and its derivative at ``x``."""
        ctx.save_for_backward(x)
        d = 1 + x**2
        return torch.log(d) / 2, x / d

    @staticmethod
    def backward(ctx, grad_value, grad_derivative):
        """The gradient with respect to ``x``."""
        (x,) = ctx.saved_tensors
        d = 1 + x**2
        return x / d * grad_value + (1 - x**2) / d**2 * grad_derivative


class _MicroBlock(nn.Module):
    """x + K2 φ(K1 x), φ the Student-t potential; ``backward`` is its
    Jacobian's transpose, applied to a gradient."""

    def __init__(self, features, padding):
        super().__init__()
        self.conv1 = _Conv(features, features, 3, padding=padding)
        self.conv2 = _Conv(features, features, 3, padding=padding)
        self.derivative = None  # φ'(K1 x), kept for backward

    def forward(self, x):
        """The block's output."""
        value, self.derivative = _StudentTFunction.apply(self.conv1(x))
        return x + self.conv2(value)

    def backward(self, grad):
        """The transpose of the block's Jacobian at its last input, applied to
        ``grad``."""
        return grad + self.conv1.backward(self.derivative * self.conv2.backward(grad))


class _MacroBlock(nn.Module):
    """A U-Net of micro blocks over ``scales`` scales; ``backward`` is its
    Jacobian's transpose."""

    def __init__(self, features, scales, multiplier, padding):
        super().__init__()
        self.scales = scales
        widths = [features * multiplier**i for i in range(scales)]
        self.micro = nn.ModuleList(
            [
                nn.ModuleList(
                    [_MicroBlock(w, padding), _MicroBlock(w, padding)]
                    if i < scales - 1
                    else [_MicroBlock(w, padding)]
                )
                for i, w in enumerate(widths)
            ]
        )
        self.down = nn.ModuleList(
            [_ConvScale(widths[i], widths[i + 1], padding) for i in range(scales - 1)]
        )
        self.up = nn.ModuleList(
            [
                _ConvScale(widths[i], widths[i + 1], padding, transpose=True)
                for i in range(scales - 1)
            ]
        )

    def forward(self, x):
        """The block's outputs at each scale, from ``x`` (a list, one per
        scale; None where there is nothing yet)."""
        for i in range(self.scales - 1):
            x[i] = self.micro[i][0](x[i])
            down = self.down[i](x[i])
            x[i + 1] = down if x[i + 1] is None else x[i + 1] + down
        x[-1] = self.micro[-1][0](x[-1])
        for i in reversed(range(self.scales - 1)):
            x[i] = x[i] + self.up[i](x[i + 1], x[i].shape)
            x[i] = self.micro[i][1](x[i])
        return x

    def backward(self, grad):
        """The transpose of the block's Jacobian, applied to ``grad`` (a
        list, one per scale)."""
        for i in range(self.scales - 1):
            grad[i] = self.micro[i][1].backward(grad[i])
            up = self.up[i].backward(grad[i])
            grad[i + 1] = up if grad[i + 1] is None else grad[i + 1] + up
        grad[-1] = self.micro[-1][0].backward(grad[-1])
        for i in reversed(range(self.scales - 1)):
            grad[i] = grad[i] + self.down[i].backward(grad[i + 1], grad[i].shape)
            grad[i] = self.micro[i][0].backward(grad[i])
        return grad


class _TDVRegulariser(nn.Module):
    """R(x) = Σ w · N(x) / features, N a network of ``macro_blocks`` macro
    blocks; ``grad`` is ∇R(x), computed through the network's transposed
    blocks."""

    def __init__(self, features, scales, macro_blocks, multiplier, padding):
        super().__init__()
        self.features = features
        self.scales = scales
        self.K1 = _Conv(
            1, features, 3, zero_mean=True, bound_norm=True, padding=padding
        )
        self.macro = nn.ModuleList(
            [
                _MacroBlock(features, scales, multiplier, padding)
                for _ in range(macro_blocks)
            ]
        )
        self.KN = _Conv(features, 1, 1, padding=padding)

    def _transformation(self, x):
        x = [self.K1(x)] + [None] * (self.scales - 1)
        for block in self.macro:
            x = block(x)
        return self.KN(x[0])

    def energy(self, x):
        """R at each pixel of ``x`` (B, 1, H, W)."""
        return self._transformation(x) / self.features

    def grad(self, x):
        """∇R(x), for ``x`` (B, 1, H, W)."""
        out = self._transformation(x)
        grad = [self.KN.backward(torch.ones_like(out) / self.features)]
        grad += [None] * (self.scales - 1)
        for block in reversed(self.macro):
            grad = block.backward(grad)
        return self.K1.backward(grad[0])


class TDV(LIONmodel):
    """Total Deep Variation (Kobler et al., 2020), as a variational network:
    from the initial reconstruction (the operator's default, FDK for CT, its
    negative values set to 0), ``steps`` steps of gradient descent on

    .. math::

        E(x) = R(x) + \\lambda \\lVert A x - y \\rVert_2^2 / (2 \\lVert A \\rVert^2),

    x ← x - (T / steps) ∇R(x) - (λ / steps) Aᵀ(Ax - y) / ‖A‖², with R the
    TDV regulariser, a learned network. The step T and the weight λ are learned
    too, unconstrained. Trained end to end (e.g. by ``SupervisedSolver``);
    works with any operator with ``autograd()``. Takes the measurements.

    The regulariser is Kobler's (VLOGroup's code); the data term for CT is
    Zakhar Shumaylov's, as LION had it in 2024. Training with LION's solvers
    doesn't keep the paper's constraints on the weights (Kobler's training
    projected them after each step); here they are projected once, as they
    start.

    ``model_parameters`` (``default_parameters()``):

    - ``steps``: the gradient steps (10);
    - ``features``: the channels of the regulariser's network (16);
    - ``scales``: the scales of each macro block (5);
    - ``macro_blocks``: the macro blocks (5);
    - ``multiplier``: the channels' factor from one scale to the next (1);
    - ``stop_time``: T, as it starts (10);
    - ``data_weight``: λ, as it starts (5);
    - ``padding``: of the convolutions, "symmetric" (the paper's) or "reflect"
      (torch's, its transpose approximated by cropping: as LION had it in
      2024)."""

    def __init__(
        self,
        operator_params: OperatorParameters,
        model_parameters: Optional[LIONModelParameter] = None,
    ):
        """The network for the operator of ``operator_params``, with
        ``model_parameters`` (``default_parameters()`` if None)."""
        if operator_params is None:
            raise ValueError(
                "operator_params required: the parameters of the operator (for CT, the geometry)"
            )
        super().__init__(model_parameters, operator_params)
        p = self.model_parameters
        if p.padding not in ("symmetric", "reflect"):
            raise ValueError(
                f'padding must be "symmetric" or "reflect", got {p.padding!r}'
            )
        self._make_operator(autograd=True)
        self.op_norm = float(power_method(self.op))
        self.regulariser = _TDVRegulariser(
            p.features, p.scales, p.macro_blocks, p.multiplier, p.padding
        )
        self.stop_time = nn.Parameter(torch.tensor(float(p.stop_time)))
        self.data_weight = nn.Parameter(torch.tensor(float(p.data_weight)))

    @staticmethod
    def default_parameters() -> LIONModelParameter:
        """The TDV of LION's 2024 benchmark (see the class): 10 steps, a
        regulariser of 5 macro blocks of 5 scales, 16 channels."""
        params = LIONModelParameter()
        params.model_input_type = ModelInputType.SINOGRAM
        params.steps = 10
        params.features = 16
        params.scales = 5
        params.macro_blocks = 5
        params.multiplier = 1
        params.stop_time = 10.0
        params.data_weight = 5.0
        params.padding = "symmetric"
        return params

    def forward(self, g):
        """The reconstructions of the measurements ``g`` (B, 1, ...), of the
        operator's range shape."""
        x = self.initial_reconstruction(g, self.op).clamp(min=0)
        steps = self.model_parameters.steps
        tau = self.stop_time / steps
        for _ in range(steps):
            if x.requires_grad:  # recomputed in backward, to save memory
                grad_R = checkpoint.checkpoint(
                    self.regulariser.grad, x, use_reentrant=False
                )
            else:
                grad_R = self.regulariser.grad(x)
            data_grad = self.AT(self.A(x) - g)
            x = (
                x
                - tau * grad_R
                - (self.data_weight / steps) * data_grad / self.op_norm**2
            )
        return x

    @staticmethod
    def cite(cite_format="bib"):
        """Where TDV comes from: its citation, BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @inproceedings{kobler2020total,
            title={Total deep variation for linear inverse problems},
            author={Kobler, Erich and Effland, Alexander and Kunisch, Karl and Pock, Thomas},
            booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
            pages={7549--7558},
            year={2020}
            }
            """,
            mla="""
            Kobler, Erich, Alexander Effland, Karl Kunisch, and Thomas Pock.
            "Total deep variation for linear inverse problems."
            Proceedings of the IEEE/CVF Conference on Computer Vision and
            Pattern Recognition (2020): 7549-7558.
            """,
        )
