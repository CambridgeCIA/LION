# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Charlie Shoebridge, Ander Biguri
# Modifications: rewritten after msd_pytorch (Allard A. Hendriksen, GPL-3,
#                https://github.com/ahendriksen/msd_pytorch), in plain torch
# =============================================================================
"""The Mixed-Scale Dense network (Pelt and Sethian, 2018), as ``msd_pytorch``
(Hendriksen, 2019) implements it, in plain PyTorch.

``msd_pytorch`` is the reference implementation, but it is no longer
maintained and needs compiling (CUDA kernels, PyTorch 1.8, Python up to 3.9).
This is the same network, with the same initialisation and padding, and its
same memory use: the layers write their outputs into one buffer, and the
gradients are computed layer by layer, so memory grows with the depth, not its
square (each layer's input is all the layers before it, and storing them for
autograd one by one would grow with the square)."""

from __future__ import annotations

from math import sqrt
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType
from LION.utils.citation import citation


def _symmetric_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
    """``x`` (B, C, H, W) padded by ``pad`` pixels on each side of H and W by
    reflection that repeats the edge (index -1 is 0, -2 is 1), as
    ``msd_pytorch``'s kernels pad."""
    for dim in (2, 3):
        n = x.shape[dim]
        index = torch.arange(-pad, n + pad, device=x.device)
        index = torch.where(index < 0, -index - 1, index)
        index = torch.where(index >= n, 2 * n - index - 1, index)
        x = x.index_select(dim, index.clamp(0, n - 1))
    return x


def _conv(x, weight, bias, dilation):
    """A 3x3 convolution of ``x`` with ``dilation``, padded by symmetric
    reflection to keep its size."""
    return F.conv2d(_symmetric_pad(x, dilation), weight, bias, dilation=dilation)


class _MSDBlock(torch.autograd.Function):
    """The hidden layers: each a dilated 3x3 convolution (with a bias) of all
    the channels before it, then a ReLU, written into one buffer (B, C + depth
    * width, H, W): the input, then each layer's ``width`` channels. Backward
    goes through the layers in reverse, recomputing each layer's convolution to
    get its gradients, so only the buffer is kept."""

    @staticmethod
    def forward(ctx, x, dilations, bias, *weights):
        """The buffer: ``x``, then every hidden layer's output."""
        width = weights[0].shape[0]
        result = x.new_empty(
            x.shape[0], x.shape[1] + width * len(weights), *x.shape[2:]
        )
        result[:, : x.shape[1]] = x
        start = x.shape[1]
        for i, (weight, dilation) in enumerate(zip(weights, dilations)):
            sub_bias = bias[i * width : (i + 1) * width]
            result[:, start : start + width] = F.relu(
                _conv(result[:, :start], weight, sub_bias, dilation)
            )
            start += width
        ctx.dilations = dilations
        ctx.save_for_backward(result, bias, *weights)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """The gradients of ``x``, the biases and the weights, layer by
        layer, from the last."""
        result, bias, *weights = ctx.saved_tensors
        width = weights[0].shape[0]
        gradients = grad_output.clone()
        grad_bias = torch.zeros_like(bias)
        grad_weights = [None] * len(weights)
        end = result.shape[1]
        for i in reversed(range(len(weights))):
            start = end - width
            # through the ReLU: its output is positive where its input was
            grad_layer = gradients[:, start:end] * (result[:, start:end] > 0)
            grad_bias[i * width : (i + 1) * width] = grad_layer.sum(dim=(0, 2, 3))
            with torch.enable_grad():
                layer_input = result[:, :start].detach().requires_grad_()
                weight = weights[i].detach().requires_grad_()
                output = _conv(layer_input, weight, None, ctx.dilations[i])
                grad_input, grad_weights[i] = torch.autograd.grad(
                    output, (layer_input, weight), grad_layer
                )
            gradients[:, :start] += grad_input
            end = start
        n_in = weights[0].shape[1]
        return (gradients[:, :n_in], None, grad_bias, *grad_weights)


class MSDNet(LIONmodel):
    """Mixed-Scale Dense network (Pelt and Sethian, 2018), as ``msd_pytorch``
    (Hendriksen, 2019): a deep, narrow network of dilated 3x3 convolutions,
    each layer seeing the input and all the layers before it, ending in a 1x1
    convolution of all of them. Takes images (e.g. a reconstruction), or with
    ``model_input_type`` SINOGRAM, the measurements.

    Each hidden layer is ``width`` convolutions with a bias and a ReLU, of
    dilation ``dilations[i % len(dilations)]``, padded by symmetric reflection;
    the output layer is linear (a 1x1 convolution with a bias, no activation),
    for regression. The hidden convolutions start as in the paper (normal, std
    sqrt(2 / n), n the inputs of a layer at full depth), the biases and the
    output layer at zero (so the untrained network gives 0).

    Memory grows with the depth (the module's description says how). Backward
    can't be differentiated again (e.g. a Jacobian penalty on the network,
    ``LION.losses.jacobian_penalty``, needs it).

    ``model_parameters`` (``default_parameters()``):

    - ``in_channels``, ``out_channels``: of the input and output images (1,
      1);
    - ``width``: the channels each layer adds (1);
    - ``depth``: the number of hidden layers (100);
    - ``dilations``: repeated over the layers (1 to 10)."""

    def __init__(
        self,
        model_parameters: Optional[LIONModelParameter] = None,
    ) -> None:
        """The network for ``model_parameters`` (``default_parameters()`` if
        None)."""
        super().__init__(model_parameters)
        p = self.model_parameters
        if p.depth < 1 or p.width < 1:
            raise ValueError(
                f"depth and width must be at least 1, got {p.depth} and {p.width}"
            )
        self.dilations = [
            int(p.dilations[i % len(p.dilations)]) for i in range(p.depth)
        ]
        self.bias = nn.Parameter(torch.zeros(p.depth * p.width))
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.empty(p.width, p.in_channels + p.width * i, 3, 3))
                for i in range(p.depth)
            ]
        )
        self.final_layer = nn.Conv2d(
            p.in_channels + p.width * p.depth, p.out_channels, 1
        )
        self.reset_parameters()

    def reset_parameters(self):
        """The initial weights (see the class): as ``msd_pytorch``'s."""
        p = self.model_parameters
        n = 9 * (p.in_channels + p.width * (p.depth - 1)) + p.out_channels
        with torch.no_grad():
            for weight in self.weights:
                weight.normal_(0, sqrt(2 / n))
            self.bias.zero_()
            self.final_layer.weight.zero_()
            self.final_layer.bias.zero_()

    @staticmethod
    def default_parameters() -> LIONModelParameter:
        """The paper's network: 100 layers of width 1, dilations 1 to 10 (see
        the class)."""
        params = LIONModelParameter()
        params.model_input_type = ModelInputType.IMAGE
        params.in_channels = 1
        params.out_channels = 1
        params.width = 1
        params.depth = 100
        params.dilations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        return params

    def forward(self, x):
        """The network's output for the images ``x`` (B, in_channels, H, W)."""
        if (C := x.shape[1]) != self.model_parameters.in_channels:
            raise ValueError(
                f"Expected {self.model_parameters.in_channels} input channels, got {C}"
            )
        layers = _MSDBlock.apply(x, self.dilations, self.bias, *self.weights)
        return self.final_layer(layers)

    @staticmethod
    def cite(cite_format="bib"):
        """Where MSDNet comes from: its citation, BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @article{pelt2018mixed,
            title={A mixed-scale dense convolutional neural network for image analysis},
            author={Pelt, Dani{\\"e}l M and Sethian, James A},
            journal={Proceedings of the National Academy of Sciences},
            volume={115},
            number={2},
            pages={254--259},
            year={2018},
            publisher={National Acad Sciences}
            }
            """,
            mla="""
            Pelt, Daniël M., and James A. Sethian.
            "A mixed-scale dense convolutional neural network for image analysis."
            Proceedings of the National Academy of Sciences
            115.2 (2018): 254-259.
            """,
        )
