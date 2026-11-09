# This file is part of LION library
# License : GPL-3
#
# Author  : Charlie Shoebridge
# Modifications: -
# =============================================================================

# Implementation of Mixed-Scale Dense Network described in:
# Daniël M. Pelt and James A. Sethian
# A mixed-scale dense convolutional neural network for image analysis
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType
from LION.utils.citation import citation


class MSD_Layer(nn.Module):
    """One layer of MSDNet: ``width`` 3x3 convolutions with their own
    dilations, each on all of the layer's input channels, each followed by
    batch normalisation and ``activation``. Its output is its input with the
    new channels appended."""

    def __init__(
        self,
        in_channels: int,
        width: int,
        dilations: int | list[int],
        activation: nn.Module,
    ) -> None:
        super().__init__()
        if isinstance(dilations, int):
            dilations = [dilations] * width

        self.in_channels = in_channels
        self.width = width
        self.convs = nn.ModuleList()
        for ch in range(width):
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=3,
                dilation=dilations[ch],
                padding="same",
                padding_mode="reflect",
                bias=False,
            )

            # conv.apply(self._initialize_conv_weights) # actually decreases performance
            self.convs.append(conv)

        self.activation = nn.Sequential(nn.BatchNorm2d(1), activation)

    def _initialize_conv_weights(self, layer: nn.Module):
        assert isinstance(layer, nn.Conv2d), "layer not Conv2D"
        n_c = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
        torch.nn.init.normal_(layer.weight, 0, np.sqrt(2 / n_c))

    def forward(self, x: torch.Tensor):  # x will be all of the previous channels to use
        """``x`` with the ``width`` new channels appended."""
        new_channels = [x]
        for i in range(self.width):
            new_channels.append(self.activation(self.convs[i](x)))

        x = torch.cat(new_channels, dim=1)

        return x


class MSDNet(LIONmodel):
    """Mixed-Scale Dense network (Pelt and Sethian, 2018): a deep, narrow
    network of dilated convolutions, every layer seeing the channels of all the
    layers before it, ending in a 1x1 convolution. Takes images (e.g. a
    reconstruction).

    ``model_parameters`` (``default_parameters()``):

    - ``in_channels``: the input image's channels (1);
    - ``width``: the channels each layer adds (1);
    - ``depth``: the number of layers (100);
    - ``dilations``: one per convolution, ``width * depth`` of them, layer by
      layer (1 to 10, repeated);
    - ``look_back_depth``: how many earlier layers each layer sees (-1: all);
    - ``final_look_back_depth``: how many layers the final 1x1 convolution sees
      (-1: all, and the input);
    - ``activation``: a ``torch.nn`` class name ("ReLU") or module."""

    def __init__(
        self,
        model_parameters: Optional[LIONModelParameter] = None,
    ) -> None:
        """The network for ``model_parameters`` (``default_parameters()`` if
        None)."""
        super().__init__(model_parameters)

        # total there should be width * depth distinct convolutions
        # so expect the same number of dilations to be given
        if (
            len(self.model_parameters.dilations)
            != self.model_parameters.width * self.model_parameters.depth
        ):
            raise ValueError(
                f"Expected {self.model_parameters.width*self.model_parameters.depth} dilation sizes to be given, instead given {len(self.model_parameters.dilations)}"
            )

        # look_back_depth should be no greater than number of hidden layers + 1
        if self.model_parameters.look_back_depth > self.model_parameters.depth + 1:
            raise ValueError(
                f"Lookback depth={self.model_parameters.look_back_depth} can not be bigger than total number of layers={self.model_parameters.depth + 1}"
            )

        self.layers = nn.ModuleList()
        first_layer = MSD_Layer(
            self.model_parameters.in_channels,
            self.model_parameters.width,
            self.model_parameters.dilations[: self.model_parameters.width],
            self._make_activation(),
        )
        self.layers.append(first_layer)
        for layer_idx in range(1, self.model_parameters.depth):
            self.layers.append(
                MSD_Layer(
                    self._count_channels_to_use(layer_idx),
                    self.model_parameters.width,
                    self.model_parameters.dilations[
                        layer_idx
                        * self.model_parameters.width : (layer_idx + 1)
                        * self.model_parameters.width
                    ],
                    self._make_activation(),
                )
            )

        for layer in self.layers:
            assert isinstance(layer, MSD_Layer)
            for conv in layer.convs:
                conv.apply(self._initialize_conv_weights)

        # final_look_back_depth should be no greater than number of hidden layers + 1
        if (
            self.model_parameters.final_look_back_depth
            > self.model_parameters.depth + 1
        ):
            raise ValueError(
                f"Final Lookback depth={self.model_parameters.final_look_back_depth} can not be bigger than total number of layers={self.model_parameters.depth + 1}"
            )

        if self.model_parameters.final_look_back_depth == -1:
            final_in_ch = (
                self.model_parameters.depth * self.model_parameters.width
                + self.model_parameters.in_channels
            )
        else:
            final_in_ch = (
                self.model_parameters.width
                * self.model_parameters.final_look_back_depth
            )

        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=final_in_ch,
                out_channels=1,
                kernel_size=1,
                padding="same",
                padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm2d(1),
            self._make_activation(),
        )

    def _make_activation(self) -> nn.Module:
        """A new activation module from model_parameters.activation, which can be
        a torch.nn class name (e.g. "ReLU") or an nn.Module."""
        activation = self.model_parameters.activation
        if isinstance(activation, str):
            if not isinstance(getattr(nn, activation, None), type):
                raise ValueError(f"torch.nn has no activation '{activation}'")
            return getattr(nn, activation)()
        return activation

    def _initialize_conv_weights(self, layer: nn.Module):
        assert isinstance(layer, nn.Conv2d), "layer not Conv2D"
        n_c = (
            np.prod(layer.weight.shape[2:])
            * (
                self.model_parameters.in_channels
                + self.model_parameters.width * (self.model_parameters.depth - 1)
            )
            + 1
        )
        torch.nn.init.normal_(layer.weight, 0, np.sqrt(2 / n_c))

    def _count_channels_to_use(self, layer) -> int:
        return (
            self.model_parameters.width * self.model_parameters.look_back_depth
            if layer >= self.model_parameters.look_back_depth
            and self.model_parameters.look_back_depth != -1
            else self.model_parameters.width * layer + self.model_parameters.in_channels
        )

    def forward(self, x):
        """The network's output image for the images ``x`` (B, in_channels, H,
        W)."""
        if (C := x.shape[1]) != self.model_parameters.in_channels:
            raise ValueError(
                f"Expected {self.model_parameters.in_channels} input channels, instead got {C}"
            )
        start_collecting = False
        final_lookbacks = (
            [] if self.model_parameters.final_look_back_depth != -1 else [x]
        )  # not a huge fan of this, think of a better way
        for i, layer in enumerate(range(self.model_parameters.depth)):
            x = self.layers[layer](x)[
                :,
                (
                    -self.model_parameters.look_back_depth * self.model_parameters.width
                    if self.model_parameters.look_back_depth != -1
                    else 0
                ) :,
            ]
            # x now contains all layers to pass forward aswell as newly calculated one
            if (
                self.model_parameters.depth - i
                <= self.model_parameters.final_look_back_depth
                or self.model_parameters.final_look_back_depth == -1
            ):  # need all the following layers for final layer
                start_collecting = True
            if start_collecting:
                final_lookbacks.append(
                    x[:, -self.model_parameters.width :]
                )  # head layer

        final_lookbacks = torch.cat(final_lookbacks, dim=1)
        x = self.final_layer(final_lookbacks)
        return x

    @staticmethod
    def default_parameters() -> LIONModelParameter:

        """The paper's network: 100 layers of width 1, dilations 1 to 10, every
        layer seeing all earlier ones (see the class)."""
        params = LIONModelParameter()
        params.model_input_type = ModelInputType.IMAGE
        params.in_channels = 1
        params.width = 1
        params.depth = 100
        params.dilations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10
        params.look_back_depth = -1
        params.final_look_back_depth = -1
        params.activation = "ReLU"

        return params

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
