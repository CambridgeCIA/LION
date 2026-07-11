# This file is part of LION library
# License : BSD-3
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

from LION.models.LIONmodel import LIONmodel, LIONModelParameter


class MSD_Layer(nn.Module):
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
        new_channels = [x]
        for i in range(self.width):
            new_channels.append(self.activation(self.convs[i](x)))

        x = torch.cat(new_channels, dim=1)

        return x


class MSDNet(LIONmodel):
    def __init__(
        self,
        model_parameters: Optional[LIONModelParameter] = None,
        geometry_parameters=None,
    ) -> None:
        """Mixed-Scale Dense Neural Network based on:
        A mixed-scale dense convolutional neural network for image analysis, Daniël M. Pelt and James A. Sethian

        Args:
            model_parameters (Optional[LIONModelParameter]):
                includes:
                    in_channels (int): number of channels in input image
                    width: number of channels in each hidden layer
                    depth: desired depth of network (not including input and output layers, i.e number of hidden layers)
                    dilations: vectorized matrix of dilations s_ij. i from 0 to width, j from 0 to depth,
                        e.g first (width) entries correspond to dilations for first layer.
                    look_back_depth: how many layers back to use when computing channels in a given layer. -1 = use all layers
                    final_look_back: how many layers to use to construct output image
                    activation: the activation function to be used between layers.
        """
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
            self.model_parameters.activation,
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
                    self.model_parameters.activation,
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
            self.model_parameters.activation,
        )

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
    def default_parameters() -> MSD_Params:

        params = LIONModelParameter()
        params.in_channels = 1
        params.width = 1
        params.depth = 100
        params.dilations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10
        params.look_back_depth = -1
        params.final_look_back_depth = -1
        params.activation = "ReLU"

        return params

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print("Pelt, Daniël M., and James A. Sethian.")
            print(
                '"A mixed-scale dense convolutional neural network for image analysis."'
            )
            print("\x1b[3mProceedings of the National Academy of Sciences  \x1b[0m")
            print("115.2 (2018): 254-259.")
        elif cite_format == "bib":
            string = """
            @article{pelt2018mixed,
            title={A mixed-scale dense convolutional neural network for image analysis},
            author={Pelt, Dani{\"e}l M and Sethian, James A},
            journal={Proceedings of the National Academy of Sciences},
            volume={115},
            number={2},
            pages={254--259},
            year={2018},
            publisher={National Acad Sciences}
            }
            """
            print(string)
        else:
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )
