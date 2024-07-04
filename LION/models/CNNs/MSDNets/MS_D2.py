# This file is part of LION library
# License : BSD-3
#
# Author  : Charlie Shoebridge
# Modifications: -
# =============================================================================

# Implementation of Mixed-Scale Dense Network described in:
# Daniël M. Pelt and James A. Sethian
# A mixed-scale dense convolutional neural network for image analysis

import torch.nn as nn
import torch
from typing import Optional
from LION.utils.parameter import LIONParameter
from LION.models.LIONmodel import LIONmodel
from LION.utils.activations import ACTIVATIONS


class MSD_Net(LIONmodel):
    def __init__(
        self,
        model_parameters: Optional[LIONParameter] = None,
    ) -> None:
        """_summary_

        Args:
            model_parameters (Optional[LIONParameter]):
                expects:
                    in_channels (int): number of channels in input image
                    width: number of channels in each hidden layer
                    depth: desired depth of network (not including input and output layers, i.e number of hidden layers)
                    dilations: vectorized matrix of dilations s_ij. i from 0 to width, j from 0 to depth,
                        e.g first (width) entries correspond to dilations for first layer.
                    look_back_depth: how many layers back to use when computing channels in a given layer. -1 = use all layers
                    final_look_back: how many layers to use to construct output image
                    activation: the activation function to be used between layers.
                        One of:
                            ReLU,
                            sigmoid,
                            ...
        """
        super().__init__(model_parameters)

        self.in_channels = self.model_parameters.in_channels
        self.width = self.model_parameters.width
        self.depth = self.model_parameters.depth
        self.dilations = self.model_parameters.dilations
        self.look_back_depth = self.model_parameters.look_back_depth
        self.final_look_back = self.model_parameters.final_look_back
        self.activation = nn.Sequential(nn.BatchNorm2d(1))
        # this is bad practice, think of a better way to do this
        try:
            self.activation.append(ACTIVATIONS[(acti:=self.model_parameters.activation)]())
        except KeyError:
            print(f"Activation '{acti}' not recognised. Expected one of {ACTIVATIONS.keys()}. Defaulting to ReLU.")
            self.activation.append(nn.ReLU())
        # total there should be width * depth distinct convolutions
        # so expect the same number of dilations to be given
        if len(self.dilations) != self.width * self.depth:
            raise ValueError(
                f"Expected {self.width*self.depth} dilation sizes to be given, instead given {len(self.dilations)}"
            )

        # look_back_depth should be no greater than number of hidden layers + 1
        if self.look_back_depth > self.depth + 1:
            raise ValueError(
                f"Lookback depth={self.look_back_depth} can not be bigger than total number of layers={self.depth + 1}"
            )

        convs = nn.ModuleDict()
        for i, d in enumerate(self.dilations):
            layer = i // self.width
            channel = i % self.width

            # number of input channels depends on whether we use original input to construct this layer
            in_ch = self._count_channels_to_use(layer)

            # bias set to false since we perform a BatchNorm after each Conv, so bias would be cancelled out anyway
            convs.add_module(
                f"{layer}_{channel}",
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=1,
                    kernel_size=3,
                    dilation=d,
                    padding="same",
                    padding_mode="reflect",
                    bias=False,
                ),
            )

        self.convs = convs

        if self.look_back_depth == -1:
            final_in_ch = self.depth * self.width + self.in_channels
        else:
            final_in_ch = self.width * self.look_back_depth
        self.final_conv = nn.Conv2d(
            in_channels=final_in_ch, out_channels=1, kernel_size=1
        )

    def _count_channels_to_use(self, layer) -> int:
        return (
            self.width * self.look_back_depth
            if layer >= self.look_back_depth and self.look_back_depth != -1
            else self.width * layer + self.in_channels
        )
    

    def forward(self, x: torch.Tensor):
        # maybe just combine these into a multi-channel tensor?
        B, C, W, H = x.shape
        if C != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, instead got {C}"
            )
        # allocate lookback tensor maximally
        lookbacks = torch.empty((B, C + self.width * self.depth, W, H), device=x.device)
        lookbacks[:, :C, :, :] = x

        for layer in range(self.depth):
            for channel in range(self.width):
                if self.look_back_depth > layer or self.look_back_depth == -1: # lookback includes input, x
                    start_ch_idx = 0
                else: # doesn't include input
                    start_layer = layer - self.look_back_depth
                    start_ch_idx = C + start_layer * self.width
                # get a reference to the relevant lookback channels...
                z_ij = lookbacks[:, C + layer * self.width: C + (layer + 1) * self.width]
                # apply convolution to them
                z_ij = self.convs[f"{layer}_{channel}"](
                    lookbacks[:, start_ch_idx : start_ch_idx + self._count_channels_to_use(layer), :, :]
                )
                z_ij = self.activation(z_ij)

        # apply final convolution
        return self.activation(self.final_conv(lookbacks))

    @staticmethod
    def default_parameters():
        in_channels = 1
        width, depth = 1, 100
        dilations = []
        for i in range(depth):
            for j in range(width):
                dilations.append((((i * width) + j) % 10) + 1)
        params = LIONParameter(
            in_channels=in_channels,
            width=width,
            depth=depth,
            dilations=dilations,
            look_back_depth=-1,
            final_look_back=-1,
            activation="ReLU",
        )
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
