# This file is part of LION library
# License : BSD-3
#
# Author  : Charlie Shoebridge
# Modifications: -
# =============================================================================

# Implementation of Mixed-Scale Dense Network described in:
# Daniël M. Pelt and James A. Sethian
# A mixed-scale dense convolutional neural network for image analysis

import numpy as np
import torch.nn as nn
import torch
from typing import Optional
from LION.models.LIONmodel import LIONmodel, ModelInputType, ModelParams


class MSD_Params(ModelParams):
    def __init__(
        self,
        in_channels: int,
        width: int,
        depth: int,
        dilations: list[int],
        look_back_depth: int,
        final_look_back_depth: int,
        activation: nn.Module,
    ):
        super().__init__(model_input_type=ModelInputType.NOISY_RECON)
        self.in_channels: int = in_channels
        self.width: int = width
        self.depth: int = depth
        self.dilations: list[int] = dilations
        self.look_back_depth: int = look_back_depth
        self.final_look_back_depth: int = final_look_back_depth
        self.activation: nn.Module = activation


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
        new_channels = []
        for i in range(self.width):
            new_channels.append(self.convs[i](x))
        # with torch.no_grad():
        x = torch.cat(new_channels, dim=1)

        x = self.activation(x)

        return x


class MSD_Net(LIONmodel):
    def __init__(
        self,
        model_parameters: Optional[MSD_Params] = None,
    ) -> None:
        """Mixed-Scale Dense Neural Network based on:
        A mixed-scale dense convolutional neural network for image analysis, Daniël M. Pelt and James A. Sethian

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
        """
        super().__init__(model_parameters)
        # it should definitely be not None here, as if it was none it's been set to default params,
        # which from here returns an MSD_Param
        # unfortunately static type checker isn't smart enough to figure that out, so might complain a bit.
        # We assert to make it happy and to ensure the model is actually initialized correctly.
        assert (
            self.model_parameters is not None
            and isinstance(self.model_parameters, MSD_Params)
        ), f"Failed to initialize MSD_Net model with given parameters: type {type(self.model_parameters)} is not acceptable for MSD_Params"

        self.in_channels = self.model_parameters.in_channels
        self.width = self.model_parameters.width
        self.depth = self.model_parameters.depth
        self.dilations = self.model_parameters.dilations
        self.look_back_depth = self.model_parameters.look_back_depth
        self.final_look_back_depth = self.model_parameters.final_look_back_depth
        self.activation = self.model_parameters.activation
        
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

        self.layers = nn.ModuleList()
        first_layer = MSD_Layer(
            self.in_channels, self.width, self.dilations[: self.width], self.activation
        )
        self.layers.append(first_layer)
        for layer_idx in range(1, self.depth):
            self.layers.append(
                MSD_Layer(
                    self._count_channels_to_use(layer_idx),
                    self.width,
                    self.dilations[
                        layer_idx * self.width : (layer_idx + 1) * self.width
                    ],
                    self.activation,
                )
            )

        for layer in self.layers:
            assert isinstance(layer, MSD_Layer)
            for conv in layer.convs:
                conv.apply(self._initialize_conv_weights)

        # final_look_back_depth should be no greater than number of hidden layers + 1
        if self.final_look_back_depth > self.depth + 1:
            raise ValueError(
                f"Final Lookback depth={self.final_look_back_depth} can not be bigger than total number of layers={self.depth + 1}"
            )

        if self.final_look_back_depth == -1:
            final_in_ch = self.depth * self.width + self.in_channels
        else:
            final_in_ch = self.width * self.final_look_back_depth

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
            self.activation
        )

    def _initialize_conv_weights(self, layer: nn.Module):
        assert isinstance(layer, nn.Conv2d), "layer not Conv2D"
        n_c = (
            np.product(layer.weight.shape[2:])
            * (self.in_channels + self.width * (self.depth - 1))
            + 1
        )
        torch.nn.init.normal_(layer.weight, 0, np.sqrt(2 / n_c))

    def _count_channels_to_use(self, layer) -> int:
        return (
            self.width * self.look_back_depth
            if layer >= self.look_back_depth and self.look_back_depth != -1
            else self.width * layer + self.in_channels
        )

    def forward(self, x):
        if (C := x.shape[1]) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, instead got {C}"
            )

        for layer in range(self.depth):
            prev = x
            x = self.layers[layer](x)
            # with torch.no_grad():
            x = torch.cat(
                [
                    prev[
                        :,
                        -self.look_back_depth * self.width
                        if self.look_back_depth != -1
                        else 0 :,
                    ],
                    x,
                ],
                dim=1,
            )  # x now contains all previous layers aswell as newly calculated one
        x = self.final_layer(x)
        return x

    @staticmethod
    def default_parameters() -> MSD_Params:
        in_channels = 1
        width, depth = 1, 100
        dilations = []
        for i in range(depth):
            for j in range(width):
                dilations.append((((i * width) + j) % 10) + 1)
        params = MSD_Params(
            in_channels=in_channels,
            width=width,
            depth=depth,
            dilations=dilations,
            look_back_depth=-1,
            final_look_back_depth=-1,
            activation=nn.ReLU(),
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
