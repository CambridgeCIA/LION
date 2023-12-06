# This file is part of LION library
# License : GPL-3
#
# Author  : Emilien Valat
# Modifications: Ander Biguri
# =============================================================================


import pathlib
from typing import Dict

import torch.nn as nn
import torch

from LION.models import LIONmodel
from LION.utils.parameter import Parameter


class InceptionLayer(nn.Module):
    def __init__(
        self,
        dimension: int,
        input_channels: int,
        output_channels: int,
        n_filters: int,
        dtype=torch.float,
    ) -> None:
        super(InceptionLayer, self).__init__()
        if dimension == 1:
            self.conv1 = nn.Conv1d(input_channels, n_filters, 1, padding=0, dtype=dtype)
            self.conv3 = nn.Conv1d(input_channels, n_filters, 3, padding=1, dtype=dtype)
            self.conv5 = nn.Conv1d(input_channels, n_filters, 5, padding=2, dtype=dtype)
            self.conv7 = nn.Conv1d(input_channels, n_filters, 7, padding=3, dtype=dtype)
            self.collection_filter = nn.Conv1d(
                4 * n_filters, output_channels, 7, padding=3, dtype=dtype
            )

        elif dimension == 2:
            self.conv1 = nn.Conv2d(input_channels, n_filters, 1, padding=0, dtype=dtype)
            self.conv3 = nn.Conv2d(input_channels, n_filters, 3, padding=1, dtype=dtype)
            self.conv5 = nn.Conv2d(input_channels, n_filters, 5, padding=2, dtype=dtype)
            self.conv7 = nn.Conv2d(input_channels, n_filters, 7, padding=3, dtype=dtype)
            self.collection_filter = nn.Conv2d(
                4 * n_filters, output_channels, 7, padding=3, dtype=dtype
            )

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.filtering = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.1),
            self.collection_filter,
            nn.LeakyReLU(negative_slope=0.1),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.filtering(
            torch.cat(
                [
                    self.conv1(input_tensor),
                    self.conv3(input_tensor),
                    self.conv5(input_tensor),
                    self.conv7(input_tensor),
                ],
                dim=1,
            )
        )


class DownModule(nn.Module):
    def __init__(self, dimension: int, input_channels: int, output_channels: int):
        super().__init__()
        if dimension == 1:
            self.down = nn.Sequential(
                nn.Conv1d(input_channels, output_channels, 4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv1d(output_channels, output_channels, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv1d(output_channels, output_channels, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
            )
        elif dimension == 2:
            self.down = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(output_channels, output_channels, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(output_channels, output_channels, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.down(input_tensor)


class UpModule(nn.Module):
    def __init__(self, dimension: int, input_channels: int, output_channels: int):
        super().__init__()
        if dimension == 1:
            self.up = nn.ConvTranspose1d(
                input_channels, input_channels, 4, stride=2, padding=1
            )
            self.conv1 = nn.Conv1d(input_channels, output_channels, 5, 1, 2)
            self.conv2 = nn.Conv1d(2 * output_channels, output_channels, 5, 1, 2)
        elif dimension == 2:
            self.up = nn.ConvTranspose2d(
                input_channels, input_channels, 4, stride=2, padding=1
            )
            self.conv1 = nn.Conv2d(input_channels, output_channels, 5, 1, 2)
            self.conv2 = nn.Conv2d(2 * output_channels, output_channels, 5, 1, 2)
        self.l_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(
        self, input_tensor: torch.Tensor, skp_connection: torch.Tensor
    ) -> torch.Tensor:
        x_0 = self.l_relu(self.up(input_tensor))
        x_1 = self.l_relu(self.conv1(x_0))
        return self.l_relu(self.conv2(torch.cat((x_1, skp_connection), 1)))


class InceptionUnet(LIONmodel.LIONmodel):
    def __init__(self, model_parameters: Parameter = None):

        super().__init__(model_parameters)

        # unwrapp model parameters for easiness to read in the constructor
        dimension = self.model_parameters.dimensions
        input_channels = self.model_parameters.input_channels
        output_channels = self.model_parameters.output_channels
        n_filters = self.model_parameters.n_filters

        if self.model_parameters.activation_function == "sigmoid":
            self.last_layer = nn.Sigmoid()
        elif self.model_parameters.activation_function == "l_relu":
            self.last_layer = nn.LeakyReLU(negative_slope=0.1)
        else:
            raise NotImplementedError(
                "Only activation function implemented for last layer are sigmoid and l_relu"
            )

        if self.model_parameters.inception:
            self.conv1 = InceptionLayer(dimension, input_channels, n_filters, n_filters)
            self.conv2 = InceptionLayer(dimension, n_filters, n_filters, n_filters)
        else:
            if dimension == 1:
                self.conv1 = nn.Conv1d(input_channels, n_filters, 5, 1, 2)
                self.conv2 = nn.Conv1d(n_filters, n_filters, 5, 1, 2)
            elif dimension == 2:
                self.conv1 = nn.Conv2d(input_channels, n_filters, 5, 1, 2)
                self.conv2 = nn.Conv2d(n_filters, n_filters, 5, 1, 2)

        self.down1 = DownModule(dimension, n_filters, 2 * n_filters)
        self.down2 = DownModule(dimension, 2 * n_filters, 4 * n_filters)
        self.down3 = DownModule(dimension, 4 * n_filters, 8 * n_filters)
        self.down4 = DownModule(dimension, 8 * n_filters, 16 * n_filters)
        self.down5 = DownModule(dimension, 16 * n_filters, 32 * n_filters)
        self.up1 = UpModule(dimension, 32 * n_filters, 16 * n_filters)
        self.up2 = UpModule(dimension, 16 * n_filters, 8 * n_filters)
        self.up3 = UpModule(dimension, 8 * n_filters, 4 * n_filters)
        self.up4 = UpModule(dimension, 4 * n_filters, 2 * n_filters)
        self.up5 = UpModule(dimension, 2 * n_filters, n_filters)

        if self.model_parameters.inception:
            self.conv3 = InceptionLayer(
                dimension, n_filters + input_channels, 2 * output_channels, n_filters
            )
            self.conv4 = InceptionLayer(
                dimension, 2 * output_channels, output_channels, n_filters
            )
        else:
            if dimension == 1:
                self.conv3 = nn.Conv1d(
                    n_filters + input_channels, 2 * output_channels, 5, 1, 2
                )
                self.conv4 = nn.Conv1d(2 * output_channels, output_channels, 5, 1, 2)
            elif dimension == 2:
                self.conv3 = nn.Conv2d(
                    n_filters + input_channels, 2 * output_channels, 5, 1, 2
                )
                self.conv4 = nn.Conv2d(2 * output_channels, output_channels, 5, 1, 2)

        self.l_relu = nn.LeakyReLU(negative_slope=0.1)

    @staticmethod
    def default_parameters():
        param = Parameter()
        param.dimensions = 2
        param.input_channels = 1
        param.output_channels = 1
        param.n_filters = 8
        param.activation_function = "sigmoid"
        param.inception = True
        return param

    def forward(self, input_tensor: torch.Tensor):
        s_0 = self.l_relu(self.conv1(input_tensor))
        s_1 = self.l_relu(self.conv2(s_0))
        s_2 = self.down1(s_1)
        s_3 = self.down2(s_2)
        s_4 = self.down3(s_3)
        s_5 = self.down4(s_4)
        u_0 = self.down5(s_5)
        u_1 = self.up1(u_0, s_5)
        u_2 = self.up2(u_1, s_4)
        u_3 = self.up3(u_2, s_3)
        u_4 = self.up4(u_3, s_2)
        u_5 = self.up5(u_4, s_1)
        return self.last_layer(
            self.conv4(self.l_relu(self.conv3(torch.cat((u_5, input_tensor), 1))))
        )
