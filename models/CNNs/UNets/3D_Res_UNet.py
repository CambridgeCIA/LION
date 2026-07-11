# This file is part of LION library
# License : BSD-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================


import torch
import torch.nn as nn
from LION.models import LIONmodel
from LION.utils.parameter import LIONParameter

# Implementation of:

# Yu, Hui, et al.
# "Design of lung nodules segmentation and recognition algorithm based on deep learning"
# BMC bioinformatics 22 (2021): 1-21.


class ResidualUnit(nn.Module):
    def __init__(self, channels, relu_type="ReLU", kernel_size=3):
        super().__init__()
        # input parsing:
        layers = len(channels) - 1
        if layers < 1:
            raise ValueError("At least one layer required")

        # convolutional layers
        layer_list = []
        for ii in range(layers):
            layer_list.append(
                nn.Conv3d(
                    channels[ii], channels[ii + 1], kernel_size, padding=1, bias=True
                )
            )
            layer_list.append(nn.BatchNorm3d(channels[ii + 1]))

            if ii < layers - 1 or relu_last:
                if relu_type == "ReLU":
                    layer_list.append(torch.nn.ReLU())
                elif relu_type == "LeakyReLU":
                    layer_list.append(torch.nn.LeakyReLU())
                elif relu_type != "None":
                    raise ValueError("Wrong ReLu type " + relu_type)
        self.block = nn.Sequential(*layer_list)

        # conv for residual layer
        self.res_block = nn.Conv3d(channels[0], channels[-1], kernel_size=1, padding=0)

    def forward(self, x):
        return self.block(x) + self.res_block(x)


class Down(nn.Module):
    """Downscaling with maxpool"""

    def __init__(self):
        super().__init__()
        self.pool = nn.Sequential(nn.MaxPool3d(2))

    def forward(self, x):
        return self.pool(x)


class Up(nn.Module):
    """Upscaling with transpose conv"""

    def __init__(self, channels, stride=2, relu_type="ReLU"):
        super().__init__()
        kernel_size = 2
        layer_list = []
        layer_list.append(
            nn.ConvTranspose3d(
                channels[0],
                channels[1],
                kernel_size,
                padding=0,
                output_padding=0,
                stride=stride,
                bias=False,
            )
        )
        layer_list.append(nn.BatchNorm3d(channels[1]))
        if relu_type == "ReLU":
            layer_list.append(nn.ReLU())
        elif relu_type == "LeakyReLU":
            layer_list.append(nn.LeakyReLU())
        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.block(x)


class ResUNet3D(LIONmodel.LIONmodel):
    def __init__(self, model_parameters: LIONParameter = None):
        super().__init__(model_parameters)

        self.block_down_1 = ResidualUnit(model_parameters.down_1_channels)
        self.down_1 = Down()
        self.block_down_2 = ResidualUnit(model_parameters.down_2_channels)
        self.down_1 = Down()
        self.block_down_3 = ResidualUnit(model_parameters.down_3_channels)
        self.down_1 = Down()
        self.block_down_4 = ResidualUnit(model_parameters.down_4_channels)

        self.block_bottom = ResidualUnit(model_parameters.latent_channels)

        self.up_1 = Up(
            [
                model_parameters.latent_channels[-1],
                model_parameters.up_1_channels[0] // 2,
            ]
        )
        self.block_up_1 = ResidualUnit(model_parameters.up_1_channels)
        self.up_2 = Up(
            [model_parameters.up_1_channels[-1], model_parameters.up_2_channels[0] // 2]
        )
        self.block_up_2 = ResidualUnit(model_parameters.up_2_channels)
        self.up_3 = Up(
            [model_parameters.up_2_channels[-1], model_parameters.up_3_channels[0] // 2]
        )
        self.block_up_3 = ResidualUnit(model_parameters.up_3_channels)
        self.up_4 = Up(
            [model_parameters.up_3_channels[-1], model_parameters.up_4_channels[0] // 2]
        )
        self.block_up_4 = ResidualUnit(model_parameters.up_4_channels)

        self.block_last = nn.Sequential(
            nn.Conv3d(
                model_parameters.last_block[0],
                model_parameters.last_block[1],
                model_parameters.last_block[2],
                padding=0,
            )
        )

    @staticmethod
    def default_parameters():
        ResUNet3D = LIONModelParameter()
        ResUNet3D.down_1_channels = [1, 8, 8]
        ResUNet3D.down_2_channels = [8, 16, 16]
        ResUNet3D.down_3_channels = [16, 32, 32]
        ResUNet3D.down_4_channels = [32, 64, 64]

        ResUNet3D.latent_channels = [64, 128, 128]

        ResUNet3D.up_1_channels = [128, 64, 64]
        ResUNet3D.up_2_channels = [64, 32, 32]
        ResUNet3D.up_3_channels = [32, 16, 16]
        ResUNet3D.up_4_channels = [16, 8, 8]

        ResUNet3D.last_block = [8, 1, 1]

        ResUNet3D.activation = "ReLU"

        return ResUNet3D

    def forward(self, x):
        block_1_res = self.block_1_down(x)
        block_2_res = self.block_2_down(self.down_1(block_1_res))
        block_3_res = self.block_3_down(self.down_2(block_2_res))
        block_4_res = self.block_4_down(self.down_3(block_3_res))

        res = self.block_bottom(self.down_4(block_4_res))

        res = self.block_1_up(torch.cat((block_4_res, self.up_1(res)), dim=1))
        res = self.block_2_up(torch.cat((block_3_res, self.up_2(res)), dim=1))
        res = self.block_3_up(torch.cat((block_2_res, self.up_3(res)), dim=1))
        res = self.block_4_up(torch.cat((block_1_res, self.up_4(res)), dim=1))
        res = self.block_last(res)
        res = torch.nn.Sigmoid(res)
        return res
