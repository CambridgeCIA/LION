# This file is part of LION library
# License : BSD-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================


from typing import Optional
import torch
import torch.nn as nn
from LION.models import LIONmodel
from LION.utils.parameter import LIONParameter
import LION.CTtools.ct_geometry as ct
from LION.classical_algorithms.fdk import fdk

# Implementation of:

# Jin, Kyong Hwan, et al.
# "Deep convolutional neural network for inverse problems in imaging."
# IEEE Transactions on Image Processing 26.9 (2017): 4509-4522.
# DOI: 10.1109/TIP.2017.2713099


class ConvBlock(nn.Module):
    def __init__(self, channels, relu_type="ReLU", relu_last=True, kernel_size=3):
        super().__init__()
        # input parsing:

        layers = len(channels) - 1
        if layers < 1:
            raise ValueError("At least one layer required")
        # convolutional layers
        layer_list = []
        for ii in range(layers):
            layer_list.append(
                nn.Conv2d(
                    channels[ii], channels[ii + 1], kernel_size, padding=1, bias=False
                )
            )
            layer_list.append(nn.BatchNorm2d(channels[ii + 1]))
            if ii < layers - 1 or relu_last:
                if relu_type == "ReLU":
                    layer_list.append(torch.nn.ReLU())
                elif relu_type == "LeakyReLU":
                    layer_list.append(torch.nn.LeakyReLU())
                elif relu_type != "None":
                    raise ValueError("Wrong ReLu type " + relu_type)
        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downscaling with maxpool"""

    def __init__(self):
        super().__init__()
        self.pool = nn.Sequential(nn.MaxPool2d(2))

    def forward(self, x):
        return self.pool(x)


class Up(nn.Module):
    """Upscaling with transpose conv"""

    def __init__(self, channels, stride=2, relu_type="ReLU"):
        super().__init__()
        kernel_size = 3
        layer_list = []
        layer_list.append(
            nn.ConvTranspose2d(
                channels[0],
                channels[1],
                kernel_size,
                padding=1,
                output_padding=1,
                stride=stride,
                bias=False,
            )
        )
        layer_list.append(nn.BatchNorm2d(channels[1]))
        if relu_type == "ReLU":
            layer_list.append(nn.ReLU())
        elif relu_type == "LeakyReLU":
            layer_list.append(nn.LeakyReLU())
        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.block(x)


class FBPConvNetParams(LIONmodel.ModelParams):
    def __init__(
        self,
        down_1_channels: list[int],
        down_2_channels: list[int],
        down_3_channels: list[int],
        down_4_channels: list[int],
        latent_channels: list[int],
        up_1_channels: list[int],
        up_2_channels: list[int],
        up_3_channels: list[int],
        up_4_channels: list[int],
        last_block: list[int],
        activation: str,
    ):
        super().__init__(LIONmodel.ModelInputType.SINOGRAM)
        self.down_1_channels = down_1_channels
        self.down_2_channels = down_2_channels
        self.down_3_channels = down_3_channels
        self.down_4_channels = down_4_channels

        self.latent_channels = latent_channels

        self.up_1_channels = up_1_channels
        self.up_2_channels = up_2_channels
        self.up_3_channels = up_3_channels
        self.up_4_channels = up_4_channels

        self.last_block = last_block

        self.activation = activation


class FBPConvNet(LIONmodel.LIONmodel):
    def __init__(
        self,
        geometry_parameters: ct.Geometry,
        model_parameters: Optional[LIONmodel.ModelParams] = None,
    ):

        assert (
            geometry_parameters is not None
        ), "Geometry parameters required for FBPConvNet."

        super().__init__(model_parameters, geometry_parameters)
        self._make_operator()
        # standard FBPConvNet (As per paper):

        # Down blocks
        self.block_1_down = ConvBlock(
            self.model_parameters.down_1_channels,
            relu_type=self.model_parameters.activation,
        )
        self.down_1 = Down()
        self.block_2_down = ConvBlock(
            self.model_parameters.down_2_channels,
            relu_type=self.model_parameters.activation,
        )
        self.down_2 = Down()
        self.block_3_down = ConvBlock(
            self.model_parameters.down_3_channels,
            relu_type=self.model_parameters.activation,
        )
        self.down_3 = Down()
        self.block_4_down = ConvBlock(
            self.model_parameters.down_4_channels,
            relu_type=self.model_parameters.activation,
        )
        self.down_4 = Down()

        # "latent space"
        self.block_bottom = ConvBlock(
            self.model_parameters.latent_channels,
            relu_type=self.model_parameters.activation,
        )

        # Up blocks
        self.up_1 = Up(
            [
                self.model_parameters.latent_channels[-1],
                self.model_parameters.up_1_channels[0] // 2,
            ],
            relu_type=self.model_parameters.activation,
        )
        self.block_1_up = ConvBlock(
            self.model_parameters.up_1_channels,
            relu_type=self.model_parameters.activation,
        )
        self.up_2 = Up(
            [
                self.model_parameters.up_1_channels[-1],
                self.model_parameters.up_2_channels[0] // 2,
            ],
            relu_type=self.model_parameters.activation,
        )
        self.block_2_up = ConvBlock(
            self.model_parameters.up_2_channels,
            relu_type=self.model_parameters.activation,
        )
        self.up_3 = Up(
            [
                self.model_parameters.up_2_channels[-1],
                self.model_parameters.up_3_channels[0] // 2,
            ],
            relu_type=self.model_parameters.activation,
        )
        self.block_3_up = ConvBlock(
            self.model_parameters.up_3_channels,
            relu_type=self.model_parameters.activation,
        )
        self.up_4 = Up(
            [
                self.model_parameters.up_3_channels[-1],
                self.model_parameters.up_4_channels[0] // 2,
            ],
            relu_type=self.model_parameters.activation,
        )
        self.block_4_up = ConvBlock(
            self.model_parameters.up_4_channels,
            relu_type=self.model_parameters.activation,
        )

        self.block_last = nn.Sequential(
            nn.Conv2d(
                self.model_parameters.last_block[0],
                self.model_parameters.last_block[1],
                self.model_parameters.last_block[2],
                padding=0,
            )
        )

    @staticmethod
    def default_parameters():
        FBPConvNet_params = FBPConvNetParams(
            [1, 64, 64, 64],
            [64, 128, 128],
            [128, 256, 256],
            [256, 512, 512],
            [512, 1024, 1024],
            [1024, 512, 512],
            [512, 256, 256],
            [256, 128, 128],
            [128, 64, 64],
            [64, 1, 1],
            "ReLU",
        )

        return FBPConvNet_params

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print(" Jin, Kyong Hwan, et al.")
            print(
                '"Deep convolutional neural network for inverse problems in imaging."'
            )
            print("\x1B[3mIEEE Transactions on Image Processing \x1B[0m")
            print(" 26.9 (2017): 4509-4522.")
        elif cite_format == "bib":
            string = """
            @article{jin2017deep,
            title={Deep convolutional neural network for inverse problems in imaging},
            author={Jin, Kyong Hwan and McCann, Michael T and Froustey, Emmanuel and Unser, Michael},
            journal={IEEE transactions on image processing},
            volume={26},
            number={9},
            pages={4509--4522},
            year={2017},
            publisher={IEEE}
            }"""
            print(string)
        else:
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )

    def forward(self, x):
        B, C, W, H = x.shape

        image = fdk(x, self.op)
        block_1_res = self.block_1_down(image)
        block_2_res = self.block_2_down(self.down_1(block_1_res))
        block_3_res = self.block_3_down(self.down_2(block_2_res))
        block_4_res = self.block_4_down(self.down_3(block_3_res))

        res = self.block_bottom(self.down_4(block_4_res))
        res = self.block_1_up(torch.cat((block_4_res, self.up_1(res)), dim=1))
        res = self.block_2_up(torch.cat((block_3_res, self.up_2(res)), dim=1))
        res = self.block_3_up(torch.cat((block_2_res, self.up_3(res)), dim=1))
        res = self.block_4_up(torch.cat((block_1_res, self.up_4(res)), dim=1))
        res = self.block_last(res)

        return image + res
