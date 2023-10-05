# This file is part of AItomotools library
# License : BSD-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================


import torch
import torch.nn as nn
from AItomotools.models import AItomomodel
from AItomotools.utils.parameter import Parameter

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
        kernel_size = 3
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
    """Downscaling with transpose conv"""

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


class FBPConvNet(AItomomodel.AItomoModel):
    def __init__(self, model_parameters: Parameter = None):
        if model_parameters is None:
            model_parameters = FBPConvNet.default_parameters()
        super().__init__(model_parameters)

        # standard FBPConvNet (As per paper):

        # Down blocks
        self.block_1_down = ConvBlock(
            model_parameters.down_1_channels, relu_type=model_parameters.activation
        )
        self.down_1 = Down()
        self.block_2_down = ConvBlock(
            model_parameters.down_2_channels, relu_type=model_parameters.activation
        )
        self.down_2 = Down()
        self.block_3_down = ConvBlock(
            model_parameters.down_3_channels, relu_type=model_parameters.activation
        )
        self.down_3 = Down()
        self.block_4_down = ConvBlock(
            model_parameters.down_4_channels, relu_type=model_parameters.activation
        )
        self.down_4 = Down()

        # "latent space"
        self.block_bottom = ConvBlock(
            model_parameters.latent_channels, relu_type=model_parameters.activation
        )

        # Up blocks
        self.up_1 = Up(
            [
                model_parameters.latent_channels[-1],
                model_parameters.up_1_channels[0] // 2,
            ],
            relu_type=model_parameters.activation,
        )
        self.block_1_up = ConvBlock(
            model_parameters.up_1_channels, relu_type=model_parameters.activation
        )
        self.up_2 = Up(
            [
                model_parameters.up_1_channels[-1],
                model_parameters.up_2_channels[0] // 2,
            ],
            relu_type=model_parameters.activation,
        )
        self.block_2_up = ConvBlock(
            model_parameters.up_2_channels, relu_type=model_parameters.activation
        )
        self.up_3 = Up(
            [
                model_parameters.up_2_channels[-1],
                model_parameters.up_3_channels[0] // 2,
            ],
            relu_type=model_parameters.activation,
        )
        self.block_3_up = ConvBlock(
            model_parameters.up_3_channels, relu_type=model_parameters.activation
        )
        self.up_4 = Up(
            [
                model_parameters.up_3_channels[-1],
                model_parameters.up_4_channels[0] // 2,
            ],
            relu_type=model_parameters.activation,
        )
        self.block_4_up = ConvBlock(
            model_parameters.up_4_channels, relu_type=model_parameters.activation
        )

        self.block_last = nn.Sequential(
            nn.Conv2d(
                model_parameters.last_block[0],
                model_parameters.last_block[1],
                model_parameters.last_block[2],
                padding=0,
            )
        )

    @staticmethod
    def default_parameters():
        FBPConvNet_params = Parameter()
        FBPConvNet_params.down_1_channels = [1, 64, 64, 64]
        FBPConvNet_params.down_2_channels = [64, 128, 128]
        FBPConvNet_params.down_3_channels = [128, 256, 256]
        FBPConvNet_params.down_4_channels = [256, 512, 512]

        FBPConvNet_params.latent_channels = [512, 1024, 1024]

        FBPConvNet_params.up_1_channels = [1024, 512, 512]
        FBPConvNet_params.up_2_channels = [512, 256, 256]
        FBPConvNet_params.up_3_channels = [256, 128, 128]
        FBPConvNet_params.up_4_channels = [128, 64, 64]

        FBPConvNet_params.last_block = [64, 1, 1]

        FBPConvNet_params.activation = "ReLU"

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
        return x + res
