import torch
import torch.nn as nn

# Implementation of:

# Jin, Kyong Hwan, et al.
# "Deep convolutional neural network for inverse problems in imaging."
# IEEE Transactions on Image Processing 26.9 (2017): 4509-4522.
# DOI: 10.1109/TIP.2017.2713099


class ConvBlock(nn.Module):
    def __init__(
        self, layers, channels, relu_type="ReLU", relu_last=True, kernel_size=3
    ):
        super().__init__()
        # input parsing:
        if len(channels) != layers + 1:
            raise ValueError(
                "Second input (channels) should have as many elements as layers your network has"
            )
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
        if relu_type == "ReLu":
            layer_list.append(nn.ReLU())
        elif relu_type == "LeakyReLU":
            layer_list.append(nn.LeakyReLU())
        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.block(x)


class FBPConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # standard FBPConvNet (As per paper):

        # Down blocks
        self.block_1_down = ConvBlock(3, [1, 64, 64, 64])
        self.down_1 = Down()
        self.block_2_down = ConvBlock(2, [64, 128, 128])
        self.down_2 = Down()
        self.block_3_down = ConvBlock(2, [128, 256, 256])
        self.down_3 = Down()
        self.block_4_down = ConvBlock(2, [256, 512, 512])
        self.down_4 = Down()

        # "latent space"
        self.block_bottom = ConvBlock(2, [512, 1024, 1024])

        # Up blocks
        self.up_1 = Up([1024, 512])
        self.block_1_up = ConvBlock(2, [1024, 512, 512])
        self.up_2 = Up([512, 256])
        self.block_2_up = ConvBlock(2, [512, 256, 256])
        self.up_3 = Up([256, 128])
        self.block_3_up = ConvBlock(2, [256, 128, 128])
        self.up_4 = Up([128, 64])
        self.block_4_up = ConvBlock(2, [128, 64, 64])

        self.block_last = nn.Sequential(nn.Conv2d(64, 1, 1, padding=0))

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
