# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Maximilian B. Kiss, Ander Biguri
# =============================================================================
"""The U-Net of the noise paper (Kiss et al., 2025): the original U-Net
(Ronneberger et al., 2015) as the paper's code had it (``UNet_3`` and
``FBPUNet``, 2024). It differs from LION's ``UNet``: batch normalisation, 64
to 512 channels, and any image size (the upsampled features are padded to
their skip connection's size), so it takes 2DeteCT's 3600x956 sinograms."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType


def double_conv(in_ch, out_ch):
    """Two 3x3 convolutions, each with batch normalisation and a ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Down(nn.Module):
    """Max pooling, then a double convolution."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    """A transposed convolution, padded to the skip connection's size and
    concatenated with it, then a double convolution."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2)
        )
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(LIONmodel):
    """The noise paper's U-Net: four downsamplings, 64 to 512 channels, batch
    normalisation (see the module). ``model_parameters.model_input_type``:
    SINOGRAM to denoise sinograms (the network on the measurements), IMAGE for
    FBP+U-Net (the network on the FDK of the measurements, which solvers and
    reconstructors give it)."""

    def __init__(self, model_parameters=None):
        super().__init__(model_parameters)
        self.inc = double_conv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, 1, 1)

    @staticmethod
    def default_parameters():
        """On images (FBP+U-Net); set ``model_input_type`` to SINOGRAM to
        denoise sinograms."""
        parameters = LIONModelParameter()
        parameters.model_input_type = ModelInputType.IMAGE
        return parameters

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
