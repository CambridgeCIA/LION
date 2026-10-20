# This file is part of LION library
# License : GPL-3
#
# Author  : Oliver Coughlan
# Modifications: Ander Biguri
# =============================================================================

import torch
import torch.nn as nn

from collections import OrderedDict

from LION.utils.parameter import LIONParameter
from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType

##Code for UNet
class UNet(LIONmodel):
    def __init__(self, model_parameters=None):
        # set properties of UNet
        if model_parameters is None:
            model_parameters = UNet.default_parameters()
        super().__init__(model_parameters)

        self.encoder1 = self.block(
            model_parameters.inChan,
            model_parameters.baseDim,
            model_parameters.noGrp,
            model_parameters.dropFac,
            "enc1",
        )
        self.pool1 = nn.MaxPool2d(model_parameters.kerSiz, stride=2)
        self.encoder2 = self.block(
            model_parameters.baseDim,
            model_parameters.baseDim * 2,
            model_parameters.noGrp,
            model_parameters.dropFac,
            "enc2",
        )
        self.pool2 = nn.MaxPool2d(model_parameters.kerSiz, stride=2)
        self.encoder3 = self.block(
            model_parameters.baseDim * 2,
            model_parameters.baseDim * 4,
            model_parameters.noGrp,
            model_parameters.dropFac,
            "enc3",
        )
        self.pool3 = nn.MaxPool2d(model_parameters.kerSiz, stride=2)
        self.encoder4 = self.block(
            model_parameters.baseDim * 4,
            model_parameters.baseDim * 8,
            model_parameters.noGrp,
            model_parameters.dropFac,
            "enc4",
        )
        self.pool4 = nn.MaxPool2d(model_parameters.kerSiz, stride=2)

        self.base = self.block(
            model_parameters.baseDim * 8,
            model_parameters.baseDim * 16,
            model_parameters.noGrp,
            model_parameters.dropFac,
            name="base",
        )

        self.upconv4 = nn.ConvTranspose2d(
            model_parameters.baseDim * 16,
            model_parameters.baseDim * 8,
            model_parameters.kerSiz,
            stride=2,
        )
        self.decoder4 = self.block(
            model_parameters.baseDim * 16,
            model_parameters.baseDim * 8,
            model_parameters.noGrp,
            model_parameters.dropFac,
            name="dec4",
        )
        self.upconv3 = nn.ConvTranspose2d(
            model_parameters.baseDim * 8,
            model_parameters.baseDim * 4,
            model_parameters.kerSiz,
            stride=2,
        )
        self.decoder3 = self.block(
            model_parameters.baseDim * 8,
            model_parameters.baseDim * 4,
            model_parameters.noGrp,
            model_parameters.dropFac,
            name="dec3",
        )
        self.upconv2 = nn.ConvTranspose2d(
            model_parameters.baseDim * 4,
            model_parameters.baseDim * 2,
            model_parameters.kerSiz,
            stride=2,
        )
        self.decoder2 = self.block(
            model_parameters.baseDim * 4,
            model_parameters.baseDim * 2,
            model_parameters.noGrp,
            model_parameters.dropFac,
            name="dec2",
        )
        self.upconv1 = nn.ConvTranspose2d(
            model_parameters.baseDim * 2,
            model_parameters.baseDim,
            model_parameters.kerSiz,
            stride=2,
        )
        self.decoder1 = self.block(
            model_parameters.baseDim * 2,
            model_parameters.baseDim,
            model_parameters.noGrp,
            model_parameters.dropFac,
            name="dec1",
        )

        self.conv = nn.Conv2d(
            model_parameters.baseDim, model_parameters.outChan, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        base = self.base(self.pool4(enc4))

        dec4 = self.upconv4(base)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)

    def block(self, inChan, outChan, noGrp, dropFac, name):
        layers = [
            (
                name + "conv1",
                nn.Conv2d(inChan, outChan, kernel_size=3, padding=1, bias=True),
            ),
            (name + "gn1", nn.GroupNorm(noGrp, outChan)),
            (name + "relu1", nn.ReLU(True)),
            (name + "dr1", nn.Dropout(p=dropFac)),
            (
                name + "conv2",
                nn.Conv2d(outChan, outChan, kernel_size=3, padding=1, bias=True),
            ),
        ]
        # Older versions named both norms "gn1", so the second one silently
        # replaced the first and blocks had a single GroupNorm. Parameters saved
        # before `norm_after_second_conv` existed keep that architecture, so
        # their weights still load.
        if getattr(self.model_parameters, "norm_after_second_conv", False):
            layers.append((name + "gn2", nn.GroupNorm(noGrp, outChan)))
        layers += [
            (name + "relu2", nn.ReLU(True)),
            (name + "dr2", nn.Dropout(p=dropFac)),
        ]
        return nn.Sequential(OrderedDict(layers))

    @staticmethod
    def default_parameters():
        param = LIONModelParameter()
        param.model_input_type = ModelInputType.IMAGE
        param.inChan = 1
        param.outChan = 1
        param.baseDim = 32
        param.dropFac = 0
        param.kerSiz = 2
        param.noGrp = 32
        # GroupNorm after both convolutions of each block (False reproduces the
        # single-norm blocks of LION versions before this option existed)
        param.norm_after_second_conv = True

        return param
