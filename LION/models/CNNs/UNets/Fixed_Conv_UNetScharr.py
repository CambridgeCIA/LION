# This file is part of LION library
# License : BSD-3
#
# Author  : Oliver Coughlan
# Modifications: Ander Biguri
# Further Modifications: Alex, Dana, Zach
# =============================================================================

#find change by searching for conv_func

#from DUGAN's paper

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScharrOperator(nn.Module):
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

        self.register_buffer('conv_x', torch.Tensor([[3, 0, -3], [10, 0, -10], [3, 0, -3]])[None, None, :, :] / 4)
        self.register_buffer('conv_y', torch.Tensor([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])[None, None, :, :] / 4)

    def forward(self, x):
        b, c, h, w = x.shape
        # if c > 1:
        #     x = x.view(b * c, 1, h, w)
        # for k in range(c)
        grad_x = F.conv2d(x.cpu(), self.conv_x.cpu(), bias=None, stride=1, padding=1)
        grad_y = F.conv2d(x.cpu(), self.conv_y.cpu(), bias=None, stride=1, padding=1)

        # new_x = x.reshape(b, 3, h,w)
        new_x = torch.empty((b, 3, h, w), dtype=torch.float32)
        #x = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.epsilon)

        x_dim = x.size()
        for i in range(x_dim[0]):
            new_x[i][0] = x[i][0]
            new_x[i][1] = grad_x[i][0]
            new_x[i][2] = grad_y[i][0]

        
        # x = x.view(b, c, h, w)

        return new_x
    
sobel = ScharrOperator()

def conv_func(x):
    return sobel.forward(x)


from collections import OrderedDict

from LION.utils.parameter import LIONParameter
from LION.models import LIONmodel

##Code for Fixed_Conv_UNet
class Fixed_Conv_UNet(LIONmodel.LIONmodel):
    def __init__(self, model_parameters=None):
        # set properties of Fixed_Conv_UNet
        if model_parameters is None:
            model_parameters = Fixed_Conv_UNet.default_parameters()
        super().__init__(model_parameters)



# def block(self, inChan, outChan, noGrp, dropFac, name):
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
        modified_input = conv_func(x)
        enc1 = self.encoder1(modified_input.cuda())
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
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(inChan, outChan, kernel_size=3, padding=1, bias=True),
                    ),
                    (name + "gn1", nn.GroupNorm(noGrp, outChan)),
                    (name + "relu1", nn.ReLU(True)),
                    (name + "dr1", nn.Dropout(p=dropFac)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            outChan, outChan, kernel_size=3, padding=1, bias=True
                        ),
                    ),
                    (name + "gn1", nn.GroupNorm(noGrp, outChan)),
                    (name + "relu2", nn.ReLU(True)),
                    (name + "dr2", nn.Dropout(p=dropFac)),
                ]
            )
        )

    @staticmethod
    def default_parameters():
        param = LIONParameter()
        param.inChan = 3
        param.outChan = 1
        param.baseDim = 32
        param.dropFac = 0
        param.kerSiz = 2
        param.noGrp = 32
        param.model_input_type = LIONmodel.ModelInputType.IMAGE

        return param
