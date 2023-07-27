# This file is part of AItomotools library
# License : BSD-3
#
# Author  : Oliver Coughlan
# Modifications: Ander Biguri
# =============================================================================


import numpy as np
import torch
import torch.nn as nn

from ts_algorithms import fdk
from collections import OrderedDict


import AItomotools.CTtools.ct_geometry as ct

from AItomotools.utils.parameter import Parameter
from AItomotools.models import AItomomodel

##Code for UNet
class UNet(AItomomodel.AItomoModel):
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
        param = Parameter()
        param.inChan = 1
        param.outChan = 1
        param.baseDim = 32
        param.dropFac = 0
        param.kerSiz = 2
        param.noGrp = 32
        return param


class ItNet(AItomomodel.AItomoModel):
    def __init__(
        self, geometry_parameters: ct.Geometry, model_parameters: Parameter = None
    ):
        if geometry_parameters is None:
            raise ValueError("Geometry parameters required. ")
        if model_parameters is None:
            model_parameters = ItNet.default_parameters()
        super().__init__(model_parameters, geometry_parameters)

        # Create layers per iteration
        for i in range(model_parameters.n_iters):
            self.add_module(f"Unet_{i}", UNet(model_parameters.Unet_params))

        # Create pytorch compatible operators and send them to autograd
        self.make_operator()

        # Define step size
        if self.model_parameters.step_size is None:
            # compute step size
            self.model_parameters.step_size = np.array(
                [1] * self.model_parameters.n_iters
            )

        elif not hasattr(self.model_parameters.step_size, "__len__"):
            self.model_parameters.step_size = np.array(
                self.model_parameters.step_size * self.model_parameters.n_iters
            )
        elif len(self.model_parameters.step_size) == self.model_parameters.n_iters:
            self.model_parameters.step_size = np.array(self.model_parameters.step_size)
        else:
            raise ValueError("Step size not understood")

        #  Are we learning the step? (with the above initialization)
        if self.model_parameters.learned_step:
            # Enforce positivity by making it 10^step
            if self.model_parameters.step_positive:
                self.step_size = nn.ParameterList(
                    [
                        nn.Parameter(
                            torch.ones(1)
                            * 10 ** np.log10(self.model_parameters.step_size[i])
                        )
                        for i in range(self.model_parameters.n_iters)
                    ]
                )
            # Negatives OK
            else:
                self.step_size = nn.ParameterList(
                    [
                        nn.Parameter(torch.ones(1) * self.model_parameters.step_size[i])
                        for i in range(self.model_parameters.n_iters)
                    ]
                )
        else:
            self.step_size = (
                torch.ones(self.model_parameters.n_iters)
                * self.model_parameters.step_size
            )

    @staticmethod
    def default_parameters():
        param = Parameter()
        param.learned_step = True
        param.step_positive = False
        param.step_size = [1.1183, 1.3568, 1.4271, 0.0808]
        param.n_iters = 4
        param.Unet_params = UNet.default_parameters()
        param.mode = "ct"
        return param

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print("Genzel, Martin, Jan Macdonald, and Maximilian März. ")
            print(
                '"AAPM DL-Sparse-View CT Challenge Submission Report: Designing an Iterative Network for Fanbeam-CT with Unknown Geometry." '
            )
            print("\x1B[3marXiv preprint \x1B[0m")
            print("arXiv:2106.00280 (2021).")

        elif cite_format == "bib":
            string = """
            @article{genzel2021aapm,
            title={AAPM DL-Sparse-View CT Challenge Submission Report: Designing an Iterative Network for Fanbeam-CT with Unknown Geometry},
            author={Genzel, Martin and Macdonald, Jan and M{\"a}rz, Maximilian},
            journal={arXiv preprint arXiv:2106.00280},
            year={2021}
            }"""
            print(string)
        else:
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )

    def forward(self, sino):

        B, C, W, H = sino.shape
        img = sino.new_zeros(B, 1, *self.geo.image_shape[1:])
        update = sino.new_zeros(B, 1, *self.geo.image_shape[1:])
        # Start from FDK
        for i in range(sino.shape[0]):
            img[i] = fdk(self.op, sino[i])

        for i in range(self.model_parameters.n_iters):
            unet = getattr(self, f"Unet_{i}")
            img = unet(img)

            for j in range(img.shape[0]):
                update[j] = self.step_size[i] * fdk(self.op, self.op(img[j]) - sino[j])
            img = img - update

        return img
