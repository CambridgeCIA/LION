# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# From: https://github.com/XXXXX
# Authors: Ferdia Sherry, Max Kiss, Ander Biguri
# =============================================================================

import torch.nn as nn
import torch
#import math
#import pdb

from LION.models import LIONmodel
#from LION.utils.math import power_method
from LION.utils.parameter import LIONParameter
import LION.CTtools.ct_geometry as ct

import tomosipo as ts
from tomosipo.torch_support import to_autograd
from ts_algorithms import fdk

#in_channels =3

class FBPDnCNN(LIONmodel.LIONmodel):
    def __init__(
        self, geometry_parameters: ct.Geometry, 
        in_channels=1,
        kernel_size=(3, 3),
        int_channels=64,
        blocks=20,
        residual=True,
        bias_free=True,
        model_parameters: LIONParameter = None
    ):
        super().__init__(model_parameters, geometry_parameters)
        self._make_operator()
        self._bias_free = bias_free
        self._residual = residual
        self.lift = torch.nn.Conv2d(
            in_channels,
            int_channels,
            kernel_size,
            padding=tuple(k // 2 for k in kernel_size),
            bias=not bias_free,
        )
        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    int_channels,
                    int_channels,
                    kernel_size,
                    padding=tuple(k // 2 for k in kernel_size),
                    bias=not bias_free,
                )
                for _ in range(blocks - 2)
            ]
        )
        self.bns = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm2d(int_channels, affine=not bias_free)
                for _ in range(blocks - 2)
            ]
        )
        self.project = torch.nn.Conv2d(
            int_channels,
            in_channels,
            kernel_size,
            padding=tuple(k // 2 for k in kernel_size),
            bias=not bias_free,
        )

    def _set_weights_zero(self):
        for conv in self.convs:
            conv.weight.data.zero_()
            if not self._bias_free:
                conv.bias.data.zero_()

    @staticmethod
    def default_parameters():
        FBPDnCNN_params = LIONParameter()

        return FBPDnCNN_params

    def forward(self, x):
        B, C, W, H = x.shape

        image = x.new_zeros(B, 1, *self.geo.image_shape[1:])
        for i in range(B):
            aux = fdk(self.op, x[i, 0])
            aux = torch.clip(aux, min=0)
            image[i] = aux

        z = torch.nn.functional.relu(self.lift(image))
        for conv, bn in zip(self.convs, self.bns):
            z = torch.nn.functional.leaky_relu(bn(conv(z)))
        if self._residual:
            return image - self.project(z)
        else:
            return self.project(z)
