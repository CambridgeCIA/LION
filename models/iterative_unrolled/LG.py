# This file is part of LION library
# License : GPL-3
#
# Authors: Zak Shumaylov, Max Kiss, Ander Biguri
# Based on: https://arxiv.org/pdf/1704.04058.pdf
# =============================================================================

from LION.models.LIONmodel import LIONmodel, LIONModelParameter, ModelInputType

from LION.utils.math import power_method
from LION.utils.parameter import LIONParameter
import LION.CTtools.ct_geometry as ct
import LION.CTtools.ct_utils as ct_utils
import LION.utils.utils as ai_utils

import numpy as np
from pathlib import Path
import warnings

import tomosipo as ts
from tomosipo.torch_support import to_autograd
from ts_algorithms import fdk
from LION.utils.math import power_method

import torch
import torch.nn as nn
import torch.nn.functional as F


class LGblock(nn.Module):
    def __init__(self, channels):
        super(LGblock, self).__init__()

        layers = len(channels) - 1

        layer_list = []
        for ii in range(layers):
            layer_list.append(nn.Conv2d(channels[ii], channels[ii + 1], 3, padding=1))
            # Have PReLUs all the way except the last
            if ii < layers - 1:
                layer_list.append(torch.nn.ReLU())
        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.block(x)


class LG(LIONmodel.LIONmodel):
    def __init__(self, geometry: ct.Geometry, model_parameters: LIONParameter = None):
        super().__init__(model_parameters, geometry)
        self.geometry = geometry
        self.channels = self.model_parameters.channels
        self.M = self.channels[-1] - 1
        self.n_iters = self.model_parameters.n_iters

        self._make_operator()
        gd_step = 1 / (power_method(self.op)) ** 2

        for i in range(self.n_iters):
            self.add_module(f"{i}_conv", LGblock(self.model_parameters.channels))

        self.step_size = nn.ParameterList(
            [nn.Parameter(torch.ones(1) * gd_step) for i in range(self.n_iters)]
        )
        self.tv_step_size = nn.ParameterList(
            [nn.Parameter(torch.ones(1) * 1e-4) for i in range(self.n_iters)]
        )

    @staticmethod
    def default_parameters():
        LG_params = LIONModelParameter()
        LG_params.channels = [8, 32, 32, 6]
        LG_params.n_iters = 5
        LG_params.model_input_type = ModelInputType.SINOGRAM
        return LG_params

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print("Adler, Jonas, and Öktem, Ozan.")
            print(
                '"Solving ill-posed inverse problems using iterative deep neural networks."'
            )
            print("Inverse Problems")
            print("33.12 (2017): 124007.")
        elif cite_format == "bib":
            string = """
            @article{adler2018learned,
            title={Learned primal-dual reconstruction},
            author={Adler, Jonas and {\"O}ktem, Ozan},
            journal={IEEE transactions on medical imaging},
            volume={37},
            number={6},
            pages={1322--1332},
            year={2018},
            publisher={IEEE}
            }"""
            print(string)
        else:
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )

    def gradientTVnorm2D(self, f):
        Gx = torch.diff(f, dim=2)
        Gy = torch.diff(f, dim=3)
        tvg = torch.zeros_like(f)

        Gx = torch.cat(
            (torch.zeros_like(Gx[:, :, :1, :]), Gx), dim=2
        )  # Pad Gx with zeros
        Gy = torch.cat(
            (torch.zeros_like(Gy[:, :, :, :1]), Gy), dim=3
        )  # Pad Gy with zeros

        nrm = torch.sqrt(Gx**2 + Gy**2 + 1e-6)

        tvg[:, :, :, :] = (
            tvg[:, :, :, :] + (Gx[:, :, :, :] + Gy[:, :, :, :]) / nrm[:, :, :, :]
        )
        tvg[:, :, :-1, :] = tvg[:, :, :-1, :] - Gx[:, :, 1:, :] / nrm[:, :, 1:, :]
        tvg[:, :, :, :-1] = tvg[:, :, :, :-1] - Gy[:, :, :, 1:] / nrm[:, :, :, 1:]

        return tvg

    def forward(self, x):

        B, C, W, H = x.shape

        f = x.new_zeros(B, 1, *self.geometry.image_shape[1:])
        for i in range(B):
            aux = fdk(self.op, x[i, 0])
            aux = torch.clip(aux, min=0)
            f[i] = aux

        tmp = torch.zeros(f.shape).type_as(f)
        s = tmp.clone()
        for i in range(self.M - 1):
            s = torch.cat((s, tmp), dim=1)

        for i in range(self.n_iters):
            conv = getattr(self, f"{i}_conv")
            del_L = self.step_size[i] * self.AT(x - self.A(f))
            del_S = self.tv_step_size[i] * self.gradientTVnorm2D(f)
            output = conv(torch.cat([f, s, del_L, del_S], dim=1))
            # output last channel
            f = f + output[:, self.M : self.M + 1]
            s = nn.ReLU()(output[:, 0 : self.M])
        return f
