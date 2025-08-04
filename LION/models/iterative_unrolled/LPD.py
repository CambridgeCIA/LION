# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================


from LION.models.LIONmodel import LIONmodel, ModelInputType

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

import torch
import torch.nn as nn
import torch.nn.functional as F


class dataProximal(nn.Module):
    """
    CNN block of the dual variable
    """

    def __init__(self, layers, channels, conv_bias, instance_norm=False):

        super().__init__()
        # imput parsing
        if len(channels) != layers + 1:
            raise ValueError(
                "Second input (channels) should have as many elements as layers your network has"
            )
        if layers < 1:
            raise ValueError("At least one layer required")
        # convolutional layers
        layer_list = []
        for ii in range(layers):
            if instance_norm:
                layer_list.append(nn.InstanceNorm2d(channels[ii]))
            # PReLUs and 3x3 kernels all the way except the last
            if ii < layers - 1:
                layer_list.append(
                    nn.Conv2d(
                        channels[ii], channels[ii + 1], 3, padding=1, bias=conv_bias
                    )
                )
                layer_list.append(nn.PReLU())
            else:
                layer_list.append(
                    nn.Conv2d(
                        channels[ii], channels[ii + 1], 1, padding=0, bias=conv_bias
                    )
                )
        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.block(x)


class RegProximal(nn.Module):
    """
    CNN block of the primal variable
    """

    def __init__(self, layers, channels, conv_bias, instance_norm=False):
        super().__init__()
        if len(channels) != layers + 1:
            raise ValueError(
                "Second input (channels) should have as many elements as layers your network has"
            )
        if layers < 1:
            raise ValueError("At least one layer required")

        layer_list = []
        for ii in range(layers):
            if instance_norm:
                layer_list.append(nn.InstanceNorm2d(channels[ii]))
            # PReLUs and 3x3 kernels all the way except the last
            if ii < layers - 1:
                layer_list.append(
                    nn.Conv2d(
                        channels[ii], channels[ii + 1], 3, padding=1, bias=conv_bias
                    )
                )
                layer_list.append(nn.PReLU())
            else:
                layer_list.append(
                    nn.Conv2d(
                        channels[ii], channels[ii + 1], 1, padding=0, bias=conv_bias
                    )
                )
        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.block(x)


class LPD(LIONmodel):
    """Learned Primal Dual network"""

    def __init__(
        self,
        geometry: ct.Geometry,
        model_parameters: LIONParameter = None,
    ):

        if geometry is None:
            raise ValueError("Geometry parameters required. ")

        super().__init__(model_parameters, geometry)
        # Pass all relevant parameters to internal storage.
        # AItomotmodel does this:
        # self.geometry = geometry
        # self.model_parameters = model_parameters

        # Create layers per iteration
        for i in range(self.model_parameters.n_iters):
            self.add_module(
                f"{i}_primal",
                RegProximal(
                    layers=len(self.model_parameters.reg_channels) - 1,
                    channels=self.model_parameters.reg_channels,
                    conv_bias=self.model_parameters.conv_bias,
                    instance_norm=self.model_parameters.instance_norm,
                ),
            )
            self.add_module(
                f"{i}_dual",
                dataProximal(
                    layers=len(self.model_parameters.data_channels) - 1,
                    channels=self.model_parameters.data_channels,
                    conv_bias=self.model_parameters.conv_bias,
                    instance_norm=self.model_parameters.instance_norm,
                ),
            )

        # Create pytorch compatible operators and send them to autograd
        self._make_operator()

        # Define step size
        if self.model_parameters.step_size is None:
            print("Step size is None, computing it with power method")
            # compute step size
            self.model_parameters.step_size = 1 / power_method(self.op)
        # Are we learning the step? (with the above initialization)
        if self.model_parameters.learned_step:
            # Enforce positivity by making it 10^step
            if self.model_parameters.step_positive:
                self.lambda_dual = nn.ParameterList(
                    [
                        nn.Parameter(
                            torch.ones(1)
                            * 10 ** np.log10(self.model_parameters.step_size)
                        )
                        for i in range(self.model_parameters.n_iters)
                    ]
                )
                self.lambda_primal = nn.ParameterList(
                    [
                        nn.Parameter(
                            torch.ones(1)
                            * 10 ** np.log10(self.model_parameters.step_size)
                        )
                        for i in range(self.model_parameters.n_iters)
                    ]
                )
            # Negatives OK
            else:
                self.lambda_dual = nn.ParameterList(
                    [
                        nn.Parameter(torch.ones(1) * self.model_parameters.step_size)
                        for i in range(self.model_parameters.n_iters)
                    ]
                )
                self.lambda_primal = nn.ParameterList(
                    [
                        nn.Parameter(torch.ones(1) * self.model_parameters.step_size)
                        for i in range(self.model_parameters.n_iters)
                    ]
                )
        else:
            self.lambda_dual = (
                torch.ones(self.model_parameters.n_iters)
                * self.model_parameters.step_size
            )
            self.lambda_primal = (
                torch.ones(self.model_parameters.n_iters)
                * self.model_parameters.step_size
            )

    @staticmethod
    def default_parameters():
        LPD_params = LIONParameter()
        LPD_params.n_iters = 10
        LPD_params.data_channels = [7, 32, 32, 5]
        LPD_params.reg_channels = [6, 32, 32, 5]
        LPD_params.learned_step = False
        LPD_params.step_size = None
        LPD_params.step_positive = False
        LPD_params.mode = "ct"
        LPD_params.instance_norm = False
        LPD_params.conv_bias = True
        LPD_params.model_input_type = ModelInputType.SINOGRAM
        return LPD_params

    @staticmethod
    def continous_LPD_paper():
        LPD_params = LIONParameter()
        LPD_params.n_iters = 5
        LPD_params.data_channels = [7, 32, 32, 32, 5]
        LPD_params.reg_channels = [6, 32, 32, 32, 5]
        LPD_params.learned_step = True
        LPD_params.step_size = None
        LPD_params.step_positive = True
        LPD_params.mode = "ct"
        LPD_params.instance_norm = True
        LPD_params.conv_bias = False
        LPD_params.model_input_type = ModelInputType.SINOGRAM
        return LPD_params

    @staticmethod
    def __dual_step(g, h, f, module):
        x = torch.cat((h, f, g), dim=1)
        out = module(x)
        return h + out

    @staticmethod
    def __primal_step(f, update, module):
        x = torch.cat((f, update), dim=1)
        out = module(x)
        return f + out

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print("Adler, Jonas, and Öktem, Ozan.")
            print('"Learned primal-dual reconstruction."')
            print("\x1B[3mIEEE transactions on medical imaging \x1B[0m")
            print("37.6 (2018): 1322-1332.")
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

    def forward(self, g):
        """
        g: sinogram input
        """
        B, C, W, H = g.shape

        if C != 1:
            raise NotImplementedError("Only 2D CT images supported")

        if len(self.geometry.angles) != W or self.geometry.detector_shape[1] != H:
            raise ValueError("geo description and sinogram size do not match")

        # initialize parameters
        h = g.new_zeros(B, 5, W, H)
        f_primal = g.new_zeros(B, 5, *self.geometry.image_shape[1:])
        for i in range(B):
            aux = fdk(self.op, g[i, 0])
            aux = torch.clip(aux, min=0)
            for channel in range(5):
                f_primal[i, channel] = aux

        for i in range(self.model_parameters.n_iters):
            primal_module = getattr(self, f"{i}_primal")
            dual_module = getattr(self, f"{i}_dual")
            f_dual = self.A(f_primal[:, :1])
            h = self.__dual_step(g, h, f_dual, dual_module)

            update = self.lambda_dual[i] * self.AT(h[:, :1])
            f_primal = self.__primal_step(f_primal, update, primal_module)

        return f_primal[:, 0:1]
