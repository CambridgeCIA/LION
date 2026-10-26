# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# Modifications: -
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

from LION.utils.tomosipo_import import import_tomosipo


import torch
import torch.nn as nn
import torch.nn.functional as F


def fdk(op, sino):
    """ts_algorithms' FDK, imported on first use (LION imports without the CT
    backend)."""
    return import_tomosipo("ts_algorithms", "LPD").fdk(op, sino)


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


def _drop_lambda_primal(module, state_dict, prefix, *args):
    """Ignore the (unused) lambda_primal steps of old checkpoints."""
    for key in [k for k in state_dict if k.startswith(prefix + "lambda_primal")]:
        del state_dict[key]


def _check_channels(params):
    """Number of primal/dual channels, from the output of the blocks.

    The dual block sees [h, A f, g] (n_dual + 2 channels) and the primal block
    [f, A^T h] (n_primal + 1 channels); both output an update of their variable.
    """
    n_dual, n_primal = params.data_channels[-1], params.reg_channels[-1]
    if params.data_channels[0] != n_dual + 2 or params.reg_channels[0] != n_primal + 1:
        raise ValueError(
            "data_channels must start with data_channels[-1] + 2 channels and "
            "reg_channels with reg_channels[-1] + 1, got "
            f"data_channels={params.data_channels}, reg_channels={params.reg_channels}"
        )
    # channel of the primal variable projected in the dual step: 1 as in Adler &
    # Oktem (the second primal channel); parameter files from before this option
    # existed used channel 0
    channel = getattr(params, "dual_step_primal_channel", 0)
    if not 0 <= channel < n_primal:
        raise ValueError(
            f"dual_step_primal_channel={channel} needs at least {channel + 1} primal channels"
        )
    return n_primal, n_dual, channel


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

        _check_channels(self.model_parameters)

        # Create pytorch compatible operators and send them to autograd
        self._make_operator()

        # Define step size
        if self.model_parameters.step_size is None:
            print("Step size is None, computing it with power method")
            # compute step size
            self.model_parameters.step_size = float(1 / power_method(self.op))
        # Step size of each iteration: the primal update adds step * A^T(h).
        # (A lambda_primal used to be created too, but was never used.)
        n_iters = self.model_parameters.n_iters
        step_size = self.model_parameters.step_size
        if self.model_parameters.learned_step:
            self.lambda_dual = nn.ParameterList(
                [nn.Parameter(torch.ones(1) * step_size) for _ in range(n_iters)]
            )
        else:
            self.register_buffer(
                "lambda_dual", torch.ones(n_iters) * step_size, persistent=False
            )
        # checkpoints saved before lambda_primal was removed still load
        self.register_load_state_dict_pre_hook(_drop_lambda_primal)

    @staticmethod
    def default_parameters():
        LPD_params = LIONModelParameter()
        LPD_params.n_iters = 10
        LPD_params.data_channels = [7, 32, 32, 5]
        LPD_params.reg_channels = [6, 32, 32, 5]
        LPD_params.learned_step = False
        LPD_params.step_size = None
        LPD_params.step_positive = False
        LPD_params.mode = "ct"
        LPD_params.instance_norm = False
        LPD_params.conv_bias = True
        # project the second primal channel in the dual step, as in Adler & Oktem
        LPD_params.dual_step_primal_channel = 1
        LPD_params.model_input_type = ModelInputType.SINOGRAM
        return LPD_params

    @staticmethod
    def continous_LPD_paper():
        LPD_params = LIONModelParameter()
        LPD_params.n_iters = 5
        LPD_params.data_channels = [7, 32, 32, 32, 5]
        LPD_params.reg_channels = [6, 32, 32, 32, 5]
        LPD_params.learned_step = True
        LPD_params.step_size = None
        LPD_params.step_positive = True
        LPD_params.mode = "ct"
        LPD_params.instance_norm = True
        LPD_params.conv_bias = False
        # project the second primal channel in the dual step, as in Adler & Oktem
        LPD_params.dual_step_primal_channel = 1
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

        n_primal, n_dual, c = _check_channels(self.model_parameters)

        # initialize parameters
        h = g.new_zeros(B, n_dual, W, H)
        f_primal = g.new_zeros(B, n_primal, *self.geometry.image_shape[1:])
        for i in range(B):
            aux = fdk(self.op, g[i, 0])
            aux = torch.clip(aux, min=0)
            for channel in range(n_primal):
                f_primal[i, channel] = aux

        for i in range(self.model_parameters.n_iters):
            primal_module = getattr(self, f"{i}_primal")
            dual_module = getattr(self, f"{i}_dual")
            f_dual = self.A(f_primal[:, c : c + 1])
            h = self.__dual_step(g, h, f_dual, dual_module)

            step = self.lambda_dual[i]
            if self.model_parameters.step_positive:
                step = step.abs()  # learned steps stay positive
            update = step * self.AT(h[:, :1])
            f_primal = self.__primal_step(f_primal, update, primal_module)

        return f_primal[:, 0:1]
