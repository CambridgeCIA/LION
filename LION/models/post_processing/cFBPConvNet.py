# This file is part of AItomotools library
# License : BSD-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================

from warnings import warn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint
from ts_algorithms import fdk
import LION.CTtools.ct_geometry as ct

from LION.models.LIONmodel import LIONmodel, ModelInputType
from LION.utils.parameter import LIONParameter

# Implementation of: continuous version of FBPConvNet


class InitialVelocity(nn.Module):
    """
    Initial velocity for second order ODE
    """

    def __init__(self, channels: int):
        super(InitialVelocity, self).__init__()

        self.initial_velocity = nn.Sequential(
            nn.InstanceNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.InstanceNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
        )

    def forward(self, x0):
        out = self.initial_velocity(x0)
        return torch.cat((x0, out), dim=1)


class ConvSODEFunc(nn.Module):
    """
    Second order ODE block
    """

    def __init__(self, channels, relu_type="ReLU", relu_last=True, kernel_size=3):
        super(ConvSODEFunc, self).__init__()
        # input parsing:
        self.nfe = 0  # Number of function evaluations
        layers = len(channels) - 1
        if layers < 1:
            raise ValueError("At least one layer required")
        # convolutional layers
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

    def forward(self, t, x):
        cutoff = int(x.shape[1] / 2)
        z = x[:, :cutoff]
        v = x[:, cutoff:]
        inp = torch.cat((z, v), dim=1)
        self.nfe += 1
        return self.block(inp)


class ODEBlock(nn.Module):
    def __init__(
        self,
        odefunc: nn.Module,
        tol: float = 1e-3,
        adjoint: bool = False,
    ):
        """
        Args:
            odefunc (nn.Module): the module to be evaluated
            tol (float): tolerance for the ODE solver
            adjoint (bool): whether to use the adjoint method for gradient calculation
            max_num_steps (int): maximum number of steps for ODE solver
        """
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, x, eval_times=None, solver="rk4"):
        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter
        self.odefunc.nfe = 0

        if eval_times is None:
            integration_time = torch.tensor([0, 1]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)

        if self.adjoint:
            out = odeint_adjoint(
                self.odefunc,
                x,
                integration_time,
                rtol=self.tol,
                atol=self.tol,
                method=solver,
            )
        else:
            out = odeint(
                self.odefunc,
                x,
                integration_time,
                rtol=self.tol,
                atol=self.tol,
                method=solver,
            )

        if eval_times is None:
            return out[1]  # Return only final time
        else:
            return out


class ConvODEFunc(nn.Module):
    def __init__(self, channels, relu_type="ReLU", relu_last=True, kernel_size=3):
        super(ConvODEFunc, self).__init__()
        # input parsing:
        self.nfe = 0  # Number of function evaluations
        layers = len(channels) - 1
        if layers < 1:
            raise ValueError("At least one layer required")
        # convolutional layers
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

    def forward(self, t, x):
        self.nfe += 1
        return self.block(x)


class Down(nn.Module):
    """Downscaling with maxpool"""

    def __init__(self):
        super().__init__()
        self.pool = nn.Sequential(nn.MaxPool2d(2))

    def forward(self, x):
        return self.pool(x)


class Up(nn.Module):
    """Upscaling with transpose conv"""

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


class cFBPConvNet(LIONmodel.LIONmodel):
    def __init__(
        self,
        geometry_parameters: ct.Geometry,
        model_parameters: LIONParameter = None,
    ):

        warn(
            "cFBPConvNet has not been succesfully tested yet, use with caution. \n If you manage to get it working, contact the authors."
        )
        super().__init__(model_parameters, geometry_parameters)
        self._make_operator()

        self.second_order = self.model_parameters.do_second_order
        self.solver = self.model_parameters.solver

        # continuous version of FBPConvNet:
        if self.second_order:
            self.initial_velocity = InitialVelocity(
                self.model_parameters.down_1_channels
            )
            # Down blocks
            ode_d1 = ConvSODEFunc(
                channels=self.model_parameters.down_1_channels,
                relu_type=self.model_parameters.activation,
            )
            ode_d2 = ConvSODEFunc(
                channels=self.model_parameters.down_2_channels,
                relu_type=self.model_parameters.activation,
            )
            ode_d3 = ConvSODEFunc(
                channels=self.model_parameters.down_3_channels,
                relu_type=self.model_parameters.activation,
            )
            ode_d4 = ConvSODEFunc(
                channels=self.model_parameters.down_4_channels,
                relu_type=self.model_parameters.activation,
            )
            # "latent space"
            ode_bottom = ConvSODEFunc(
                channels=self.model_parameters.latent_channels,
                relu_type=self.model_parameters.activation,
            )
            # Up blocks
            ode_u1 = ConvSODEFunc(
                channels=self.model_parameters.up_1_channels,
                relu_type=self.model_parameters.activation,
            )
            ode_u2 = ConvSODEFunc(
                channels=self.model_parameters.up_2_channels,
                relu_type=self.model_parameters.activation,
            )
            ode_u3 = ConvSODEFunc(
                channels=self.model_parameters.up_3_channels,
                relu_type=self.model_parameters.activation,
            )
            ode_u4 = ConvSODEFunc(
                channels=self.model_parameters.up_4_channels,
                relu_type=self.model_parameters.activation,
            )
        else:
            # Down blocks
            ode_d1 = ConvODEFunc(
                channels=self.model_parameters.down_1_channels,
                relu_type=self.model_parameters.activation,
            )
            ode_d2 = ConvODEFunc(
                channels=self.model_parameters.down_2_channels,
                relu_type=self.model_parameters.activation,
            )
            ode_d3 = ConvODEFunc(
                channels=self.model_parameters.down_3_channels,
                relu_type=self.model_parameters.activation,
            )
            ode_d4 = ConvODEFunc(
                channels=self.model_parameters.down_4_channels,
                relu_type=self.model_parameters.activation,
            )
            # "latent space"
            ode_bottom = ConvODEFunc(
                channels=self.model_parameters.latent_channels,
                relu_type=self.model_parameters.activation,
            )
            # Up blocks
            ode_u1 = ConvODEFunc(
                channels=self.model_parameters.up_1_channels,
                relu_type=self.model_parameters.activation,
            )
            ode_u2 = ConvODEFunc(
                channels=self.model_parameters.up_2_channels,
                relu_type=self.model_parameters.activation,
            )
            ode_u3 = ConvODEFunc(
                channels=self.model_parameters.up_3_channels,
                relu_type=self.model_parameters.activation,
            )
            ode_u4 = ConvODEFunc(
                channels=self.model_parameters.up_4_channels,
                relu_type=self.model_parameters.activation,
            )

        self.input_1x1_conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.model_parameters.down_1_channels[0],
            kernel_size=1,
        )
        self.block_1_down = ODEBlock(
            ode_d1,
            tol=self.model_parameters.tol,
            adjoint=self.model_parameters.ode_adjoint,
        )
        self.conv_down1_2 = nn.Conv2d(
            in_channels=self.model_parameters.down_1_channels[-1],
            out_channels=self.model_parameters.down_2_channels[0],
            kernel_size=1,
        )
        self.down_1 = Down()
        self.block_2_down = ODEBlock(
            ode_d2,
            tol=self.model_parameters.tol,
            adjoint=self.model_parameters.ode_adjoint,
        )
        self.conv_down2_3 = nn.Conv2d(
            in_channels=self.model_parameters.down_2_channels[-1],
            out_channels=self.model_parameters.down_3_channels[0],
            kernel_size=1,
        )
        self.down_2 = Down()
        self.block_3_down = ODEBlock(
            ode_d3,
            tol=self.model_parameters.tol,
            adjoint=self.model_parameters.ode_adjoint,
        )
        self.conv_down3_4 = nn.Conv2d(
            in_channels=self.model_parameters.down_3_channels[-1],
            out_channels=self.model_parameters.down_4_channels[0],
            kernel_size=1,
        )
        self.down_3 = Down()
        self.block_4_down = ODEBlock(
            ode_d4,
            tol=self.model_parameters.tol,
            adjoint=self.model_parameters.ode_adjoint,
        )
        self.conv_down4_bottom = nn.Conv2d(
            in_channels=self.model_parameters.down_4_channels[-1],
            out_channels=self.model_parameters.latent_channels[0],
            kernel_size=1,
        )
        self.down_4 = Down()
        self.block_bottom = ODEBlock(
            ode_bottom,
            tol=self.model_parameters.tol,
            adjoint=self.model_parameters.ode_adjoint,
        )
        self.up_1 = Up(
            [
                self.model_parameters.latent_channels[-1],
                self.model_parameters.up_1_channels[0],
            ],
            relu_type=self.model_parameters.activation,
        )
        self.conv_up1_2 = nn.Conv2d(
            in_channels=self.model_parameters.latent_channels[-1],
            out_channels=self.model_parameters.up_1_channels[0],
            kernel_size=1,
        )
        self.block_1_up = ODEBlock(
            ode_u1,
            tol=self.model_parameters.tol,
            adjoint=self.model_parameters.ode_adjoint,
        )
        self.up_2 = Up(
            [
                self.model_parameters.up_1_channels[-1],
                self.model_parameters.up_2_channels[0],
            ],
            relu_type=self.model_parameters.activation,
        )
        self.conv_up2_3 = nn.Conv2d(
            in_channels=self.model_parameters.up_1_channels[-1],
            out_channels=self.model_parameters.up_2_channels[0],
            kernel_size=1,
        )
        self.block_2_up = ODEBlock(
            ode_u2,
            tol=self.model_parameters.tol,
            adjoint=self.model_parameters.ode_adjoint,
        )
        self.up_3 = Up(
            [
                self.model_parameters.up_2_channels[-1],
                self.model_parameters.up_3_channels[0],
            ],
            relu_type=self.model_parameters.activation,
        )
        self.conv_up3_4 = nn.Conv2d(
            in_channels=self.model_parameters.up_2_channels[-1],
            out_channels=self.model_parameters.up_3_channels[0],
            kernel_size=1,
        )
        self.block_3_up = ODEBlock(
            ode_u3,
            tol=self.model_parameters.tol,
            adjoint=self.model_parameters.ode_adjoint,
        )
        self.up_4 = Up(
            [
                self.model_parameters.up_3_channels[-1],
                self.model_parameters.up_4_channels[0],
            ],
            relu_type=self.model_parameters.activation,
        )

        self.block_4_up = ODEBlock(
            ode_u4,
            tol=self.model_parameters.tol,
            adjoint=self.model_parameters.ode_adjoint,
        )
        self.conv_up4_last = nn.Conv2d(
            in_channels=self.model_parameters.up_4_channels[-1] * 2,
            out_channels=self.model_parameters.last_block[0],
            kernel_size=1,
        )
        self.block_last = nn.Sequential(
            nn.Conv2d(
                self.model_parameters.last_block[0],
                self.model_parameters.last_block[1],
                self.model_parameters.last_block[2],
                padding=0,
            )
        )

    @staticmethod
    def default_parameters():
        params = LIONParameter()
        params = ModelInputType.IMAGE
        params.down_1_channels = [64, 64, 64, 64]
        params.down_2_channels = [128, 128, 128]
        params.down_3_channels = [256, 256, 256]
        params.down_4_channels = [512, 512, 512]

        params.latent_channels = [1024, 1024, 1024]

        params.up_1_channels = [512, 512, 512]
        params.up_2_channels = [256, 256, 256]
        params.up_3_channels = [128, 128, 128]
        params.up_4_channels = [64, 64, 64]

        params.last_block = [64, 1, 1]

        params.activation = "ReLU"

        params.tol = (1e-3,)
        params.ode_adjoint = (False,)
        params.do_second_order = (False,)
        params.solver = ("rk4",)
        return params

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print(" Jin, Kyong Hwan, et al.")
            print('"Continuous Learned Primal Dual"')
            print("\x1B[3m ArXiv \x1B[0m")
            print("(2024): 2405.02478.")
        elif cite_format == "bib":
            string = """
            @article{runkel2024continuous,
            title={Continuous Learned Primal Dual},
            author={Runkel, Christina and Biguri, Ander and Sch{\"o}nlieb, Carola-Bibiane},
            journal={arXiv preprint arXiv:2405.02478},
            year={2024}
            }"""
            print(string)
        else:
            raise AttributeError(
                'cite_format not understood, only "MLA" and "bib" supported'
            )

    def forward(self, x):
        B, C, W, H = x.shape

        image = x.new_zeros(B, 1, *self.geo.image_shape[1:])
        for i in range(B):
            aux = fdk(self.op, x[i, 0])
            aux = torch.clip(aux, min=0)
            image[i] = aux

        if self.second_order:
            x = self.initial_velocity(x)
        inp = self.input_1x1_conv(x)
        block_1_res = self.block_1_down(inp)
        block_2_res = self.block_2_down(self.down_1(self.conv_down1_2(block_1_res)))
        block_3_res = self.block_3_down(self.down_2(self.conv_down2_3(block_2_res)))
        block_4_res = self.block_4_down(self.down_3(self.conv_down3_4(block_3_res)))
        res = self.block_bottom(self.down_4(self.conv_down4_bottom(block_4_res)))
        res = self.block_1_up(
            self.conv_up1_2(torch.cat((block_4_res, self.up_1(res)), dim=1))
        )
        res = self.block_2_up(
            self.conv_up2_3(torch.cat((block_3_res, self.up_2(res)), dim=1))
        )
        res = self.block_3_up(
            self.conv_up3_4(torch.cat((block_2_res, self.up_3(res)), dim=1))
        )
        res = self.block_4_up(
            self.conv_up4_last(torch.cat((block_1_res, self.up_4(res)), dim=1))
        )
        res = self.block_last(res)
        return x + res
