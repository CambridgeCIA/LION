from AItomotools.utils.math import power_method
from AItomotools.utils.parameter import Parameter
import AItomotools.CTtools.ct_geometry as ct

import numpy as np

import tomosipo as ts
from tomosipo.torch_support import to_autograd
from ts_algorithms import  fdk

import torch
import torch.nn as nn
import torch.nn.functional as F

class dataProximal(nn.Module):
    """
    CNN block of the dual variable
    """

    def __init__(self, layers, channels):

        super().__init__()
        # imput parsing
        if len(channels) != layers+1:
            raise ValueError(
                "Second input (channels) should have as many elements as layers your network has"
            )
        if layers < 1:
            raise ValueError("At least one layer required")
        # convolutional layers
        layer_list = []
        for ii in range(layers):
            layer_list.append(nn.Conv2d(channels[ii], channels[ii + 1], 3, padding=1, bias=False))
            # Have PReLUs all the way except the last
            if ii < layers - 1:
                layer_list.append(torch.nn.PReLU())
        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.block(x)


class RegProximal(nn.Module):
    """
    CNN bloack of the primal variable
    """

    def __init__(self, layers, channels):
        super().__init__()
        if len(channels) != layers+1:
            raise ValueError(
                "Second input (channels) should have as many elements as layers your network has"
            )
        if layers < 1:
            raise ValueError("At least one layer required")

        layer_list = []
        for ii in range(layers):
            layer_list.append(nn.Conv2d(channels[ii], channels[ii + 1], 3,padding=1, bias=False))
            # Have PReLUs all the way except the last
            if ii < layers - 1:
                layer_list.append(torch.nn.PReLU())
        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.block(x)


class LPD(nn.Module):
    def __init__(
        self,
        n_iters,
        geo,
        angles,
        mode="ct",
        data_channels=[7, 32, 32, 5],
        reg_channels=[6, 32, 32, 5],
        learned_step=True,
        step_size=None,
        step_positive=True   # I found it hard to achieve good performance with this set to false. 
    ):
        super().__init__()

        self.geo=geo
        self.n_iters = n_iters
        for i in range(n_iters):
            self.add_module(
                f"{i}_primal", RegProximal(layers=len(reg_channels)-1, channels=reg_channels)
            )
            self.add_module(
                f"{i}_dual", dataProximal(layers=len(data_channels)-1, channels=data_channels)
            )

        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn(10, 10)) for i in range(10)]
        )
        op = self.__make_operators(geo,angles, mode)
        self.op=op
        self.A = to_autograd(op)
        self.AT = to_autograd(op.T)

        if step_size is None:
            # compute step size
            self.step_size = 1 / power_method(op)
        else:
            self.step_size=step_size
        if learned_step:
            if step_positive:
                self.lambda_dual = nn.ParameterList(
                    [nn.Parameter(torch.ones(1)* 10 ** np.log10(self.step_size)) for i in range(n_iters)]
                )
                self.lambda_primal = nn.ParameterList(
                    [nn.Parameter(torch.ones(1)* 10 ** np.log10(self.step_size)) for i in range(n_iters)]
                )
            else:
                self.lambda_dual = nn.ParameterList(
                    [nn.Parameter(torch.ones(1) * self.step_size) for i in range(n_iters)]
                )
                self.lambda_primal = nn.ParameterList(
                    [nn.Parameter(torch.ones(1) * self.step_size) for i in range(n_iters)]
                )
        else:
            self.lambda_dual = torch.ones(n_iters) * self.step_size
            self.lambda_primal = torch.ones(n_iters) * self.step_size


    @staticmethod
    def default_parameters(mode='ct'):
        LPD_params=Parameter()
        LPD_params.n_iters=10
        LPD_params.mode=mode
        LPD_params.data_channels=[7, 32, 32, 5]
        LPD_params.reg_channels=[6, 32, 32, 5]
        LPD_params.learned_step=True
        LPD_params.step_size=None
        LPD_params.step_positive=True
        return LPD_params
        
    @staticmethod
    def __make_operators(geo, angles, mode='ct'):
        if mode.lower() != "ct":
            raise NotImplementedError("Only CT operators supported")
        vg = ts.volume(shape=geo.nVoxel, size=geo.sVoxel)
        pg = ts.cone(
            angles=angles,
            shape=geo.detector_shape,
            size=geo.detector_size,
            src_orig_dist=geo.DSO,
            src_det_dist=geo.DSD,
        )
        A = ts.operator(vg, pg)
        return A

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


    def forward(self, g):
        """
        geo: tigre style geo
        """

        B, C, W, H = g.shape

        if C != 1:
            raise NotImplementedError("Only 2D CT images supported")

        # TODO: have a default geo?
        if len(self.geo.angles) != W or self.geo.nDetector[1] != H:
            raise ValueError("geo description and sinogram size do not match")

        # initialize parameters
        h = g.new_zeros(B, 5, W, H)
        f_primal = g.new_zeros(B, 5, *self.geo.nVoxel[1:])
        for i in range(B):
            aux=fdk(self.op, g[i,0])
            for channel in range(5):
                f_primal[i,channel]=aux
        
        for i in range(self.n_iters):
            primal_module = getattr(self, f"{i}_primal")
            dual_module = getattr(self, f"{i}_dual")
            f_dual = self.A(f_primal[:, :1]).cuda()
            h = self.__dual_step(g, h, f_dual, dual_module)

            update = self.lambda_dual[i]*self.AT(h[:, :1]).cuda()
            f_primal = self.__primal_step(f_primal, update, primal_module)

        return f_primal[:, 0:1]
