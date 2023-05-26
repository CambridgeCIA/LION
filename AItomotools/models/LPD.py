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
    CNN block of the primal variable
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
            layer_list.append(nn.Conv2d(channels[ii], channels[ii + 1], 3, padding=1, bias=False))
            # Have PReLUs all the way except the last
            if ii < layers - 1:
                layer_list.append(torch.nn.PReLU())
        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.block(x)


class LPD(nn.Module):
    """ Learn Primal Dual network
    """
    def __init__(
        self,
        geometry_parameters:ct.Geometry,
        model_parameters:Parameter
    ):
        super().__init__()
        # Pass all relevant parameters to internal storage. 
        self.geo=geometry_parameters
        self.model_parameters = model_parameters

        # Create layers per iteration
        for i in range(self.model_parameters.n_iters):
            self.add_module(
                f"{i}_primal", RegProximal(layers=len(self.model_parameters.reg_channels)-1, channels=self.model_parameters.reg_channels)
            )
            self.add_module(
                f"{i}_dual", dataProximal(layers=len(self.model_parameters.data_channels)-1, channels=self.model_parameters.data_channels)
            )
  
        # Create pytorch compatible operators and send them to aiutograd
        op = self.__make_operators(self.geo, self.model_parameters.mode)
        self.op=op
        self.A = to_autograd(op, num_extra_dims=1)
        self.AT = to_autograd(op.T, num_extra_dims=1)

        # Define step size
        if self.model_parameters.step_size is None:
            # compute step size
            self.model_parameters.step_size = 1 / power_method(op)

        
        # Are we learning the step? (with the above initialization)
        if self.model_parameters.learned_step:
            # Enforce positivity by making it 10^step
            if self.model_parameters.step_positive:
                self.lambda_dual = nn.ParameterList(
                    [nn.Parameter(torch.ones(1)* 10 ** np.log10(self.model_parameters.step_size)) for i in range(self.model_parameters.n_iters)]
                )
                self.lambda_primal = nn.ParameterList(
                    [nn.Parameter(torch.ones(1)* 10 ** np.log10(self.model_parameters.step_size)) for i in range(self.model_parameters.n_iters)]
                )
            # Negatives OK
            else:
                self.lambda_dual = nn.ParameterList(
                    [nn.Parameter(torch.ones(1) * self.model_parameters.step_size) for i in range(n_iters)]
                )
                self.lambda_primal = nn.ParameterList(
                    [nn.Parameter(torch.ones(1) * self.model_parameters.step_size) for i in range(n_iters)]
                )
        else:
            self.lambda_dual = torch.ones(n_iters) * self.model_parameters.step_size
            self.lambda_primal = torch.ones(n_iters) * self.model_parameters.step_size

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

    def get_parameters(self):
        return self.model_parameters, self.geo

    @staticmethod
    def __make_operators(geo, mode='ct'):
        if mode.lower() != "ct":
            raise NotImplementedError("Only CT operators supported")
        vg = ts.volume(shape=geo.image_shape, size=geo.image_size)
        pg = ts.cone(
            angles=geo.angles,
            shape=geo.detector_shape,
            size=geo.detector_size,
            src_orig_dist=geo.dso,
            src_det_dist=geo.dsd,
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

    @staticmethod
    def cite(cite_format="MLA"):
        if cite_format == "MLA":
            print("Adler, Jonas, and Ozan Öktem.")
            print("\"Learned primal-dual reconstruction.\"")
            print("\x1B[3mIEEE transactions on medical imaging \x1B[0m")
            print("37.6 (2018): 1322-1332.")
        elif cite_format=="bib":
            string="""
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
            raise AttributeError("cite_format not understood, only \"MLA\" and \"bib\" supported")      
        

    def forward(self, g):
        """
        g: sinogram input
        """

        B, C, W, H = g.shape

        if C != 1:
            raise NotImplementedError("Only 2D CT images supported")

        if len(self.geo.angles) != W or self.geo.detector_shape[1] != H:
            raise ValueError("geo description and sinogram size do not match")

        # initialize parameters
        h = g.new_zeros(B, 5, W, H)
        f_primal = g.new_zeros(B, 5, *self.geo.image_shape[1:])
        for i in range(B):
            aux=fdk(self.op, g[i,0])
            for channel in range(5):
                f_primal[i,channel]=aux

        for i in range(self.model_parameters.n_iters):
            primal_module = getattr(self, f"{i}_primal")
            dual_module = getattr(self, f"{i}_dual")
            f_dual = self.A(f_primal[:, :1]).cuda()
            h = self.__dual_step(g, h, f_dual, dual_module)

            update = self.lambda_dual[i]*self.AT(h[:, :1]).cuda()
            f_primal = self.__primal_step(f_primal, update, primal_module)

        return f_primal[:, 0:1]
