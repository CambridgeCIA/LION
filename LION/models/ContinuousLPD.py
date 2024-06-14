# This file is part of AItomotools library
# License : BSD-3
#
# Author  : Ander Biguri & Christina Runkel
# Modifications: -
# =============================================================================


from LION.models import LIONmodel

from LION.utils.math import power_method
from LION.utils.parameter import Parameter
import LION.CTtools.ct_geometry as ct

# import LION.CTtools.ct_utils as ct_utils
# import LION.utils.utils as ai_utils

import numpy as np

from pathlib import Path
import warnings

# import tomosipo as ts
from tomosipo.torch_support import to_autograd
from ts_algorithms import fdk

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint


class ConvODEFunc(nn.Module):
    """
    First order ODE block
    """

    def __init__(self, layers: int, channels: list, instance_norm: bool = False):
        super(ConvODEFunc, self).__init__()
        self.nfe = 0  # Number of function evaluations
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
                    nn.Conv2d(channels[ii], channels[ii + 1], 3, padding=1, bias=False)
                )
                layer_list.append(nn.PReLU())
            else:
                layer_list.append(
                    nn.Conv2d(channels[ii], channels[ii + 1], 1, padding=0, bias=False)
                )
        self.block = nn.Sequential(*layer_list)

    def forward(self, t, x):
        self.nfe += 1
        return self.block(x)


class InitialVelocity(nn.Module):
    """
    Initial velocity for second order ODE
    """

    def __init__(self, channels: int, instance_norm: bool = False):
        super(InitialVelocity, self).__init__()
        if instance_norm:
            self.initial_velocity = nn.Sequential(
                nn.InstanceNorm2d(channels),
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                nn.PReLU(),
                nn.InstanceNorm2d(channels),
                nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
                nn.PReLU(),
            )
        else:
            self.initial_velocity = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                nn.PReLU(),
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

    def __init__(self, layers: int, channels: list, instance_norm: bool = False):
        super(ConvSODEFunc, self).__init__()
        self.nfe = 0  # Number of function evaluations
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
                        channels[ii] * 2, channels[ii + 1] * 2, 3, padding=1, bias=False
                    )
                )
                layer_list.append(nn.PReLU())
            else:
                layer_list.append(
                    nn.Conv2d(
                        channels[ii] * 2, channels[ii + 1] * 2, 1, padding=0, bias=False
                    )
                )
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


class ContinuousDataProximal(nn.Module):
    def __init__(
        self,
        layers: int,
        channels: list,
        tol: float = 1e-3,
        adjoint: bool = True,
        second_order: bool = False,
        solver: str = "rk4",
        instance_norm: bool = False,
    ):
        """
        Args:
            layers(int): number of layers
            channels (list): list of number of channels per layer
            tol (float): tolerance to be used for ODE solver
            adjoint (bool): whether to use the adjoint method to calculate the gradients
            second_order (bool): use second order ODE instead of first order if 'True'
            solver (str): ODE solver to use
        """
        super(ContinuousDataProximal, self).__init__()
        self.second_order = second_order
        self.solver = solver

        if second_order:
            self.initial_velocity = InitialVelocity(
                channels=channels[1], instance_norm=instance_norm
            )
            ode = ConvSODEFunc(
                layers=layers - 2, channels=channels[1:-1], instance_norm=instance_norm
            )
            self.last_layer = nn.Conv2d(
                in_channels=channels[-2] * 2, out_channels=channels[-1], kernel_size=1
            )
        else:
            ode = ConvODEFunc(
                layers=layers - 2, channels=channels[1:-1], instance_norm=instance_norm
            )
            self.last_layer = nn.Conv2d(
                in_channels=channels[-2], out_channels=channels[-1], kernel_size=1
            )
        self.odeblock = ODEBlock(ode, tol=tol, adjoint=adjoint)
        self.first_layer = nn.Conv2d(
            in_channels=channels[0], out_channels=channels[1], kernel_size=1
        )

    def forward(self, x):
        x = self.first_layer(x)
        if self.second_order:
            x = self.initial_velocity(x)
        print(x.shape)
        x = self.odeblock(x, solver=self.solver)
        return self.last_layer(x)


class ContinuousRegProximal(nn.Module):
    def __init__(
        self,
        layers: int,
        channels: list,
        tol: float = 1e-3,
        adjoint: bool = False,
        second_order: bool = False,
        solver: str = "rk4",
        instance_norm: bool = False,
    ):
        """
        layers(int): number of layers
            channels (list): list of number of channels per layer
            tol (float): tolerance to be used for ODE solver
            adjoint (bool): whether to use the adjoint method to calculate the gradients
            second_order (bool): use second order ODE instead of first order if 'True'
            solver (str): ODE solver to use
        """
        super(ContinuousRegProximal, self).__init__()
        self.second_order = second_order
        self.solver = solver

        if second_order:
            self.initial_velocity = InitialVelocity(
                channels=channels[1], instance_norm=instance_norm
            )
            ode = ConvSODEFunc(
                layers=layers - 2, channels=channels[1:-1], instance_norm=instance_norm
            )
            self.last_layer = nn.Conv2d(
                in_channels=channels[-2] * 2, out_channels=channels[-1], kernel_size=1
            )
        else:
            ode = ConvODEFunc(
                layers=layers - 2, channels=channels[1:-1], instance_norm=instance_norm
            )
            self.last_layer = nn.Conv2d(
                in_channels=channels[-2], out_channels=channels[-1], kernel_size=1
            )
        self.odeblock = ODEBlock(ode, tol=tol, adjoint=adjoint)
        self.first_layer = nn.Conv2d(
            in_channels=channels[0], out_channels=channels[1], kernel_size=1
        )

    def forward(self, x):
        x = self.first_layer(x)
        if self.second_order:
            x = self.initial_velocity(x)
        x = self.odeblock(x, solver=self.solver)
        return self.last_layer(x)


class ContinuousLPD(LIONmodel.LIONmodel):
    """Learn Primal Dual network with continuous blocks"""

    def __init__(
        self,
        geometry_parameters: ct.Geometry,
        model_parameters: Parameter = None,
        second_order: bool = False,
        instance_norm: bool = False,
    ):
        if geometry_parameters is None:
            raise ValueError("Geometry parameters required. ")

        super().__init__(model_parameters, geometry_parameters)
        # Pass all relevant parameters to internal storage.
        # AItomotmodel does this:
        # self.geo = geometry_parameters
        # self.model_parameters = model_parameters

        # Create layers per iteration
        for i in range(self.model_parameters.n_iters):
            self.add_module(
                f"{i}_primal",
                ContinuousRegProximal(
                    layers=len(self.model_parameters.reg_channels) - 1,
                    channels=self.model_parameters.reg_channels,
                    second_order=second_order,
                    instance_norm=instance_norm,
                ),
            )
            self.add_module(
                f"{i}_dual",
                ContinuousDataProximal(
                    layers=len(self.model_parameters.data_channels) - 1,
                    channels=self.model_parameters.data_channels,
                    second_order=second_order,
                    instance_norm=instance_norm,
                ),
            )

        # Create pytorch compatible operators and send them to aiutograd
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
        LPD_params = Parameter()
        LPD_params.n_iters = 10
        LPD_params.data_channels = [7, 32, 32, 32, 5]
        LPD_params.reg_channels = [6, 32, 32, 32, 5]
        LPD_params.learned_step = False
        LPD_params.step_size = None
        LPD_params.step_positive = False
        LPD_params.mode = "ct"

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
            print("Adler, Jonas, and Ozan Öktem.")
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

        if len(self.geo.angles) != W or self.geo.detector_shape[1] != H:
            raise ValueError("geo description and sinogram size do not match")

        # initialize parameters
        h = g.new_zeros(B, 5, W, H)
        f_primal = g.new_zeros(B, 5, *self.geo.image_shape[1:])
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
    
    @staticmethod
    def _load_parameter_file(fname, supress_warnings=False):
        # Load the actual parameters
        ##############################
        options = Parameter()
        options.load(fname.with_suffix(".json"))
        if hasattr(options, "geometry_parameters"):
            options.geometry_parameters = ct.Geometry._init_from_parameter(
                options.geometry_parameters
            )
        # Error check
        ################################
        # Check if model has been changed since save.
        # if not hasattr(options, "commit_hash") and not supress_warnings:
        #     warnings.warn(
        #         "\nNo commit hash found. This model was not saved with the standard AItomotools function and it will likely fail to load.\n"
        #     )
        # else:
        #     curr_commit_hash = ai_utils.get_git_revision_hash()
        #     curr_class_path = cls.current_file()
        #     curr_aitomomodel_path = Path(__file__)
        #     if (
        #         ai_utils.check_if_file_changed_git(
        #             curr_class_path, options.commit_hash, curr_commit_hash
        #         )
        #         or ai_utils.check_if_file_changed_git(
        #             curr_aitomomodel_path, options.commit_hash, curr_commit_hash
        #         )
        #         and not supress_warnings
        #     ):
        #         warnings.warn(
        #             f"\nThe code for the model has changed since it was saved, loading it may fail. This model was saved in {options.commit_hash}\n"
        #         )
        return options
    
    @staticmethod
    def _load_data(fname, supress_warnings=False):
        # Make it a Path if needed
        if isinstance(fname, str):
            fname = Path(fname)
        # Check compatible suffixes:
        if fname.with_suffix(".pt").is_file():
            fname = fname.with_suffix(".pt")
        elif fname.with_suffix(".pth").is_file():
            fname = fname.with_suffix(".pth")
        # Load the actual pythorch saved data
        data = torch.load(
            fname,
            map_location=torch.device(torch.cuda.current_device()),
        )
        if len(data) > 1 and not supress_warnings:
            # this should be only 1 thing, but you may be loading a checkpoint or may have saved more data.  Its OK, but we warn.
            warnings.warn(
                "\nSaved file contains more than 1 object, but only model_state_dict is being loaded.\n Call load_checkpoint() to load checkpointed model.\n"
            )
        return data

    @staticmethod
    def load(fname, instance_norm=False, supress_warnings=False):
        """
        Function that loads a model from memory.
        """
        # Make it a Path if needed
        if isinstance(fname, str):
            fname = Path(fname)

        options = ContinuousLPD._load_parameter_file(fname)
        # Check if model name matches the one that is loading it
        # load data
        data = ContinuousLPD._load_data(fname)
        # Some models need geometry, some others not.
        # This initializes the model itself (cls)
        if hasattr(options, "geometry_parameters"):
            model = ContinuousLPD(
                model_parameters=options.model_parameters,
                geometry_parameters=options.geometry_parameters,
                instance_norm=instance_norm,
            )
        else:
            model = ContinuousLPD(
                model_parameters=options.model_parameters, instance_norm=instance_norm
            )

        # Load the data into the model we created.
        model.to(torch.cuda.current_device())
        model.load_state_dict(data.pop("model_state_dict"))

        return model, options, data
