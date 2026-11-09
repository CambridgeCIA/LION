# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri & Christina Runkel
# Modifications: -
# =============================================================================


from LION.models.iterative_unrolled.LPD import LPD
from LION.models.LIONmodel import LIONModelParameter, ModelInputType

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
from LION.utils.citation import citation


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
        do_secon_order: bool = False,
        solver: str = "rk4",
        instance_norm: bool = False,
    ):
        """
        Args:
            layers(int): number of layers
            channels (list): list of number of channels per layer
            tol (float): tolerance to be used for ODE solver
            adjoint (bool): whether to use the adjoint method to calculate the gradients
            do_secon_order (bool): use second order ODE instead of first order if 'True'
            solver (str): ODE solver to use
        """
        super(ContinuousDataProximal, self).__init__()
        self.do_secon_order = do_secon_order
        self.solver = solver

        if do_secon_order:
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
        if self.do_secon_order:
            x = self.initial_velocity(x)
        x = self.odeblock(x, solver=self.solver)
        return self.last_layer(x)


class ContinuousRegProximal(nn.Module):
    def __init__(
        self,
        layers: int,
        channels: list,
        tol: float = 1e-3,
        adjoint: bool = False,
        do_secon_order: bool = False,
        solver: str = "rk4",
        instance_norm: bool = False,
    ):
        """
        layers(int): number of layers
            channels (list): list of number of channels per layer
            tol (float): tolerance to be used for ODE solver
            adjoint (bool): whether to use the adjoint method to calculate the gradients
            do_secon_order (bool): use second order ODE instead of first order if 'True'
            solver (str): ODE solver to use
        """
        super(ContinuousRegProximal, self).__init__()
        self.do_secon_order = do_secon_order
        self.solver = solver

        if do_secon_order:
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
        if self.do_secon_order:
            x = self.initial_velocity(x)
        x = self.odeblock(x, solver=self.solver)
        return self.last_layer(x)


class cLPD(LPD):
    """Learned Primal Dual with continuous blocks (Runkel et al.): LPD whose
    primal and dual blocks are neural ODEs."""

    def _iteration_blocks(self):
        """The primal and dual blocks of one iteration: continuous (ODE)
        blocks, first or second order (``do_secon_order``)."""
        params = self.model_parameters
        primal = ContinuousRegProximal(
            layers=len(params.reg_channels) - 1,
            channels=params.reg_channels,
            do_secon_order=params.do_secon_order,
            instance_norm=params.instance_norm,
        )
        dual = ContinuousDataProximal(
            layers=len(params.data_channels) - 1,
            channels=params.data_channels,
            do_secon_order=params.do_secon_order,
            instance_norm=params.instance_norm,
        )
        return primal, dual

    @staticmethod
    def default_parameters():
        params = LIONModelParameter()
        params.n_iters = 10
        params.data_channels = [7, 32, 32, 32, 5]
        params.reg_channels = [6, 32, 32, 32, 5]
        params.learned_step = False
        params.step_size = None
        params.step_positive = False
        params.mode = "ct"
        params.do_secon_order = False
        params.instance_norm = False
        # project the second primal channel in the dual step, as in Adler & Oktem
        params.dual_step_primal_channel = 1
        params.model_input_type = ModelInputType.SINOGRAM
        return params

    @staticmethod
    def continous_LPD_paper():
        params = LIONModelParameter()
        params.n_iters = 10
        params.data_channels = [7, 32, 32, 32, 5]
        params.reg_channels = [6, 32, 32, 32, 5]
        params.learned_step = True
        params.step_size = None
        params.step_positive = False
        params.mode = "ct"
        params.do_secon_order = False
        params.instance_norm = False
        # project the second primal channel in the dual step, as in Adler & Oktem
        params.dual_step_primal_channel = 1
        params.model_input_type = ModelInputType.SINOGRAM
        return params

    @staticmethod
    def cite(cite_format="bib"):
        """Where cLPD comes from: its citation, BibTeX ("bib") or "MLA"."""
        return citation(
            cite_format,
            bib="""
            @article{runkel2024continuous,
            title={Continuous Learned Primal Dual},
            author={Runkel, Christina and Biguri, Ander and Sch{\\"o}nlieb, Carola-Bibiane},
            journal={arXiv preprint arXiv:2405.02478},
            year={2024}
            }
            """,
            mla="""
            Runkel, Christina, Ander Biguri, and Carola-Bibiane Schönlieb.
            "Continuous Learned Primal Dual"
             ArXiv 
            (2024): 2405.02478.
            """,
        )
