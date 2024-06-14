# This file is part of AItomotools library
# License : BSD-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================


from LION.models import LIONmodel

from LION.utils.math import power_method
from LION.utils.parameter import Parameter
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

    def __init__(self, layers, channels, instance_norm=False):

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
                    nn.Conv2d(channels[ii], channels[ii + 1], 3, padding=1, bias=False)
                )
                layer_list.append(nn.PReLU())
            else:
                layer_list.append(
                    nn.Conv2d(channels[ii], channels[ii + 1], 1, padding=0, bias=False)
                )
        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.block(x)


class RegProximal(nn.Module):
    """
    CNN block of the primal variable
    """

    def __init__(self, layers, channels, instance_norm=False):
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
                    nn.Conv2d(channels[ii], channels[ii + 1], 3, padding=1, bias=False)
                )
                layer_list.append(nn.PReLU())
            else:
                layer_list.append(
                    nn.Conv2d(channels[ii], channels[ii + 1], 1, padding=0, bias=False)
                )
        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.block(x)


class LPD(LIONmodel.LIONmodel):
    """Learn Primal Dual network"""

    def __init__(
        self,
        geometry_parameters: ct.Geometry,
        model_parameters: Parameter = None,
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
                RegProximal(
                    layers=len(self.model_parameters.reg_channels) - 1,
                    channels=self.model_parameters.reg_channels,
                    instance_norm=instance_norm,
                ),
            )
            self.add_module(
                f"{i}_dual",
                dataProximal(
                    layers=len(self.model_parameters.data_channels) - 1,
                    channels=self.model_parameters.data_channels,
                    instance_norm=instance_norm,
                ),
            )

        # Create pytorch compatible operators and send them to autograd
        self._make_operator()

        # Define step size
        if self.model_parameters.step_size is None:
            print("Step size is None, computing it with power method")
            # compute step size
            self.model_parameters.step_size = 1 / power_method(self.op)
        print(self.model_parameters.step_size)
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

        options = LPD._load_parameter_file(fname)
        # Check if model name matches the one that is loading it
        # load data
        data = LPD._load_data(fname)
        # Some models need geometry, some others not.
        # This initializes the model itself (cls)
        if hasattr(options, "geometry_parameters"):
            model = LPD(
                model_parameters=options.model_parameters,
                geometry_parameters=options.geometry_parameters,
                instance_norm=instance_norm,
            )
        else:
            model = LPD(
                model_parameters=options.model_parameters, instance_norm=instance_norm
            )

        # Load the data into the model we created.
        model.to(torch.cuda.current_device())
        model.load_state_dict(data.pop("model_state_dict"))

        return model, options, data