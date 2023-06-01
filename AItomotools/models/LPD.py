from AItomotools.utils.math import power_method
from AItomotools.utils.parameter import Parameter
import AItomotools.CTtools.ct_geometry as ct
import AItomotools.CTtools.ct_utils as ct_utils
import AItomotools.utils.utils as ai_utils

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

    def __init__(self, layers, channels):

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
            layer_list.append(
                nn.Conv2d(channels[ii], channels[ii + 1], 3, padding=1, bias=False)
            )
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
        if len(channels) != layers + 1:
            raise ValueError(
                "Second input (channels) should have as many elements as layers your network has"
            )
        if layers < 1:
            raise ValueError("At least one layer required")

        layer_list = []
        for ii in range(layers):
            layer_list.append(
                nn.Conv2d(channels[ii], channels[ii + 1], 3, padding=1, bias=False)
            )
            # Have PReLUs all the way except the last
            if ii < layers - 1:
                layer_list.append(torch.nn.PReLU())
        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.block(x)


class LPD(nn.Module):
    """Learn Primal Dual network"""

    def __init__(self, model_parameters: Parameter, geometry_parameters: ct.Geometry):
        super().__init__()
        # Pass all relevant parameters to internal storage.
        self.geo = geometry_parameters
        self.model_parameters = model_parameters

        # Create layers per iteration
        for i in range(self.model_parameters.n_iters):
            self.add_module(
                f"{i}_primal",
                RegProximal(
                    layers=len(self.model_parameters.reg_channels) - 1,
                    channels=self.model_parameters.reg_channels,
                ),
            )
            self.add_module(
                f"{i}_dual",
                dataProximal(
                    layers=len(self.model_parameters.data_channels) - 1,
                    channels=self.model_parameters.data_channels,
                ),
            )

        # Create pytorch compatible operators and send them to aiutograd
        op = self.__make_operator(self.geo, self.model_parameters.mode)
        self.op = op
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
                        for i in range(n_iters)
                    ]
                )
                self.lambda_primal = nn.ParameterList(
                    [
                        nn.Parameter(torch.ones(1) * self.model_parameters.step_size)
                        for i in range(n_iters)
                    ]
                )
        else:
            self.lambda_dual = torch.ones(n_iters) * self.model_parameters.step_size
            self.lambda_primal = torch.ones(n_iters) * self.model_parameters.step_size

    @staticmethod
    def default_parameters(mode="ct"):
        LPD_params = Parameter()
        LPD_params.n_iters = 10
        LPD_params.mode = mode
        LPD_params.data_channels = [7, 32, 32, 5]
        LPD_params.reg_channels = [6, 32, 32, 5]
        LPD_params.learned_step = True
        LPD_params.step_size = None
        LPD_params.step_positive = True
        return LPD_params

    def get_parameters(self):
        return self.model_parameters, self.geo

    @staticmethod
    def __make_operator(geo, mode="ct"):
        if mode.lower() == "ct":
            A = ct_utils.make_operator(geo)
        else:
            raise NotImplementedError("Only CT operators supported")
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

        # All classes should have this method.

    # This shouls save all relevant information to complete reproduce models
    def save(self, fname, **kwargs):
        """
        Saves model given a filename.
        While its not enforced, the following Parameters are expected from kwargs:
        - 'dataset' : Parameter describing the dataset creation and handling.
        - 'training': Parameter describing the training algorithm and procedures
        - 'geometry': If the model itself has no scan geometry parameter, but the dataset was created with some geometry

        If you want to save the model for training later (i.e. checkpoiting), use save_checkpoint()
        """
        # Make it a Path if needed
        if isinstance(fname, str):
            fname = Path(fname)
        # We should always save models with the git hash they were created. Models may change, and if loading at some point breaks
        # we need to at least know exactly when the model was saved, to at least be able to reproduce.
        commit_hash = ai_utils.get_git_revision_hash()
        # Parse kwargs
        dataset_params = []
        if "dataset" in kwargs:
            dataset_params = kwargs.pop("dataset")
        else:
            warnings.warn(
                "Expected 'dataset' parameter! Only ignore if you really don't have it."
            )
        training = []
        if "training" in kwargs:
            training = kwargs.pop("training")
        else:
            warnings.warn(
                "Expected 'training' parameter! Only ignore if ythere has been no training."
            )

        loss = []
        if "loss" in kwargs:
            loss = kwargs.pop("loss")
        epoch = []
        if "epoch" in kwargs:
            epoch = kwargs.pop("epoch")
        optimizer = []
        if "optimizer" in kwargs:
            optimizer = kwargs.pop("optimizer")

        # (optional)
        geo = []
        if "geometry" in kwargs:
            geo = kwargs.pop("geometry")
        elif hasattr(self, "geo") and self.geo:
            geo = self.geo
        else:
            warnings.warn(
                "Expected 'geometry' parameter! Only ignore if tomographic reconstruction was not part of the model."
            )

        if kwargs:  # if not empty yet
            raise ValueError(
                "The following parameters are not understood: " + str(list(kwargs.keys))
            )

        ## Make a super Parameter()
        options = Parameter()

        options.model_parameters = self.model_parameters
        if geo:
            options.geometry = geo
        if dataset_params:
            options.dataset_params = dataset_params
        if training:
            options.training = training

        ## Make a dictionary of relevant values
        dic = {"model_state_dict": self.state_dict()}
        if loss:
            dic["loss"] = loss
        if epoch:
            dic["epoch"] = epoch
        if optimizer:
            dic["optimizer_state_dict"] = optimizer.state_dict()

        # Do the save:
        options.save(fname)
        torch.save(dic, fname.with_suffix(".pt"))

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
            for channel in range(5):
                f_primal[i, channel] = aux

        for i in range(self.model_parameters.n_iters):
            primal_module = getattr(self, f"{i}_primal")
            dual_module = getattr(self, f"{i}_dual")
            f_dual = self.A(f_primal[:, :1]).cuda()
            h = self.__dual_step(g, h, f_dual, dual_module)

            update = self.lambda_dual[i] * self.AT(h[:, :1]).cuda()
            f_primal = self.__primal_step(f_primal, update, primal_module)

        return f_primal[:, 0:1]
