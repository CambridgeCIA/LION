#%% This is a template of a model for AItomotools.
#
# It gives an easy start on how to build a model that fits well with the library.
# This file has text and examples explaining what you need.
# Use ut as a template to build your models on.
#
# The reason to not have all models inherint from this is future-proofing. There is too much complexity in AI to start making
# strict limits on what the models can have.

#%% Imports

# You will want to import Parameter, as all models must save and use Parameters.
from AItomotools.utils.parameter import Parameter

# We will need utilities
import AItomotools.utils.utils as ai_utils

# (optional) Given this is a tomography library, it is likely that you will want to load geometries of the tomogprahic problem you are solving, e.g. a ct_geometry
import AItomotools.CTtools.ct_geometry as ct
import AItomotools.CTtools.ct_utils as ct_utils

# (optinal) If your model uses the operator (e.g. the CT operator), you may want to load it here. E.g. for tomosipo:
import tomosipo as ts
from tomosipo.torch_support import to_autograd

# some standard imports, e.g.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from pathlib import Path

# Lets define the class. This demo just shows a very simple model that uses both the opeartor of CT and a CNN layer, for demostrational purposes.
# The model makes no sense, it only sits here for coding demostration purposes.


class myModel(nn.Module):
    """My model netweork (title)
    Some more info about it.
    """

    # Initialization of the models should have only "Parameter()" classes. These should be "topic-wise", with at minimum 1 parameter object being passed.
    # e.g. a Unet will have only 1 parameter (model_parameters), but the Learned Primal Dual will have 2, one for the model parameters and another one
    # for the geometry parameters of the inverse problem.
    def __init__(
        self,
        model_parameters: Parameter,  # model parameters
        geometry_parameters: ct.Geometry,  # (optional) if your model uses an operator, you may need its parameters. e.g. ct geometry parameters for tomosipo operators
    ):
        super().__init__()  # Initialize parent classes.

        # Pass all relevant parameters to internal storage.
        self.geo = geometry_parameters
        self.model_parameters = model_parameters

        # (optional) if your model is for CT reconstruction, you may need the CT operator defined with e.g. tomosipo. This is how its done.
        # model_parameters.mode contains the tomographic mode, e.g. 'ct'
        op = self.__make_operator(self.geo, self.model_parameters.mode)
        self.op = op
        self.A = to_autograd(op, num_extra_dims=1)
        self.AT = to_autograd(op.T, num_extra_dims=1)

        ##### EXAMPLE #####
        # make some NN layers, maybe defined by the Parameter file
        #
        # in this case, model_parameters has .bias (True/False) and .channels (list with number of channels in each layer).
        # for example, model_parameters.channels=[7 10 5 1], and model_parameters.bias=False
        #
        # Albeit this is here for demosntrational purposes, do not clog the __init__ function, add classes that contain subblocks (check LPD.py)
        layer_list = []
        for ii in range(len(self.model_parameters.channels - 1)):
            layer_list.append(
                nn.Conv2d(
                    self.model_parameters.channels[ii],
                    self.model_parameters.channels[ii + 1],
                    3,
                    padding=s1,
                    bias=self.model_parameters.bias,
                )
            )
            # Have PReLUs all the way except the last
            if ii < layers - 1:
                layer_list.append(torch.nn.PReLU())
        self.block = nn.Sequential(*layer_list)

    # All classes in AItomotools must have a static method called default_parameters().
    # This should return the parameters from the paper the model is from
    @staticmethod
    def default_parameters(mode="ct") -> Parameter:
        # create empty object
        model_params = Parameter()
        ##### EXAMPLE #####
        model_params.mode = mode
        model_params.channels = [7, 10, 5, 1]
        model_params.bias = False

        return model_params

    # All classes should have this method, just change the amount of Parameters it returns of you have more/less
    def get_parameters(self):
        return self.model_parameters, self.geo

    # All classes shoudl have this method. Yhis is the example for Learned Primal Dual.
    # You can obtain this exact text from Google Scholar's page of the paper.
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

    # Mandatory function, saves model for training
    def save_checkpoint(self, fname, epoch, loss, optimizer, **kwargs):
        """
        This is like save, but saves a checkpoint of the model.
        Its essentailly a wrapper of save() with mandatory values
        """

        if "training" not in kwargs:
            raise ValueError("The mandatory Parameter 'training' not inputed.")
        assert isinstance(optimizer, torch.optim.Optimizer)
        self.save(
            filename, "epoch", epoch, "loss", loss, "optimizer", optimizer, **kwargs
        )

    # (optional) if your model uses a CT operator, this will create it, for tomosipo backend.
    @staticmethod
    def __make_operator(geo, mode="ct"):
        if mode.lower() == "ct":
            A = ct_utils.make_operator(geo)
        else:
            raise NotImplementedError("Only CT operators supported")
        return A

    # Mandatory for all models, the forwar pass.
    def forward(self, g):
        """
        g: sinogram input
        """
        B, C, W, H = g.shape
        # Have some input parsing
        if len(self.geo.angles) != W or self.geo.detector_shape[1] != H:
            raise ValueError("geo description and sinogram size do not match")

        # (optional) if your code is only 2D
        if C != 1:
            raise NotImplementedError("Only 2D CT images supported")

        # Your code.
        f_out

        return f_out
