#%% This is a base class for AItomotools.
#
# All classes must derive from this one.
# It definest a bunch of auxiliary functions
#

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

# some numerical standard imports, e.g.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# imports related to class
from abc import ABC, abstractmethod, ABCMeta

# Some other imports
import warnings
from pathlib import Path
import subprocess


class AItomotoModel(nn.Module, ABC):
    """
    Base class for all models in the toolbox,
    """

    # Initialization of the models should have only "Parameter()" classes. These should be "topic-wise", with at minimum 1 parameter object being passed.
    # e.g. a Unet will have only 1 parameter (model_parameters), but the Learned Primal Dual will have 2, one for the model parameters and another one
    # for the geometry parameters of the inverse problem.
    def __init__(
        self,
        model_parameters: Parameter,  # model parameters
        geometry_parameters: ct.Geometry = None,  # (optional) if your model uses an operator, you may need its parameters. e.g. ct geometry parameters for tomosipo operators
    ):
        super().__init__()  # Initialize parent classes.
        __metaclass__ = ABCMeta  # make class abstract.

        # Pass all relevant parameters to internal storage.
        self.geo = geometry_parameters
        self.model_parameters = model_parameters

    # This should return the parameters from the paper the model is from
    @staticmethod
    @abstractmethod  # crash if not defined in derived class
    def default_parameters(mode="ct") -> Parameter:
        pass

    # makes operator and make it pytorch compatible.
    def make_operator(self):
        if self.model_parameters.mode.lower() != "ct":
            raise NotImplementedError("Only CT operators supported")
        if self.geo is not None:
            self.op = ct_utils.make_operator(self.geo)
            self.A = to_autograd(self.op, num_extra_dims=1)
            self.AT = to_autograd(self.op.T, num_extra_dims=1)
        else:
            raise AttributeError("Can't make operator without geo parameters.")

    # All classes should have this method, just change the amount of Parameters it returns of you have more/less
    def get_parameters(self):
        if self.geo is not None:
            return self.model_parameters, self.geo
        else:
            return self.model_parameters

    # All classes should have this method. This is the example for Learned Primal Dual.
    # You can obtain this exact text from Google Scholar's page of the paper.
    @staticmethod
    def cite(cite_format="MLA"):
        pass

    #     if cite_format == "MLA":
    #         print("Adler, Jonas, and Ozan Öktem.")
    #         print('"Learned primal-dual reconstruction."')
    #         print("\x1B[3mIEEE transactions on medical imaging \x1B[0m")
    #         print("37.6 (2018): 1322-1332.")
    #     elif cite_format == "bib":
    #         string = """
    #         @article{adler2018learned,
    #         title={Learned primal-dual reconstruction},
    #         author={Adler, Jonas and {\"O}ktem, Ozan},
    #         journal={IEEE transactions on medical imaging},
    #         volume={37},
    #         number={6},
    #         pages={1322--1332},
    #         year={2018},
    #         publisher={IEEE}
    #         }"""
    #         print(string)
    #     else:
    #         raise AttributeError(
    #             'cite_format not understood, only "MLA" and "bib" supported'
    #         )

    # This shoudl save all relevant information to completely reproduce models
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
                "Expected 'training' parameter! Only ignore if there has been no training."
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
        # We should always save models with the git hash they were created. Models may change, and if loading at some point breaks
        # we need to at least know exactly when the model was saved, to at least be able to reproduce.
        options.commit_hash = ai_utils.get_git_revision_hash()
        options.model_parameters = self.model_parameters
        if geo:
            options.geometry_parameters = geo
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
        options.save(fname.with_suffix(".json"))
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

    # Loads model.
    @classmethod
    def load(cls, fname, supress_warnings=False):
        """
        Function that loads a model from memory.
        """
        # Make it a Path if needed
        if isinstance(fname, str):
            fname = Path(fname)

        # Load the actual parameters
        options = Parameter()
        options.load(fname.with_suffix(".json"))

        # Check if model has been changed since save.
        if not hasattr(options, "commit_hash") and not supress_warnings:
            warnings.warn(
                "No commit hash found. This model was not saved with the standard AItomotools function and it will likely fail to load."
            )
        else:
            curr_commit_hash = ai_utils.get_git_revision_hash()
            curr_path = cls.current_file()
            if (
                ai_utils.check_if_file_changed_git(
                    curr_path, options.commit_hash, curr_commit_hash
                )
                and not supress_warnings
            ):
                warnings.warn(
                    f"The code for the model has changed since it was saved, loading it may fail. This model was saved in {options.commit_hash}"
                )

        data = torch.load(fname.with_suffix(".pt"))
        if len(data) > 1 and not supress_warnings:
            warnings.warn(
                "Saved file contains more than 1 object, but only model_state_dict is being loaded.\n Call load_checkpint() to load checkpointed model."
            )

        if hasattr(options, "geometry_parameters"):
            model = cls(options.model_parameters, options.geometry_parameters)
        else:
            model = cls(options.model_parameters)

        data = torch.load(fname.with_suffix(".pt"))
        for key in data:
            if not (key == "model_state_dict") and not supress_warnings:
                warnings.warn(f"Saved parameter '{key}' ignored at loading model")
        model.load_state_dict(data["model_state_dict"])
        return model

    @staticmethod
    def load_checkpoint(self, fname):
        # Make it a Path if needed
        if isinstance(fname, str):
            fname = Path(fname)
        self.load()
        return 1, 2

    @classmethod
    def current_file(cls):
        import sys

        module = sys.modules[cls.__module__]
        fname = Path(module.__file__)
        # This will be in the install path, so lets get a relative path. Assuming user here will be working on AItomotools folder, which they may not be.
        parts = fname.resolve().parts[fname.resolve().parts.index("models") - 1 :]
        return Path(*parts)
