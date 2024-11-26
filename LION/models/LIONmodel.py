# This file is part of LION library
# License : BSD-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================


#%% This is a base class for LION models.
#
# All classes must derive from this one.
# It definest a bunch of auxiliary functions
#

#%% Imports

# You will want to import LIONParameter, as all models must save and use Parameters.
from enum import Enum
from typing import Optional
from LION.utils.parameter import LIONParameter

# We will need utilities
import LION.utils.utils as ai_utils

# (optional) Given this is a tomography library, it is likely that you will want to load geometries of the tomogprahic problem you are solving, e.g. a ct_geometry
import LION.CTtools.ct_geometry as ct
import LION.CTtools.ct_utils as ct_utils

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


class ModelInputType(int, Enum):
    SINOGRAM = 0
    NOISY_RECON = 1
    IMAGE = 1


# it is the job of the subclass constructor to specify input_type
class ModelParams(LIONParameter):
    def __init__(self, model_input_type, **kwargs):
        super().__init__(**kwargs)
        self.model_input_type = model_input_type


class LIONmodel(nn.Module, ABC):
    """
    Base class for all models in the toolbox,
    """

    # Initialization of the models should have only "LIONParameter()" classes. These should be "topic-wise", with at minimum 1 parameter object being passed.
    # e.g. a Unet will have only 1 parameter (model_parameters), but the Learned Primal Dual will have 2, one for the model parameters and another one
    # for the geometry parameters of the inverse problem.
    def __init__(
        self,
        model_parameters: Optional[ModelParams],  # model parameters
        geometry: Optional[
            ct.Geometry
        ] = None,  # (optional) if your model uses an operator, you may need its parameters. e.g. ct geometry parameters for tomosipo operators
    ):
        super().__init__()  # Initialize parent classes.
        __metaclass__ = ABCMeta  # make class abstract.

        if model_parameters is None:
            model_parameters = self.default_parameters()
        # Pass all relevant parameters to internal storage.
        self.geometry = geometry
        self.model_parameters = model_parameters

    # This should return the parameters from the paper the model is from
    @staticmethod
    @abstractmethod  # crash if not defined in derived class
    def default_parameters(mode="ct") -> ModelParams:
        pass

    # makes operator and make it pytorch compatible.
    def _make_operator(self):
        # if self.model_parameters.mode.lower() != "ct":
        #     raise NotImplementedError("Only CT operators supported")
        if hasattr(self, "geometry") and self.geometry is not None:
            self.op = ct_utils.make_operator(self.geometry)
            self.A = to_autograd(self.op, num_extra_dims=1)
            self.AT = to_autograd(self.op.T, num_extra_dims=1)
        else:
            raise AttributeError("Can't make operator without geometry parameters.")

    # All classes should have this method, just change the amount of Parameters it returns of you have more/less
    def get_parameters(self):
        if self.geometry is not None:
            return self.model_parameters, self.geometry
        else:
            return self.model_parameters

    def get_input_type(self) -> ModelInputType:
        return self.model_parameters.model_input_type

    # All classes should have this method. This is the example for Learned Primal Dual.
    # You can obtain this exact text from Google Scholar's page of the paper.
    @staticmethod
    def cite(cite_format="MLA"):
        print("cite not implemented for selected method")
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
        - 'dataset' : LIONParameter describing the dataset creation and handling.
        - 'training': LIONParameter describing the training algorithm and procedures
        - 'geometry': If the model itself has no scan geometry parameter, but the dataset was created with some geometry

        If you want to save the model for training later (i.e. checkpoiting), use save_checkpoint()
        """
        # Make it a Path if needed
        if isinstance(fname, str):
            fname = Path(fname)

        # Create dict of saved data:
        ##########################
        dic = {"model_state_dict": self.state_dict()}

        # Parse kwargs
        ################
        dataset_params = np.empty(0)
        if "dataset" in kwargs:
            dataset_params = kwargs.pop("dataset")
            dic["dataset_params"] = dataset_params

        else:
            warnings.warn(
                "\nExpected 'dataset' parameter! Only ignore if you really don't have it.\n"
            )
        training = np.empty(0)
        if "training" in kwargs:
            training = kwargs.pop("training")
            dic["training_params"] = training

        else:
            warnings.warn(
                "\nExpected 'training' parameter! Only ignore if there has been no training.\n"
            )

        if "loss" in kwargs:
            dic["loss"] = kwargs.pop("loss")
        epoch = np.empty(0)
        if "epoch" in kwargs:
            dic["epoch"] = kwargs.pop("epoch")
        optimizer = np.empty(0)
        if "optimizer" in kwargs:
            dic["optimizer_state_dict"] = kwargs.pop("optimizer")

        # (optional)
        geometry = []
        if "geometry" in kwargs:
            geometry = kwargs.pop("geometry")
            dic["geometry"] = geometry
        elif hasattr(self, "geometry") and self.geometry is not None:
            geometry = self.geometry
            dic["geometry"] = geometry
        else:
            warnings.warn(
                "Expected 'geometry' parameter! Only ignore if tomographic reconstruction was not part of the model."
            )

        if kwargs:  # if not empty yet
            raise ValueError(
                "The following parameters are not understood: "
                + str(list(kwargs.keys()))
            )
        # Prepare parameters to be saved
        ##########################
        # These are for human readability, but we redundantly save it in the data too
        # Make a super LIONParameter()
        options = LIONParameter()
        # We should always save models with the git hash they were created. Models may change, and if loading at some point breaks
        # we need to at least know exactly when the model was saved, to at least be able to reproduce.
        try:
            options.commit_hash = ai_utils.get_git_revision_hash()
        except:
            warnings.warn("\nCould not get git hash.\n")
        options.model_name = self.__class__.__name__
        options.model_parameters = self.model_parameters
        if geometry:
            options.geometry = geometry
        if dataset_params:
            options.dataset_params = dataset_params
        if training:
            options.training = training

        # Do the save:

        options.save(fname.with_suffix(".json"))
        torch.save(dic, fname.with_suffix(".pt"))

    # Mandatory function, saves model for training
    def save_checkpoint(self, fname, epoch, loss, optimizer, training_param, **kwargs):
        """
        This is like save, but saves a checkpoint of the model.
        Its essentailly a wrapper of save() with mandatory values
        """

        assert isinstance(optimizer, torch.optim.Optimizer)
        self.save(
            fname,
            epoch=epoch,
            loss=loss,
            optimizer=optimizer.state_dict(),
            training=training_param,
            **kwargs,
        )

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

    @classmethod
    def _load_parameter_file(cls, fname, supress_warnings=False):
        # Load the actual parameters
        ##############################
        options = LIONParameter()
        options.load(fname.with_suffix(".json"))
        if hasattr(options, "geometry"):
            options.geometry = ct.Geometry.init_from_parameter(options.geometry)
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

    @classmethod
    def load(cls, fname, supress_warnings=False):
        """
        Function that loads a model from memory.
        """
        # Make it a Path if needed
        if isinstance(fname, str):
            fname = Path(fname)

        options = LIONmodel._load_parameter_file(fname)
        # Check if model name matches the one that is loading it
        if options.model_name != cls.__name__:
            warnings.warn(
                f"\nSaved model is from a class with a different name than current, likely load will fail. \nCurrent class name: {cls.__name__}, Saved model class name: {options.model_name}\n"
            )

        # load data
        data = LIONmodel._load_data(fname)
        # Some models need geometry, some others not.
        # This initializes the model itself (cls)
        if hasattr(options, "geometry"):
            model = cls(
                model_parameters=options.model_parameters,
                geometry=options.geometry,
            )
        else:
            model = cls(model_parameters=options.model_parameters)

        # Load the data into the model we created.
        model.to(torch.cuda.current_device())
        model.load_state_dict(data.pop("model_state_dict"))

        return model, options, data

    @classmethod
    def load_checkpoint(cls, fname):
        # Make it a Path if needed
        if isinstance(fname, str):
            fname = Path(fname)

        options = cls._load_parameter_file(fname)
        # Check if model name matches the one that is loading it
        if options.model_name != cls.__name__:
            warnings.warn(
                f"\nSaved model is from a class with a different name than current, likely load will fail. \nCurrent class name: {cls.__name__}, Saved model class name: {options.model_name}\n"
            )
        # load data
        data = LIONmodel._load_data(fname, supress_warnings=True)
        # Some models need geometry, some others not.
        # This initializes the model itself (cls)
        if hasattr(options, "geometry"):
            model = cls(
                model_parameters=options.model_parameters,
                geometry=options.geometry,
            )
        else:
            model = cls(model_parameters=options.model_parameters)
        # Load the data into the model we created.
        model.to(torch.cuda.current_device())
        model.load_state_dict(data.pop("model_state_dict"))

        return model, options.unpack(), data

    @classmethod
    def _current_file(cls):
        import sys

        module = sys.modules[cls.__module__]
        fname = Path(module.__file__)
        # This will be in the install path, so lets get a relative path. Assuming user here will be working on AItomotools folder, which they may not be.
        parts = fname.resolve().parts[fname.resolve().parts.index("models") - 1 :]
        return Path(*parts)

    @staticmethod
    def final_file_exists(fname, stop_code=False):
        if isinstance(fname, str):
            fname = Path(fname)
        exists = fname.is_file()
        if stop_code and exists:
            print("Final version found, no need to loop further, exiting")
            exit()
        return exists

    @classmethod
    def load_checkpoint_if_exists(
        cls, fname, model, optimiser, total_loss, verbose=True
    ):
        if isinstance(fname, str):
            fname = Path(fname)
        checkpoints = sorted(list(fname.parent.glob(fname.name)))
        if checkpoints:
            model, options, data = cls.load_checkpoint(
                fname.parent.joinpath(checkpoints[-1])
            )
            optimiser.load_state_dict(data["optimizer_state_dict"])
            start_epoch = data["epoch"]
            total_loss = data["loss"]
            model.train()
        else:
            print(f"checkpoint {fname} not found, failed to load.")
            return model, optimiser, 0, total_loss, None
        return model, optimiser, start_epoch, total_loss, data

    @staticmethod
    def _read_min_validation(filename):
        """
        Given a filename for saved models, reads the validaton loss and returns it.
        This is useful when both checkpointing and validation are done in the same training, and the training partially stops.
        One can load the checkpoint, but the minimum validation loss is not saved in the checkpoint, so this function reads it from the filename,
        which should be the minimum validation model.
        """
        if isinstance(filename, str):
            filename = Path(filename)
        data = LIONmodel._load_data(filename, supress_warnings=True)
        loss = data["loss"]
        if type(loss) is np.ndarray:
            if len(loss) > 1:
                warnings.warn(
                    "More than one loss found in file, which suggests that it was not a minimum validation file. Returning last loss."
                )
                loss = loss[-1]
            else:
                loss = loss[0]
        return loss
