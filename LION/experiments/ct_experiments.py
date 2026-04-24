# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

import LION.CTtools.ct_geometry as ctgeo

# import LION.CTtools.ct_utils as ct
# from LION.data_loaders.deteCT import deteCT
# from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
from LION.utils.parameter import LIONParameter


# Design: Each `{x}_experiments.py` file contains a set of `Experiment` classes, each representing a FIXED experiment of type `x` (e.g. `x = "ct"`).
#         Each file should also contain one (and only one) subclass of `LIONParameter` defining shared parameters for the experiments in that file ONLY.
#         This parameter object should NOT be used for experiments not defined in that file.
#         Even if an experiment somewhere else use very similar parameters,
#         it should define its own parameter class in its file.
# Pros:
#   - Aid documentation:
#     We eventually need to document the parameters anyway.
#     Documenting in each experiment class is very repetitive.
#     Documenting straight into the code instead of a separate documentation file makes it easier to check and update.
#   - Ensure consistency, early bug detection:
#     E.g., Say we need to add a new parameter for all experiment classes.
#     If there are 9 experiment classes, we would need to add it to `default_parameters` of each class anyway.
#     With this, we need to do one more addition to this parameter class.
#     In exchange, this helps make it clear,
#     showing proper warnings and errors if, for example, we forget to add it to one of the `default_parameters` of the experiment classes.
#   - If users don't define new experiment classes and try to pass custom parameters,
#     they don't need to know this exists.
#     Since this is a subclass of `LIONParameter`,
#     as long as users know the right parameters to pass to the experiment,
#     they can still use `LIONParameter` without knowing this exists.
# Potential cons:
#   - One more layer of abstraction for devs/maintainers.
#     However, we need to make the parameters the same for all experiments in this file anyway, and it's only used in this file (no branching to other parts of the repo) so it should take very little effort to maintain.
#   - Writing new experiment classes in user's main script
#   - Adding new experiment classes into this file may be slower?
#   - Adding
#     Technically, new devs can also just use `LIONParameter` only.
#     However, if we use this convention everywhere,
#     and not make it clear in documentation that this is not a requirement,
#     they may just copy and try to modify and then feel forced to use this convention,
#     which may lead to more time spent on learning the convention.
#     However, as seen below, this is a very minimal design so learning it even without explicit instructions (just copy and modify) should be very quick.
#     New devs need to ensure consistency anyway and add documentation anyway,
#     this actually helps them (see pros above).
@dataclass
class CTReconParameter(LIONParameter):
    """Common parameters for CT reconstruction experiments.

    Note for experiment developers: If you copy this file and try to modify it,
    know that a class like this is not a requirement.
    You can just use `LIONParameter` instead (replace the casting in `__init__` of experiment class with simple assignment).
    If you do decide to make a class like this, make sure to change the name as appropriate.
    Please don't import this class in other experiment files even if the parameters are the same.

    Parameters
    ----------
    name : str
        Name of the experiment.
    geometry : ctgeo.Geometry
        Geometry parameters for CT forward operator.
    noise_params : LIONParameter
        Settings for sinogram noise simulation.
    data_loader_params : LIONParameter
        Parameters to build the data loader.
    """

    name: str
    geometry: ctgeo.Geometry
    noise_params: LIONParameter
    data_loader_params: LIONParameter


@dataclass
class Experiment(ABC):
    """CT reconstruction experiment base class.

    Defines the common interface for CT reconstruction experiments (e.g., low-dose, limited-angle, etc.).
    """

    # """
    # Parameters
    # ----------
    # name : str
    #     Name of the experiment.
    # geometry : ctgeo.Geometry
    #     Geometry parameters for CT forward operator.
    # noise_params : LIONParameter
    #     Settings for sinogram noise simulation.
    # data_loader_params : LIONParameter
    #     Parameters to build the data loader.
    # """

    # name: str
    # geometry: ctgeo.Geometry
    # noise_params: LIONParameter
    # data_loader_params: LIONParameter
    # dataset: str = "LIDC-IDRI"

    def __init__(
        self,
        experiment_params: LIONParameter | None = None,
        dataset: str = "LIDC-IDRI",
        datafolder: str | None = None,
    ):
        """Base class for CT reconstruction experiments.

        Defines the common interface for CT reconstruction experiments (e.g., low-dose, limited-angle, etc.).

        Intended usage: Create a new subclass inheriting from `Experiment` and implement `default_parameters` method.
        The subclass represents a FIXED experiment and should NOT accept `experiment_params` in the constructor.
        Instead, the parameters should be hardcoded in the `default_parameters` method.
        However, the experiment should still be able to use different datasets and datafolders, so the subclass constructor should still accept those.

        Parameters
        ----------
        experiment_params : LIONParameter, optional
            Custom experiment parameters. If None, default parameters will be used. Default is None.
            NOT RECOMMENDED. The purpose of the experiment class is to have reliable and repeatable experiments.
            If you want to change the parameters, please create a new class with the new parameters.
        dataset : str, optional
            The name of the dataset to be used for the experiment. Default is "LIDC-IDRI". See `Experiment.__get_dataset` for supported datasets.
        datafolder : str or Path, optional
            The folder where the dataset is stored. If None, the default folder will be used. Default is None.
        """
        super().__init__()  # Initialize parent classes.
        if experiment_params is None:
            self.param: CTReconParameter = cast(
                CTReconParameter, self.default_parameters(dataset=dataset)
            )
        else:
            # Expect users to provide correct parameters as defined in `CTReconParameter`.
            # Should fail otherwise. Good for early bug detection
            # (compared to failing only when parameters are actually used).
            self.param = cast(CTReconParameter, experiment_params)
        if datafolder is not None:
            self.param.data_loader_params.folder = datafolder
        # self.geometry = self.param.geometry
        self.dataset = dataset
        if hasattr(self.param, "noise_params"):
            self.sino_fun = lambda sino, I0=self.param.noise_params.I0, sigma=self.param.noise_params.sigma, cross_talk=self.param.noise_params.cross_talk: ct.sinogram_add_noise(
                sino, I0=I0, sigma=sigma, cross_talk=cross_talk
            )

    # Pro: Change if `self.param` is changed.
    # TODO: Should geometry just be inferred from `data_loader_params.geometry`?
    @property
    def geometry(self):
        # return self.param.data_loader_params.geometry
        return self.param.geometry

    @staticmethod
    @abstractmethod  # crash if not defined in derived class
    def default_parameters(dataset: str) -> LIONParameter:
        """Default experiment parameters."""

    def __get_dataset(self, mode: str):

        if self.dataset == "LIDC-IDRI":
            dataloader = LIDC_IDRI(
                mode=mode,
                parameters=self.param.data_loader_params,
                geometry_parameters=self.geometry,
            )
            dataloader.set_sinogram_transform(self.sino_fun)

        elif self.dataset == "2DeteCT":
            if hasattr(self.param, "noise_params"):
                warnings.warn(
                    "You are setting noise parameters for a dataset that comes with real measured data. The noise will be added on top of the real measured noise\n Note that only noise_params.I0 and noise_params.cross_talk will be used"
                )
                self.param.data_loader_params.noise_params.I0 = (
                    self.param.noise_params.I0
                )
                self.param.data_loader_params.noise_params.cross_talk = (
                    self.param.noise_params.cross_talk
                )
                self.param.data_loader_params.add_noise = True
            dataloader = deteCT(
                mode=mode,
                geometry_params=self.geometry,
                parameters=self.param.data_loader_params,
            )
        else:
            raise NotImplementedError(f"Dataset {self.dataset} not implemented")
        return dataloader

    def get_training_dataset(self):
        return self.__get_dataset("train")

    def get_validation_dataset(self):
        return self.__get_dataset("validation")

    def get_testing_dataset(self):
        return self.__get_dataset("test")

    def __str__(self):
        return f"Experiment parameters: \n {self.param} \n Dataset: \n {self.dataset} \n Geometry parameters: \n {self.geometry}"

    @staticmethod
    def get_dataset_parameters(dataset, geometry=None):
        # if dataset == "LIDC-IDRI":
        #     return LIDC_IDRI.default_parameters(geometry=geometry)
        # if dataset == "2DeteCT":
        #     return deteCT.default_parameters()
        # else:
        #     raise NotImplementedError(f"Dataset {dataset} not implemented")
        return None


class ExtremeLowDoseCTRecon(Experiment):
    """Extremely low dose experiment.

    Fixed values (also see `ExtremeLowDoseCTRecon.default_parameters()`):
        - 10% of clinical dose
        - I0=1000
        - sigma=5
        - cross_talk=0.05
        - default geometry (see `ctgeo.Geometry.default_parameters()`)
    """

    def __init__(self, dataset: str = "LIDC-IDRI", datafolder: str | None = None):
        super().__init__(None, dataset, datafolder)

    # # TODO:
    # #   - Given `dataset` may be different from `self.dataset`?
    # #   - Should geometry just be inferred from `data_loader_params.geometry`?
    # @staticmethod
    # def default_parameters(dataset="LIDC-IDRI"):
    #     param = LIONParameter()
    #     param.name = "Extremely low dose full angular sampling experiment"
    #     # Parameters for the geometry
    #     param.geometry = ctgeo.Geometry.default_parameters()
    #     # Parameters for the noise in the sinogram.
    #     # Default, 10% of clinical dose.
    #     param.noise_params = LIONParameter()
    #     param.noise_params.I0 = 1000
    #     param.noise_params.sigma = 5
    #     param.noise_params.cross_talk = 0.05
    #     param.data_loader_params = Experiment.get_dataset_parameters(
    #         dataset, geometry=param.geometry
    #     )
    #     return param

    # Using the constructor is less painful than repeatedly writing `param.field = value` for each field?
    # Also more in line with the idea of fixed experiments (discourage runtime changes).
    # (Writing `param.field = value` for a non-dict-like class is technically allowed in Python
    # but doesn't seem right...)
    # TODO:
    #   - Given `dataset` may be different from `self.dataset`?
    #   - Should geometry just be inferred from `data_loader_params.geometry`?
    @staticmethod
    def default_parameters(dataset="LIDC-IDRI") -> LIONParameter:
        """Default extremely low dose CT reconstruction experiment's parameters."""
        geometry = ctgeo.Geometry.default_parameters()
        return CTReconParameter(
            name="Extremely low dose full angular sampling experiment",
            geometry=geometry,
            # Default, 10% of clinical dose.
            noise_params=LIONParameter(I0=1000, sigma=5, cross_talk=0.05),
            data_loader_params=Experiment.get_dataset_parameters(dataset, geometry),
        )


class LowDoseCTRecon(Experiment):
    """Low dose experiment.

    Fixed values (also see `LowDoseCTRecon.default_parameters()`):
        - 10% of clinical dose (TODO: same as `ExtremeLowDoseCTRecon`?
        Maybe comment in `LowDoseCTRecon.default_parameters()` is outdated?)
        - I0=3500
        - sigma=5
        - cross_talk=0.05
        - default geometry (see `ctgeo.Geometry.default_parameters()`)
    """

    def __init__(self, experiment_params=None, dataset="LIDC-IDRI", datafolder=None):

        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI"):
        param = LIONParameter()
        param.name = "Low dose full angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.default_parameters()
        # Parameters for the noise in the sinogram.
        # Default, 10% of clinical dose.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 3500
        param.noise_params.sigma = 5
        param.noise_params.cross_talk = 0.05
        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geometry=param.geometry
        )

        return param


class LimitedAngleCTRecon(Experiment):
    """Limited angle experiment.

    Fixed values (also see `LimitedAngleCTRecon.default_parameters()`):
        - 50% of clinical dose
        - I0=10000
        - sigma=5
        - cross_talk=0.05
        - sparse angle geometry (see `ctgeo.Geometry.sparse_angle_parameters()`)
    """

    def __init__(self, experiment_params=None, dataset="LIDC-IDRI", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI"):
        param = LIONParameter()
        param.name = "Clinical dose limited angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_angle_parameters()
        # Parameters for the noise in the sinogram.
        # Default, 50% of clinical dose.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 10000
        param.noise_params.sigma = 5
        param.noise_params.cross_talk = 0.05
        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geometry=param.geometry
        )

        return param


class LimitedAngleLowDoseCTRecon(Experiment):
    def __init__(self, experiment_params=None, dataset="LIDC-IDRI", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI"):
        param = LIONParameter()
        param.name = "Clinical dose limited angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_angle_parameters()
        # Parameters for the noise in the sinogram.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 3500
        param.noise_params.sigma = 5
        param.noise_params.cross_talk = 0.05

        if dataset == "LIDC-IDRI":
            # Parameters for the LIDC-IDRI dataset
            param.data_loader_params = LIDC_IDRI.default_parameters(
                geometry=param.geometry, task="reconstruction"
            )
            param.data_loader_params.max_num_slices_per_patient = 5
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented")
        return param


class LimitedAngleExtremeLowDoseCTRecon(Experiment):
    def __init__(self, experiment_params=None, dataset="LIDC-IDRI", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI"):
        param = LIONParameter()
        param.name = "Clinical dose limited angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_angle_parameters()
        # Parameters for the noise in the sinogram.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 1000
        param.noise_params.sigma = 5
        param.noise_params.cross_talk = 0.05

        if dataset == "LIDC-IDRI":
            # Parameters for the LIDC-IDRI dataset
            param.data_loader_params = LIDC_IDRI.default_parameters(
                geometry=param.geometry, task="reconstruction"
            )
            param.data_loader_params.max_num_slices_per_patient = 5
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented")
        return param


class SparseAngleCTRecon(Experiment):
    def __init__(self, experiment_params=None, dataset="LIDC-IDRI", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI"):
        param = LIONParameter()
        param.name = "Clinical dose sparse angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_view_parameters()
        # Parameters for the noise in the sinogram.
        # Default, 50% of clinical dose.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 10000
        param.noise_params.sigma = 5
        param.noise_params.cross_talk = 0.05

        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geometry=param.geometry
        )

        return param


class SparseAngleLowDoseCTRecon(Experiment):
    def __init__(self, experiment_params=None, dataset="LIDC-IDRI", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI"):
        param = LIONParameter()
        param.name = "Clinical dose sparse angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_view_parameters()
        # Parameters for the noise in the sinogram.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 3500
        param.noise_params.sigma = 5
        param.noise_params.cross_talk = 0.05

        if dataset == "LIDC-IDRI":
            # Parameters for the LIDC-IDRI dataset
            param.data_loader_params = LIDC_IDRI.default_parameters(
                geometry=param.geometry, task="reconstruction"
            )
            param.data_loader_params.max_num_slices_per_patient = 5
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented")
        return param


class SparseAngleExtremeLowDoseCTRecon(Experiment):
    def __init__(self, experiment_params=None, dataset="LIDC-IDRI", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI"):
        param = LIONParameter()
        param.name = "Clinical dose sparse angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_view_parameters()
        # Parameters for the noise in the sinogram.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 1000
        param.noise_params.sigma = 5
        param.noise_params.cross_talk = 0.05

        if dataset == "LIDC-IDRI":
            # Parameters for the LIDC-IDRI dataset
            param.data_loader_params = LIDC_IDRI.default_parameters(
                geometry=param.geometry, task="reconstruction"
            )
            param.data_loader_params.max_num_slices_per_patient = 5
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented")
        return param


class clinicalCTRecon(Experiment):
    def __init__(self, experiment_params=None, dataset="LIDC-IDRI", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI"):
        param = LIONParameter()
        param.name = "Clinical dose full angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.default_parameters()
        # Parameters for the noise in the sinogram.
        # Default, 50% of clinical dose.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 10000
        param.noise_params.sigma = 5
        param.noise_params.cross_talk = 0.05

        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geometry=param.geometry
        )

        return param
