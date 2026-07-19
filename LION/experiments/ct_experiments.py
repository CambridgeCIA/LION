"""Reusable CT experiment definitions for LION datasets."""

# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================


import numpy as np
import torch
import pathlib
import warnings
from abc import ABC, abstractmethod, ABCMeta

from LION.utils.parameter import LIONParameter
import LION.CTtools.ct_geometry as ctgeo
import LION.CTtools.ct_utils as ct
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
from LION.data_loaders.deteCT import deteCT


class Experiment(ABC):
    """Bind CT geometry, measurement noise, and dataset parameters.

    Derived classes define :meth:`default_parameters`; instances then create
    consistent training, validation, and testing datasets for that protocol.
    """

    def __init__(
        self,
        experiment_params=None,
        dataset="LIDC-IDRI",
        datafolder=None,
        image_scaling=1.0,
    ):

        super().__init__()  # Initialize parent classes.
        __metaclass__ = ABCMeta  # make class abstract.
        self.experiment_params = experiment_params
        if experiment_params is None:
            self.experiment_params = self.default_parameters(
                dataset=dataset, image_scaling=image_scaling
            )
        else:
            self.experiment_params = experiment_params
        self.param = self.experiment_params
        if datafolder is not None:
            self.param.data_loader_params.folder = datafolder
        self.geometry = self.experiment_params.geometry
        self.dataset = dataset
        self.sino_fun = None
        if hasattr(self.param, "noise_params"):
            self.sino_fun = lambda sino, I0=self.param.noise_params.I0, sigma=self.param.noise_params.sigma, sigma_blur=self.param.noise_params.sigma_blur: ct.sinogram_add_noise(
                sino, I0=I0, sigma=sigma, sigma_blur=sigma_blur
            )

    @staticmethod
    @abstractmethod  # crash if not defined in derived class
    def default_parameters():
        pass

    def __get_dataset(self, mode):

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
                self.param.data_loader_params.noise_params.sigma_blur = (
                    self.param.noise_params.sigma_blur
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
        """Construct the training split dataset."""
        return self.__get_dataset("train")

    def get_validation_dataset(self):
        """Construct the validation split dataset."""
        return self.__get_dataset("validation")

    def get_testing_dataset(self):
        """Construct the testing split dataset."""
        return self.__get_dataset("test")

    def __str__(self):
        return f"Experiment parameters: \n {self.param} \n Dataset: \n {self.dataset} \n Geometry parameters: \n {self.geometry}"

    @staticmethod
    def get_dataset_parameters(dataset, geometry=None):
        """Return default loader parameters for a supported dataset."""
        if dataset == "LIDC-IDRI":
            return LIDC_IDRI.default_parameters(geometry=geometry)
        if dataset == "2DeteCT":
            return deteCT.default_parameters()
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented")


def _padis_lidc_fanbeam_parameters(
    *, dataset: str, image_scaling: float, view_count: int, angle_span: float
) -> LIONParameter:
    if dataset != "LIDC-IDRI":
        raise NotImplementedError(f"Dataset {dataset} not implemented")
    param = LIONParameter()
    span_degrees = float(np.degrees(angle_span))
    param.name = (
        f"PaDIS noise-free {view_count}-view {span_degrees:g}-degree "
        "LIDC fan-beam CT experiment"
    )
    param.geometry = ctgeo.Geometry.default_parameters(image_scaling=image_scaling)
    param.geometry.angles = np.linspace(0, angle_span, view_count, endpoint=False)
    param.view_count = view_count
    param.angle_span = angle_span
    # PaDIS forward-projects its [0, 1] model-domain CT images directly.
    param.measurement_source = "normal"
    param.data_loader_params = LIDC_IDRI.default_parameters(
        geometry=param.geometry, task="image_prior"
    )
    return param


class _PaDISFanBeamCTReconBase(Experiment):
    """Noise-free fan-beam PaDIS experiment in LION's LIDC geometry."""

    view_count: int
    angle_span: float = 2 * np.pi

    def __init__(
        self,
        experiment_params=None,
        dataset="LIDC-IDRI",
        datafolder=None,
        image_scaling=1.0,
    ):
        super().__init__(experiment_params, dataset, datafolder, image_scaling)

    @classmethod
    def default_parameters(cls, dataset="LIDC-IDRI", image_scaling=1.0):
        return _padis_lidc_fanbeam_parameters(
            dataset=dataset,
            image_scaling=image_scaling,
            view_count=cls.view_count,
            angle_span=cls.angle_span,
        )


class PaDISFanBeam8CTRecon(_PaDISFanBeamCTReconBase):
    """Noise-free eight-view full-angle LIDC fan-beam experiment."""

    view_count = 8


class PaDISFanBeam20CTRecon(_PaDISFanBeamCTReconBase):
    """Noise-free 20-view full-angle LIDC fan-beam experiment."""

    view_count = 20


class PaDISFanBeam60CTRecon(_PaDISFanBeamCTReconBase):
    """Noise-free 60-view full-angle LIDC fan-beam experiment."""

    view_count = 60


class PaDISFanBeam120LimitedCTRecon(_PaDISFanBeamCTReconBase):
    """Noise-free 20-view LIDC fan-beam experiment spanning 120 degrees."""

    view_count = 20
    angle_span = 2 * np.pi / 3


class PaDISFanBeam180CTRecon(PaDISFanBeam120LimitedCTRecon):
    """Compatibility alias for the paper-facing limited-angle fan-beam row."""


class ExtremeLowDoseCTRecon(Experiment):
    def __init__(
        self,
        experiment_params=None,
        dataset="LIDC-IDRI",
        datafolder=None,
        image_scaling=1.0,
    ):
        super().__init__(experiment_params, dataset, datafolder, image_scaling)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI", image_scaling=1.0):
        param = LIONParameter()
        param.name = "Extremely low dose full angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.default_parameters(image_scaling=image_scaling)
        # Parameters for the noise in the sinogram.
        # Default, 10% of clinical dose.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 1000
        param.noise_params.sigma = 5
        param.noise_params.sigma_blur = 0.3015
        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geometry=param.geometry
        )
        return param


class LowDoseCTRecon(Experiment):
    def __init__(
        self,
        experiment_params=None,
        dataset="LIDC-IDRI",
        datafolder=None,
        image_scaling=1.0,
    ):

        super().__init__(experiment_params, dataset, datafolder, image_scaling)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI", image_scaling=1.0):
        param = LIONParameter()
        param.name = "Low dose full angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.default_parameters(image_scaling=image_scaling)
        print("LDCTRecon", param.geometry)
        # Parameters for the noise in the sinogram.
        # Default, 10% of clinical dose.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 3500
        param.noise_params.sigma = 5
        param.noise_params.sigma_blur = 0.3015
        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geometry=param.geometry
        )

        return param


class LimitedAngleCTRecon(Experiment):
    def __init__(
        self,
        experiment_params=None,
        dataset="LIDC-IDRI",
        datafolder=None,
        image_scaling=1.0,
    ):
        super().__init__(experiment_params, dataset, datafolder, image_scaling)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI", image_scaling=1.0):
        param = LIONParameter()
        param.name = "Clinical dose limited angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_angle_parameters(
            image_scaling=image_scaling
        )
        # Parameters for the noise in the sinogram.
        # Default, 50% of clinical dose.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 10000
        param.noise_params.sigma = 5
        param.noise_params.sigma_blur = 0.3015
        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geometry=param.geometry
        )

        return param


class LimitedAngleLowDoseCTRecon(Experiment):
    def __init__(
        self,
        experiment_params=None,
        dataset="LIDC-IDRI",
        datafolder=None,
        image_scaling=1.0,
    ):
        super().__init__(experiment_params, dataset, datafolder, image_scaling)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI", image_scaling=1.0):
        param = LIONParameter()
        param.name = "Clinical dose limited angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_angle_parameters(
            image_scaling=image_scaling
        )
        # Parameters for the noise in the sinogram.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 3500
        param.noise_params.sigma = 5
        param.noise_params.sigma_blur = 0.3015

        if dataset == "LIDC-IDRI":
            # Parameters for the LIDC-IDRI dataset
            param.data_loader_params = LIDC_IDRI.default_parameters(
                geometry=param.geometry, task="reconstruction"
            )
            param.data_loader_params.max_num_slices_per_patient = 4
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented")
        return param


class LimitedAngleExtremeLowDoseCTRecon(Experiment):
    def __init__(
        self,
        experiment_params=None,
        dataset="LIDC-IDRI",
        datafolder=None,
        image_scaling=1.0,
    ):
        super().__init__(experiment_params, dataset, datafolder, image_scaling)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI", image_scaling=1.0):
        param = LIONParameter()
        param.name = "Clinical dose limited angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_angle_parameters(
            image_scaling=image_scaling
        )
        # Parameters for the noise in the sinogram.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 1000
        param.noise_params.sigma = 5
        param.noise_params.sigma_blur = 0.3015

        if dataset == "LIDC-IDRI":
            # Parameters for the LIDC-IDRI dataset
            param.data_loader_params = LIDC_IDRI.default_parameters(
                geometry=param.geometry, task="reconstruction"
            )
            param.data_loader_params.max_num_slices_per_patient = 4
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented")
        return param


class SparseAngleCTRecon(Experiment):
    def __init__(
        self,
        experiment_params=None,
        dataset="LIDC-IDRI",
        datafolder=None,
        image_scaling=1.0,
    ):
        super().__init__(experiment_params, dataset, datafolder, image_scaling)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI", image_scaling=1.0):
        param = LIONParameter()
        param.name = "Clinical dose sparse angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_view_parameters(
            image_scaling=image_scaling
        )
        # Parameters for the noise in the sinogram.
        # Default, 50% of clinical dose.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 10000
        param.noise_params.sigma = 5
        param.noise_params.sigma_blur = 0.3015

        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geometry=param.geometry
        )

        return param


class SparseAngleLowDoseCTRecon(Experiment):
    def __init__(
        self,
        experiment_params=None,
        dataset="LIDC-IDRI",
        datafolder=None,
        image_scaling=1.0,
    ):
        super().__init__(experiment_params, dataset, datafolder, image_scaling)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI", image_scaling=1.0):
        param = LIONParameter()
        param.name = "Clinical dose sparse angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_view_parameters(
            image_scaling=image_scaling
        )
        # Parameters for the noise in the sinogram.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 3500
        param.noise_params.sigma = 5
        param.noise_params.sigma_blur = 0.3015

        if dataset == "LIDC-IDRI":
            # Parameters for the LIDC-IDRI dataset
            param.data_loader_params = LIDC_IDRI.default_parameters(
                geometry=param.geometry, task="reconstruction"
            )
            param.data_loader_params.max_num_slices_per_patient = 4
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented")
        return param


class SparseAngleExtremeLowDoseCTRecon(Experiment):
    def __init__(
        self,
        experiment_params=None,
        dataset="LIDC-IDRI",
        datafolder=None,
        image_scaling=1.0,
    ):
        super().__init__(experiment_params, dataset, datafolder, image_scaling)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI", image_scaling=1.0):
        param = LIONParameter()
        param.name = "Clinical dose sparse angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_view_parameters(
            image_scaling=image_scaling
        )
        # Parameters for the noise in the sinogram.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 1000
        param.noise_params.sigma = 5
        param.noise_params.sigma_blur = 0.3015

        if dataset == "LIDC-IDRI":
            # Parameters for the LIDC-IDRI dataset
            param.data_loader_params = LIDC_IDRI.default_parameters(
                geometry=param.geometry, task="reconstruction"
            )
            param.data_loader_params.max_num_slices_per_patient = 4
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented")
        return param


class clinicalCTRecon(Experiment):
    def __init__(
        self,
        experiment_params=None,
        dataset="LIDC-IDRI",
        datafolder=None,
        image_scaling=1.0,
    ):
        super().__init__(experiment_params, dataset, datafolder, image_scaling)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI", image_scaling=1.0):
        param = LIONParameter()
        param.name = "Clinical dose full angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.default_parameters(image_scaling=image_scaling)
        # Parameters for the noise in the sinogram.
        # Default, 50% of clinical dose.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 10000
        param.noise_params.sigma = 5
        param.noise_params.sigma_blur = 0.3015

        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geometry=param.geometry
        )

        return param
