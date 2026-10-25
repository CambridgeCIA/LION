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

# Photons per detector pixel in vacuum of the simulated experiments (physical
# noise model, see Experiment.noise_parameters). Chosen by looking at
# LIDC-IDRI reconstructions (scripts/noise_study, 25 Oct 2026); before, the
# relative model used 10000, 3500 and 1000.
CLINICAL_I0 = 100000
LOW_DOSE_I0 = 30000
EXTREMELY_LOW_DOSE_I0 = 7000


class Experiment(ABC):
    def __init__(self, experiment_params=None, dataset="LIDC-IDRI", datafolder=None):

        super().__init__()  # Initialize parent classes.
        __metaclass__ = ABCMeta  # make class abstract.
        self.experiment_params = experiment_params
        if experiment_params is None:
            self.experiment_params = self.default_parameters(dataset=dataset)
        else:
            self.experiment_params = experiment_params
        self.param = self.experiment_params
        if datafolder is not None:
            self.param.data_loader_params.folder = datafolder
        self.geometry = self.experiment_params.geometry
        self.dataset = dataset
        # Noise added to the simulated sinograms (None: clean sinograms)
        self.sino_fun = None
        if hasattr(self.param, "noise_params"):
            noise = self.param.noise_params
            model = getattr(noise, "noise_model", None)
            if model is None:
                # saved before noise_model existed (25 Oct 2026): the relative
                # model, and before legacy_noise_model (22 Oct 2026) exactly
                # LION's original noise
                model = "relative"
                legacy = getattr(noise, "legacy_noise_model", True)
            else:
                legacy = getattr(noise, "legacy_noise_model", False)
            kwargs = dict(
                legacy=legacy,
                model=model,
                mass_attenuation=getattr(
                    noise, "mass_attenuation", ct.WATER_MASS_ATTENUATION[60]
                ),
                # missing: LION's old rule for starved rays (see sinogram_add_noise)
                min_counts=getattr(noise, "min_counts", None),
            )
            self.sino_fun = lambda sino, I0=noise.I0, sigma=noise.sigma, sigma_blur=noise.sigma_blur: ct.sinogram_add_noise(
                sino, I0=I0, sigma=sigma, sigma_blur=sigma_blur, **kwargs
            )

    @staticmethod
    def noise_parameters(I0):
        """
        Noise of LION's simulated experiments: the physical model of
        ct_utils.sinogram_add_noise, with I0 photons per detector pixel in
        vacuum, water at 60 keV, electronic noise sigma = 5 photons, detector
        cross-talk, and starved rays floored at 1 photon.
        """
        noise = LIONParameter()
        noise.noise_model = "physical"
        noise.I0 = I0
        noise.sigma = 5
        noise.sigma_blur = 0.3015
        noise.mass_attenuation = ct.WATER_MASS_ATTENUATION[60]
        noise.min_counts = 1.0
        return noise

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
                    "You are setting noise parameters for a dataset that comes with real measured data. The noise will be added on top of the real measured noise\n Note that only noise_params.I0 and noise_params.sigma_blur will be used"
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
        return self.__get_dataset("train")

    def get_validation_dataset(self):
        return self.__get_dataset("validation")

    def get_testing_dataset(self):
        return self.__get_dataset("test")

    def __str__(self):
        return f"Experiment parameters: \n {self.param} \n Dataset: \n {self.dataset} \n Geometry parameters: \n {self.geometry}"

    @staticmethod
    def get_dataset_parameters(dataset, geometry=None):
        if dataset == "LIDC-IDRI":
            return LIDC_IDRI.default_parameters(geometry=geometry)
        if dataset == "2DeteCT":
            return deteCT.default_parameters()
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented")


class ExtremeLowDoseCTRecon(Experiment):
    def __init__(self, experiment_params=None, dataset="LIDC-IDRI", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI"):
        param = LIONParameter()
        param.name = "Extremely low dose full angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.default_parameters()
        # Noise in the sinogram (see Experiment.noise_parameters)
        param.noise_params = Experiment.noise_parameters(EXTREMELY_LOW_DOSE_I0)
        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geometry=param.geometry
        )
        return param


class LowDoseCTRecon(Experiment):
    def __init__(self, experiment_params=None, dataset="LIDC-IDRI", datafolder=None):

        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI"):
        param = LIONParameter()
        param.name = "Low dose full angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.default_parameters()
        # Noise in the sinogram (see Experiment.noise_parameters)
        param.noise_params = Experiment.noise_parameters(LOW_DOSE_I0)
        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geometry=param.geometry
        )

        return param


class LimitedAngleCTRecon(Experiment):
    def __init__(self, experiment_params=None, dataset="LIDC-IDRI", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI"):
        param = LIONParameter()
        param.name = "Clinical dose limited angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_angle_parameters()
        # Noise in the sinogram (see Experiment.noise_parameters)
        param.noise_params = Experiment.noise_parameters(CLINICAL_I0)
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
        param.name = "Low dose limited angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_angle_parameters()
        # Noise in the sinogram (see Experiment.noise_parameters)
        param.noise_params = Experiment.noise_parameters(LOW_DOSE_I0)

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
        param.name = "Extremely low dose limited angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_angle_parameters()
        # Noise in the sinogram (see Experiment.noise_parameters)
        param.noise_params = Experiment.noise_parameters(EXTREMELY_LOW_DOSE_I0)

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
        # Noise in the sinogram (see Experiment.noise_parameters)
        param.noise_params = Experiment.noise_parameters(CLINICAL_I0)

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
        param.name = "Low dose sparse angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_view_parameters()
        # Noise in the sinogram (see Experiment.noise_parameters)
        param.noise_params = Experiment.noise_parameters(LOW_DOSE_I0)

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
        param.name = "Extremely low dose sparse angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_view_parameters()
        # Noise in the sinogram (see Experiment.noise_parameters)
        param.noise_params = Experiment.noise_parameters(EXTREMELY_LOW_DOSE_I0)

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
        # Noise in the sinogram (see Experiment.noise_parameters)
        param.noise_params = Experiment.noise_parameters(CLINICAL_I0)

        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geometry=param.geometry
        )

        return param
