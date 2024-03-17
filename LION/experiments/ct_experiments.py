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
    def __init__(self, experiment_params=None, dataset="LIDC-IDRI", datafolder=None):

        super().__init__()  # Initialize parent classes.
        __metaclass__ = ABCMeta  # make class abstract.

        if experiment_params is None:
            self.experiment_params = self.default_parameters(dataset=dataset)
        else:
            self.experiment_params = experiment_params
        self.param = self.experiment_params
        if datafolder is not None:
            self.param.data_loader_params.folder = datafolder
        self.geo = self.experiment_params.geo
        self.dataset = dataset
        if hasattr(self.param, "noise_params"):
            self.sino_fun = lambda sino, I0=self.param.noise_params.I0, sigma=self.param.noise_params.sigma, cross_talk=self.param.noise_params.cross_talk: ct.sinogram_add_noise(
                sino, I0=I0, sigma=sigma, cross_talk=cross_talk
            )

    @staticmethod
    @abstractmethod  # crash if not defined in derived class
    def default_parameters():
        pass

    def get_training_dataset(self):

        if self.dataset == "LIDC-IDRI":
            dataloader = LIDC_IDRI(
                mode="train",
                parameters=self.param.data_loader_params,
                geometry_parameters=self.geo,
            )
            dataloader.set_sinogram_transform(self.sino_fun)

        elif self.dataset == "2DeteCT":
            dataloader = deteCT(
                mode="train",
                geometry_params=self.geo,
                parameters=self.param.data_loader_params,
            )
            if hasattr(self.param, "noise_params"):
                warnings.warn(
                    "Noise simulating parameters are not used 2DeteCT dataset, as it comes with real measured data"
                )
        else:
            raise NotImplementedError(f"Dataset {self.dataset} not implemented")
        return dataloader

    def get_validation_dataset(self):
        if self.dataset == "LIDC-IDRI":
            dataloader = LIDC_IDRI(
                mode="validation",
                parameters=self.param.data_loader_params,
                geometry_parameters=self.geo,
            )
            dataloader.set_sinogram_transform(self.sino_fun)

        elif self.dataset == "2DeteCT":
            dataloader = deteCT(
                mode="validation",
                geometry_params=self.geo,
                parameters=self.param.data_loader_params,
            )
            if hasattr(self.param, "noise_params"):
                warnings.warn(
                    "Noise simulating parameters are not used 2DeteCT dataset, as it comes with real measured data"
                )
        else:
            raise NotImplementedError(f"Dataset {self.dataset} not implemented")

        return dataloader

    def get_testing_dataset(self):
        if self.dataset == "LIDC-IDRI":
            dataloader = LIDC_IDRI(
                mode="test",
                parameters=self.param.data_loader_params,
                geometry_parameters=self.geo,
            )
            dataloader.set_sinogram_transform(self.sino_fun)

        elif self.dataset == "2DeteCT":
            dataloader = deteCT(
                mode="test",
                geometry_params=self.geo,
                parameters=self.param.data_loader_params,
            )
            if hasattr(self.param, "noise_params"):
                warnings.warn(
                    "Noise simulating parameters are not used 2DeteCT dataset, as it comes with real measured data"
                )
        else:
            raise NotImplementedError(f"Dataset {self.dataset} not implemented")
        return dataloader

    def __str__(self):
        return f"Experiment parameters: \n {self.param} \n Dataset: \n {self.dataset} \n Geometry parameters: \n {self.geo}"

    @staticmethod
    def get_dataset_parameters(dataset, geo=None):
        if dataset == "LIDC-IDRI":
            return LIDC_IDRI.default_parameters(geo=geo)
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
        param.geo = ctgeo.Geometry.default_parameters()
        # Parameters for the noise in the sinogram.
        # Default, 10% of clinical dose.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 1000
        param.noise_params.sigma = 5
        param.noise_params.cross_talk = 0.05
        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geo=param.geo
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
        param.geo = ctgeo.Geometry.default_parameters()
        # Parameters for the noise in the sinogram.
        # Default, 10% of clinical dose.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 3500
        param.noise_params.sigma = 5
        param.noise_params.cross_talk = 0.05
        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geo=param.geo
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
        param.geo = ctgeo.Geometry.sparse_angle_parameters()
        # Parameters for the noise in the sinogram.
        # Default, 50% of clinical dose.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 10000
        param.noise_params.sigma = 5
        param.noise_params.cross_talk = 0.05
        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geo=param.geo
        )

        return param


class SparseAngleCTRecon(Experiment):
    def __init__(self, experiment_params=None, dataset="LIDC-IDRI", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI"):
        param = LIONParameter()
        param.name = "Clinical dose sparse angular sampling experiment"
        # Parameters for the geometry
        param.geo = ctgeo.Geometry.sparse_view_parameters()
        # Parameters for the noise in the sinogram.
        # Default, 50% of clinical dose.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 10000
        param.noise_params.sigma = 5
        param.noise_params.cross_talk = 0.05

        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geo=param.geo
        )

        return param


class clinicalCTRecon(Experiment):
    def __init__(self, experiment_params=None, dataset="LIDC-IDRI", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI"):
        param = LIONParameter()
        param.name = "Clinical dose full angular sampling experiment"
        # Parameters for the geometry
        param.geo = ctgeo.Geometry.default_parameters()
        # Parameters for the noise in the sinogram.
        # Default, 50% of clinical dose.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 10000
        param.noise_params.sigma = 5
        param.noise_params.cross_talk = 0.05

        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geo=param.geo
        )

        return param
