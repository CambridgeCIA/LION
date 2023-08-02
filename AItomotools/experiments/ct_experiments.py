# =============================================================================
# This file is part of AItomotools library
# License : BSD-3
#
# Author  : Ander Biguri
# =============================================================================


import numpy as np
import torch
import pathlib
import warnings

from AItomotools.utils.parameter import Parameter
import AItomotools.CTtools.ct_geometry as ctgeo
import AItomotools.CTtools.ct_utils as ct
from AItomotools.data_loaders.LIDC_IDRI import LIDC_IDRI


class ExtremeLowDoseCTRecon:
    def __init__(self, experiment_params=None):

        if experiment_params is None:
            experiment_params = ExtremeLowDoseCTRecon.default_parameters()

        self.param = experiment_params
        self.geo = experiment_params.geo

        self.sino_fun = lambda sino, I0=self.param.noise_params.I0, sigma=self.param.noise_params.sigma, cross_talk=self.param.noise_params.cross_talk: ct.sinogram_add_noise(
            sino, I0=I0, sigma=sigma, cross_talk=cross_talk
        )

    @staticmethod
    def default_parameters():
        param = Parameter()
        param.name = "Low dose full angular sampling experiment"
        # Parameters for the geometry
        param.geo = ctgeo.Geometry.default_parameters()
        # Parameters for the noise in the sinogram.
        # Default, 10% of clinical dose.
        param.noise_params = Parameter()
        param.noise_params.I0 = 1000
        param.noise_params.sigma = 5
        param.noise_params.cross_talk = 0.05
        # Parameters for the LIDC-IDRI dataset
        param.data_loader_params = LIDC_IDRI.default_parameters(
            geo=param.geo, task="reconstruction"
        )
        param.data_loader_params.max_num_slices_per_patient = 5
        return param

    def get_training_dataset(self):
        dataloader = LIDC_IDRI(
            mode="training", parameters=self.param.data_loader_params
        )
        dataloader.set_sinogram_transform(self.sino_fun)
        return dataloader

    def get_validation_dataset(self):
        dataloader = LIDC_IDRI(
            mode="validation", parameters=self.param.data_loader_params
        )
        dataloader.set_sinogram_transform(self.sino_fun)
        return dataloader

    def get_testing_dataset(self):
        dataloader = LIDC_IDRI(mode="testing", parameters=self.param.data_loader_params)
        dataloader.set_sinogram_transform(self.sino_fun)
        return dataloader


class LowDoseCTRecon:
    def __init__(self, experiment_params=None):

        if experiment_params is None:
            experiment_params = LowDoseCTRecon.default_parameters()

        self.param = experiment_params
        self.geo = experiment_params.geo

        self.sino_fun = lambda sino, I0=self.param.noise_params.I0, sigma=self.param.noise_params.sigma, cross_talk=self.param.noise_params.cross_talk: ct.sinogram_add_noise(
            sino, I0=I0, sigma=sigma, cross_talk=cross_talk
        )

    @staticmethod
    def default_parameters():
        param = Parameter()
        param.name = "Low dose full angular sampling experiment"
        # Parameters for the geometry
        param.geo = ctgeo.Geometry.default_parameters()
        # Parameters for the noise in the sinogram.
        # Default, 10% of clinical dose.
        param.noise_params = Parameter()
        param.noise_params.I0 = 3500
        param.noise_params.sigma = 5
        param.noise_params.cross_talk = 0.05
        # Parameters for the LIDC-IDRI dataset
        param.data_loader_params = LIDC_IDRI.default_parameters(
            geo=param.geo, task="reconstruction"
        )
        param.data_loader_params.max_num_slices_per_patient = 5
        return param

    def get_training_dataset(self):
        dataloader = LIDC_IDRI(
            mode="training", parameters=self.param.data_loader_params
        )
        dataloader.set_sinogram_transform(self.sino_fun)
        return dataloader

    def get_validation_dataset(self):
        dataloader = LIDC_IDRI(
            mode="validation", parameters=self.param.data_loader_params
        )
        dataloader.set_sinogram_transform(self.sino_fun)
        return dataloader

    def get_testing_dataset(self):
        dataloader = LIDC_IDRI(mode="testing", parameters=self.param.data_loader_params)
        dataloader.set_sinogram_transform(self.sino_fun)
        return dataloader


class LimitedAngleCTRecon:
    def __init__(self, experiment_params=None):
        if experiment_params is None:
            experiment_params = ExtremeLowDoseCTRecon.default_parameters()

        self.param = experiment_params
        self.geo = experiment_params.geo
        self.sino_fun = lambda sino, I0=self.param.noise_params.I0, sigma=self.param.noise_params.sigma, cross_talk=self.param.noise_params.cross_talk: ct.sinogram_add_noise(
            sino, I0=I0, sigma=sigma, cross_talk=cross_talk
        )

    @staticmethod
    def default_parameters():
        param = Parameter()
        param.name = "Mid dose sparse angular sampling experiment"
        # Parameters for the geometry
        param.geo = ctgeo.Geometry.sparse_view_parameters()
        # Parameters for the noise in the sinogram.
        # Default, 50% of clinical dose.
        param.noise_params = Parameter()
        param.noise_params.I0 = 10000
        param.noise_params.sigma = 5
        param.noise_params.cross_talk = 0.05
        # Parameters for the LIDC-IDRI dataset
        param.data_loader_params = LIDC_IDRI.default_parameters(
            geo=param.geo, task="reconstruction"
        )
        param.data_loader_params.max_num_slices_per_patient = 5

        return param

    def get_training_dataset(self):
        dataloader = LIDC_IDRI(
            mode="training", parameters=self.param.data_loader_params
        )
        dataloader.set_sinogram_transform(self.sino_fun)
        return dataloader

    def get_validation_dataset(self):
        dataloader = LIDC_IDRI(
            mode="validation", parameters=self.param.data_loader_params
        )
        dataloader.set_sinogram_transform(self.sino_fun)
        return dataloader

    def get_testing_dataset(self):
        dataloader = LIDC_IDRI(mode="testing", parameters=self.param.data_loader_params)
        dataloader.set_sinogram_transform(self.sino_fun)
        return dataloader
