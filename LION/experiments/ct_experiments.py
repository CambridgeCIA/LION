# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# =============================================================================
"""
LION's CT experiments: a geometry, a dataset and, for simulated data, the
noise added to the sinograms (``param.noise_params``).

Noise of the simulated (LIDC-IDRI) experiments, since 25 October 2026
---------------------------------------------------------------------
``Experiment.noise_parameters(I0)``: the physical noise model of
``ct_utils.sinogram_add_noise``. The images are densities (g/cm^3, see the
LIDC-IDRI data loader), the geometries are in mm, so the line integrals give
the real transmission through water at 80 keV, and ``I0`` is the number of
photons per detector pixel in vacuum. Thicker patients get noisier data, as in
a scanner, and photon-starved rays are floored at 1 photon before the log. The
detector is energy-integrating (electronic noise, cross-talk) unless
``detector="photon_counting"``.

The doses (``CLINICAL_I0``, ``LOW_DOSE_I0``, ``EXTREMELY_LOW_DOSE_I0``) were
chosen by Ander by looking at FDK reconstructions of small, typical and large
LIDC-IDRI patients.

Before, the experiments used the relative model: each sinogram divided by its
maximum, so every patient's thickest ray kept I0/e photons (I0 = 10000, 3500
and 1000). That is not how a scanner works; the noise did not depend on the
size of the patient. Parameter files saved before ``noise_model`` existed
still use the relative model (``noise_model = "relative"``), with the errors
LION's noise had before October 2026 fixed (cross-talk across angles, batch
maximum, 1e-6 rule for starved rays), so their results are close to, but not
bit-identical with, the old ones. New results are not comparable with old
results of the same experiment name.

Measured 2DeteCT data have their own noise; see the deteCT data loader.
"""

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
# noise model, water at 80 keV; see Experiment.noise_parameters). Chosen by
# Ander on 25 Oct 2026 by looking at LIDC-IDRI reconstructions. With the
# relative model, the experiments used 10000, 3500 and 1000.
CLINICAL_I0 = 20000
LOW_DOSE_I0 = 5000
EXTREMELY_LOW_DOSE_I0 = 1000


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
            kwargs = dict(
                # saved before noise_model existed (25 Oct 2026): the relative model
                model=getattr(noise, "noise_model", "relative"),
                detector=getattr(noise, "detector", "energy_integrating"),
                mass_attenuation=getattr(
                    noise, "mass_attenuation", ct.DEFAULT_MASS_ATTENUATION
                ),
                min_counts=getattr(noise, "min_counts", 1.0),
            )
            self.sino_fun = lambda sino, I0=noise.I0, sigma=noise.sigma, sigma_blur=noise.sigma_blur: ct.sinogram_add_noise(
                sino, I0=I0, sigma=sigma, sigma_blur=sigma_blur, **kwargs
            )

    @staticmethod
    def noise_parameters(I0, detector="energy_integrating"):
        """
        Noise of LION's simulated experiments: the physical model of
        ct_utils.sinogram_add_noise, with I0 photons per detector pixel in
        vacuum, water at 80 keV, and starved rays floored at 1 photon. The
        default energy-integrating detector has electronic noise (sigma = 5
        photons) and cross-talk; detector="photon_counting" has neither (see
        ct_utils.sinogram_add_noise).
        """
        noise = LIONParameter()
        noise.noise_model = "physical"
        noise.I0 = I0
        noise.sigma = 5
        noise.sigma_blur = 0.3015
        noise.detector = detector
        noise.mass_attenuation = ct.DEFAULT_MASS_ATTENUATION
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
