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
        self.experiment_params = experiment_params
        if experiment_params is None:
            self.experiment_params = self.default_parameters(dataset=dataset)
        else:
            self.experiment_params = experiment_params
        self.param = self.experiment_params
        if datafolder is not None:
            if dataset == "2DeteCT":
                self.param.data_loader_params.path_to_dataset = pathlib.Path(datafolder)
            else:
                self.param.data_loader_params.folder = datafolder
        self.geometry = self.experiment_params.geometry
        self.dataset = dataset
        # Noise added to the simulated sinograms (None: clean sinograms)
        self.sino_fun = None
        if hasattr(self.param, "noise_params"):
            noise = self.param.noise_params
            # parameter files from before legacy_noise_model existed used the old model
            legacy = getattr(noise, "legacy_noise_model", True)
            self.sino_fun = lambda sino, I0=noise.I0, sigma=noise.sigma, sigma_blur=noise.sigma_blur: ct.sinogram_add_noise(
                sino, I0=I0, sigma=sigma, sigma_blur=sigma_blur, legacy=legacy
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
                raise ValueError(
                    "2DeteCT is measured data: its experiments use the measured "
                    "modes (e.g. mode1 for low dose) instead of noise_params. To "
                    "add noise on top of the measurements, set "
                    "data_loader_params.add_noise and data_loader_params.noise_params "
                    "(I0, cross_talk) instead."
                )
            dataloader = deteCT(
                mode=mode,
                geometry_params=self.geometry,
                parameters=self.param.data_loader_params,
            )
        else:
            raise NotImplementedError(f"Dataset {self.dataset} not implemented")
        return dataloader

    @staticmethod
    def measured_2detect_parameters(
        name: str, angles=slice(None), dose: str = "clinical"
    ) -> LIONParameter:
        """Parameters of an experiment on the measured 2DeteCT data.

        Its geometry is fixed by the scan (3600 angles over 360 degrees), so an
        experiment can only select some of the measured angles (``angles``, an
        index or slice into them). Dose levels are measured modes: "clinical"
        uses mode2 sinograms, "low" the low-dose mode1 ones; the targets are
        mode2 reconstructions. There is no noise to simulate.
        """
        input_modes = {"clinical": "mode2", "low": "mode1"}
        if dose not in input_modes:
            raise ValueError(
                f"2DeteCT has measured data for the doses {list(input_modes)}, not {dose!r}"
            )
        param = LIONParameter()
        param.name = f"{name}, on the measured 2DeteCT data"
        param.geometry = deteCT.get_default_geometry()
        param.geometry.angles = np.asarray(param.geometry.angles)[angles]
        param.data_loader_params = Experiment.get_dataset_parameters("2DeteCT")
        param.data_loader_params.task = "sino2recon"
        param.data_loader_params.input_mode = input_modes[dose]
        param.data_loader_params.target_mode = "mode2"
        return param

    @staticmethod
    def not_possible_with_2detect(name: str, reason: str):
        raise ValueError(
            f"{name} is not possible with the measured 2DeteCT data: {reason}"
        )

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
        if dataset == "2DeteCT":
            Experiment.not_possible_with_2detect(
                "Extremely low dose full angular sampling experiment",
                "there is no measured mode with an extremely low dose",
            )
        param = LIONParameter()
        param.name = "Extremely low dose full angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.default_parameters()
        # Parameters for the noise in the sinogram.
        # Default, 10% of clinical dose.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 1000
        param.noise_params.sigma = 5
        param.noise_params.sigma_blur = 0.3015
        # True reproduces the noise of LION before October 2026 (see sinogram_add_noise)
        param.noise_params.legacy_noise_model = False
        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geometry=param.geometry
        )
        return param


class LowDoseCTRecon(Experiment):
    def __init__(self, experiment_params=None, dataset="LIDC-IDRI", datafolder=None):

        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI"):
        if dataset == "2DeteCT":
            return Experiment.measured_2detect_parameters(
                "Low dose full angular sampling experiment",
                angles=slice(None),
                dose="low",
            )
        param = LIONParameter()
        param.name = "Low dose full angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.default_parameters()
        # Parameters for the noise in the sinogram.
        # Default, 10% of clinical dose.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 3500
        param.noise_params.sigma = 5
        param.noise_params.sigma_blur = 0.3015
        # True reproduces the noise of LION before October 2026 (see sinogram_add_noise)
        param.noise_params.legacy_noise_model = False
        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geometry=param.geometry
        )

        return param


class LimitedAngleCTRecon(Experiment):
    def __init__(self, experiment_params=None, dataset="LIDC-IDRI", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI"):
        if dataset == "2DeteCT":
            return Experiment.measured_2detect_parameters(
                "Clinical dose limited angular sampling experiment",
                angles=slice(0, 600, 10),
                dose="clinical",
            )  # 60 views over 60 degrees
        param = LIONParameter()
        param.name = "Clinical dose limited angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_angle_parameters()
        # Parameters for the noise in the sinogram.
        # Default, 50% of clinical dose.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 10000
        param.noise_params.sigma = 5
        param.noise_params.sigma_blur = 0.3015
        # True reproduces the noise of LION before October 2026 (see sinogram_add_noise)
        param.noise_params.legacy_noise_model = False
        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geometry=param.geometry
        )

        return param


class LimitedAngleLowDoseCTRecon(Experiment):
    def __init__(self, experiment_params=None, dataset="LIDC-IDRI", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI"):
        if dataset == "2DeteCT":
            return Experiment.measured_2detect_parameters(
                "Low dose limited angular sampling experiment",
                angles=slice(0, 600, 10),
                dose="low",
            )  # 60 views over 60 degrees
        param = LIONParameter()
        param.name = "Low dose limited angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_angle_parameters()
        # Parameters for the noise in the sinogram.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 3500
        param.noise_params.sigma = 5
        param.noise_params.sigma_blur = 0.3015
        # True reproduces the noise of LION before October 2026 (see sinogram_add_noise)
        param.noise_params.legacy_noise_model = False

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
        if dataset == "2DeteCT":
            Experiment.not_possible_with_2detect(
                "Extremely low dose limited angular sampling experiment",
                "there is no measured mode with an extremely low dose",
            )
        param = LIONParameter()
        param.name = "Extremely low dose limited angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_angle_parameters()
        # Parameters for the noise in the sinogram.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 1000
        param.noise_params.sigma = 5
        param.noise_params.sigma_blur = 0.3015
        # True reproduces the noise of LION before October 2026 (see sinogram_add_noise)
        param.noise_params.legacy_noise_model = False

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
        if dataset == "2DeteCT":
            return Experiment.measured_2detect_parameters(
                "Clinical dose sparse angular sampling experiment",
                angles=slice(None, None, 72),
                dose="clinical",
            )  # 50 views, every 7.2 degrees
        param = LIONParameter()
        param.name = "Clinical dose sparse angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_view_parameters()
        # Parameters for the noise in the sinogram.
        # Default, 50% of clinical dose.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 10000
        param.noise_params.sigma = 5
        param.noise_params.sigma_blur = 0.3015
        # True reproduces the noise of LION before October 2026 (see sinogram_add_noise)
        param.noise_params.legacy_noise_model = False

        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geometry=param.geometry
        )

        return param


class SparseAngleLowDoseCTRecon(Experiment):
    def __init__(self, experiment_params=None, dataset="LIDC-IDRI", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI"):
        if dataset == "2DeteCT":
            return Experiment.measured_2detect_parameters(
                "Low dose sparse angular sampling experiment",
                angles=slice(None, None, 72),
                dose="low",
            )  # 50 views, every 7.2 degrees
        param = LIONParameter()
        param.name = "Low dose sparse angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_view_parameters()
        # Parameters for the noise in the sinogram.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 3500
        param.noise_params.sigma = 5
        param.noise_params.sigma_blur = 0.3015
        # True reproduces the noise of LION before October 2026 (see sinogram_add_noise)
        param.noise_params.legacy_noise_model = False

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
        if dataset == "2DeteCT":
            Experiment.not_possible_with_2detect(
                "Extremely low dose sparse angular sampling experiment",
                "there is no measured mode with an extremely low dose",
            )
        param = LIONParameter()
        param.name = "Extremely low dose sparse angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.sparse_view_parameters()
        # Parameters for the noise in the sinogram.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 1000
        param.noise_params.sigma = 5
        param.noise_params.sigma_blur = 0.3015
        # True reproduces the noise of LION before October 2026 (see sinogram_add_noise)
        param.noise_params.legacy_noise_model = False

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
        if dataset == "2DeteCT":
            return Experiment.measured_2detect_parameters(
                "Clinical dose full angular sampling experiment",
                angles=slice(None),
                dose="clinical",
            )
        param = LIONParameter()
        param.name = "Clinical dose full angular sampling experiment"
        # Parameters for the geometry
        param.geometry = ctgeo.Geometry.default_parameters()
        # Parameters for the noise in the sinogram.
        # Default, 50% of clinical dose.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 10000
        param.noise_params.sigma = 5
        param.noise_params.sigma_blur = 0.3015
        # True reproduces the noise of LION before October 2026 (see sinogram_add_noise)
        param.noise_params.legacy_noise_model = False

        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geometry=param.geometry
        )

        return param
