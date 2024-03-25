# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri, Max Kiss
# =============================================================================


import numpy as np
import torch
import pathlib
import warnings
from abc import ABC, abstractmethod, ABCMeta

from LION.utils.parameter import LIONParameter
import LION.CTtools.ct_geometry as ctgeo
import LION.CTtools.ct_utils as ct
from LION.data_loaders.deteCT import deteCT

from LION.experiments.ct_experiments import Experiment


class CTBenchmarkingExperiment(Experiment):
    def __init__(self, experiment_params=None, dataset="2DeteCT", datafolder=None):
        dataset = "2DeteCT"
        if dataset != "2DeteCT":
            raise ValueError(
                "Benchmarking experiments only supports 2DeteCT dataset, currently"
            )
        super().__init__(experiment_params, dataset, datafolder)


class FullDataCTRecon(CTBenchmarkingExperiment):
    def __init__(self, datafolder=None):
        super().__init__(experiment_params=None, datafolder=datafolder)

    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        param = LIONParameter()

        param.name = "Full Data CT reconstruction using the full angular sampling experiment from mode 2"

        # Parameters for the geometry
        param.geo = deteCT.get_default_geometry()

        param.data_loader_params = Experiment.get_dataset_parameters("2DeteCT")

        # Change the data loader to be sino2recon
        param.data_loader_params.task = "sino2recon"
        param.data_loader_params.input_mode = "mode2"
        param.data_loader_params.target_mode = "mode2"

        return param


class LimitedAngle150CTRecon(CTBenchmarkingExperiment):
    def __init__(self, experiment_params=None, dataset="2DeteCT", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        param = LIONParameter()

        param.name = (
            "150 degrees Limited Angle CT reconstruction experiment from mode 2"
        )

        # Parameters for the geometry
        param.geo = deteCT.get_default_geometry()

        # Changing the angles of the geometry in the 2DeteCT loading:
        param.geo.angles = param.geo.angles[:1500]  # to have projections 1 to 1,501
        # The data loader will automatically subsample the sinogram for you.

        # leave this untouched
        param.data_loader_params = Experiment.get_dataset_parameters("2DeteCT")

        # Change the data loader to be sino2recon
        param.data_loader_params.task = "sino2recon"
        param.data_loader_params.input_mode = "mode2"
        param.data_loader_params.target_mode = "mode2"

        return param


class LimitedAngle120CTRecon(CTBenchmarkingExperiment):
    def __init__(self, experiment_params=None, dataset="2DeteCT", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        param = LIONParameter()

        param.name = (
            "120 degrees Limited Angle CT reconstruction experiment from mode 2"
        )

        # Parameters for the geometry
        param.geo = deteCT.get_default_geometry()

        # Changing the angles of the geometry in the 2DeteCT loading:
        param.geo.angles = param.geo.angles[:1200]  # to have projections 1 to 1,201
        # The data loader will automatically subsample the sinogram for you.

        # leave this untouched
        param.data_loader_params = Experiment.get_dataset_parameters("2DeteCT")

        # Change the data loader to be sino2recon
        param.data_loader_params.task = "sino2recon"
        param.data_loader_params.input_mode = "mode2"
        param.data_loader_params.target_mode = "mode2"

        return param


class LimitedAngle90CTRecon(CTBenchmarkingExperiment):
    def __init__(self, experiment_params=None, dataset="2DeteCT", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        param = LIONParameter()

        param.name = "90 degrees Limited Angle CT reconstruction experiment from mode 2"

        # Parameters for the geometry
        param.geo = deteCT.get_default_geometry()

        # Changing the angles of the geometry in the 2DeteCT loading:
        param.geo.angles = param.geo.angles[:900]  # to have projections 1 to 901
        # The data loader will automatically subsample the sinogram for you.

        # leave this untouched
        param.data_loader_params = Experiment.get_dataset_parameters("2DeteCT")

        # Change the data loader to be sino2recon
        param.data_loader_params.task = "sino2recon"
        param.data_loader_params.input_mode = "mode2"
        param.data_loader_params.target_mode = "mode2"

        return param


class LimitedAngle60CTRecon(CTBenchmarkingExperiment):
    def __init__(self, experiment_params=None, dataset="2DeteCT", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        param = LIONParameter()

        param.name = "60 degrees Limited Angle CT reconstruction experiment from mode 2"

        # Parameters for the geometry
        param.geo = deteCT.get_default_geometry()

        # Changing the angles of the geometry in the 2DeteCT loading:
        param.geo.angles = param.geo.angles[:600]  # to have projections 1 to 601
        # The data loader will automatically subsample the sinogram for you.

        # leave this untouched
        param.data_loader_params = Experiment.get_dataset_parameters("2DeteCT")

        # Change the data loader to be sino2recon
        param.data_loader_params.task = "sino2recon"
        param.data_loader_params.input_mode = "mode2"
        param.data_loader_params.target_mode = "mode2"

        return param


class SparseAngle720CTRecon(CTBenchmarkingExperiment):
    def __init__(self, experiment_params=None, dataset="2DeteCT", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        param = LIONParameter()

        param.name = "720 angles Sparse Angle CT reconstruction experiment from mode 2"

        # Parameters for the geometry
        param.geo = deteCT.get_default_geometry()

        # Changing the angles of the geometry in the 2DeteCT loading:
        param.geo.angles = param.geo.angles[
            ::5
        ]  # to have every 5th projection (in total 720)
        # The data loader will automatically subsample the sinogram for you.

        # leave this untouched
        param.data_loader_params = Experiment.get_dataset_parameters("2DeteCT")

        # Change the data loader to be sino2recon
        param.data_loader_params.task = "sino2recon"
        param.data_loader_params.input_mode = "mode2"
        param.data_loader_params.target_mode = "mode2"

        return param


class SparseAngle360CTRecon(CTBenchmarkingExperiment):
    def __init__(self, experiment_params=None, dataset="2DeteCT", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        param = LIONParameter()

        param.name = "360 angles Sparse Angle CT reconstruction experiment from mode 2"

        # Parameters for the geometry
        param.geo = deteCT.get_default_geometry()

        # Changing the angles of the geometry in the 2DeteCT loading:
        param.geo.angles = param.geo.angles[
            ::10
        ]  # to have every 10th projection (in total 360)
        # The data loader will automatically subsample the sinogram for you.

        # leave this untouched
        param.data_loader_params = Experiment.get_dataset_parameters("2DeteCT")

        # Change the data loader to be sino2recon
        param.data_loader_params.task = "sino2recon"
        param.data_loader_params.input_mode = "mode2"
        param.data_loader_params.target_mode = "mode2"

        return param


class SparseAngle180CTRecon(CTBenchmarkingExperiment):
    def __init__(self, experiment_params=None, dataset="2DeteCT", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        param = LIONParameter()

        param.name = "180 angles Sparse Angle CT reconstruction experiment from mode 2"

        # Parameters for the geometry
        param.geo = deteCT.get_default_geometry()

        # Changing the angles of the geometry in the 2DeteCT loading:
        param.geo.angles = param.geo.angles[
            ::20
        ]  # to have every 20th projection (in total 180)
        # The data loader will automatically subsample the sinogram for you.

        # leave this untouched
        param.data_loader_params = Experiment.get_dataset_parameters("2DeteCT")

        # Change the data loader to be sino2recon
        param.data_loader_params.task = "sino2recon"
        param.data_loader_params.input_mode = "mode2"
        param.data_loader_params.target_mode = "mode2"

        return param


class SparseAngle120CTRecon(CTBenchmarkingExperiment):
    def __init__(self, experiment_params=None, dataset="2DeteCT", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        param = LIONParameter()

        param.name = "120 angles Sparse Angle CT reconstruction experiment from mode 2"

        # Parameters for the geometry
        param.geo = deteCT.get_default_geometry()

        # Changing the angles of the geometry in the 2DeteCT loading:
        param.geo.angles = param.geo.angles[
            ::30
        ]  # to have every 30th projection (in total 120)
        # The data loader will automatically subsample the sinogram for you.

        # leave this untouched
        param.data_loader_params = Experiment.get_dataset_parameters("2DeteCT")

        # Change the data loader to be sino2recon
        param.data_loader_params.task = "sino2recon"
        param.data_loader_params.input_mode = "mode2"
        param.data_loader_params.target_mode = "mode2"

        return param


class SparseAngle90CTRecon(CTBenchmarkingExperiment):
    def __init__(self, experiment_params=None, dataset="2DeteCT", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        param = LIONParameter()

        param.name = "90 angles Sparse Angle CT reconstruction experiment from mode 2"

        # Parameters for the geometry
        param.geo = deteCT.get_default_geometry()

        # Changing the angles of the geometry in the 2DeteCT loading:
        param.geo.angles = param.geo.angles[
            ::40
        ]  # to have every 40th projection (in total 90)
        # The data loader will automatically subsample the sinogram for you.

        # leave this untouched
        param.data_loader_params = Experiment.get_dataset_parameters("2DeteCT")

        # Change the data loader to be sino2recon
        param.data_loader_params.task = "sino2recon"
        param.data_loader_params.input_mode = "mode2"
        param.data_loader_params.target_mode = "mode2"

        return param


class SparseAngle60CTRecon(CTBenchmarkingExperiment):
    def __init__(self, experiment_params=None, dataset="2DeteCT", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        param = LIONParameter()

        param.name = "60 angles Sparse Angle CT reconstruction experiment from mode2"

        # Parameters for the geometry
        param.geo = deteCT.get_default_geometry()

        # Changing the angles of the geometry in the 2DeteCT loading:
        param.geo.angles = param.geo.angles[
            ::60
        ]  # to have every 60th projection (in total 60)
        # The data loader will automatically subsample the sinogram for you.

        # leave this untouched
        param.data_loader_params = Experiment.get_dataset_parameters("2DeteCT")

        # Change the data loader to be sino2recon
        param.data_loader_params.task = "sino2recon"
        param.data_loader_params.input_mode = "mode2"
        param.data_loader_params.target_mode = "mode2"

        return param


class LowDoseCTRecon(CTBenchmarkingExperiment):
    def __init__(self, experiment_params=None, dataset="2DeteCT", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        param = LIONParameter()

        param.name = "Denoising experiment with sinograms from mode1 and reconstructions from mode2"

        # Parameters for the geometry
        param.geo = deteCT.get_default_geometry()

        # leave this untouched
        param.data_loader_params = Experiment.get_dataset_parameters("2DeteCT")

        # Change the data loader to be sino2recon
        param.data_loader_params.task = "sino2recon"
        param.data_loader_params.input_mode = "mode1"
        param.data_loader_params.target_mode = "mode2"

        return param


class BeamHardeningCTRecon(CTBenchmarkingExperiment):
    def __init__(self, experiment_params=None, dataset="2DeteCT", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        param = LIONParameter()

        param.name = "Beam Hardening Reduction experiment with sinograms from mode3 and reconstructions from mode2"

        # Parameters for the geometry
        param.geo = deteCT.get_default_geometry()

        # leave this untouched
        param.data_loader_params = Experiment.get_dataset_parameters("2DeteCT")

        # Change the data loader to be sino2recon
        param.data_loader_params.task = "sino2recon"
        param.data_loader_params.input_mode = "mode3"
        param.data_loader_params.target_mode = "mode2"

        return param
