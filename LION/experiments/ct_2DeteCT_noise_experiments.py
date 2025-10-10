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


class CTNoiseExperiment(Experiment):
    def __init__(self, experiment_params=None, dataset="2DeteCT", datafolder=None):
        dataset = "2DeteCT"
        if dataset != "2DeteCT":
            raise ValueError(
                "CTNoiseExperiment experiments only supports 2DeteCT dataset, currently"
            )
        super().__init__(experiment_params, dataset, datafolder)


class sample_delete_when_happy(CTNoiseExperiment):
    def __init__(self, datafolder=None):
        super().__init__(experiment_params=None, datafolder=datafolder)

    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        param = LIONParameter()

        param.name = "xxxx"

        # Parameters for the geometry
        param.geometry = deteCT.get_default_geometry()

        param.data_loader_params = Experiment.get_dataset_parameters("2DeteCT")

        # Change the data loader to be sino2recon
        param.data_loader_params.task = "sino2sino"
        param.data_loader_params.input_mode = "mode2"
        param.data_loader_params.target_mode = "mode1"

        param.data_loader_params.add_noise = True
        param.data_loader_params.noise_params = LIONParameter()
        param.data_loader_params.noise_params.I0 = 10000
        param.data_loader_params.noise_params.cross_talk = (
            0.05  # leave as is, from XCIST
        )
        return param
