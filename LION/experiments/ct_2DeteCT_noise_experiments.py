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
