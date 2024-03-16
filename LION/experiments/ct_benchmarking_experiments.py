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
from LION.data_loaders.deteCT import deteCT

from LION.experiments.ct_experiments import Experiment


class ExampleClassForMax(Experiment):
    def __init__(self, experiment_params=None, dataset="2DeteCT", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="2DeteCT"):
        param = LIONParameter()

        param.name = "Max Kiss is a great title giver"

        # Parameters for the geometry
        param.geo = deteCT.get_default_geometry()
        # 2DeteCT will error if the geometry doesn not match. This error will happen when the dataLoader is set.
        # if you want to test if it errors, use the following code in a script:
        # dataset= deteCT(geometry_params=geo,mode="train")
        # deteCT has now a function that allows _some_ flexibility on the geometry,
        # e.g. detector and image shape changes, or angle redefinition (as long as they are part of the original)

        # The following code shows examples of how to change the geometry in a calid way
        param.geo.angles = param.geo.angles[::2]  # or any other subsampling
        param.geo.detector_shape = (1, 100)  # or any other shape
        param.geo.image_shape = (1, 100, 100)  # or any other shape
        # but when image shape is changed, you need to chagne image size[0] too, because tomosipo gets confused otherwise
        param.geo.image_size[0] = 1024 / 100  # original/current

        # leave this untouched
        param.data_loader_params = Experiment.get_dataset_parameters(dataset)
        return param
