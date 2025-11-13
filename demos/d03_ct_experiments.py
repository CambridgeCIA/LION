# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================


import numpy as np
import os

#%% This demo covers the way you can load pre-defined CT experiment parameters.
# It is commont to see "low dose CT" or "limited anlge CT", but these descriptors are not deterministic enough, and many papers in the literature use different
# geometries and definitions. LION provides a set of experiments inbuilt and easy to use, and allows users to submit new ones.
# This demo shows how to define and use such experiments.

#%% 1: load experiment class
import LION.experiments.ct_experiments as ct_experiments

#%% 2: Loading an experiment

# There are several Experiments defined for you to use. You can load/define them as:
experiment = ct_experiments.clinicalCTRecon()
# you can also pass optional parameters. As with all classes in LION, you can pass them the experiment LIONParameter class as:
experiment_parameters = ct_experiments.clinicalCTRecon.default_parameters()
experiment = ct_experiments.clinicalCTRecon(experiment_params=experiment_parameters)
# however this is HIGHLY DISCOURAGED. The purpose of the experiment class is to have reliable and repeatable experiments.
# Please create a new Experiment if you want to change their parameters.

# you can also change the data loader. By default, Experiments use the LIDC-IDRI data loader.
experiment = ct_experiments.clinicalCTRecon(dataset="LIDC-IDRI")
experiment = ct_experiments.clinicalCTRecon(dataset="2DeteCT")

# you can print the experiment parameters
print(experiment)
#%% 3: Experiment types
# Currently, this is an exhaustive list of all experiments inplemented:

# Standard fan beam CT, properly sampeld and with not too much poisson noise:
experiment = ct_experiments.clinicalCTRecon()
# Same geometry but lower dose (more noise). About a third of photon counts of the clinical case
experiment = ct_experiments.LowDoseCTRecon()
# Same geometry but even lower dose. About a 10% of clinical dose.
experiment = ct_experiments.ExtremeLowDoseCTRecon()
# Limited view (angle) tomography. Noise levels as clinical CT, 60 angles of range
experiment = ct_experiments.LimitedAngleCTRecon()
# Sparse view (angle) tomography. Noise levels as clinical ct, but 50 angles over the full circle
experiment = ct_experiments.SparseAngleCTRecon()

#%% 4: How to use the Experiments output

# The return object, "experiment" is generally used to generate your data loaders and to pass CT geometry
# #information to the operators.

# Important uses are: passing CT geometry to other functions
experiment.geo
# e.g. you may want to forward project some image, or make a CT operator (see d_01)
from LION.CTtools.ct_utils import make_operator

op = make_operator(experiment.geo)

# Inspect experiment parameters
experiment.param
print(experiment.param)

# Produce and define DataLoaders. These are torch DataLoader class objects.
# Some of the DataLoaders need parameters defined (e.g. LIDC-IDRI needs the number of slices per patient for 2D),
# these values are also in the experiment.param, in:
experiment.param.data_loader_params
print(experiment.param.data_loader_params)
# Then, you can get your DataLoaders, as:
training_dataset = experiment.get_training_dataset()
validation_dataset = experiment.get_validation_dataset()
testing_dataset = experiment.get_testing_dataset()

#%% 5 Creating a new experiment

# if you have a new experiment, you should share it with the community by adding it to the LION/experiments/ct_experiments.py file
# for reproducibility purposes. Make a PR!
# However, if you are testing/playing with options, you may want to create it in your own file. This is easy.

# Import abstract class
from LION.experiments.ct_experiments import Experiment

# Import other classes you will need (these are imported in LION/experiments/ct_experiments.py)
from LION.utils.parameter import LIONParameter
import LION.CTtools.ct_geometry as ctgeo
import LION.CTtools.ct_utils as ct
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI


# Create a class inheriting from Experiment, and make an initialization.
# Please, make sure the __init__ has exactly this trace, at minimum. You can add things.
class TestExperiment(Experiment):
    def __init__(self, experiment_params=None, dataset="LIDC-IDRI"):
        super().__init__(experiment_params, dataset)

    # The only other method you need to define is default_parameters(), e.g. this is the one for clinicalCT
    @staticmethod
    def default_parameters(dataset="LIDC-IDRI"):
        param = LIONParameter()
        param.name = "Clinical dose experiment"
        # Parameters for the geometry
        param.geo = ctgeo.Geometry.default_parameters()
        # Parameters for the noise in the sinogram.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 10000
        param.noise_params.sigma = 5
        param.noise_params.cross_talk = 0.05

        if dataset == "LIDC-IDRI":
            # Parameters for the LIDC-IDRI dataset
            param.data_loader_params = LIDC_IDRI.default_parameters(
                geo=param.geo, task="reconstruction"
            )
            param.data_loader_params.max_num_slices_per_patient = 5
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented")
        return param


# Now you have a class, you can just call it
experiment = TestExperiment()
