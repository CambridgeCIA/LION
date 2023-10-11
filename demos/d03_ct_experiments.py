# =============================================================================
# This file is part of LION library
# License : BSD-3
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

#%% 2:

experiment = ct_experiments.ExtremeLowDoseCTRecon()
