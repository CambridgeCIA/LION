# =============================================================================
# This file is part of LION library
# License : BSD-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================


import numpy as np
import os

#%% Demo on LIONParameter class, used all across the toolbox to save and load parameters reproducibly.
# This demo shows how the tool manages the parameters of geometries, models and results.
# While as a user, you may not need to know most of this, it is convenient you learn about it. Your models and results should use these parameters
# so you can reproduce your experiments.
#%% 1-LIONParameters
from LION.utils.parameter import LIONParameter

# You can make a parameter of whatever you want.
my_params = LIONParameter()
# It is now empty.
print("Parameters:")
print(my_params)

# Lets fill it, you can put whatever you want
my_params.fruit = "banana"
my_params.num_fruits = np.array([4, 2, 12])
print("Parameters:")
print(my_params)
print("")

# You can save it
my_params.save("fruit.json")
# Do read now the json file with your favourite text editor. You can see that its quite easy to read, and even modify, if you want to.
# Exercise 1: Modify the file, and see what its loaded after.

# You can also load it:
new_my_params = LIONParameter()
new_my_params.load("fruit.json")
print("Parameters:")
print(new_my_params)
print("")

# lets clean up:
os.remove("fruit.json")

#%% 2-Special instances of LIONParameters.
# The tool provides (and you can make) special LIONParameters sets (that inherit from LIONParameter class).
# In particular, for CT, we have the Geometry() class, describing a CT geometry.
# You can load it as:
from LION.CTtools.ct_geometry import Geometry

# it also has a default
geo = Geometry.default_parameters()
print("Geometry:")
print(geo)
print("")

# You can save it and load is as before:
geo.save("geo.json")

geo2 = Geometry()
geo2.load("geo.json")

# lets clean up:
os.remove("geo.json")


#%% 3-Parameters inside ML-models.
# As the main goal of these is ensure reproducibility, all ML methods inside the toolbox have default_parameters and ways to create them from parameters.
# Lets use LPD as an example.
from LION.models.iterative_unrolled.LPD import LPD

# All ML-models will have this:
LPD_params = LPD.default_parameters()
print("Default LPD parameters")
print("-" * 30)
print(LPD_params)
print("")

# You can also initialize all ML-models from parameter files.
# As this is a tomography ML tool, you need to also give it geometry parameters (only models that require it).
model = LPD(geo, LPD_params)
# tada!

# You can now get such parameters:
LPD_param_2, LPD_geo = model.get_parameters()

# Notice that LPD computes the right step_size if given "none"
print("Computed LPD parameters")
print("-" * 30)
print(LPD_param_2)
print("")
