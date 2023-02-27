import numpy as np
import torch
import tomosipo as ts
import matplotlib.pyplot as plt

## This scripts generates the data available at:
from AItomotools.utils.paths import path_projections_luna
print(path_projections_luna)

#%% This scripts loands data from LUNA16, randomly slices the images, and stores the result.
# Then it simulates forward projections of a particular geometry and adds realistic noise of different levels to it.
# For the testing set, the slices that contain nodules are used. 

