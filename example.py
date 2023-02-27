import numpy as np
import torch
import tomosipo as ts
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import skimage
import AItomotools.CTtools.ct_utils as ct

## This scripts generates the data available at:
from AItomotools.utils.paths import path_projections_luna_2d, path_luna
path_projections_luna=path_projections_luna_2d
print(path_projections_luna)

data=np.load(path_projections_luna+"/testing_nodule/clean/image_000015_slice_004.npy")
mask=np.load(path_projections_luna+"/testing_nodule/clean/mask_000015_slice_004.npy")

print(data.shape)

plt.figure()
plt.subplot(121)
plt.imshow(np.squeeze(data))
plt.subplot(122)
plt.imshow(np.squeeze(mask))
plt.savefig("test.png")