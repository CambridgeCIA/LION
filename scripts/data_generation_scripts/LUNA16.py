# =============================================================================
# This file is part of AItomotools library
# License : BSD-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================


import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import tomosipo as ts
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import skimage
import AItomotools.CTtools.ct_geometry as ct
from AItomotools.utils.parameter import Parameter

## This scripts generates the data available at:
from AItomotools.utils.paths import LUNA_PROCESSED_DATASET_PATH, LUNA_DATASET_PATH

print(LUNA_PROCESSED_DATASET_PATH)


# Lets make a Parameter() describing the dataset

luna_parameter = Parameter()
luna_parameter.raw_data_location = LUNA_DATASET_PATH
luna_parameter.data_location = LUNA_PROCESSED_DATASET_PATH
#%% This scripts loands data from LUNA16, randomly slices the images, and stores the result.
# Then it simulates forward project_utilsions of a particular geometry and adds realistic noise of different levels to it.
# For the testing set, the slices that contain nodules are used.

# make dir.
Path(LUNA_PROCESSED_DATASET_PATH).mkdir(parents=True, exist_ok=True)
Path(LUNA_PROCESSED_DATASET_PATH.joinpath("training")).mkdir(
    parents=True, exist_ok=True
)  # Training data
Path(LUNA_PROCESSED_DATASET_PATH.joinpath("validation")).mkdir(
    parents=True, exist_ok=True
)  # Validation data
Path(LUNA_PROCESSED_DATASET_PATH.joinpath("testing")).mkdir(
    parents=True, exist_ok=True
)  # Testing data
Path(LUNA_PROCESSED_DATASET_PATH.joinpath("testing_nodule")).mkdir(
    parents=True, exist_ok=True
)  # Testing data containing nodule slices

from AItomotools.data_loaders.LUNA16.pre_processing import LUNA16

luna_dataset = LUNA16(LUNA_DATASET_PATH, load_metadata=True)
luna_dataset.unit = "normal"
# lets set a seed for reproducibility:
np.random.seed(42)

# More Parameters:
luna_parameter.num_patients = len(luna_dataset.images)
luna_parameter.training_pct = 0.8
luna_parameter.validation_pct = 0.1
luna_parameter.testing_pct = 0.1

luna_parameter.slice_selection_mode = "uniform_skip"
luna_parameter.slice_selection_step = 4
# We are going to simulate 2D slices, so using the entire luna will likely bee too much. Lets generate 6 slices for each LUNA dataset
# and 4 will go ot training, 1 to validation, 1 to testing
data_index = 0
slices = []
luna_parameter.training_patients = []
luna_parameter.validation_patients = []
luna_parameter.testing_patients = []

for i in tqdm(range(len(luna_dataset.images))):
    luna_dataset.load_data(i)

    if luna_parameter.slice_selection_mode == "uniform_skip":
        slice_indices = list(
            range(
                0, luna_dataset.images[i].shape[0], luna_parameter.slice_selection_step
            )
        )
    else:
        raise ValueError("Unimplemented sampling scheme")
    # lets process each of these slices
    for idx, slice in enumerate(slice_indices):
        if i < len(luna_dataset.images) * luna_parameter.training_pct:
            subfolder = "training"
            luna_parameter.training_patients.append(i)
        elif i < len(luna_dataset.images) * (
            luna_parameter.training_pct + luna_parameter.validation_pct
        ):
            subfolder = "validation"
            luna_parameter.validation_patients.append(i)
        else:
            subfolder = "testing"
            luna_parameter.testing_patients.append(i)

        image = luna_dataset.images[i].data[slice]
        image = np.expand_dims(image, 0)
        image = skimage.transform.resize(image, (1, 512, 512))

        # Save clean image and sinogram for supervised training.
        torch.save(
            torch.from_numpy(image),
            LUNA_PROCESSED_DATASET_PATH.joinpath(
                subfolder + "/image_" + f"{i:04}" + "_" + f"{idx:04}" + ".pt"
            ),
        )

    luna_dataset.unload_data(i)  # free memory


#%%  Nodules
# Now, albeit we have build a good typical train/validate/test set, the LUNA dataset has something extra interesting: Lung nodules.
# These are 3D, i.e. they have few slices thickness. So lets extract_utils these slices and put them together
Path(LUNA_PROCESSED_DATASET_PATH.joinpath("testing_nodule")).mkdir(
    parents=True, exist_ok=True
)
Path(LUNA_PROCESSED_DATASET_PATH.joinpath("testing_nodule/metadata")).mkdir(
    parents=True, exist_ok=True
)

luna_parameter.save(LUNA_PROCESSED_DATASET_PATH.joinpath("parameter.json"))

nodule_index = 0
for i in tqdm(range(len(luna_dataset.images))):
    if not luna_dataset.images[i].nodules:  # if it has no nodules, skip
        continue
    luna_dataset.load_data(i)
    image = luna_dataset.images[i]  # not a copy!
    # If you want to do posterior image analysis. e.g. radiomics, the spacing in which the image is at, in mm, is quite important.
    # it is possible that the images where originaly in different spacing (they were, at least in z) and thus knowing exacly the spacing
    # after resampling them is important for the nodule metadata.
    resampled_spacing = (
        image.spacing * np.array([1, *image.shape[1:]]) / np.array([1, 512, 512])
    )

    # First, we need to find where the nodules are. We need a "nodule mask"
    # There are two ways of doing that. First, we can have a "block-label" of where the cube of the nodule lies.
    #                                   Second, we can rely on the basic Otsu thresholding segmentation
    #
    # Lets save both, but you should not believe the Otsu one...

    # Make "block-label"
    image.make_nodule_only_mask("all")
    nodule_slices, mask_slices = image.get_croped_nodule_slices("all")

    # for each nodule
    for nodule_slice, mask_slice in zip(nodule_slices, mask_slices):
        # save spacing of this nodule
        torch.save(
            torch.from_numpy(resampled_spacing),
            LUNA_PROCESSED_DATASET_PATH.joinpath(
                "testing_nodule/metadata/nodule_spacing_" + str(nodule_index) + ".pt"
            ),
        )

        # for every slice in each nodule
        for slice_index, (ns, ms) in enumerate(zip(nodule_slice, mask_slice)):
            ns = np.expand_dims(ns, axis=0)
            ns = skimage.transform.resize(ns, (1, 512, 512))
            ms = np.expand_dims(ms, axis=0)
            ms = skimage.transform.resize(ms, (1, 512, 512)) > 0

            # Save clean image and sinogram for supervised training.
            torch.save(
                torch.from_numpy(ns),
                LUNA_PROCESSED_DATASET_PATH.joinpath(
                    "testing_nodule/image_"
                    + f"{nodule_index:06}"
                    + "_slice_"
                    + f"{slice_index:03}"
                    + ".pt"
                ),
            )
            torch.save(
                torch.from_numpy(ms),
                LUNA_PROCESSED_DATASET_PATH.joinpath(
                    "testing_nodule/mask_"
                    + f"{nodule_index:06}"
                    + "_slice_"
                    + f"{slice_index:03}"
                    + ".pt"
                ),
            )

        nodule_index += 1
    luna_dataset.unload_data(i)
