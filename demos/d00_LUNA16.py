import numpy as np
import torch
import tomosipo as ts
import matplotlib.pyplot as plt

#%% Demo on how to load and manipulate LUNA16
# ===================================================================================================================
# by: Ander Biguri

# AItomotools comes with data loaders for common datasets, e.g. LUNA16.
# If you only want to use the datasets, this demo is not useful to you. If you are at CIA group at unviersity of Camrbidge, the data
# should have been processed for you adn available in the relevant folders (LUNA_DATASET_PATH). Otherwise, you should go to
# AItomotools/scripts/data_generation_scripts/LUNA16.py, which will generate an instance of the dataset in pytorch tensors that are easy to load, and reproducible.

# However, you may have a unique model, a case where a particular instance of the dataset is required, that the "standard preprocessing" that
# AItomotools does not cover. In that case, you may want to learn about how to use the methods in the LUNA class to process the data, which is
# the purpose of this demo.

# TL;DR: This demo is only useful if you want to generate your own isntance of LUNA that is not the standard.

# ===================================================================================================================

#%% Data paths
# To avoid unnecesary repetition of data, we have much of the data already stored and processed in shared folders.
from AItomotools.utils.paths import LUNA_DATASET_PATH

# Lets assume here that you need a new instance of this dataset, or to process it for the first time.
# However, assume that what you need is likely already been processed, and you should use that dataset.

#%% Lets see how to use the LUNA16 data loader.
from AItomotools.data_loaders.LUNA16 import LUNA16

# The data loader will do all the job for you, but its expecting a certain format. Simply:
# .
# └── LUNA16
#     annotations.csv
#     ├── seg-lungs-LUNA16
#     ├── subset0
#     ├── subset1
#     ├── subset2
#     ├── subset3
#     ├── subset4
#     ├── subset5
#     ├── subset6
#     ├── subset7
#     ├── subset8
#     └── subset9
#

# To dowload them, run ./AItomotools/data_loaders/LUNA16/download_luna16.py

# Create the data Loader, this is empty.
luna_dataset = LUNA16(LUNA_DATASET_PATH, load_metadata=False, verbose=True)
# You can do this with the constructor, but I set it to false.
# This will load the metadata (image names, nodules etc), but will not load actual data (for memory reasons)
luna_dataset.load_metadata()
# If you want to load the actual data, you need to actually load it manually. This will load metadata for every image
# and convert it to the right unit. You can do that with
image_to_load = 0
luna_dataset.unit = "HU"  # default is HUs, can be normal or mu
luna_dataset.load_data(image_to_load)
# You can also load a range of them
luna_dataset.load_data(list(range(11, 17)))
# For your own safety, this does not have a "load all" utility, albeit you can check len(luna_dataset.images)
# to know the number of images available. But this is likely more memory than the RAM you have available

# You can also unload the data from memory.
luna_dataset.unload_data(image_to_load)
luna_dataset.unload_data(list(range(11, 17)))
luna_dataset.unload_data()  # just unload all

# We can also apply transforms to all loaded images (plus metadata)
luna_dataset.from_HU_to_mu()
# luna_dataset.from_HU_to_normal()

#%% Manipulating the data inside.

# The LUNA dataset is 3D CT images, and its likely that you (yes you!) are doing 2D recon. Therefore you may want to load
# or use just some of the slices. In fact if you were to use all of the slices it will likely be too much data.
luna_dataset.load_data(0)
data = luna_dataset.images[0]  # this is a 3D np.array
# It is however likely that you want an images of the same size, such are CT scanners.
# Given a desired resulution, you can resample the image as
original_size = luna_dataset.images[0].shape
luna_dataset.images[0].resample([original_size[0], 256, 256])

print("")
print("Original size: " + str(original_size))
print("Resampled size: " + str(luna_dataset.images[0].shape))

# You can also crop the 3D image in a set of desired slices. This also modifies the relevant metadata.
luna_dataset.images[0].crop_z(50)

# The LUNA datasets comes with annotations.csv, that gives locations and radious of lung nodules in the images.
# You may want to keep only the slices with a lung nodule, and ignore the rest.
# TODO whats wrong with the first one?
for image in luna_dataset.images[10:]:
    if image.nodules:
        # Load the image
        image.load_data()
        # Make binary masks (Segmentations) of the nodules (Otsu trhesholding, needs to improve)
        # TODO does "seg-lungs-LUNA16" contain nodules?
        image.make_binary_mask_nodule(
            "all"
        )  # you can also select individual nodules, thre may be more thhan one
        # TODO make the data change, as "crop_z" does
        croped_slices_list, nodule_mask_list = image.get_croped_nodule_slices("all")
        print("")
        print("Number of nodules found: " + str(len(image.nodules)))
        print(
            "Slices with the first lung nodule: " + str(croped_slices_list[0].shape[0])
        )
        print("Metadata of first nodule")
        print(image.nodules[0])

        plt.figure()
        plt.subplot(121)
        plt.imshow(croped_slices_list[0][3])
        plt.clim(0, 2)
        plt.subplot(122)
        plt.imshow(nodule_mask_list[0][3])
        plt.clim(0, 2)
        plt.savefig("nodule.png")

        break  # lets just do one


# TODO show how to input this into ct_simulation.py
