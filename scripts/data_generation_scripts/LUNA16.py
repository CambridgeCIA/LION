import numpy as np
import torch
import tomosipo as ts
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import skimage
import AItomotools.CTtools.ct_geometry as ct

## This scripts generates the data available at:
from AItomotools.utils.paths import LUNA_PROCESSED_DATASET_PATH, LUNA_DATASET_PATH

print(LUNA_PROCESSED_DATASET_PATH)

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

from AItomotools.data_loaders.LUNA16 import LUNA16

luna_dataset = LUNA16(LUNA_DATASET_PATH, load_metadata=True)
luna_dataset.unit = "normal"
# lets set a seed for reproducibility:
np.random.seed(42)

geo = ct.Geometry()
geo.default_geo()
geo.save(LUNA_PROCESSED_DATASET_PATH.joinpath("geometry.json"))
# Define operator (fan beam)
# TODO save metadata for this.
vg = ts.volume(shape=geo.image.shape, size=geo.image_size)
pg = ts.cone(
    angles=360,
    shape=geo.detector_shape,
    size=geo.detector_size,
    src_orig_dist=geo.dso,
    src_det_dist=geo.dsd,
)
A = ts.operator(vg, pg)

# We are going to simulate 2D slices, so using the entire luna will likely bee too much. Lets generate 6 slices for each LUNA dataset
# and 4 will go ot training, 1 to validation, 1 to testing
data_index = 0
for i in tqdm(range(len(luna_dataset.images))):
    luna_dataset.load_data(i)
    slice_indices = np.random.randint(0, luna_dataset.images[i].shape[0], size=6)
    # lets process each of these slices
    for idx, slice in enumerate(slice_indices):
        if idx == 0:
            subfolder = "testing"
        elif idx == 1:
            subfolder = "validation"
        else:
            subfolder = "training"

        image = luna_dataset.images[i].data[slice]
        image = np.expand_dims(image, 0)
        image = skimage.transform.resize(image, (1, 512, 512))
        sino = A(image)  # forward operator

        # Save clean image and sinogram for supervised training.
        torch.save(
            torch.from_numpy(image),
            LUNA_PROCESSED_DATASET_PATH.joinpath(
                subfolder + "/image_" + f"{data_index:06}" + ".pt"
            ),
        )
        torch.save(
            torch.from_numpy(sino),
            LUNA_PROCESSED_DATASET_PATH.joinpath(
                subfolder + "/sino_" + f"{data_index:06}" + ".pt"
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
            sino = A(ns)

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

            torch.save(
                torch.from_numpy(sino),
                LUNA_PROCESSED_DATASET_PATH.joinpath(
                    "testing_nodule/sino_"
                    + f"{nodule_index:06}"
                    + "_slice_"
                    + f"{slice_index:03}"
                    + ".pt"
                ),
            )

        nodule_index += 1
    luna_dataset.unload_data(i)
