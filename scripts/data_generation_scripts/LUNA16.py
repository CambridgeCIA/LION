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

#%% This scripts loands data from LUNA16, randomly slices the images, and stores the result.
# Then it simulates forward projections of a particular geometry and adds realistic noise of different levels to it.
# For the testing set, the slices that contain nodules are used. 

# make dir.
Path(path_projections_luna).mkdir(parents=True, exist_ok=True)
Path(path_projections_luna+"training").mkdir(parents=True, exist_ok=True) #Training data
Path(path_projections_luna+"validation").mkdir(parents=True, exist_ok=True) # Validation data
Path(path_projections_luna+"testing").mkdir(parents=True, exist_ok=True) # Testing data
Path(path_projections_luna+"testing_nodule").mkdir(parents=True, exist_ok=True) # Testing data containing nodule slices

from AItomotools.data_loaders.LUNA16 import LUNA16
luna_dataset=LUNA16(path_luna,load_metadata=True)
luna_dataset.unit="normal" 
# lets set a seed for reproducibility:
np.random.seed(42)

# Define operator (fan beam)
# TODO save metadata for this.
vg = ts.volume(shape=(1,512,512), size=(5, 300, 300))
pg = ts.cone(angles=360, shape=(1, 900), size=(1, 900), src_orig_dist=575, src_det_dist=1050)
A = ts.operator(vg, pg)

# We want to simulate different levels of noise. Noise in CT is best defined by the number of counts in the detector in air, so lower==more noise. 
# I0=10000 ~ medical scanner
noise_level=[10000,5000,3500,2000,1000]
for noise in noise_level:
    Path(path_projections_luna+"training/"+str(noise)).mkdir(parents=True, exist_ok=True) #Training data
    Path(path_projections_luna+"validation/"+str(noise)).mkdir(parents=True, exist_ok=True) # Validation data
    Path(path_projections_luna+"testing/"+str(noise)).mkdir(parents=True, exist_ok=True) # Testing data
    Path(path_projections_luna+"testing_nodule/"+str(noise)).mkdir(parents=True, exist_ok=True) # Testing data containing nodule slices

Path(path_projections_luna+"training/"+"clean").mkdir(parents=True, exist_ok=True) #Training data
Path(path_projections_luna+"validation/"+"clean").mkdir(parents=True, exist_ok=True) #Training data
Path(path_projections_luna+"testing/"+"clean").mkdir(parents=True, exist_ok=True) #Training data
Path(path_projections_luna+"testing_nodule/"+"clean").mkdir(parents=True, exist_ok=True) #Training data

# We are going to simulate 2D slices, so using the entire luna will likely bee too much. Lets generate 6 slices for each LUNA dataset
# and 4 will go ot training, 1 to validation, 1 to testing
data_index=0
for i in tqdm(range(len(luna_dataset.images))):
    luna_dataset.load_data(i)
    slice_indices=np.random.randint(0,luna_dataset.images[i].shape[0],size=6)
    # lets process each of these slices
    for idx, slice in enumerate(slice_indices):
        if idx==0:
            subfolder="testing"
        elif idx==1:
            subfolder="validation"
        else:
            subfolder="training"

        image=luna_dataset.images[i].data[slice]
        image=np.expand_dims(image, 0)
        image=skimage.transform.resize(image,(1,512,512))
        sino=A(image) # forward operator

        # Save clean image and sinogram for supervised training.
        np.save(path_projections_luna+subfolder+"/clean/image_"+f'{data_index:06}'+".npy",image)
        np.save(path_projections_luna+subfolder+"/clean/sino_"+f'{data_index:06}'+".npy",sino)

        # Simulate all noise levels, and save. 
        for noise in noise_level:
            sino_noisy=ct.sinogram_add_noise(sino, I0=noise)
            np.save(path_projections_luna+subfolder+"/"+str(noise)+"/sino_"+f'{data_index:06}'+".npy",sino_noisy)
        data_index+=1
    luna_dataset.unload_data(i) #free memory


#%%  Nodules
# Now, albeit we have build a good typical train/validate/test set, the LUNA dataset has something extra interesting: Lung nodules. 
# These are 3D, i.e. they have few slices thickness. So lets extract these slices and put them together

Path(path_projections_luna+"testing_nodule/"+"metadata").mkdir(parents=True, exist_ok=True) 

nodule_index=0
for i in tqdm(range(len(luna_dataset.images))):
    if not luna_dataset.images[i].nodules: #if it has no nodules, skip
        continue
    luna_dataset.load_data(i)
    image=luna_dataset.images[i] # not a copy!
    # If you want to do posterior image analysis. e.g. radiomics, the spacing in which the image is at, in mm, is quite important. 
    # it is possible that the images where originaly in different spacing (they were, at least in z) and thus knowing exacly the spacing
    # after resampling them is important for the nodule metadata. 
    resampled_spacing=image.spacing*np.array([1, *image.shape[1:]])/np.array([1, 512, 512])

    # First, we need to find where the nodules are. We need a "nodule mask"
    # There are two ways of doing that. First, we can have a "block-label" of where the cube of the nodule lies. 
    #                                   Second, we can rely on the basic Otsu thresholding segmentation 
    #
    # Lets save both, but you should not believe the Otsu one...

    # Make "block-label"
    image.make_nodule_only_mask("all")
    nodule_slices , mask_slices = image.get_croped_nodule_slices("all")
    
    # for each nodule
    for nodule_slice, mask_slice in zip(nodule_slices, mask_slices):
        # save spacing of this nodule
        np.save(path_projections_luna+'/testing_nodule/metadata/nodule_spacing_'+str(nodule_index)+".npy",resampled_spacing)

        # for every slice in each nodule
        for slice_index, (ns, ms) in enumerate(zip(nodule_slice,mask_slice)):
            ns=np.expand_dims(ns,axis=0)
            ns=skimage.transform.resize(ns,(1,512,512))
            ms=np.expand_dims(ms,axis=0)
            ms=skimage.transform.resize(ms,(1,512,512))>0
            sino = A(ns)

             # Save clean image and sinogram for supervised training.
            np.save(path_projections_luna+"/testing_nodule/clean/image_"+f'{nodule_index:06}'+"_slice_"+f'{slice_index:03}'+".npy",ns)
            np.save(path_projections_luna+"/testing_nodule/clean/mask_"+f'{nodule_index:06}'+"_slice_"+f'{slice_index:03}'+".npy",ms)

            np.save(path_projections_luna+"/testing_nodule/clean/sino_"+f'{nodule_index:06}'+"_slice_"+f'{slice_index:03}'+".npy",sino)

            # Simulate all noise levels, and save. 
            for noise in noise_level:
                sino_noisy=ct.sinogram_add_noise(sino, I0=noise)
                np.save(path_projections_luna+"/testing_nodule/"+str(noise)+"/sino_"+f'{nodule_index:06}'+"_slice_"+f'{slice_index:03}'+".npy",sino_noisy)
        nodule_index+=1
    luna_dataset.unload_data(i)