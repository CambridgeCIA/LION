import numpy as np
import torch
import tomosipo as ts
from CTtools.ct_utils import *
#%% Demo on how to create a sinogram from an image and simulate projections, for 2D, using tomosipo and AItomotools
#===================================================================================================================
# by: Ander Biguri


#%% Create image, process
# Create a phantom containing a small cube. In your code, this will be your data
phantom = np.ones((512,512))*-1000  #lets assume its 512^2
phantom[200:250, 200:250, 200:250] = 300

# As we want a 2D image but tomosipo deals with 3D data, lets make it 3D with a singleton z dimension
phantom = np.expand_dims(phantom, 0)

# CT images can be in different units. The phantom avobe has been defined in Hounsfield Units (HUs)
# CTtools.ct_utils has a series of functions to transfrom from 3 units: Hounsfield Units (HUs), linear attenuation coefficien (mus) or normalized images 

phantom_mu=from_HU_to_mu(phantom)
phantom_normal=from_HU_to_normal(phantom)

# We can also breakdown the HU image into a segmented tissue ID image. Check function for documentatio of tissue idx.
phantom_tissues=from_HU_to_material_id(phantom)

# Lets use the mu, as this is the real measurement

phantom=phantom_mu

#%% Create operator 

# Define volume geometry. Shape is the shape in elements, and size is in real life units (e.g. mm)
# Note: we make Z "thick" just because we are going to simulate 2D. for 3D be more careful with this value.
vg = ts.volume(shape=(1,*phantom.shape[1:]), size=(5, 300, 300))
# Define acquisition geometry. We want fan beam, so lets make a "cone" beam and make it 2D. We also need sod and sdd, so we set them up to something medically reasonable.
pg = ts.cone(angles=360, shape=(1, 900), size=(1, 900), src_orig_dist=575, src_det_dist=1050)
# A is now an operator. 
A = ts.operator(vg, pg)

#%% Create sinograms, simulate noise. 

# ct_utils.py contain functions to fo realistic CT simulations. 

# in its most simple form, a fan-beam CT acquisition is defined by:
# image size in mm
# detector size in mm
# detector size in pixles
# distance source object
# distance source detector
# number of angles

# We can use the following function
sino=forward_projection_fan(phantom,size=(5,300,300),sino_shape=(1,900),sino_size=(1,900),DSD=1050,DSO=575,backend="tomosipo",angles=360)
# But given we already defined an operator, we can just do:
sino=A(phantom)

# For noise simulation, a good approximation of CT noise is to add Poisson noise to the non-log transformed sinograms,
# with some gaussian noise to account for the detector electronic noise and detector crosstalk. 
sino_noisy=sinogram_add_noise(proj, I0=1000, sigma=5,crosstalk=0.05,flat_field=None,dark_field=None)