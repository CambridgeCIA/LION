import numpy as np
import torch
import tomosipo as ts
import matplotlib.pyplot as plt
import AItomotools.CTtools.ct_utils as ct
import pathlib

#%% Demo on how to create a sinogram from an image and simulate projections, for 2D, using tomosipo and AItomotools
# ===================================================================================================================
# by: Ander Biguri

# If you are here is because you want to do tomography with AI.
# Therefore, to start with, we need to learn how to do tomography.
# This demo teaches how to define a tomographic geometry, an image, and produce sinograms (CT measurements) from it.
# It also shows how to simulate realistic CT noise.
# AItomotools has various backends, but we are using tomosipo in this case.


#%% Create image, process
# Create a phantom containing a small cube. In your code, this will be your data
phantom = np.ones((512, 512)) * -1000  # lets assume its 512^2
phantom[200:250, 200:250] = 300

# As we want a 2D image but tomosipo deals with 3D data, lets make it 3D with a singleton z dimension
phantom = np.expand_dims(phantom, 0)

# CT images can be in different units. The phantom avobe has been defined in Hounsfield Units (HUs)
# CTtools.ct_utils has a series of functions to transfrom from 3 units: Hounsfield Units (HUs), linear attenuation coefficien (mus) or normalized images

phantom_mu = ct.from_HU_to_mu(phantom)
phantom_normal = ct.from_HU_to_normal(phantom)

# We can also breakdown the HU image into a segmented tissue ID image. Check function definition for documentatio of tissue idx.
phantom_tissues = ct.from_HU_to_material_id(phantom)

# Lets use the mu, as this is the real measurement (mu is what would measure, as HUs are a post-processing change of units that doctors like)
phantom = phantom_mu

#%% Create operator

# We need to define a "virtual CT machine" that simualtes the data. This is, in mathematical terms, an "operator".

# Define volume geometry. Shape is the shape in elements, and size is in real life units (e.g. mm)
# Note: we make Z "thick" just because we are going to simulate 2D. for 3D be more careful with this value.
vg = ts.volume(shape=(1, *phantom.shape[1:]), size=(5, 300, 300))
# Define acquisition geometry. We want fan beam, so lets make a "cone" beam and make it 2D. We also need sod and sdd, so we set them up to something medically reasonable.
pg = ts.cone(
    angles=360, shape=(1, 900), size=(1, 900), src_orig_dist=575, src_det_dist=1050
)
# A is now an operator.
A = ts.operator(vg, pg)

#%% Using Geometry class.
# The above example shows how to define tomosipo operators, but for AItomotools, you should be using the Parameter class, in particular the CT Geometry class.
import AItomotools.CTtools.ct_geometry as ctgeo

# Create empty Geometry
geo = ctgeo.Geometry()
# Fill it with default values
geo.default_geo()
# Print the geo (these are the values you can ser)
print(geo)
# Dave the geo in JSON
geo.save(pathlib.Path("geo.json"))
geo.load(pathlib.Path("geo.json"))

#%% CPU or GPU?

# The only thing needed to make the tomosipo operator work in GPU instead of GPU is just make it a torch tensor.
dev = torch.device("cuda")
phantom = torch.from_numpy(phantom).to(dev)


#%% Create sinograms, simulate noise.

# ct_utils.py contain functions to for realistic CT simulations.
# In essence, this is the same as using the operator we defined above.

# We can use the following function
sino = ct.forward_projection_fan(phantom, geo, backend="tomosipo")
# But given we already defined an operator, we can just do (its the same):
sino = A(phantom)

# For noise simulation, a good approximation of CT noise is to add Poisson noise to the non-log transformed sinograms,
# with some gaussian noise to account for the detector electronic noise and detector crosstalk.
# A typical CT scan in a hospital will have I0=10000 photon counts in air. I0=1000 will produce an severely noisy image.
# You should be cool with not touching the rest of the parameters.
sino_noisy = ct.sinogram_add_noise(
    sino, I0=10000, sigma=5, crosstalk=0.05, flat_field=None, dark_field=None
)

#%% Plot sinograms
sino = sino.detach().cpu().numpy()
sino_noisy = sino_noisy.detach().cpu().numpy()

plt.figure()
plt.subplot(121)
plt.imshow(sino[0].T)
plt.colorbar()
plt.subplot(122)
plt.imshow(sino_noisy[0].T)
plt.colorbar()
plt.savefig("Sino.png")
