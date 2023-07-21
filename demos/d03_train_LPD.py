# =============================================================================
# This file is part of AItomotools library
# License : BSD-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================


import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from AItomotools.utils.parameter import Parameter
from AItomotools.data_loaders.luna16_dataset import Luna16Dataset
from AItomotools.models.LPD import LPD
import AItomotools.CTtools.ct_geometry as ctgeo
import AItomotools.CTtools.ct_utils as ct

#%% Demo to show how to load, define, train, a model.
# Use this as a template to train your networks and design new ones

# Lets set the GPU.
device = torch.device("cuda:1")

#%% Creating our LPD model

# first, lets define the CT geometry we are simulating. More info in demo 1.
geom = ctgeo.Geometry()
geom.default_geo()

# Low, lets create the mode. We can get the default parameters by inspecting the default_parameters() method.
default_params = LPD.default_parameters()
print(default_params)

# You can initialize with default parameters eithre by giving them as an input or leaving the parameter input empty.
# The following 3 lines are equivalent

model = LPD(geometry_parameters=geom, model_parameters=default_params)
model = LPD(geometry_parameters=geom)
model = LPD(geom)
# NOTE: LDP initializataion for param.step_size=None is computionally expensive, comment these lines out if you are testing


# Lets change a parameter just to check:
default_params.step_positive = True  # A.Biguri: my experience is that this is better
model = LPD(geometry_parameters=geom, model_parameters=default_params)

# move it to GPU
model.to(device)
#%% Lets define the dataset.
# We are using LUNA16 for this case.
mode = "training"
luna16_training = Luna16Dataset(device, mode, geom)
# We can pass what we want the sinograms to go trhough at loading, e.g. adding noise. Easiest way is with a lamnda
sino_fun = lambda sino: ct.sinogram_add_noise(sino, I0=1000)
luna16_training.set_sinogram_transform(sino_fun)
# And the training parameters of the dataset:
batch_size = 2
luna16_dataloader = DataLoader(luna16_training, batch_size, shuffle=False)

#%% Training parameters
train_param = Parameter()
train_param.epochs = 10
train_param.learning_rate = 1e-4

train_param.loss = "MSELoss"
loss_fcn = torch.nn.MSELoss()

train_param.optimiser = "adam"
optimiser = torch.optim.Adam(model.parameters(), lr=train_param.learning_rate)
#%% simple training loop

for epoch in range(train_param.epochs):
    epoch_loss = 0
    for index, (sinogram, target_reconstruction) in tqdm(enumerate(luna16_dataloader)):
        optimiser.zero_grad()
        reconstruction = model(sinogram)
        loss = loss_fcn(reconstruction, target_reconstruction)
        loss.backward()
        epoch_loss += loss.item()
        optimiser.step()
        if index % 100 == 0:
            plt.subplot(121)
            plt.imshow(reconstruction[0, 0].float().detach().cpu())
            plt.clim(0, 1)
            plt.subplot(122)
            plt.imshow(target_reconstruction[0, 0].float().detach().cpu())
            plt.clim(0, 1)
            plt.savefig("current_reconstruction.jpg")
            plt.clf()

    model.save(
        "test.pt", training=train_param, dataset=luna16_training.pre_processing_params
    )
    model.save_checkpoint(
        "test_check.pt",
        epoch,
        loss,
        optimiser,
        train_param,
        dataset=luna16_training.pre_processing_params,
    )
