# =============================================================================
# This file is part of AItomotools library
# License : BSD-3
#
# Author  : Ander Biguri
# Modifications: -
# =============================================================================


#%% Learned Primal Dual
#
# This script attempts to replicate the experiments descrived in
#
# Adler, Jonas, and Ozan Öktem.
# "Learned primal-dual reconstruction."
# IEEE transactions on medical imaging
# 37.6 (2018): 1322-1332.
#
# Any difference to the paper will be explicitly stated.
#


#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pathlib
import AItomotools.CTtools.ct_geometry as ctgeo
import AItomotools.CTtools.ct_utils as ct
from AItomotools.data_loaders.luna16_dataset import Luna16Dataset
from AItomotools.models.LPD import LPD
from AItomotools.utils.parameter import Parameter


#%% Chose device:
device = torch.device("cuda:2")
savefolder = pathlib.Path("/store/DAMTP/ab2860/wip_models")

#%% EXPERIMENT 2: Patient data
##################################################
#%% Create CT geometry
geom = ctgeo.Geometry()
geom.default_geo()

geom.dsd = 1000
geom.dso = 500
geom.image_shape = [1, 512, 512]
geom.image_size = [300 / 512, 300, 300]  # Unknown from the article, assumed
geom.detector_shape = [1, 1000]
geom.detector_size = [1, 1000]  # Unknown from the article, assumed
geom.angles = np.linspace(0, 2 * np.pi, 1000, endpoint=False)

#%% Dataset
#
# This is different than the original paper.
# The article uses:
#
# "reconstruction of simulated data from human abdomen CT scans as provided by
# Mayo Clinic for the AAPM Low Dose CT Grand Challenge
# [24]. The data includes full dose CT scans from 10 patients,
# of which we used 9 for training and 1 for evaluation. We used
# the 3 mm slice thickness reconstructions, resulting in 2168
# training images, each 512 × 512 pixel in size."
#
# In here, we will be instead using the LUNA dataset.
# Same amount of slices for training, except they come from a wider range of patients.


luna16_training = Luna16Dataset(device, mode="training", geo=geom)
# They use I0=1000, sigma=5, cross_talk=0
sino_fun = lambda sino: ct.sinogram_add_noise(sino, I0=1000, sigma=5, cross_talk=0)
luna16_training.set_sinogram_transform(sino_fun)
# Use the same amount of training
luna16_subset = torch.utils.data.Subset(luna16_training, range(2168))

batch_size = 1
luna16_dataloader = DataLoader(luna16_subset, batch_size, shuffle=True)


#%% Model
# Default model is already from the paper.
default_parameters = LPD.default_parameters()
# This makes the LPD calculate the step size for the backprojection, which in my experience results in much much better pefromace
# as its all in the correct scale.
default_parameters.step_size = None
model = LPD(geom, default_parameters).to(device)

# Weigths initialized
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(
            m.weight, gain=0.5
        )  # gain=0.5 is necesary for appropiate scale, but unkown why.
        if m.bias is not None:
            m.bias.data.fill_(0.0)


model.apply(init_weights)

#%% Optimizer
train_param = Parameter()

# loss fn
loss_fcn = torch.nn.MSELoss()
train_param.optimiser = "adam"

# optimizer
train_param.epochs = 462  # ~(10e5/2168)
train_param.learning_rate = 1e-3
train_param.betas = (0.9, 0.99)
train_param.loss = "MSELoss"
optimiser = torch.optim.Adam(
    model.parameters(), lr=train_param.learning_rate, betas=train_param.betas
)

# learning parameter update
steps = len(luna16_dataloader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)

model.train()

loss_val = np.zeros((train_param.epochs * len(luna16_dataloader)))
#%% Do training!
total_loss = np.zeros(train_param.epochs)
from ts_algorithms import fdk

for epoch in range(train_param.epochs):
    epoch_loss = []
    for index, (sinogram, target_reconstruction) in tqdm(enumerate(luna16_dataloader)):

        optimiser.zero_grad()
        reconstruction = model(sinogram)
        loss = loss_fcn(reconstruction, target_reconstruction)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        epoch_loss.append(loss.item())

        optimiser.step()
        scheduler.step()

        if index % 100 == 0:
            plt.subplot(121)
            plt.imshow(reconstruction[0, 0].float().detach().cpu())
            plt.clim(0, 1)
            plt.colorbar()
            plt.subplot(122)
            plt.imshow(target_reconstruction[0, 0].float().detach().cpu())
            plt.clim(0, 1)
            plt.savefig("current_reconstruction.png")
            plt.clf()

    # restart scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)

    if epoch % 50:
        model.save_checkpoint(
            savefolder.joinpath(f"LPD_check_{epoch}.pt"),
            epoch,
            loss,
            optimiser,
            train_param,
            dataset=luna16_training.pre_processing_params,
        )

    # Plot stuff
    total_loss[epoch] = sum(epoch_loss) / len(epoch_loss)
    plt.plot(
        np.linspace(1, train_param.epochs, train_param.epochs).astype(int), total_loss
    )
    plt.savefig("Loss.png")
    plt.clf()
model.save(savefolder.joinpath("LPD_as_paper.pt"))
