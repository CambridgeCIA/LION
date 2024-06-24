# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Max Kiss, Ander Biguri
# =============================================================================

#%% 0 - Imports
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Standard imports
import matplotlib.pyplot as plt
import pathlib
from skimage.metrics import structural_similarity as ssim

# Torch imports
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

# LION imports
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_learned_denoising_experiments as ct_denoising

# Just a temporary SSIM that takes torch tensors (will be added to LION at some point)
def my_ssim(x: torch.tensor, y: torch.tensor):
    x = x.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    return ssim(x, y, data_range=x.max() - x.min())


#%% 1 - Settings
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Device
device = torch.device("cuda:0")
torch.cuda.set_device(device)

# Define your data paths
savefolder = pathlib.Path("/export/scratch3/mbk/LION/noise_paper/trained_models/testing_debugging/")

# Filenames and patterns
final_result_fname = savefolder.joinpath("UNet_ArtNoise.pt")
checkpoint_fname = "UNet_ArtNoise_check_*.pt"  # if you use LION checkpointing, remember to have wildcard (*) in the filename
validation_fname = savefolder.joinpath("UNet_ArtNoise_min_val.pt")

#%% 2 - Define experiment
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# These are all the experiments we need to run for the noise paper

# Experimental noisy dataset
#experiment = ct_denoising.ExperimentalNoiseDenoising()

# Simulated noisy dataset
experiment = ct_denoising.ArtificialNoiseDenoising()


#%% 3 - Obtaining Datasets from experiments
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
training_data = experiment.get_training_dataset()
validation_data = experiment.get_validation_dataset()
testing_data = experiment.get_testing_dataset()

# smaller dataset for testing if this template works for you.
##############################################################
# REMOVE THIS CHUNK IN THE FINAL VERSION

#indices = torch.arange(50)
#training_data = data_utils.Subset(training_data, indices)
#validation_data = data_utils.Subset(validation_data, indices)
#testing_data = data_utils.Subset(testing_data, indices)

# REMOVE THIS CHUNK IN THE FINAL VERSION
##############################################################

#%% 4 - Define Data Loader
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is standard pytorch, no LION here.

batch_size = 1
training_dataloader = DataLoader(training_data, batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size, shuffle=False)
testing_dataloader = DataLoader(testing_data, batch_size, shuffle=False)

#%% 5 - Load Model
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We want to train MS_D networks but also U-Nets - comment out the one you do not need
from LION.models.CNNs.UNets.UNet_3 import UNet

# If you are happy with the default parameters, you can just do
model = UNet(experiment.geo).to(device)
# Remember to use `experiment.geo` as an input, so the model knows the operator
default_parameters = UNet.default_parameters()
model = UNet(experiment.geo, default_parameters).to(device)

#%% 6 - Define Loss and Optimizer
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is standard pytorch, no LION here.

# loss fn
loss_fcn = torch.nn.MSELoss()
optimiser = "adam"

# optimizer
epochs = 100
learning_rate = 1e-4
betas = (0.9, 0.99)
loss = "MSELoss"
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas)

#%% 7 - Define Solver
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# if your model is trained with a supported solver (for now only supervisedSolver and Noise2Inverse_solver), you can use the following code.
# If you have a custom training loop, you can just use that, pure pytorch is supported.
# Note: LIONmodel comes with a few quite useful functions, so you might want to use it even if you have a custom training loop. e.g. model.save_checkpoint() etc.
# Read demo d04_LION_models.py for more info.

# You know how to write pytorch loops, so let me show you how to use LION for training.

from LION.optimizers.supervised_learning import supervisedSolver
from LION.optimizers.Noise2Inverse_solver import Noise2Inverse_solver

# create solver
solver = supervisedSolver(model, optimiser, loss_fcn, verbose=True)

# YOU CAN IGNORE THIS. You can 100% just write your own pytorch training loop.
# LIONSover is just a convenience class that does some stuff for you, no need to use it.

# set data
solver.set_training(training_dataloader)
# Set validation. If non defined by user, it uses the loss function to validate.
# If this is set, it will save the model with the lowest validation loss automatically, given the validation_fname
solver.set_validation(
    validation_dataloader, validation_freq=10, validation_fname=validation_fname
)
# Set testing. Second input has to be a function that accepts torch tensors and returns a scalar
solver.set_testing(testing_dataloader, my_ssim)
# set checkpointing procedure. It will automatically checkpoint your models.
# If load_checkpoint=True, it will load the last checkpoint available in disk (useful for partial training loops in HPC)
solver.set_checkpointing(
    savefolder,
    checkpoint_fname=checkpoint_fname,
    checkpoint_freq=10,
    load_checkpoint=False,
)


#%% 8 - TRAIN!
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
solver.train(epochs)

# delete checkpoints if finished
#solver.clean_checkpoints()

# save final result
solver.save_final_results(final_result_fname)

# Save the training.
plt.figure()
plt.semilogy(solver.train_loss)
plt.savefig("loss_UNet_ArtNoise.png")

# Now your savefolder should have the min validation and the final result.

#%% 9 - TEST
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# if verbose it will print mean+std
result_vals_nparray = solver.test()
