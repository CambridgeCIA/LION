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
final_result_fname = savefolder.joinpath("FBPUNet_ExpNoise.pt")
checkpoint_fname = "FBPUNet_ExpNoise_check_*.pt"  # if you use LION checkpointing, remember to have wildcard (*) in the filename
validation_fname = savefolder.joinpath("FBPUNet_ExpNoise_min_val.pt")

#%% 2 - Define experiment
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# These are all the experiments we need to run for the noise paper

# Experimental noisy dataset
experiment = ct_denoising.ExperimentalNoiseDenoisingRecon()

# Simulated noisy dataset
#experiment = ct_denoising.ArtificialNoiseDenoisingRecon()


#%% 3 - Obtaining Datasets from experiments
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
training_data = experiment.get_training_dataset()
validation_data = experiment.get_validation_dataset()
testing_data = experiment.get_testing_dataset()

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
from LION.models.post_processing.FBPUNet import FBPUNet
model = FBPUNet(experiment.geo).to(device)

#from LION.models.post_processing.FBPMSDNet import FBPMS_D
#model = FBPMS_D(experiment.geo).to(device)

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
plt.savefig("loss_FBPUNet_ExpNoise.png")

# Now your savefolder should have the min validation and the final result.

#%% 9 - TEST
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# if verbose it will print mean+std
result_vals_nparray = solver.test()
