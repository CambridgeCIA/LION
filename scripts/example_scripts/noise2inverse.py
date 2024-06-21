#%% This example shows how to train Learned Primal dual for full angle, noisy measurements.

#%% Imports

# Standard imports
import matplotlib.pyplot as plt
import pathlib
from skimage.metrics import structural_similarity as ssim
from ts_algorithms import fdk

# Torch imports
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

# Lion imports
from LION.models.CNNs.MS_D import MS_D
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
from LION.optimizers.Noise2Inverse_solver import Noise2Inverse_solver


def my_ssim(x, y):
    x = x.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    return ssim(x, y, data_range=x.max() - x.min())


#%%
# % Chose device:
device = torch.device("cuda:1")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/ab2860/trained_models/test_debbuging/")
final_result_fname = savefolder.joinpath("Noise2Inverse_MSD.pt")
checkpoint_fname = "Noise2Inverse_MSD_check_*.pt"
validation_fname = savefolder.joinpath("Noise2Inverse_MSD_min_val.pt")
#
#%% Define experiment

experiment = ct_experiments.LowDoseCTRecon(dataset="LIDC-IDRI")

#%% Dataset
lidc_dataset = experiment.get_training_dataset()

# smaller dataset for example. Remove this for full dataset
indices = torch.arange(100)
lidc_dataset = data_utils.Subset(lidc_dataset, indices)


#%% Define DataLoader
# Use the same amount of training


batch_size = 10
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=False)
lidc_test = DataLoader(experiment.get_testing_dataset(), batch_size, shuffle=False)

#%% Model
# Default model is already from the paper.
model = MS_D().to(device)

#%% Optimizer
train_param = LIONParameter()

# loss fn
loss_fcn = torch.nn.MSELoss()
train_param.optimiser = "adam"

# optimizer
train_param.epochs = 100
train_param.learning_rate = 1e-4
train_param.betas = (0.9, 0.99)
train_param.loss = "MSELoss"
optimiser = torch.optim.Adam(
    model.parameters(), lr=train_param.learning_rate, betas=train_param.betas
)

#%% Train
# create solver
noise2inverse_parameters = Noise2Inverse_solver.default_parameters()
noise2inverse_parameters.sino_splits = (
    4  # its default anyway, but this is how you can modify it
)
noise2inverse_parameters.base_algo = (
    fdk  # its default anyway, but this is how you can modify it
)
solver = Noise2Inverse_solver(
    model,
    optimiser,
    loss_fcn,
    verbose=True,
    geo=experiment.geo,
    optimizer_params=noise2inverse_parameters,
)

# YOU CAN IGNORE THIS. You can 100% just write your own pytorch training loop.
# LIONSover is just a convinience class that does some stuff for you, no need to use it.

# set data
solver.set_training(lidc_dataloader)
solver.set_testing(lidc_test, my_ssim)

# set checkpointing procedure
solver.set_checkpointing(savefolder, checkpoint_fname, 10, load_checkpoint=False)
# train
solver.train(train_param.epochs)
# delete checkpoints if finished
solver.clean_checkpoints()
# save final result
solver.save_final_results(final_result_fname)

# test

solver.test()

plt.figure()
plt.semilogy(solver.train_loss[1:])
plt.savefig("loss.png")
