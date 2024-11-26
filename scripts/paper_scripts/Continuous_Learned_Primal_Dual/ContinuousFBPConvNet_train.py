# %% This example shows how to train ContinuousFBPConvNet for full angle, noisy measurements.


# %% Imports
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pathlib
import LION.CTtools.ct_geometry as ctgeo
import LION.CTtools.ct_utils as ct
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
from LION.models.post_processing.cFBPConvNet import cFBPConvNet

from LION.utils.parameter import LIONParameter
from ts_algorithms import fdk

import LION.experiments.ct_experiments as ct_experiments


#%% README

# This script trains the Continuous FBP ConvNet model for low dose CT reconstruction.
# This is part of the Continous Learned Primal Dual paper.
# cFBPConvNet never got sufficiently good results in this paper, so it was not included in the final version.


#%% Input parser

parser = argparse.ArgumentParser()
parser.add_argument("--geometry", type=str)
parser.add_argument("--dose", type=str)

parser.add_argument("--lr", type=float, default=1e-4)
args = parser.parse_args()

# %% Define experiment
if args.geometry == "full":
    if args.dose == "low":
        experiment = ct_experiments.LowDoseCTRecon()
    elif args.dose == "extreme_low":
        experiment = ct_experiments.ExtremeLowDoseCTRecon()
    else:
        raise ValueError("Dose not recognised")
elif args.geometry == "limited_angle":
    if args.dose == "low":
        experiment = ct_experiments.LimitedAngleLowDoseCTRecon()
    elif args.dose == "extreme_low":
        experiment = ct_experiments.LimitedAngleExtremeLowDoseCTRecon()
    else:
        raise ValueError("Dose not recognised")
elif args.geometry == "sparse_angle":
    if args.dose == "low":
        experiment = ct_experiments.SparseAngleLowDoseCTRecon()
    elif args.dose == "extreme_low":
        experiment = ct_experiments.SparseAngleExtremeLowDoseCTRecon()
    else:
        raise ValueError("Dose not recognised")
else:
    raise ValueError("Geometry not recognised")


# Now the experiment is set.


# %% Define required filepaths
# % Chose device:
device = torch.device("cuda:0")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/your/working/directory")
model_name = "CFBPConvNet"

final_result_fname = savefolder.joinpath(
    f"{model_name}_final_iter_{args.dose}_{args.geometry}.pt"
)
checkpoint_fname = savefolder.joinpath(
    f"{model_name}_check_{args.dose}_{args.geometry}_*.pt"
)
validation_fname = savefolder.joinpath(
    f"{model_name}_min_val_{args.dose}_{args.geometry}.pt"
)


# %% Get the  Datasets and DataLoader
training_dataset = experiment.get_training_dataset()
validation_dataset = experiment.get_validation_dataset()
testing_dataset = experiment.get_testing_dataset()

batch_size = 2
training_dataloader = DataLoader(training_dataset, batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size, shuffle=True)
testing_dataloader = DataLoader(testing_dataset, 1, shuffle=False)
# %% Define Model
# Default model is already from the paper.
model_params = cFBPConvNet.default_parameters()
model_params.adjoint = True
model_params.tol = 1e-5
model = cFBPConvNet(experiment.geo, model_params).to(device)


# %% Define solver parameters
train_param = LIONParameter()

# loss fn
loss_fcn = torch.nn.MSELoss()
train_param.optimiser = "adam"

# optimizer
train_param.epochs = 500
train_param.learning_rate = args.lr
train_param.betas = (0.9, 0.99)
train_param.loss = "MSELoss"
optimiser = torch.optim.Adam(
    model.parameters(), lr=train_param.learning_rate, betas=train_param.betas
)

from LION.optimizers.supervised_learning import SupervisedSolver


# create solver
solver = SupervisedSolver(model, optimiser, loss_fcn, verbose=True)

# set data
solver.set_training(training_dataloader)
# Set validation. If non defined by user, it uses the loss function to validate.
# If this is set, it will save the model with the lowest validation loss automatically, given the validation_fname
solver.set_validation(
    validation_dataloader, validation_freq=10, validation_fname=validation_fname
)
# Set testing. Second input has to be a function that accepts torch tensors and returns a scalar
# set checkpointing procedure. It  will automatically checkpoint your models.
# If load_checkpoint=True, it will load the last checkpoint available in disk (useful for partial training loops in HPC)
solver.set_checkpointing(
    savefolder,
    checkpoint_fname=checkpoint_fname,
    checkpoint_freq=10,
    load_checkpoint=False,
)

# %% Train
solver.train(train_param.epochs)


# save final result
solver.save_final_results(final_result_fname)
