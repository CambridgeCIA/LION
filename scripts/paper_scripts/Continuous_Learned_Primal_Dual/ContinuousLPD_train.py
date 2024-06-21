# %% This example shows how to train Continuous LPD for full angle, noisy measurements.


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

from LION.models.iterative_unrolled.cLPD import cLPD
from LION.utils.parameter import LIONParameter
from LION.utils.utils import str2bool
from ts_algorithms import fdk


import LION.experiments.ct_experiments as ct_experiments


# %%
# arguments for argparser
parser = argparse.ArgumentParser()
parser.add_argument("--geometry", type=str)
parser.add_argument("--dose", type=str)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--second_order", type=str2bool)
parser.add_argument("--instance_norm", type=str2bool)

# %%
args = parser.parse_args()
# % Chose device:
device = torch.device("cuda:0")
torch.cuda.set_device(device)

model_name = "ContinuousLPD"

if args.dose == "low":
    savefolder = pathlib.Path("/your/work/in/progress/folder/low_dose/")
elif args.dose == "extreme_low":
    savefolder = pathlib.Path("/your/work/in/progress/folder/low_dose/")
else:
    raise ValueError("Dose not recognised")

datafolder = pathlib.Path("/home/cr661/rds/hpc-work/store/LION/data/LIDC-IDRI/")
final_result_fname = savefolder.joinpath(
    f"{model_name}_final_iterBS2smallerLR_no_adjoint_in{args.instance_norm}_{args.dose}_{args.geometry}.pt"
)
checkpoint_fname = savefolder.joinpath(
    f"ContinuousLPD_checkBS2smallerLR_no_adjoint_in{args.instance_norm}_{args.dose}_{args.geometry}*.pt"
)
validation_fname = savefolder.joinpath(
    f"ContinuousLPD_min_valBS2smallerLR_no_adjoint_in{args.instance_norm}_{args.dose}_{args.geometry}.pt"
)
#
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

# %% Dataset
lidc_dataset = experiment.get_training_dataset()
lidc_dataset_val = experiment.get_validation_dataset()

# %% Define DataLoader
# Use the same amount of training
batch_size = 2
training_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
validation_dataloader = DataLoader(lidc_dataset_val, batch_size, shuffle=True)

# %% Model
# Default model is already from the paper.
default_parameters = cLPD.continous_LPD_paper()
# This makes the LPD calculate the step size for the backprojection, which in my experience results in much much better pefromace
# as its all in the correct scale.
default_parameters.instance_norm = args.instance_norm
default_parameters.do_second_order = args.second_order
print(f"Training ContinuousLPD with second order: {args.second_order}")


model = cLPD(
    geometry_parameters=experiment.geo, model_parameters=default_parameters
).to(device)


# %% Optimizer
train_param = LIONParameter()

# loss fn
loss_fcn = torch.nn.MSELoss()
train_param.optimiser = "adam"

# optimizer
train_param.epochs = 100
train_param.learning_rate = args.lr
train_param.betas = (0.9, 0.99)
train_param.loss = "MSELoss"
optimiser = torch.optim.Adam(
    model.parameters(), lr=train_param.learning_rate, betas=train_param.betas
)

from LION.optimizers.supervised_learning import supervisedSolver


# create solver
solver = supervisedSolver(model, optimiser, loss_fcn, verbose=True)

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
