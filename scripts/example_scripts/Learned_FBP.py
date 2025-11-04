# %% This example shows how to train Learned Primal dual for full angle, noisy measurements.

# %% Imports

# Standard imports
import matplotlib.pyplot as plt
import pathlib
from skimage.metrics import structural_similarity as ssim

# Torch imports
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

# Lion imports
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
from LION.optimizers.SupervisedSolver import SupervisedSolver


def my_ssim(x, y):
    x = x.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    return ssim(x, y, data_range=x.max() - x.min())


# %%
# % Chose device:
device = torch.device("cuda:0")
torch.cuda.set_device(device)

# Define your data paths
savefolder = pathlib.Path("/store/LION/as3628/trained_models/")
# Creates the folders if they does not exist
savefolder.mkdir(parents=True, exist_ok=True)
final_result_fname_DeepFBP = "DeepFBP.pt"
checkpoint_fname_DeepFBP = "DeepFBP_check_*.pt"
validation_fname_DeepFBP = "DeepFBP_min_val.pt"
final_result_fname_DeepFusionBP = "DeepFusionBP.pt"
checkpoint_fname_DeepFusionBP = "DeepFusionBP_check_*.pt"
validation_fname_DeepFusionBP = "DeepFusionBP_min_val.pt"
final_result_fname_FusionFBP = "FusionFBP.pt"
checkpoint_fname_FusionFBP = "FusionFBP_check_*.pt"
validation_fname_FusionFBP = "FusionFBP_min_val.pt"

# %% Define experiment

experiment = ct_experiments.LowDoseCTRecon(dataset="LIDC-IDRI")
# %% Dataset
lidc_dataset = experiment.get_training_dataset()
lidc_dataset_val = experiment.get_validation_dataset()
# smaller dataset for example. Remove this for full dataset
indices = torch.arange(1)
lidc_dataset = data_utils.Subset(lidc_dataset, indices)
lidc_dataset_val = data_utils.Subset(lidc_dataset_val, indices)




# %% Define DataLoader
# Use the same amount of training

# get only one sample
batch_size = 1
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=False)
lidc_validation = DataLoader(lidc_dataset_val, batch_size, shuffle=False)
# I just want to check if the models are working so I'll use de training as testing to check if the SSIM loss is near 1 
lidc_test = DataLoader(lidc_dataset, batch_size, shuffle=False)


# %% Models
from LION.models.learned_fbp.DeepFBP import DeepFBPNetwork
from LION.models.learned_fbp.DeepFusionBP import DeepFusionBPNetwork
from LION.models.learned_fbp.FusionFBP import FusionFBPNetwork

# selects default parameters and create models 
default_parameters_DeepFBP = DeepFBPNetwork.default_parameters()
default_parameters_DeepFBP.n_iters = 5
model_DeepFBP = DeepFBPNetwork(experiment.geometry, default_parameters_DeepFBP)

default_parameters_DeepFusionBP = DeepFusionBPNetwork.default_parameters()
default_parameters_DeepFusionBP.n_iters = 5
model_DeepFusionBP = DeepFusionBPNetwork(experiment.geometry, default_parameters_DeepFusionBP)

default_parameters_FusionFBP = FusionFBPNetwork.default_parameters()
default_parameters_FusionFBP.n_iters = 5
model_FusionFBP = FusionFBPNetwork(experiment.geometry, default_parameters_FusionFBP)




# Define general optimizer
# 
# # %% Optimizer
train_param = LIONParameter()

# loss fn
loss_fcn = torch.nn.MSELoss()
train_param.optimiser = "adam"

# optimizer
train_param.epochs = 100
train_param.learning_rate = 1e-4
train_param.betas = (0.9, 0.99)
train_param.loss = "MSELoss"




##################################################### DeepFBP #####################################################

#define specific optimizer for each model
optimiser = torch.optim.Adam(
    model_DeepFBP.parameters(), lr=train_param.learning_rate, betas=train_param.betas
)

# %% Train
# create solver
solver = SupervisedSolver(
    model_DeepFBP, optimiser, loss_fcn, verbose=True, save_folder=savefolder
)

# YOU CAN IGNORE THIS. You can 100% just write your own pytorch training loop.
# LIONSover is just a convinience class that does some stuff for you, no need to use it.

# set data
solver.set_training(lidc_dataloader)
solver.set_validation(lidc_validation, 10, validation_fname=validation_fname_DeepFBP)
solver.set_testing(lidc_test, my_ssim)

# set checkpointing procedure
solver.set_checkpointing(
    checkpoint_fname_DeepFBP, 10, load_checkpoint_if_exists=False, save_folder=savefolder
)
# train
solver.train(train_param.epochs)
# delete checkpoints if finished
solver.clean_checkpoints()
# save final result
solver.save_final_results(final_result_fname_DeepFBP, savefolder)

# test

test_losses = solver.test()   
print(f"\nDeepFBP Mean test loss = {test_losses.mean():.6f}")
print(f"DeepFBP Std test loss  = {test_losses.std():.6f}")

plt.figure()
plt.semilogy(solver.train_loss[1:])
plt.savefig("loss_DeepFBP.png")







##################################################### DeepFusionBP #####################################################

#define specific optimizer for each model
optimiser = torch.optim.Adam(
    model_DeepFusionBP.parameters(), lr=train_param.learning_rate, betas=train_param.betas
)

# %% Train
# create solver
solver = SupervisedSolver(
    model_DeepFusionBP, optimiser, loss_fcn, verbose=True, save_folder=savefolder
)

# YOU CAN IGNORE THIS. You can 100% just write your own pytorch training loop.
# LIONSover is just a convinience class that does some stuff for you, no need to use it.

# set data
solver.set_training(lidc_dataloader)
solver.set_validation(lidc_validation, 10, validation_fname=validation_fname_DeepFusionBP)
solver.set_testing(lidc_test, my_ssim)

# set checkpointing procedure
solver.set_checkpointing(
    checkpoint_fname_DeepFusionBP, 10, load_checkpoint_if_exists=False, save_folder=savefolder
)
# train
solver.train(train_param.epochs)
# delete checkpoints if finished
solver.clean_checkpoints()
# save final result
solver.save_final_results(final_result_fname_DeepFusionBP, savefolder)

# test

test_losses = solver.test()   
print(f"\nDeepFusionBP Mean test loss = {test_losses.mean():.6f}")
print(f"DeepFusionBP Std test loss  = {test_losses.std():.6f}")

plt.figure()
plt.semilogy(solver.train_loss[1:])
plt.savefig("loss_DeepFusionBP.png")







##################################################### FusionFBP #####################################################


#define specific optimizer for each model
optimiser = torch.optim.Adam(
    model_FusionFBP.parameters(), lr=train_param.learning_rate, betas=train_param.betas
)

# %% Train
# create solver
solver = SupervisedSolver(
    model_FusionFBP, optimiser, loss_fcn, verbose=True, save_folder=savefolder
)

# YOU CAN IGNORE THIS. You can 100% just write your own pytorch training loop.
# LIONSover is just a convinience class that does some stuff for you, no need to use it.

# set data
solver.set_training(lidc_dataloader)
solver.set_validation(lidc_validation, 10, validation_fname=validation_fname_FusionFBP)
solver.set_testing(lidc_test, my_ssim)

# set checkpointing procedure
solver.set_checkpointing(
    checkpoint_fname_FusionFBP, 10, load_checkpoint_if_exists=False, save_folder=savefolder
)
# train
solver.train(train_param.epochs)
# delete checkpoints if finished
solver.clean_checkpoints()
# save final result
solver.save_final_results(final_result_fname_FusionFBP, savefolder)

# test

test_losses = solver.test()   
print(f"\nFusionFBP Mean test loss = {test_losses.mean():.6f}")
print(f"FusionFBP Std test loss  = {test_losses.std():.6f}")

plt.figure()
plt.semilogy(solver.train_loss[1:])
plt.savefig("loss_FusionFBP.png")


