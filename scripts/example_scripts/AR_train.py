#%% This example shows how to train MSDNet for full angle, noisy measurements.


#%% Imports
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Subset
import pathlib
from LION.models.CNNs.MSDNets.MS_D2 import MSD_Net
from LION.models.LIONmodel import ModelInputType
from LION.models.iterative_unrolled.ItNet import UNet
from LION.models.iterative_unrolled.LPD import LPD
from LION.models.learned_regularizer.AR import AR
from LION.optimizers.losses.WGANloss import WGANloss
from LION.optimizers.supervised_learning import SupervisedSolver
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments


#%%
# % Chose device:
device = torch.device("cuda:1")
torch.cuda.set_device(device)

# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/cs2186/trained_models/test_debugging/")

final_result_fname = "ar_final_iter.pt"
checkpoint_fname = "ar_check_*.pt"
validation_fname = "ar_min_val.pt"
#
#%% Define experiment
# experiment = ct_experiments.LowDoseCTRecon(datafolder=datafolder)
experiment = ct_experiments.clinicalCTRecon()
#%% Dataset
lidc_dataset = experiment.get_training_dataset()
lidc_dataset_val = experiment.get_validation_dataset()
lidc_dataset_test = experiment.get_testing_dataset()

#%% Define DataLoader
# Use the same amount of training
batch_size = 1
lidc_dataset = Subset(lidc_dataset, [i for i in range(5)])
lidc_dataset_val = Subset(lidc_dataset_val, [i for i in range(5)])
lidc_dataset_test = Subset(lidc_dataset_test, [i for i in range(5)])
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
lidc_validation = DataLoader(lidc_dataset_val, batch_size, shuffle=True)
lidc_test = DataLoader(lidc_dataset_test, batch_size, shuffle=True)
#%% Model
width, depth = 1, 3
dilations = []
for i in range(depth):
    for j in range(width):
        dilations.append((((i * width) + j) % 10) + 1)
model = UNet().to(device)
model.model_parameters.model_input_type = ModelInputType.IMAGE
regularizer = AR(model, experiment.geo)

#%% Optimizer
@dataclass
class TrainParam(LIONParameter):
    optimiser: str
    epochs: int
    learning_rate: float
    betas: Tuple[float, float]
    loss: str
    accumulation_steps: int


train_param = TrainParam("adam", 1, 1e-3, (0.9, 0.99), "MSELoss", 1)

optimizer = Adam(
    model.parameters(), lr=train_param.learning_rate, betas=train_param.betas
)


train_loss = WGANloss()
val_loss = nn.MSELoss()
#%% Solver
solver = SupervisedSolver(
    regularizer, optimizer, train_loss, experiment.geo, device=device, verbose=True
)

solver.set_saving(savefolder, final_result_fname)
solver.set_checkpointing(checkpoint_fname)
solver.set_training(lidc_dataloader)
solver.set_validation(lidc_validation, 1, val_loss, validation_fname)
solver.set_testing(lidc_test)
solver.set_normalization(True)

# train regularizer
solver.train(3)

#%% Use regularizer
