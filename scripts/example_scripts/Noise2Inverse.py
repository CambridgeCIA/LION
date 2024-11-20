from LION.classical_algorithms.fdk import fdk
from LION.optimizers.noise2inverse import Noise2InverseParams, Noise2InverseSolver
from LION.models.CNNs.MSDNet import MSDNet
import LION.experiments.ct_experiments as ct_experiments
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
import torch.nn as nn
import torch


# Set Device
device = torch.device("cuda:0")


# Define experiment
experiment = ct_experiments.LowDoseCTRecon()

# Data to train
batch_size = 1
dataloader = DataLoader(experiment.get_training_dataset(), batch_size, True)

# Define model
model = MSDNet()

# Create optimizer and loss function
optimizer = Adam(model.parameters())
loss_fn = nn.MSELoss()

# LION's Noise2InverseSolver.
# Noise to inverse requires certain user specifications.
# first, the number of splits of the sinogram
splits = 4
# then, the algorithm being used for the base-recosntruction that the model will denoise
algo = fdk
# last, an strategy on how to split the sinogram,
strategy = Noise2InverseSolver.X_one_strategy(splits)
# initialize the solver parameters
solver_params = Noise2InverseParams(
    splits, algo, strategy
)  # this is default, but if you want to define your own...

# Initialize the solver as the other solvers in LION
solver = Noise2InverseSolver(
    model,
    optimizer,
    loss_fn,
    solver_params,
    experiment.geo,
    verbose=True,
    device=device,
)

solver.set_training(dataloader)

solver.set_saving("path/to/save/folder", "finalresult.pt")
solver.set_checkpointing("checkpointfname_*.pt", 3)

epochs = 5
solver.train(epochs)

solver.save_final_results()
solver.clean_checkpoints()
