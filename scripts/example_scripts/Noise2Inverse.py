from LION.classical_algorithms.fdk import fdk
from LION.optimizers.Noise2InverseSolver import Noise2InverseSolver
from LION.models.CNNs.MSD_pytorch import MSD_pytorch
import LION.experiments.ct_experiments as ct_experiments
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
import torch.nn as nn
import torch
import pathlib


# Set Device
#%%
# % Chose device:
device = torch.device("cuda:1")
torch.cuda.set_device(device)

# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/ab2860/trained_models/test_debbuging/")
final_result_fname = "N2I.pt"
checkpoint_fname = "N2I_check_*.pt"

# Define experiment
experiment = ct_experiments.LowDoseCTRecon()

# Data to train
batch_size = 1
dataloader = DataLoader(experiment.get_training_dataset(), batch_size, True)

# Define model. Any model should work but the original paper used MSDNet
model = MSD_pytorch()

# Create optimizer and loss function
optimizer = Adam(model.parameters())
loss_fn = nn.MSELoss()


Noise2InverseSolver.cite()

# LION's Noise2InverseSolver.
n2i_params = Noise2InverseSolver.default_parameters()
# Noise to inverse requires certain user specifications.
# first, the number of splits of the sinogram
n2i_params.splits = 4
# then, the algorithm being used for the base-recosntruction that the model will denoise
n2i_params.algo = fdk
# last, an strategy on how to split the sinogram,
n2i_params.strategy = Noise2InverseSolver.X_one_strategy(n2i_params.splits)

# Initialize the solver as the other solvers in LION
solver = Noise2InverseSolver(
    model,
    optimizer,
    loss_fn,
    n2i_params,
    experiment.geometry,
    verbose=True,
    device=device,
)

solver.set_training(dataloader)

solver.set_checkpointing(checkpoint_fname, 3, save_folder=savefolder)

epochs = 100
solver.train(epochs)

solver.save_final_results(final_result_fname, savefolder)
solver.clean_checkpoints()


## It is important to remember that the noise2inverse inference does not work as an standard model
# Use the solver class to reconstruct (later to be replaced by recosntructor class maybe).

testing_dataloader = DataLoader(experiment.get_testing_dataset(), 1, True)
sinogram, target = next(iter(testing_dataloader))
solver.reconstruct(sinogram)
# Or just use solver.test() to run the entire testing set.
