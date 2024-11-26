from LION.classical_algorithms.fdk import fdk
from LION.optimizers.SelfSupervisedSolver import SelfSupervisedSolver
from LION.models.CNNs.UNets.Unet import UNet
import LION.experiments.ct_experiments as ct_experiments
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
import torch.nn as nn
import torch
import pathlib
from LION.losses.SURE import SURE
import torch.utils.data as data_utils

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

indices = torch.arange(1)
lidc_dataset = data_utils.Subset(experiment.get_training_dataset(), indices)
# Data to train
batch_size = 1
dataloader = DataLoader(lidc_dataset, batch_size, False)

# Define model. Any model should work but the original paper used MSDNet
model = UNet()

# Create optimizer and loss function
optimizer = Adam(model.parameters())
SURE.cite()
loss_fn = SURE(noise_std=0.07)

# Initialize the solver as the other solvers in LION
solver = SelfSupervisedSolver(
    model,
    optimizer,
    loss_fn,
    geometry=experiment.geometry,
)


solver.set_training(dataloader)

solver.set_checkpointing(
    checkpoint_fname, 50, save_folder=savefolder, load_checkpoint_if_exists=False
)

epochs = 500
solver.train(epochs)

solver.save_final_results(final_result_fname, savefolder)
solver.clean_checkpoints()


# %% Test
testing_dataset = experiment.get_training_dataset()
testing_dataloader = DataLoader(testing_dataset, 1, False)
sino, target = next(iter(testing_dataloader))
noisy = fdk(sino, experiment.geometry)
solver.model.eval()
output = solver.model(noisy)

import matplotlib.pyplot as plt

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(noisy[0, 0].cpu().numpy())
plt.clim(0, 3)
plt.title("Noisy")
plt.subplot(1, 3, 2)
plt.imshow(output[0, 0].detach().cpu().numpy())
plt.clim(0, 3)

plt.title("Denoised")
plt.subplot(1, 3, 3)
plt.imshow(target[0, 0].cpu().numpy())
plt.clim(0, 3)

plt.title("Ground Truth")
plt.savefig("test.png", dpi=300)
# %%
