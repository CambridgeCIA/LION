from LION.classical_algorithms.fdk import fdk
from Sparse2InverseSolver import Sparse2InverseSolver
from LION.models.CNNs.UNets.Unet import UNet
import LION.experiments.ct_experiments as ct_experiments
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
import torch.nn as nn
import torch
import pathlib
import torch.utils.data as data_utils
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from LION.metrics.haarpsi import HAARPsi

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set Device
#%%
# % Chose device:
device = torch.device("cuda:1")
torch.cuda.set_device(device)

# Define your data paths
savefolder = pathlib.Path("/store/LION/ea692/LION/LION/trained_models/Sparse2Inverse/Train/SparseAngleLowDoseCTRecon")
# Creates the folders if they does not exist
savefolder.mkdir(parents=True, exist_ok=True)
final_result_fname = "S2I.pt"
checkpoint_fname = "S2I_check_*.pt"

# Define experiment
experiment = ct_experiments.SparseAngleLowDoseCTRecon()
train_dataset = experiment.get_training_dataset()
#30 sinograms for the experiment
indices = torch.arange(30)
train_dataset = data_utils.Subset(train_dataset, indices)

# Data to train
batch_size = 1
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Define model. In the original paper used UNet
model = UNet()

# Create optimizer and loss function
optimizer = Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

#Sparse2InverseSolver.
s2i_params = Sparse2InverseSolver.default_parameters()
# Sparse to inverse requires certain user specifications.
s2i_params.sino_split_count = 4
s2i_params.recon_fn = fdk

# Initialize the solver as the other solvers in LION
solver = Sparse2InverseSolver(
    model,
    optimizer,
    loss_fn,
    solver_params=s2i_params,
    geometry=experiment.geometry,
    verbose=True,
    device=device,
)

solver.set_training(dataloader)
solver.set_checkpointing(checkpoint_fname, 100, save_folder=savefolder)

epochs = 100

solver.train(epochs)
solver.save_final_results(final_result_fname, savefolder)
solver.clean_checkpoints()

# Test using the training data
savefolder = pathlib.Path("/home/ea692/LION/LION/trained_models/Sparse2Inverse/Test/SparseAngleLowDoseCTRecon/SparseVSNoise/30sin2000ep/64Angles_Haarpsi_and_SSIM")
savefolder.mkdir(parents=True, exist_ok=True)

model.eval()
solver_params = Sparse2InverseSolver.default_parameters()
solver_params.sino_split_count = 4
solver_params.recon_fn = fdk
optimizer = Adam(model.parameters())
#Not used directly, the solver defines its own loss.
loss_fn = nn.MSELoss()

solver_sparse = Sparse2InverseSolver(
    model,
    optimizer,
    loss_fn,
    solver_params=solver_params,
    geometry=experiment.geometry,
    verbose=False,
    device=device,
)

#Normalization in order to ensure a fair comparison of structural and perceptual image quality.
def normalize_01(x,y):
    x = (x - y.min())/ (y.max() - y.min())
    x[x>1]=1
    x[x<0]=0
    return x

#SSIM metric
def my_ssim(x, y):
    x = x.detach().squeeze().cpu()
    y = y.detach().squeeze().cpu()
    
    target_n = normalize_01(y,y)
    sparse_n = normalize_01(x,y)
    return ssim(target_n, sparse_n, data_range=1)
    
model.eval()
solver.set_testing(dataloader, my_ssim)
solver.test()
