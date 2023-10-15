#%% Noise2Inverse train

#%% Imports

# Basic science imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# basic python imports
from tqdm import tqdm
import pathlib
import copy

# LION imports
import LION.CTtools.ct_utils as ct
from LION.models.MS_D import MS_D
from LION.utils.parameter import Parameter
import LION.experiments.ct_experiments as ct_experiments
from ts_algorithms import fdk


#%%
# % Chose device:
device = torch.device("cuda:0")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/ab2860/low_dose/")
datafolder = pathlib.Path("/local/scratch/public/AItomotools/processed/LIDC-IDRI/")
final_result_fname = savefolder.joinpath("Noise2Inverse_final_iter.pt")
checkpoint_fname = savefolder.joinpath("Noise2Inverse_check_*.pt")
#
#%% Define experiment
experiment = ct_experiments.LowDoseCTRecon(datafolder=datafolder)

#%% Dataset
dataset = experiment.get_testing_dataset()
batch_size = 1
dataloader = DataLoader(dataset, batch_size, shuffle=True)

#%% Load model
n2i, n2i_param, n2i_data = MS_D.load(final_result_fname)
n2i.eval()

# number of data splits
k = 4
op = []
geo = copy.deepcopy(n2i_param.geometry_parameters)
angles = geo.angles.copy()
for i in range(k):
    geo.angles = angles[i:-1:k]
    op.append(ct.make_operator(geo))

# loop trhough testing data
for index, (sinogram, target_reconstruction) in tqdm(enumerate(dataloader)):

    # This is FDK for comparison
    bad_recon = torch.zeros(target_reconstruction.shape, device=device)
    for sino in range(sinogram.shape[0]):
        bad_recon[sino] = fdk(dataset.operator, sinogram[sino])

    # do the FBP recon per k split
    size_noise2inv = list(target_reconstruction.shape)
    size_noise2inv.insert(0, k)
    bad_recon_n2i = torch.zeros(tuple(size_noise2inv), device=device)
    for split in range(k):
        bad_recon_n2i[split] = fdk(op[split], sinogram[0, 0, split:-1:k])
    # apply net
    n2i_out = torch.zeros(bad_recon.shape, device=device)
    for split in range(k):
        n2i_out = n2i_out + n2i(bad_recon_n2i[split])
    n2i_out = n2i_out / k
    n2i_out = n2i_out.detach().cpu().numpy()[0, 0]

    # do whatever you want with this.
