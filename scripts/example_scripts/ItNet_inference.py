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
from LION.models.iterative_unrolled.ItNet import ItNet
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
from ts_algorithms import fdk


#%%
# % Chose device:
device = torch.device("cuda:0")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/ab2860/low_dose/")
datafolder = pathlib.Path(
    "/store/DAMTP/ab2860/AItomotools/data/AItomotools/processed/LIDC-IDRI/"
)
final_result_fname = savefolder.joinpath("LPD_final_iter.pt")
checkpoint_fname = savefolder.joinpath("LPD_check_*.pt")
#
#%% Define experiment
experiment = ct_experiments.LowDoseCTRecon(datafolder=datafolder)

#%% Dataset
dataset = experiment.get_testing_dataset()
batch_size = 1
dataloader = DataLoader(dataset, batch_size, shuffle=True)

#%% Load model
itnet_model, itnet_param, itnet_data = ItNet.load(final_result_fname)
itnet_model.eval()

# loop trhough testing data
for index, (sinogram, target_reconstruction) in tqdm(enumerate(dataloader)):

    lpd_out = itnet_model(sinogram)

    # do whatever you want with this.
