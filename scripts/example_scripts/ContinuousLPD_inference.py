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
from LION.models.ContinuousLPD import ContinuousLPD
from LION.utils.parameter import Parameter
import LION.experiments.ct_experiments as ct_experiments
from ts_algorithms import fdk

#%%
%matplotlib inline

#%%
# % Chose device:
device = torch.device("cuda:0")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/cr661/LION/trained_models/low_dose/")
datafolder = pathlib.Path(
    "/store/DAMTP/ab2860/AItomotools/data/AItomotools/processed/LIDC-IDRI/"
)
final_result_fname = savefolder.joinpath("ContinuousLPD_checkBS2_0081.pt")
checkpoint_fname = savefolder.joinpath("ContinuousLPD_checkBS2_*.pt")
#
#%% Define experiment
experiment = ct_experiments.LowDoseCTRecon(datafolder=datafolder)

#%% Dataset
dataset = experiment.get_testing_dataset()
batch_size = 1
dataloader = DataLoader(dataset, batch_size, shuffle=True)

#%% Load model
lpd_model, lpd_param, lpd_data = ContinuousLPD.load(final_result_fname)
lpd_model.eval()

# loop trhough testing data
for index, (sinogram, target_reconstruction) in tqdm(enumerate(dataloader)):

    lpd_out = lpd_model(sinogram)
    plt.imshow(lpd_out[0,0,:,:].detach().numpy())
    plt.show()
    # do whatever you want with this.
