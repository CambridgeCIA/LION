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
from LION.models.iterative_unrolled.LPD import LPD
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
from ts_algorithms import fdk


#%%
# % Chose device:
device = torch.device("cuda:0")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/ab2860/trained_models/test_debbuging/")
datafolder = pathlib.Path(
    "/store/DAMTP/ab2860/AItomotools/data/AItomotools/processed/LIDC-IDRI/"
)
final_result_fname = savefolder.joinpath("LPD_final_iter.pt")
checkpoint_fname = savefolder.joinpath("LPD_check_*.pt")
#
#%% Define experiment
params = ct_experiments.LowDoseCTRecon.default_parameters()
params.data_loader_params.max_num_slices_per_patient = 1
params.data_loader_params.training_proportion = 0.02
params.data_loader_params.validation_proportion = 0.02
experiment = ct_experiments.LowDoseCTRecon(
    experiment_params=params, datafolder=datafolder
)

#%% Dataset
dataset = experiment.get_training_dataset()
batch_size = 1
dataloader = DataLoader(dataset, batch_size, shuffle=True)

#%% Load model
lpd_model, lpd_param, lpd_data = LPD.load(final_result_fname)
lpd_model.eval()

# loop trhough testing data
for index, (sinogram, target_reconstruction) in tqdm(enumerate(dataloader)):

    lpd_out = lpd_model(sinogram)
    plt.figure()
    plt.imshow(lpd_out[0, 0, :, :].cpu().detach().numpy())
    plt.clim(0, 3)
    plt.savefig("lpd_out.png")
    # do whatever you want with this.
