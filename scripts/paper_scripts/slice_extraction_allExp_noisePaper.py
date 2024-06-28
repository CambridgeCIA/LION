# Standard imports
import matplotlib.pyplot as plt
import pathlib
import imageio
from tqdm import tqdm

# Torch imports
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

# Lion imports
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_learned_denoising_experiments as ct_denoising
from ts_algorithms import fdk, sirt, tv_min, nag_ls
from ts_algorithms.tv_min import tv_min2d
from LION.CTtools.ct_utils import make_operator


def extract_recons(experiments, model, slice_indices, savename):

    model.eval()

    # Experimental Testing Data
    testing_data = experiments[0].get_testing_dataset()
    #indices = torch.arange(20)
    #testing_data = data_utils.Subset(testing_data, indices)
    testing_dataloader = DataLoader(testing_data, 1, shuffle=False)

    # prepare operator for FBP
    op = make_operator(experiments[0].geo)

    with torch.no_grad():
        for index, (data, target) in enumerate(tqdm(testing_dataloader)):
            if index in slice_indices:
                if model_name.startswith("FBP"):
                    recon = model(data.to(device)).detach().cpu().numpy().squeeze()
                else:
                    output = model(data.to(device))
                    recon = fdk(op,output[0].to(device)).detach().cpu().numpy().squeeze()
                np.save(str(savename[:-4]+"_ExpNoise_test_slice"+str(index)+".npy"),recon)

    # Artificial Testing Data
    testing_data = experiments[1].get_testing_dataset()
    #indices = torch.arange(20)
    #testing_data = data_utils.Subset(testing_data, indices)
    testing_dataloader = DataLoader(testing_data, 1, shuffle=False)

    # prepare operator for FBP
    op = make_operator(experiments[1].geo)

    with torch.no_grad():
        for index, (data, target) in enumerate(tqdm(testing_dataloader)):
            if index in slice_indices:
                if model_name.startswith("FBP"):
                    recon = model(data.to(device)).detach().cpu().numpy().squeeze()
                else:
                    output = model(data.to(device))
                    recon = fdk(op,output[0].to(device)).detach().cpu().numpy().squeeze()
                np.save(str(savename[:-4]+"_ArtNoise_test_slice"+str(index)+".npy"),recon)

    # Experimental Testing Data Recon Domain
    testing_data = experiments[2].get_testing_dataset()
    #indices = torch.arange(20)
    #testing_data = data_utils.Subset(testing_data, indices)
    testing_dataloader = DataLoader(testing_data, 1, shuffle=False)

    # prepare operator for FBP
    op = make_operator(experiments[2].geo)

    with torch.no_grad():
        for index, (data, target) in enumerate(tqdm(testing_dataloader)):
            if index in slice_indices:
                GT_recon = target[0].to(device).detach().cpu().numpy().squeeze()
                noisy_recon = fdk(op,data[0].to(device)).detach().cpu().numpy().squeeze()
                np.save(str("/export/scratch3/mbk/LION/noise_slices/NoisyRecon_ExpNoise_test_slice"+str(index)+".npy"),noisy_recon)
                np.save(str("/export/scratch3/mbk/LION/noise_slices/GTRecon_ExpNoise_test_slice"+str(index)+".npy"),GT_recon)

    return None


# Define your cuda device
device = torch.device("cuda:0")

# Define the noise level for the artificial noise
I0 = 200

# Define your slice indices that you want to extract
slice_indices = [205,366,408]

# Define the paths to the folder with the saved trained models and where the slice extractions shall be stored
savefolder = pathlib.Path("/export/scratch3/mbk/LION/noise_paper/trained_models/testing_debugging/")
slicefolder = pathlib.Path("/export/scratch3/mbk/LION/noise_slices/")


# Define the model names for the iterable evaluation
# Use min validation "min_val.pt", or final result ".pt", whichever you prefer
# ["UNet_ExpNoise_min_val.pt", "UNet_ArtNoise_min_val.pt",
model_names = ["MSD_ExpNoise_min_val.pt", "MSD_ArtNoise_min_val.pt",
"FBPUNet_ExpNoise_min_val.pt", "FBPUNet_ArtNoise_min_val.pt",
"FBPMSDNet_ExpNoise_min_val.pt", "FBPMSDNet_ArtNoise_min_val.pt"]

# Define all experiments in experiments list
experiments = []

experiments.append(ct_denoising.ExperimentalNoiseDenoising())

ArtNoise_sino_experiment = ct_denoising.ArtificialNoiseDenoising()
ArtNoise_noise_sino_default_parameters = ArtNoise_sino_experiment.default_parameters("2DeteCT")
ArtNoise_noise_sino_default_parameters.data_loader_params.noise_params.I0 = I0
ArtNoise_sino_experiment = ct_denoising.ArtificialNoiseDenoising(ArtNoise_noise_sino_default_parameters)
experiments.append(ArtNoise_sino_experiment)

experiments.append(ct_denoising.ExperimentalNoiseDenoisingRecon())

ArtNoise_recon_experiment = ct_denoising.ArtificialNoiseDenoising()
ArtNoise_noise_recon_default_parameters = ArtNoise_recon_experiment.default_parameters("2DeteCT")
ArtNoise_noise_recon_default_parameters.data_loader_params.noise_params.I0 = I0
ArtNoise_recon_experiment = ct_denoising.ArtificialNoiseDenoising(ArtNoise_noise_recon_default_parameters)
experiments.append(ArtNoise_recon_experiment)


# Import the relevant LION models for the experiments
from LION.models.CNNs.UNets.UNet_3 import UNet
from LION.models.CNNs.MS_D import MS_D
from LION.models.post_processing.FBPUNet import FBPUNet
from LION.models.post_processing.FBPMSDNet import FBPMS_D

for model_idx, model_name in enumerate(tqdm(model_names)):
    savename = str(str(slicefolder.joinpath(model_name[:-11]))+".pdf")
    if model_name.startswith("UNet"):
        model, options, data = UNet.load(savefolder.joinpath(model_name))
    elif model_name.startswith("MSD"):
        model, options, data = MS_D.load(savefolder.joinpath(model_name))
    elif model_name.startswith("FBPU"):
        model, options, data = FBPUNet.load(savefolder.joinpath(model_name))
    elif model_name.startswith("FBPM"):
        model, options, data = FBPMS_D.load(savefolder.joinpath(model_name))

    model.to(device)
    extract_recons(experiments, model, slice_indices, savename)
