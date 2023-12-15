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

from skimage.metrics import structural_similarity as ssim

# LION imports
import LION.CTtools.ct_utils as ct
from LION.models.ContinuousLPD import ContinuousLPD
from LION.models.LPD import LPD
from LION.models.FBPConvNet import FBPConvNet
from LION.models.ItNet import ItNet
from LION.utils.parameter import Parameter
import LION.experiments.ct_experiments as ct_experiments
from ts_algorithms import fdk

#%%
%matplotlib inline

#%%
def plot_outputs(clpd_out, fbp_out, bad_recon, target_reconstruction):
    fig, axs = plt.subplots(1, 4, figsize=(10, 5))

    # Plot lcpd_out
    im0 = axs[0].imshow(clpd_out[0,0,:,:].detach().cpu().numpy(), cmap='gray')
    axs[0].set_title('CLPD Output')
    im0.set_clim(0,3)
    ## Plot lpd_out
    #im1 = axs[1].imshow(lpd_out[0,0,:,:].detach().cpu().numpy(), cmap='gray')
    #axs[1].set_title('LPD Output')
    #im1.set_clim(0,3)
    # Plot FBP out
    im1 = axs[1].imshow(fbp_out[0,0,:,:].detach().cpu().numpy(), cmap='gray')
    axs[1].set_title('FBP Output')
    im1.set_clim(0,3)
    # Plot bad_recon
    im2 = axs[2].imshow(bad_recon[0,0,:,:].detach().cpu().numpy(), cmap='gray')
    axs[2].set_title('FDK')
    im2.set_clim(0,3)
    # Plot target_reconstruction
    im3 = axs[3].imshow(target_reconstruction[0,0,:,:].detach().cpu().numpy(), cmap='gray')
    axs[3].set_title('Target Reconstruction')
    im3.set_clim(0,3)
    plt.show()   

#%%
def compute_ssim(out, target_reconstruction):
    out = out.detach().squeeze().cpu().numpy()
    target_reconstruction = target_reconstruction.detach().squeeze().cpu().numpy()

    ssim_scores = []
    for i in range(out.shape[0]):
        ssim_score = ssim(out[i], target_reconstruction[i], multichannel=False, data_range=out[i].max()-out[i].min())
        ssim_scores.append(ssim_score)

    return np.mean(ssim_scores)
        
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

final_result_fname_fbp = pathlib.Path("/store/DAMTP/ab2860/trained_models/low_dose/FBPConvNet_final_iter.pt")
final_result_fname_itnet = pathlib.Path("/store/DAMTP/ab2860/trained_models/low_dose/ItNet_Unet_final_iter.pt")

final_result_fname_lpd = savefolder.joinpath("LPD_checkBS2fixed_0171.pt")
checkpoint_fname_lpd = savefolder.joinpath("LPD_checkBS2fixed_*.pt")
#%% Define experiment
experiment = ct_experiments.LowDoseCTRecon(datafolder=datafolder)

#%% Dataset
dataset = experiment.get_testing_dataset()
batch_size = 1
dataloader = DataLoader(dataset, batch_size, shuffle=False)


#%% Load model
clpd_model, clpd_param, clpd_data = ContinuousLPD.load(final_result_fname)
lpd_model, lpd_param, lpd_data = LPD.load(final_result_fname_lpd)
fbp_model, fbp_param, fbp_data = FBPConvNet.load(final_result_fname_fbp)
clpd_model.eval()
lpd_model.eval()
fbp_model.eval()


# loop through testing data
for index, (sinogram, target_reconstruction) in tqdm(enumerate(dataloader)):
    bad_recon = torch.zeros(target_reconstruction.shape, device=device)
    for sino in range(sinogram.shape[0]):
        bad_recon[sino] = fdk(dataset.operator, sinogram[sino])
    clpd_out = clpd_model(sinogram)
    lpd_out = lpd_model(sinogram)
    fbp_out = fbp_model(bad_recon)
    clpd_ssim = compute_ssim(clpd_out, target_reconstruction)
    fbp_ssim = compute_ssim(fbp_out, target_reconstruction)
    print("CLPD SSIM: ", clpd_ssim)
    print("FBP SSIM: ", fbp_ssim)
    plot_outputs(clpd_out, clpd_out, bad_recon, target_reconstruction)
    torch.cuda.empty_cache()
# %%
