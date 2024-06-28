# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author  : Max Kiss, Ander Biguri
# =============================================================================

#%% 0 - Imports
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Standard imports
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.patches as patches

from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Torch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

# LION imports
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_learned_denoising_experiments as ct_denoising
from LION.CTtools.ct_utils import make_operator
from ts_algorithms import fdk, sirt, tv_min, nag_ls


# Just a temporary SSIM that takes torch tensors (will be added to LION at some point)
def my_ssim(x: torch.tensor, y: torch.tensor, data_range=None):
    x = x.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    if data_range is None:
        data_range = x.max() - x.min()
    elif type(data_range) == torch.Tensor:
        data_range = data_range.cpu().numpy().squeeze()
    return ssim(x, y, data_range=data_range)


def my_psnr(x: torch.tensor, y: torch.tensor, data_range=None):
    x = x.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    if data_range is None:
        data_range = x.max() - x.min()
    elif type(data_range) == torch.Tensor:
        data_range = data_range.cpu().numpy().squeeze()
    return psnr(x, y, data_range=data_range)


def plot_imgs(image_list, rows,  savename, width=None, vmin=None, vmax=None, scl_fctr = None):
    if width is None:
        width = plt.rcParams["figure.figsize"][0]

    rows = rows
    cols = int(np.ceil(len(image_list)/rows))
    if scl_fctr != None:
        fig = plt.figure(figsize=(width, (rows / cols * width)/scl_fctr))
    else:
        fig = plt.figure(figsize=(width, rows / cols * width))

    grid = ImageGrid(
        fig,
        111,                # similar to subplot(111)
        nrows_ncols=(rows, cols),
        axes_pad=0.025,       # pad between axes in inch.
        label_mode="all",   # all axes get a label (which we remove later)
    )

    for i, img in enumerate(image_list):
        max_val = np.max(img)
        print(f"Max val of image {i} is {max_val}")
        
        ax = grid[i]
        ax.axis('off')
        for s in ax.spines.values():
            s.set_visible(False)

        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)

    plt.tight_layout()
    plt.savefig(savename, dpi=300)
    plt.show()

# Create instance of MSE function
MSE = nn.MSELoss()


#%% 1 - Settings
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Savefolder
savefolder = "/export/scratch3/mbk/LION/noise_paper/noise_figures/"

# Device
device = torch.device("cuda:0")
torch.cuda.set_device(device)

#%% 2 - Define experiment
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# These are all the experiments we need to run for the noise paper

# Experimental noisy dataset
ExpNoise_sino_experiment = ct_denoising.ExperimentalNoiseDenoising()


#%% 3 - Obtaining Datasets from experiments
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ExpNoise_sino_data = ExpNoise_sino_experiment.get_training_dataset()

#print(experiment.experiment_params.flat_field_correction)

##############################################################
# REMOVE THIS CHUNK IN THE FINAL VERSION
#indices = torch.arange(20)
#ExpNoise_sino_data = data_utils.Subset(ExpNoise_sino_data, indices)
# REMOVE THIS CHUNK IN THE FINAL VERSION
##############################################################

#%% 4 - Define Data Loader
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
batch_size = 1
ExpNoise_sino_dataloader = DataLoader(ExpNoise_sino_data, batch_size, shuffle=False)


# Metrics arrays
ExpNoise_sino_ssim = np.zeros(len(ExpNoise_sino_dataloader))
ExpNoise_sino_psnr = np.zeros(len(ExpNoise_sino_dataloader))
ExpNoise_sino_mse = np.zeros(len(ExpNoise_sino_dataloader))
ExpNoise_sino_means = np.zeros(len(ExpNoise_sino_dataloader))

ArtNoise_sino_ssim = np.zeros(len(ExpNoise_sino_dataloader))
ArtNoise_sino_psnr = np.zeros(len(ExpNoise_sino_dataloader))
ArtNoise_sino_mse = np.zeros(len(ExpNoise_sino_dataloader))
ArtNoise_sino_means = np.zeros(len(ExpNoise_sino_dataloader))

ExpArtNoise_sino_mse = np.zeros(len(ExpNoise_sino_dataloader))

ExpNoise_recon_ssim = np.zeros(len(ExpNoise_sino_dataloader))
ExpNoise_recon_psnr = np.zeros(len(ExpNoise_sino_dataloader))
ExpNoise_recon_mse = np.zeros(len(ExpNoise_sino_dataloader))
ExpNoise_recon_means = np.zeros(len(ExpNoise_sino_dataloader))

ArtNoise_recon_ssim = np.zeros(len(ExpNoise_sino_dataloader))
ArtNoise_recon_psnr = np.zeros(len(ExpNoise_sino_dataloader))
ArtNoise_recon_mse = np.zeros(len(ExpNoise_sino_dataloader))
ArtNoise_recon_means = np.zeros(len(ExpNoise_sino_dataloader))

ExpArtNoise_recon_mse = np.zeros(len(ExpNoise_sino_dataloader))

#fdk_ssim = np.zeros(len(testing_dataloader))
#fdk_psnr = np.zeros(len(testing_dataloader))

# Images lists
Exp_sinos = []
Exp_recons = []
Art_sinos = []
Art_recons = []

# Slice indices for images lists
#slice_indices = [2,5,7,12,18]
slice_indices = [72,134,182,220,257]

# Experimental Noise Comparisons
with torch.no_grad():
    for index, (sino, target) in enumerate(ExpNoise_sino_dataloader):
        # where sino is the experimental noisy sinogram and target is the experimental clean sinogram
        ExpNoise_sino_ssim[index] = my_ssim(target, sino)
        ExpNoise_sino_psnr[index] = my_psnr(target, sino)
        ExpNoise_sino_mse[index] = MSE(target, sino)
        ExpNoise_sino_means[index] = torch.mean(sino)

        op = make_operator(ExpNoise_sino_experiment.geo)
        Exp_recon = fdk(op,sino[0].to(device))
        GT_recon = fdk(op,target[0].to(device))

        ExpNoise_recon_ssim[index] = my_ssim(GT_recon, Exp_recon)
        ExpNoise_recon_psnr[index] = my_psnr(GT_recon, Exp_recon)
        ExpNoise_recon_mse[index] = MSE(GT_recon, Exp_recon)
        ExpNoise_recon_means[index] = torch.mean(Exp_recon)

        if index in slice_indices:
            Exp_sinos.append(-np.exp(sino[0].detach().cpu().numpy().squeeze().T))
            np.save(str(savefolder+"sino_slice_"+str(index)+"_ExpNoise.npy"),-np.exp(sino[0].detach().cpu().numpy().squeeze().T))
            Exp_recons.append(Exp_recon[0].detach().cpu().numpy().squeeze())
            np.save(str(savefolder+"recon_slice_"+str(index)+"_ExpNoise.npy"),Exp_recon[0].detach().cpu().numpy().squeeze())

print(f"Sino domain results")
print(f"Experimental Noise vs. GT SSIM: {ExpNoise_sino_ssim.mean():.4f} $\pm$ {ExpNoise_sino_ssim.std():.4f}")
print(f"Experimental Noise vs. GT PSNR: {ExpNoise_sino_psnr.mean():.4f} $\pm$ {ExpNoise_sino_psnr.std():.4f}")
print(f"Experimental Noise vs. GT MSE: {ExpNoise_sino_mse.mean():.4f} $\pm$ {ExpNoise_sino_mse.std():.4f}")
print(f"Experimental Noise Mean: {ExpNoise_sino_means.mean():.4f} $\pm$ {ExpNoise_sino_means.std():.4f}")

print(f"Recon domain results")
print(f"Experimental Noise vs. GT SSIM: {ExpNoise_recon_ssim.mean():.4f} $\pm$ {ExpNoise_recon_ssim.std():.4f}")
print(f"Experimental Noise vs. GT PSNR: {ExpNoise_recon_psnr.mean():.4f} $\pm$ {ExpNoise_recon_psnr.std():.4f}")
print(f"Experimental Noise vs. GT MSE: {ExpNoise_recon_mse.mean():.4f} $\pm$ {ExpNoise_recon_mse.std():.4f}")
print(f"Experimental Noise Mean: {ExpNoise_recon_means.mean():.4f} $\pm$ {ExpNoise_recon_means.std():.4f}")

#%% 5 - Loop through different levels of artificial noise
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

I0_list = [200,250,300,350]

for I0 in I0_list:

    # Simulated noisy dataset
    ArtNoise_sino_experiment = ct_denoising.ArtificialNoiseDenoising()
    
    ArtNoise_noise_default_parameters = ArtNoise_sino_experiment.default_parameters("2DeteCT")
    ArtNoise_noise_default_parameters.data_loader_params.noise_params.I0 = I0
    ArtNoise_sino_experiment = ct_denoising.ArtificialNoiseDenoising(ArtNoise_noise_default_parameters)
    ArtNoise_sino_data = ArtNoise_sino_experiment.get_training_dataset()
    
    ##############################################################
    # REMOVE THIS CHUNK IN THE FINAL VERSION
    #indices = torch.arange(20)
    #ArtNoise_sino_data = data_utils.Subset(ArtNoise_sino_data, indices)
    # REMOVE THIS CHUNK IN THE FINAL VERSION
    ##############################################################
    
    ArtNoise_sino_dataloader = DataLoader(ArtNoise_sino_data, batch_size, shuffle=False)  
    
    # Artificial Noise Comparisons

    with torch.no_grad():
        for index, (sino, target) in enumerate(ArtNoise_sino_dataloader):
            # where sino is the artificial noisy sinogram and target is the experimental clean sinogram
            ArtNoise_sino_ssim[index] = my_ssim(target, sino)
            ArtNoise_sino_psnr[index] = my_psnr(target, sino)
            ArtNoise_sino_mse[index] = MSE(target,sino)
            ArtNoise_sino_means[index] = torch.mean(sino)

            op = make_operator(ArtNoise_sino_experiment.geo)
            Art_recon = fdk(op,sino[0].to(device))
            GT_recon = fdk(op,target[0].to(device))

            ArtNoise_recon_ssim[index] = my_ssim(GT_recon, Art_recon)
            ArtNoise_recon_psnr[index] = my_psnr(GT_recon, Art_recon)
            ArtNoise_recon_mse[index] = MSE(GT_recon, Art_recon)
            ArtNoise_recon_means[index] = torch.mean(Art_recon)

            if index in slice_indices:
                Art_sinos.append(-np.exp(sino[0].detach().cpu().numpy().squeeze().T))
                np.save(str(savefolder+"sino_slice_"+str(index)+"_ArtNoise_I0_"+str(I0)+".npy"),-np.exp(sino[0].detach().cpu().numpy().squeeze().T))
                Art_recons.append(Art_recon[0].detach().cpu().numpy().squeeze())
                np.save(str(savefolder+"recon_slice_"+str(index)+"_ArtNoise_I0_"+str(I0)+".npy"),Art_recon[0].detach().cpu().numpy().squeeze())

            ExpArtNoise_sino_mse[index] = MSE(ExpNoise_sino_data[index][0],sino[0])
            ExpArtNoise_recon_mse[index] = MSE(fdk(op,ExpNoise_sino_data[index][0].to(device)),Art_recon[0])
    

    print(f"Sino domain results for I0 = {I0}")
    print(f"Artificial Noise vs. GT SSIM: {ArtNoise_sino_ssim.mean():.4f} $\pm$ {ArtNoise_sino_ssim.std():.4f}")
    print(f"Artificial Noise vs. GT PSNR: {ArtNoise_sino_psnr.mean():.4f} $\pm$ {ArtNoise_sino_psnr.std():.4f}")
    print(f"Artificial Noise vs. GT MSE: {ArtNoise_sino_mse.mean():.4f} $\pm$ {ArtNoise_sino_mse.std():.4f}")
    print(f"Artificial Noise Mean: {ArtNoise_sino_means.mean():.4f} $\pm$ {ArtNoise_sino_means.std():.4f}")

    print(f"Experimental Noise vs. Artificial Noise MSE: {ExpArtNoise_sino_mse.mean():.4f} +- {ExpArtNoise_sino_mse.std():.4f}")
    
    print(f"Recon domain results for I0 = {I0}")
    print(f"Artificial Noise vs. GT SSIM: {ArtNoise_recon_ssim.mean():.4f} $\pm$ {ArtNoise_recon_ssim.std():.4f}")
    print(f"Artificial Noise vs. GT PSNR: {ArtNoise_recon_psnr.mean():.4f} $\pm$ {ArtNoise_recon_psnr.std():.4f}")
    print(f"Artificial Noise vs. GT MSE: {ArtNoise_recon_mse.mean():.4f} $\pm$ {ArtNoise_recon_mse.std():.4f}")
    print(f"Artificial Noise Mean: {ArtNoise_recon_means.mean():.4f} $\pm$ {ArtNoise_recon_means.std():.4f}")

    print(f"Experimental Noise vs. Artificial Noise MSE: {ExpArtNoise_recon_mse.mean():.4f} $\pm$ {ExpArtNoise_recon_mse.std():.4f}")    
    
# TEST 2: Display examples

sinos_list = Exp_sinos
sinos_list = sinos_list + Art_sinos

plot_imgs(sinos_list, rows = 5, vmin=-12.20, vmax = -1.00, width=20, scl_fctr = 3.75, savename = str(savefolder+"Simulation_sinogram_comparison.pdf"))

recons_list = Exp_recons
recons_list = recons_list + Art_recons

plot_imgs(recons_list, rows = 5, vmin=0, vmax = 0.008, width=20, savename = str(savefolder+"Simulation_reconstruction_comparison.pdf"))
