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

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# LION imports
from LION.models.ContinuousLPD import ContinuousLPD
from LION.models.LPD import LPD
from LION.models.FBPConvNet import FBPConvNet
import LION.experiments.ct_experiments as ct_experiments
from ts_algorithms import fdk

#%%
%matplotlib inline

#%%
def plot_outputs(clpd_out, fbp_out, bad_recon, target_reconstruction, save_path):
    fig, axs = plt.subplots(1, 4, figsize=(10, 5))

    # Plot lcpd_out
    im0 = axs[0].imshow(clpd_out[0,0,:,:].detach().cpu().numpy(), cmap='gray')
    axs[0].set_title('CLPD Output')
    im0.set_clim(0,3)
    axs[0].axis('off')
    ## Plot lpd_out
    #im1 = axs[1].imshow(lpd_out[0,0,:,:].detach().cpu().numpy(), cmap='gray')
    #axs[1].set_title('LPD Output')
    #im1.set_clim(0,3)
    # Plot FBP out
    im1 = axs[1].imshow(fbp_out[0,0,:,:].detach().cpu().numpy(), cmap='gray')
    axs[1].set_title('FBP Output')
    im1.set_clim(0,3)
    axs[1].axis('off')
    # Plot bad_recon
    im2 = axs[2].imshow(bad_recon[0,0,:,:].detach().cpu().numpy(), cmap='gray')
    axs[2].set_title('FDK')
    im2.set_clim(0,3)
    axs[2].axis('off')
    # Plot target_reconstruction
    im3 = axs[3].imshow(target_reconstruction[0,0,:,:].detach().cpu().numpy(), cmap='gray')
    axs[3].set_title('Target Reconstruction')
    im3.set_clim(0,3)
    axs[3].axis('off')
    plt.show()
    plt.savefig(save_path)

#%%
def compute_ssim(out, target_reconstruction):
    out = out.detach().squeeze().cpu().numpy()
    target_reconstruction = target_reconstruction.detach().squeeze().cpu().numpy()

    ssim_scores = []
    for i in range(out.shape[0]):
        ssim_score = ssim(out[i], target_reconstruction[i], multichannel=False, data_range=out[i].max()-out[i].min())
        ssim_scores.append(ssim_score)

    return np.mean(ssim_scores)

def compute_psnr(img1, img2):
    img1 = img1.detach().squeeze().cpu().numpy()
    img2 = img2.detach().squeeze().cpu().numpy()
    return psnr(img1, img2, data_range=img1.max() - img1.min())
        
#%%
def save_ssim_psnr_values(clpd_ssim, lpd_ssim, fbp_ssim, fdk_ssim, clpd_psnr, lpd_psnr, fbp_psnr, fdk_psnr, filename):
    with open(filename, 'w') as f:
        f.write(f"CLPD SSIM: {clpd_ssim}, PSNR: {clpd_psnr}\n")
        f.write(f"LPD SSIM: {lpd_ssim}, PSNR: {lpd_psnr}\n")
        f.write(f"FBP SSIM: {fbp_ssim}, PSNR: {fbp_psnr}\n")
        f.write(f"FDK SSIM: {fdk_ssim}, PSNR: {fdk_psnr}\n")

def read_ssim_psnr_values(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        clpd_ssim, clpd_psnr = map(float, lines[0].split(":")[1].split(","))
        lpd_ssim, lpd_psnr = map(float, lines[1].split(":")[1].split(","))
        fbp_ssim, fbp_psnr = map(float, lines[2].split(":")[1].split(","))
        fdk_ssim, fdk_psnr = map(float, lines[3].split(":")[1].split(","))
    return clpd_ssim, lpd_ssim, fbp_ssim, fdk_ssim, clpd_psnr, lpd_psnr, fbp_psnr, fdk_psnr

#%%
def get_data_by_index(dataloader, index):
    for i, (sinogram, target_reconstruction) in enumerate(dataloader):
        if i == index:
            return sinogram, target_reconstruction
    raise IndexError("Index out of range of dataloader")

# %%
# plot results for index
def plot_results(index, save_path):
    with torch.no_grad():
        sinogram, target_reconstruction = get_data_by_index(dataloader, index)
        bad_recon = torch.zeros(target_reconstruction.shape, device=device)
        for sino in range(sinogram.shape[0]):
            bad_recon[sino] = fdk(dataset.operator, sinogram[sino])
        clpd_out = clpd_model(sinogram)
        #lpd_out = lpd_model(sinogram)
        fbp_out = fbp_model(bad_recon)
        plot_outputs(clpd_out, fbp_out, bad_recon, target_reconstruction, save_path=save_path)
#%%
if __name__ == "__main__":
    print("Starting evaluation.")
    # % Chose device:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    # Define your data paths
    savefolder = pathlib.Path("/store/DAMTP/cr661/LION/trained_models/low_dose")
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
    with torch.no_grad():
        clpd_model, clpd_param, clpd_data = ContinuousLPD.load(final_result_fname)
        lpd_model, lpd_param, lpd_data = LPD.load(final_result_fname_lpd)
        fbp_model, fbp_param, fbp_data = FBPConvNet.load(final_result_fname_fbp)
        clpd_model.eval()
        lpd_model.eval()
        fbp_model.eval()

        #%%

        clpd_ssim = 0.
        fbp_ssim = 0.
        lpd_ssim = 0.
        fdk_ssim = 0.
        
        clpd_psnr = 0.
        fbp_psnr = 0.
        lpd_psnr = 0.
        fdk_psnr = 0.

        # loop through testing data
        for index, (sinogram, target_reconstruction) in tqdm(enumerate(dataloader)):
            bad_recon = torch.zeros(target_reconstruction.shape, device=device)
            for sino in range(sinogram.shape[0]):
                bad_recon[sino] = fdk(dataset.operator, sinogram[sino])
            clpd_out = clpd_model(sinogram)
            lpd_out = lpd_model(sinogram)
            fbp_out = fbp_model(bad_recon)
            ssim_clpd = compute_ssim(clpd_out.detach().clone(), target_reconstruction)
            ssim_fbp = compute_ssim(fbp_out.detach().clone(), target_reconstruction)
            ssim_lpd = compute_ssim(lpd_out.detach().clone(), target_reconstruction)
            ssim_fdk = compute_ssim(bad_recon.detach().clone(), target_reconstruction)
            clpd_ssim += ssim_clpd
            fbp_ssim += ssim_fbp
            lpd_ssim += ssim_lpd
            fdk_ssim += ssim_fdk
            # Compute PSNR
            psnr_clpd = compute_psnr(clpd_out.detach().clone(), target_reconstruction)
            psnr_fbp = compute_psnr(fbp_out.detach().clone(), target_reconstruction)
            psnr_lpd = compute_psnr(lpd_out.detach().clone(), target_reconstruction)
            psnr_fdk = compute_psnr(bad_recon.detach().clone(), target_reconstruction)

            # Update PSNR sums
            clpd_psnr += psnr_clpd
            fbp_psnr += psnr_fbp
            lpd_psnr += psnr_lpd
            fdk_psnr += psnr_fdk
            print(f"CLPD SSIM: {ssim_clpd} - LPD SSIM: {ssim_lpd} - FBP SSIM: {ssim_fbp} - FDK SSIM: {ssim_fdk}")
            #plot_outputs(clpd_out, clpd_out, bad_recon, target_reconstruction)
            torch.cuda.empty_cache()

        clpd_ssim /= (index+1)
        fbp_ssim /= (index+1)
        lpd_ssim /= (index+1)
        fdk_ssim /= (index+1)

        clpd_psnr /= (index+1)
        fbp_psnr /= (index+1)
        lpd_psnr /= (index+1)
        fdk_psnr /= (index+1)
        save_ssim_psnr_values(clpd_ssim, lpd_ssim, fbp_ssim, fdk_ssim, clpd_psnr, lpd_psnr, fbp_psnr, fdk_psnr, savefolder.joinpath("ssim_psnr_values.txt"))

        #%%
        # Plot results for specific index
        index = 1
        plot_results(index=index, save_path=savefolder.joinpath(f"results_{index}.png"))





# %%
