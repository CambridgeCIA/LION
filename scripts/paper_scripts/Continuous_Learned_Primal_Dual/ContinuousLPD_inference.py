# %% Noise2Inverse train

# %% Imports
import argparse

# Basic science imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from LION.utils.utils import str2bool

# basic python imports
from tqdm import tqdm
import pathlib

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# LION imports
from LION.models.iterative_unrolled.cLPD import cLPD
from LION.models.iterative_unrolled.LPD import LPD
from LION.models.post_processing.FBPConvNet import FBPConvNet
import LION.experiments.ct_experiments as ct_experiments
from ts_algorithms import fdk


# %%
# arguments for argparser
parser = argparse.ArgumentParser()
parser.add_argument("--geometry", type=str)
parser.add_argument("--dose", type=str)
parser.add_argument("--compute_psnr_ssim", type=str2bool)
parser.add_argument("--plot_results", type=str2bool)
parser.add_argument("--instance_norm_clpd", type=str2bool)
parser.add_argument("--instance_norm_lpd", type=str2bool)
parser.add_argument("--save_outputs", type=str2bool)


# %%
def plot_outputs(clpd_out, lpd_out, bad_recon, target_reconstruction, save_path):
    fig, axs = plt.subplots(1, 4, figsize=(10, 5))

    # Plot lcpd_out
    im0 = axs[1].imshow(clpd_out[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
    axs[1].set_title("cLPD")
    im0.set_clim(0, 3)
    axs[1].axis("off")
    ## Plot lpd_out
    # im1 = axs[1].imshow(lpd_out[0,0,:,:].detach().cpu().numpy(), cmap='gray')
    # axs[1].set_title('LPD Output')
    # im1.set_clim(0,3)
    # Plot FBP out
    im1 = axs[2].imshow(lpd_out[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
    axs[2].set_title("LPD")
    im1.set_clim(0, 3)
    axs[2].axis("off")
    # Plot bad_recon
    im2 = axs[3].imshow(bad_recon[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
    axs[3].set_title("FBP")
    im2.set_clim(0, 3)
    axs[3].axis("off")
    # Plot target_reconstruction
    im3 = axs[0].imshow(
        target_reconstruction[0, 0, :, :].detach().cpu().numpy(), cmap="gray"
    )
    axs[0].set_title("Target")
    im3.set_clim(0, 3)
    axs[0].axis("off")
    plt.show()
    plt.savefig(save_path)


def save_outputs_pt(clpd_out, lpd_out, bad_recon, target_reconstruction, save_path):
    torch.save(clpd_out, save_path.with_suffix(".clpd.pt"))
    torch.save(lpd_out, save_path.with_suffix(".lpd.pt"))
    torch.save(bad_recon, save_path.with_suffix(".fdk.pt"))
    torch.save(target_reconstruction, save_path.with_suffix(".target.pt"))


# %%
def compute_ssim(out, target_reconstruction):
    out = out.detach().squeeze().cpu().numpy()
    target_reconstruction = target_reconstruction.detach().squeeze().cpu().numpy()

    ssim_scores = []
    for i in range(out.shape[0]):
        ssim_score = ssim(
            out[i],
            target_reconstruction[i],
            multichannel=False,
            data_range=target_reconstruction[i].max() - target_reconstruction[i].min(),
        )
        ssim_scores.append(ssim_score)

    return np.mean(ssim_scores)


def compute_psnr(img1, img2):
    img1 = img1.detach().squeeze().cpu().numpy()
    img2 = img2.detach().squeeze().cpu().numpy()
    return psnr(img1, img2, data_range=img2.max() - img2.min())


# %%
# def save_ssim_psnr_values_old(clpd_ssim, lpd_ssim, fbp_ssim, fdk_ssim, clpd_psnr, lpd_psnr, fbp_psnr, fdk_psnr, filename):
#    with open(filename, 'w') as f:
#        f.write(f"CLPD SSIM: {clpd_ssim}, PSNR: {clpd_psnr}\n")
#        f.write(f"LPD SSIM: {lpd_ssim}, PSNR: {lpd_psnr}\n")
#        f.write(f"FBP SSIM: {fbp_ssim}, PSNR: {fbp_psnr}\n")
#        f.write(f"FDK SSIM: {fdk_ssim}, PSNR: {fdk_psnr}\n")


def save_ssim_psnr_values(
    clpd_ssim, lpd_ssim, fdk_ssim, clpd_psnr, lpd_psnr, fdk_psnr, filename
):
    tensor_list = [clpd_ssim, lpd_ssim, fdk_ssim, clpd_psnr, lpd_psnr, fdk_psnr]
    torch.save(tensor_list, filename)


def read_ssim_psnr_values(filename):
    tensor_list = torch.load(filename)
    clpd_ssim, lpd_ssim, fdk_ssim, clpd_psnr, lpd_psnr, fdk_psnr = tensor_list
    return clpd_ssim, lpd_ssim, fdk_ssim, clpd_psnr, lpd_psnr, fdk_psnr


# def read_ssim_psnr_values_old(filename):
#    with open(filename, 'r') as f:
#        lines = f.readlines()
#       clpd_ssim, clpd_psnr = map(float, lines[0].split(":")[1].split(","))
#        lpd_ssim, lpd_psnr = map(float, lines[1].split(":")[1].split(","))
#        fbp_ssim, fbp_psnr = map(float, lines[2].split(":")[1].split(","))
#        fdk_ssim, fdk_psnr = map(float, lines[3].split(":")[1].split(","))
#    return clpd_ssim, lpd_ssim, fbp_ssim, fdk_ssim, clpd_psnr, lpd_psnr, fbp_psnr, fdk_psnr


# %%
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
        lpd_out = lpd_model(sinogram)
        # fbp_out = fbp_model(bad_recon)
        plot_outputs(
            clpd_out, lpd_out, bad_recon, target_reconstruction, save_path=save_path
        )


def save_imgs(index, save_path):
    with torch.no_grad():
        sinogram, target_reconstruction = get_data_by_index(dataloader, index)
        bad_recon = torch.zeros(target_reconstruction.shape, device=device)
        for sino in range(sinogram.shape[0]):
            bad_recon[sino] = fdk(dataset.operator, sinogram[sino])
        clpd_out = clpd_model(sinogram)
        lpd_out = lpd_model(sinogram)
        # fbp_out = fbp_model(bad_recon)
        save_outputs_pt(
            clpd_out, lpd_out, bad_recon, target_reconstruction, save_path=save_path
        )


# %%
if __name__ == "__main__":
    args = parser.parse_args()
    print("Starting evaluation.")
    # % Chose device:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    # Define your data paths
    if args.dose == "low":
        savefolder = pathlib.Path(
            "/home/cr661/rds/hpc-work/store/LION/trained_models/low_dose/"
        )
    elif args.dose == "extreme_low":
        savefolder = pathlib.Path(
            "/home/cr661/rds/hpc-work/store/LION/trained_models/extreme_low_dose/"
        )
    else:
        raise ValueError("Dose not recognised")
    datafolder = pathlib.Path("/home/cr661/rds/hpc-work/store/LION/data/LIDC-IDRI/")
    final_result_fname = savefolder.joinpath(
        f"ContinuousLPD_final_iterBS2smallLR_no_adjoint_{args.dose}_{args.geometry}.pt"
    )
    checkpoint_fname = savefolder.joinpath(
        f"ContinuousLPD_checkBS2smallLR_no_adjoint_{args.dose}_{args.geometry}*.pt"
    )

    final_result_fname_fbp = pathlib.Path(
        "/home/cr661/rds/hpc-work/store/LION/trained_models/low_dose/FBPConvNet_final_iter.pt"
    )
    final_result_fname_itnet = pathlib.Path(
        "/home/cr661/rds/hpc-work/store/LION/trained_models/low_dose/ItNet_Unet_final_iter.pt"
    )

    final_result_fname_lpd = savefolder.joinpath(
        f"LPD_final_iterBS2fixedsmallLR_no_in_{args.dose}_{args.geometry}.pt"
    )
    checkpoint_fname_lpd = savefolder.joinpath(
        f"LPD_checkBS2fixedsmallLR_no_in_{args.dose}_{args.geometry}*.pt"
    )

    # %% Define experiment
    if args.geometry == "full":
        if args.dose == "low":
            experiment = ct_experiments.LowDoseCTRecon(datafolder=datafolder)
        elif args.dose == "extreme_low":
            experiment = ct_experiments.ExtremeLowDoseCTRecon(datafolder=datafolder)
        else:
            raise ValueError("Dose not recognised")
    elif args.geometry == "limited_angle":
        if args.dose == "low":
            experiment = ct_experiments.LimitedAngleLowDoseCTRecon(
                datafolder=datafolder
            )
        elif args.dose == "extreme_low":
            experiment = ct_experiments.LimitedAngleExtremeLowDoseCTRecon(
                datafolder=datafolder
            )
        else:
            raise ValueError("Dose not recognised")
    elif args.geometry == "sparse_angle":
        if args.dose == "low":
            experiment = ct_experiments.SparseAngleLowDoseCTRecon(datafolder=datafolder)
        elif args.dose == "extreme_low":
            experiment = ct_experiments.SparseAngleExtremeLowDoseCTRecon(
                datafolder=datafolder
            )
        else:
            raise ValueError("Dose not recognised")
    else:
        raise ValueError("Geometry not recognised")

    # %% Dataset
    dataset = experiment.get_testing_dataset()
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size, shuffle=False)

    # %% Load model
    if args.compute_psnr_ssim:
        with torch.no_grad():
            clpd_model, clpd_param, clpd_data = cLPD.load(final_result_fname)
            lpd_model, lpd_param, lpd_data = LPD.load(final_result_fname_lpd)
            fbp_model, fbp_param, fbp_data = FBPConvNet.load(final_result_fname_fbp)
            clpd_model.eval()
            lpd_model.eval()
            fbp_model.eval()

            # %%

            clpd_ssim = []
            # fbp_ssim = 0.
            lpd_ssim = []
            fdk_ssim = []

            clpd_psnr = []
            # fbp_psnr = 0.
            lpd_psnr = []
            fdk_psnr = []

            # loop through testing data
            for index, (sinogram, target_reconstruction) in tqdm(enumerate(dataloader)):
                bad_recon = torch.zeros(target_reconstruction.shape, device=device)
                for sino in range(sinogram.shape[0]):
                    bad_recon[sino] = fdk(dataset.operator, sinogram[sino])
                clpd_out = clpd_model(sinogram)
                lpd_out = lpd_model(sinogram)
                # fbp_out = fbp_model(bad_recon)
                ssim_clpd = compute_ssim(
                    clpd_out.detach().clone(), target_reconstruction
                )
                # ssim_fbp = compute_ssim(fbp_out.detach().clone(), target_reconstruction)
                ssim_lpd = compute_ssim(lpd_out.detach().clone(), target_reconstruction)
                ssim_fdk = compute_ssim(
                    bad_recon.detach().clone(), target_reconstruction
                )
                clpd_ssim.append(ssim_clpd)
                # fbp_ssim += ssim_fbp
                lpd_ssim.append(ssim_lpd)
                fdk_ssim.append(ssim_fdk)
                # Compute PSNR
                psnr_clpd = compute_psnr(
                    clpd_out.detach().clone(), target_reconstruction
                )
                # psnr_fbp = compute_psnr(fbp_out.detach().clone(), target_reconstruction)
                psnr_lpd = compute_psnr(lpd_out.detach().clone(), target_reconstruction)
                psnr_fdk = compute_psnr(
                    bad_recon.detach().clone(), target_reconstruction
                )

                # Update PSNR sums
                clpd_psnr.append(psnr_clpd)
                # fbp_psnr += psnr_fbp
                lpd_psnr.append(psnr_lpd)
                fdk_psnr.append(psnr_fdk)
                print(
                    f"CLPD SSIM: {ssim_clpd} - LPD SSIM: {ssim_lpd} - FDK SSIM: {ssim_fdk}"
                )
                print(
                    f"CLPD PSNR: {psnr_clpd} - LPD PSNR: {psnr_lpd} - FDK PSNR: {psnr_fdk}"
                )
                # plot_outputs(clpd_out, clpd_out, bad_recon, target_reconstruction)
                torch.cuda.empty_cache()

            clpd_ssim = torch.tensor(clpd_ssim)
            lpd_ssim = torch.tensor(lpd_ssim)
            fdk_ssim = torch.tensor(fdk_ssim)
            clpd_psnr = torch.tensor(clpd_psnr)
            lpd_psnr = torch.tensor(lpd_psnr)
            fdk_psnr = torch.tensor(fdk_psnr)

            save_ssim_psnr_values(
                clpd_ssim=clpd_ssim,
                lpd_ssim=lpd_ssim,
                fdk_ssim=fdk_ssim,
                clpd_psnr=clpd_psnr,
                lpd_psnr=lpd_psnr,
                fdk_psnr=fdk_psnr,
                filename=savefolder.joinpath(
                    f"ssim_psnr_values_{args.dose}_{args.geometry}_cLPDIN_LPD_corrected.pt"
                ),
            )

    # %%
    # Plot results for specific index
    if args.plot_results:
        for index in range(10):
            plot_results(
                index=index,
                save_path=savefolder.joinpath(
                    f"results_{args.dose}_{args.geometry}_cLPD_LPD_{index}.png"
                ),
            )

    if args.save_outputs:
        for index in range(100):
            save_imgs(index=index, save_path=savefolder.joinpath(f"output_{index}.pt"))
