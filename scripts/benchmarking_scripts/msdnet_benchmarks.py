# %% This example shows how to train MSDNet for full angle, noisy measurements.


# %% Imports
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pathlib
from LION.CTtools.ct_utils import make_operator
from LION.classical_algorithms.fdk import fdk
from LION.models.CNNs.MSDNets.MSDNet import MSD_Params, MSDNet
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
from skimage.metrics import (
    structural_similarity as ssim,
    peak_signal_noise_ratio as psnr,
    mean_squared_error as mse,
)


def my_ssim(x: torch.Tensor, y: torch.Tensor):
    x = x.detach().cpu().numpy().squeeze()
    y = y.detach().cpu().numpy().squeeze()
    vals = []
    if x.shape[0] == 1:
        return ssim(x, y, data_range=y.max() - y.min())
    for i in range(x.shape[0]):
        vals.append(ssim(x[i], y[i], data_range=y[i].max() - y[i].min()))
    return np.array(vals).mean()


def my_psnr(x: torch.Tensor, y: torch.Tensor):
    x = x.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    vals = []
    if x.shape[0] == 1:
        return psnr(x, y, data_range=y.max() - y.min())
    for i in range(x.shape[0]):
        vals.append(psnr(x[i], y[i], data_range=y[i].max() - y[i].min()))
    return np.array(vals).mean()


# %%
# % Chose device:
device = torch.device("cuda:3")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/cs2186/trained_models/clinical_dose/")
torch.manual_seed(0)

#
# %% Define experiment

experiments = [
    ct_experiments.clinicalCTRecon(),
    ct_experiments.ExtremeLowDoseCTRecon(),
    ct_experiments.LimitedAngleCTRecon(),
    ct_experiments.SparseAngleCTRecon(),
]
f = open("msd_benchmarks.txt", "w")

for experiment in experiments:
    experiment_str = str(type(experiment)).split("ct_experiments.")[1][:-2]
    print(experiment_str)
    f.write(experiment_str + "\n")
    op = make_operator(experiment.geo)
    # %% Dataset
    lidc_dataset = experiment.get_training_dataset()
    lidc_dataset_val = experiment.get_validation_dataset()
    lidc_dataset = Subset(lidc_dataset, range(250))
    lidc_dataset_val = Subset(lidc_dataset_val, range(250))
    lidc_dataset_test = experiment.get_testing_dataset()
    lidc_dataset_test = Subset(lidc_dataset_test, range(10))

    # %% Define DataLoader
    # Use the same amount of training
    batch_size = 2
    lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
    lidc_validation = DataLoader(lidc_dataset_val, 1, shuffle=True)
    lidc_testing = DataLoader(lidc_dataset_test, 1, shuffle=True)
    # %% Model
    width, depth = 1, 100
    dilations = []
    for i in range(depth):
        for j in range(width):
            dilations.append((((i * width) + j) % 10) + 1)
    model_params = MSDParams(
        in_channels=1,
        width=width,
        depth=depth,
        dilations=dilations,
        look_back_depth=-1,
        final_look_back_depth=-1,
        activation="ReLU",
    )
    # model = MSDNet(model_parameters=model_params).to(device)
    model, data, options = MSDNet.load(f"MSD{experiment_str}.pt")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model has {count_parameters(model)} parameters")

    # %% Optimizer
    train_param = LIONParameter()

    # loss fn
    loss_fcn = torch.nn.MSELoss()
    train_param.optimiser = "adam"

    # optimizer
    train_param.epochs = 20
    train_param.learning_rate = 1e-3
    train_param.betas = (0.9, 0.99)
    train_param.loss = "MSELoss"
    train_param.accumulation_steps = 1
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=train_param.learning_rate,
        betas=train_param.betas,
    )

    # learning parameter update
    steps = len(lidc_dataloader)
    model.train()
    min_valid_loss = np.inf
    total_loss = np.zeros(train_param.epochs)
    start_epoch = 0

    total_loss = np.resize(total_loss, train_param.epochs)
    print(f"Starting iteration at epoch {start_epoch}")
    optimiser.zero_grad()

    start_time = time.time()
    total_train_time = 0
    total_validation_time = 0

    # %% train
    # for epoch in range(start_epoch, train_param.epochs):
    #     train_loss = 0.0
    #     print("Training...")
    #     model.train()
    #     for idx, (sinogram, target_reconstruction) in enumerate(tqdm(lidc_dataloader)):
    #         output = model(fdk(sinogram, op))
    #         optimiser.zero_grad()
    #         sinogram = sinogram.to(device)
    #         target_reconstruction = target_reconstruction.to(device)

    #         reconstruction = model(fdk(sinogram, op))
    #         loss = loss_fcn(reconstruction, target_reconstruction)
    #         loss = loss / train_param.accumulation_steps

    #         loss.backward()

    #         train_loss += loss.item()

    #         optimiser.step()

    #     total_loss[epoch] = train_loss
    #     total_train_time += time.time() - start_time

    #     # Validation
    #     valid_loss = 0.0
    #     model.eval()
    #     start_time = time.time()
    #     with torch.no_grad():
    #         print("Validating...")
    #         for sinogram, target_reconstruction in tqdm(lidc_validation):
    #             sinogram = sinogram.to(device)
    #             target_reconstruction.to(device)
    #             reconstruction = model(fdk(sinogram, op))
    #             loss = loss_fcn(target_reconstruction, reconstruction.to(device))
    #             valid_loss += loss.item()

    #     print(
    #         f"Epoch {epoch+1} \t\t Training Loss: {train_loss / len(lidc_dataset)} \t\t Validation Loss: {valid_loss / len(lidc_dataset_val)}"
    #     )
    #     if valid_loss < min_valid_loss:
    #         min_valid_loss = valid_loss
    #     total_validation_time += time.time() - start_time

    # f.write("------Training-----\n-")
    # f.write(f"Model has {count_parameters(model)} trainable parameters\n")
    # f.write(
    #     f"Model took {total_train_time/train_param.epochs}s /epoch to train on average\n"
    # )
    # f.write(f"Model achieved minimum training loss of {min(total_loss)}\n")
    # f.write(
    #     f"Model took {total_validation_time/train_param.epochs}s /epoch to validate on average\n"
    # )
    # f.write(f"Model achieved validation loss of {min_valid_loss}\n")

    # model.save(
    #     f"MSD{experiment_str}",
    #     epoch=train_param.epochs,
    #     training=train_param,
    #     dataset=experiment.param,
    #     geometry=experiment.geo,
    # )

    with torch.no_grad():
        model.eval()
        mses = np.array([])
        ssims = np.array([])
        psnrs = np.array([])
        for i, (sinogram, gt) in enumerate(tqdm(lidc_testing)):
            sinogram = sinogram.to(device)
            gt = gt.to(device)
            bad_recon = fdk(sinogram, op)
            bad_mse = mse(bad_recon.cpu().numpy(), gt.cpu().numpy())
            bad_ssim = my_ssim(bad_recon, gt)
            bad_psnr = my_psnr(bad_recon, gt)

            output = model(bad_recon)
            cur_mse = mse(output.cpu().numpy(), gt.cpu().numpy())
            cur_ssim = my_ssim(output, gt)
            cur_psnr = my_psnr(output, gt)

            mses = np.append(mses, cur_mse)
            ssims = np.append(ssims, cur_ssim)
            psnrs = np.append(psnrs, cur_psnr)

            plt.figure()
            plt.subplot(131)
            plt.imshow(gt[0].detach().cpu().numpy().T)
            plt.clim(torch.min(gt[0]).item(), torch.max(gt[0]).item())
            plt.gca().set_title("Ground Truth")
            plt.axis("off")
            plt.subplot(132)
            # should cap max / min of plots to actual max / min of gt
            plt.imshow(bad_recon[0].detach().cpu().numpy().T)
            plt.clim(torch.min(gt[0]).item(), torch.max(gt[0]).item())
            plt.gca().set_title("FBP")
            plt.text(0, 650, f"{bad_ssim:.2f}\n{bad_psnr:.2f}")
            plt.axis("off")
            plt.subplot(133)
            plt.imshow(output[0].detach().cpu().numpy().T)
            plt.clim(torch.min(gt[0]).item(), torch.max(gt[0]).item())
            plt.gca().set_title("MSDNet")
            plt.text(0, 650, f"{cur_ssim:.2f}\n{cur_psnr:.2f}")
            plt.axis("off")
            # reconstruct filepath with suffix i
            plt.savefig(f"msdnet{i}.png", dpi=700)
            plt.close()

        print(mses)
        print(ssims)
        print(psnrs)

        mses = mses[np.isfinite(mses)]
        ssims = ssims[np.isfinite(ssims)]
        psnrs = psnrs[np.isfinite(psnrs)]

        f.write("------Testing-------\n")
        f.write(f"Average mse: {np.mean(mses)}\n")
        f.write(f"Max mse: {max(mses)}\n")
        f.write(f"Min mse: {min(mses)}\n")
        f.write(f"Average psnr: {np.mean(psnrs)}\n")
        f.write(f"Max psnr: {max(psnrs)}\n")
        f.write(f"Min psnr: {min(psnrs)}\n")
        f.write(f"Average ssim: {np.mean(ssims)}\n")
        f.write(f"Max ssim: {max(ssims)}\n")
        f.write(f"Min ssim: {min(ssims)}\n")


f.close()
