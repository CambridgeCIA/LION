from dataclasses import dataclass
from typing import Tuple
import matplotlib.pyplot as plt
import pathlib
import torch
from torch.utils.data import DataLoader, Subset
from LION.CTtools.ct_utils import make_operator
from LION.classical_algorithms.fdk import fdk
from LION.models.LIONmodel import ModelInputType
from LION.models.iterative_unrolled.ItNet import UNet
from LION.optimizers.EquivariantSolver import EquivariantSolver
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
from skimage.metrics import (
    structural_similarity as ssim,
    peak_signal_noise_ratio as psnr,
)
import numpy as np


def my_ssim(x: torch.Tensor, y: torch.Tensor):
    if x.shape[0] == 1:
        x = x.detach().cpu().numpy().squeeze()
        y = y.detach().cpu().numpy().squeeze()
        return np.array([ssim(x, y, data_range=x.max() - x.min())])
    else:
        x = x.detach().cpu().numpy().squeeze()
        y = y.detach().cpu().numpy().squeeze()
        vals = []
        for i in range(x.shape[0]):
            vals.append(ssim(x[i], y[i], data_range=x[i].max() - x[i].min()))
        return np.array(vals)


def my_psnr(x: torch.Tensor, y: torch.Tensor):
    if x.shape[0] == 1:
        x = x.detach().cpu().numpy().squeeze()
        y = y.detach().cpu().numpy().squeeze()
        return np.array([psnr(x, y, data_range=x.max() - x.min())])
    else:
        x = x.detach().cpu().numpy().squeeze()
        y = y.detach().cpu().numpy().squeeze()
        vals = []
        for i in range(x.shape[0]):
            vals.append(psnr(x[i], y[i], data_range=x[i].max() - x[i].min()))
        return np.array(vals)


# %%
# % Chose device:
device = torch.device("cuda:0")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/cs2186/trained_models/test_debugging/")
final_result_fname = "Equivariance.pt"
checkpoint_fname = "Equivariance_check_*.pt"
validation_fname = "Equivariance_min_val.pt"
#
# %% Define experiment

experiment = ct_experiments.LowDoseCTRecon(dataset="LIDC-IDRI")

# %% Dataset
lidc_dataset = experiment.get_training_dataset()

# smaller dataset for example. Remove this for full dataset
# lidc_dataset = Subset(lidc_dataset, [i for i in range(len(lidc_dataset) // 2)])
lidc_dataset = Subset(lidc_dataset, [i for i in range(100)])


# %% Define DataLoader

batch_size = 2
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
lidc_test = DataLoader(experiment.get_testing_dataset(), 3, True)
lidc_val = DataLoader(experiment.get_validation_dataset(), 3, True)

# %% Model
model = UNet().to(device)
model.model_parameters.model_input_type = ModelInputType.IMAGE

# %% Optimizer
@dataclass
class TrainParams(LIONParameter):
    optimizer: str
    epochs: int
    learning_rate: float
    betas: Tuple[float, float]
    loss: str


train_param = TrainParams("adam", 20, 1e-4, (0.9, 0.99), "MSELoss")

# loss fn
loss_fn = torch.nn.MSELoss()

optimiser = torch.optim.Adam(
    model.parameters(), lr=train_param.learning_rate, betas=train_param.betas
)

# %% Train
# create solver
solver = EquivariantSolver(
    model,
    optimiser,
    loss_fn,
    experiment.geo,
    False,
    device=device,
)

# set data
solver.set_training(lidc_dataloader)
solver.set_validation(lidc_val, 1, my_ssim, validation_fname)
solver.set_normalization(True)
solver.set_testing(lidc_test, my_ssim)

# set checkpointing procedure
solver.set_saving(savefolder, final_result_fname)
solver.set_checkpointing(checkpoint_fname, 10)
solver.set_loading(savefolder, True)

# train
solver.train(train_param.epochs)
# delete checkpoints if finished
solver.clean_checkpoints()
# save final result
solver.save_final_results(final_result_fname)

# for after you've trained the model.
# trained_model, options, data = UNet.load(savefolder.joinpath(final_result_fname))
# solver.model = trained_model

# print("-"*20)
# print(data)

# loss_epoch = data.get('loss')
# plt.plot(loss_epoch)
# plt.yscale('log')
# plt.savefig('equivarianceloss.png')

# quit()

with open("equivarianceresults.txt", "w") as f:
    # test
    # test with ssim
    solver.testing_fn = my_ssim
    ssims = solver.test()
    f.write(f"Mean ssim: {np.mean(ssims)}\n")
    f.write(f"std ssim: {np.std(ssims)}\n")

    # test with psnr
    solver.testing_fn = my_psnr
    psnrs = solver.test()
    f.write(f"Mean psnrs: {np.mean(psnrs)}\n")
    f.write(f"std psnrs: {np.std(psnrs)}\n")

# batch worth of visualisations
op = make_operator(experiment.geo)

sino, gt = next(iter(lidc_test))
noisy_recon = fdk(sino, op)
bad_ssim = my_ssim(noisy_recon, gt)
bad_psnr = my_psnr(noisy_recon, gt)
good_recon = solver.model(noisy_recon)
good_ssim = my_ssim(good_recon, gt)
good_psnr = my_psnr(good_recon, gt)

for i in range(len(good_recon)):
    plt.figure()
    plt.subplot(131)
    plt.imshow(gt[i].detach().cpu().numpy().T)
    plt.clim(torch.min(gt[i]).item(), torch.max(gt[i]).item())
    plt.gca().set_title("Ground Truth")
    plt.subplot(132)
    # should cap max / min of plots to actual max / min of gt
    plt.imshow(noisy_recon[i].detach().cpu().numpy().T)
    plt.clim(torch.min(gt[i]).item(), torch.max(gt[i]).item())
    plt.gca().set_title(f"FBP. SSIM: {bad_ssim[i]:.2f}. PSNR: {bad_psnr[i]:.2f}")
    plt.subplot(133)
    plt.imshow(good_recon[i].detach().cpu().numpy().T)
    plt.clim(torch.min(gt[i]).item(), torch.max(gt[i]).item())
    plt.gca().set_title(f"EI. SSIM: {good_ssim[i]:.2f}. PSNR: {good_psnr[i]:.2f}")
    # reconstruct filepath with suffix i
    plt.savefig(f"equivariance_test{i}.png", dpi=700)
    plt.close()

plt.figure()
plt.semilogy(solver.train_loss[1:])
plt.savefig("loss.png")
