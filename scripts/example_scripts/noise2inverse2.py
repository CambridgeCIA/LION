from dataclasses import dataclass
from typing import Tuple
import matplotlib.pyplot as plt
import pathlib
import torch
from torch.utils.data import DataLoader, Subset
from LION.CTtools.ct_utils import make_operator
from LION.models.LIONmodel import ModelInputType
from LION.models.iterative_unrolled.ItNet import UNet
from LION.models.post_processing.FBPConvNet import FBPConvNet
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
from LION.optimizers.Noise2Inverse_solver2 import Noise2InverseSolver
from LION.optimizers.Noise2Inverse_solver import Noise2Inverse_solver
from skimage.metrics import (
    structural_similarity as ssim,
    peak_signal_noise_ratio as psnr,
)
import numpy as np


def my_ssim(x, y):
    if x.shape[0] == 1:
        x = x.detach().cpu().numpy().squeeze()
        y = y.detach().cpu().numpy().squeeze()
        return ssim(x, y, data_range=x.max() - x.min())
    else:
        x = x.detach().cpu().numpy().squeeze()
        y = y.detach().cpu().numpy().squeeze()
        vals = []
        for i in range(x.shape[0]):
            vals.append(ssim(x[i], y[i], data_range=x[i].max() - x[i].min()))
        return np.array(vals)


def my_psnr(x, y):
    if x.shape[0] == 1:
        x = x.detach().cpu().numpy().squeeze()
        y = y.detach().cpu().numpy().squeeze()
        return psnr(x, y, data_range=x.max() - x.min())
    else:
        x = x.detach().cpu().numpy().squeeze()
        y = y.detach().cpu().numpy().squeeze()
        vals = []
        for i in range(x.shape[0]):
            vals.append(psnr(x[i], y[i], data_range=x[i].max() - x[i].min()))
        return np.array(vals)


# %%
# % Chose device:
device = torch.device("cuda:2")
torch.cuda.set_device(device)
print(torch.cuda.current_device())
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/cs2186/trained_models/test_debugging/")
final_result_fname = "Noise2Inverse_MSD.pt"
checkpoint_fname = "Noise2Inverse_MSD_check_*.pt"
validation_fname = "Noise2Inverse_MSD_min_val.pt"
#
# %% Define experiment

experiment = ct_experiments.clinicalCTRecon(dataset="LIDC-IDRI")

# %% Dataset
lidc_dataset = experiment.get_training_dataset()

# smaller dataset for example. Remove this for full dataset
# lidc_dataset = Subset(lidc_dataset, [i for i in range(len(lidc_dataset) // 2)])
lidc_dataset = Subset(lidc_dataset, [i for i in range(300)])


# %% Define DataLoader

batch_size = 4
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
lidc_test = DataLoader(experiment.get_testing_dataset(), batch_size, shuffle=True)

# %% Model
# Default model is already from the paper.
model_params = UNet.default_parameters()
model_params.model_input_type = ModelInputType.IMAGE
model = UNet(model_parameters=model_params).to(device)

# %% Optimizer
@dataclass
class TrainParams(LIONParameter):
    optimizer: str
    epochs: int
    learning_rate: float
    betas: Tuple[float, float]
    loss: str


train_param = TrainParams("adam", 25, 1e-4, (0.9, 0.99), "MSELoss")

# loss fn
loss_fn = torch.nn.MSELoss()

optimiser = torch.optim.Adam(
    model.parameters(), lr=train_param.learning_rate, betas=train_param.betas
)

# %% Train
# create solver
noise2inverse_parameters = Noise2InverseSolver.default_parameters()
solver = Noise2InverseSolver(
    model, optimiser, loss_fn, noise2inverse_parameters, True, experiment.geo, device
)

# set data
solver.set_training(lidc_dataloader)
# solver.set_normalization(True)
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
# trained_model, options, data = MS_D.load(savefolder.joinpath(final_result_fname))
# solver.model = trained_model

# print("-"*20)
# print(data)

# loss_epoch = data.get('loss')
plt.plot(solver.train_loss[1:])
plt.yscale("log")
plt.savefig("loss.png")

# quit()


with open("n2i2results.txt", "w") as f:
    # test
    # test with ssim
    solver.testing_fn = my_ssim
    ssims = solver.test()
    f.write(f"Min ssim: {np.min(ssims)}")
    f.write(f"Min ssim: {np.min(ssims)}")
    f.write(f"Mean ssim: {np.mean(ssims)}\n")
    f.write(f"std ssim: {np.std(ssims)}\n")

    # test with psnr
    solver.testing_fn = my_psnr
    psnrs = solver.test()
    f.write(f"Max psnrs: {np.max(psnrs)}")
    f.write(f"Min psnrs: {np.min(psnrs)}")
    f.write(f"Mean psnrs: {np.mean(psnrs)}\n")
    f.write(f"std psnrs: {np.std(psnrs)}\n")

# batch worth of visualisations
op = make_operator(experiment.geo)

sino, gt = next(iter(lidc_test))
noisy_recon = solver.recon_fn(sino, op)
bad_ssim = my_ssim(noisy_recon, gt)
bad_psnr = my_psnr(noisy_recon, gt)
good_recon = solver.process(sino)
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
    plt.gca().set_title(f"FBP. {bad_ssim[i]:.2f}, {bad_psnr[i]:.2f}")
    plt.subplot(133)
    plt.imshow(good_recon[i].detach().cpu().numpy().T)
    plt.clim(torch.min(gt[i]).item(), torch.max(gt[i]).item())
    plt.gca().set_title(f"N2I. {good_ssim[i]:.2f}, {good_psnr[i]:.2f}")
    # reconstruct filepath with suffix i
    plt.savefig(f"n2i2_test{i}walljs.png", dpi=700)
    plt.close()

plt.figure()
plt.semilogy(solver.train_loss[1:])
plt.savefig("loss.png")
