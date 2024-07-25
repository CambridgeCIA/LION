from matplotlib import pyplot as plt
import numpy as np
from LION.CTtools.ct_utils import make_operator
from LION.classical_algorithms.fdk import fdk
import LION.experiments.ct_experiments as ct_experiments
from LION.models.CNNs.iCTNet import iCTNet
from torch.utils.data import DataLoader, Subset
from torch.optim.adam import Adam
import torch.nn as nn
import torch
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

from LION.optimizers.supervised_learning import SupervisedSolver

def my_ssim(x: torch.Tensor, y: torch.Tensor):
    if x.shape[0]==1:
        x_np = x.detach().cpu().numpy().squeeze()
        y_np = y.detach().cpu().numpy().squeeze()
        return ssim(x_np, y_np, data_range=y_np.max() - y_np.min())
    else: 
        x_np = x.detach().cpu().numpy().squeeze()
        y_np = y.detach().cpu().numpy().squeeze()
        vals=[]
        for i in range(x.shape[0]):
            vals.append(ssim(x_np[i], y_np[i], data_range=y[i].max() - y[i].min()))
        return np.array(vals)

def my_psnr(x: torch.Tensor, y: torch.Tensor):
    if x.shape[0]==1:
        x_np = x.detach().cpu().numpy().squeeze()
        y_np = y.detach().cpu().numpy().squeeze()
        return psnr(x_np, y_np, data_range=y_np.max() - y_np.min())
    else: 
        x_np = x.detach().cpu().numpy().squeeze()
        y_np = y.detach().cpu().numpy().squeeze()
        vals=[]
        for i in range(x.shape[0]):
            vals.append(psnr(x_np[i], y_np[i], data_range=y_np[i].max() - y_np[i].min()))
        return np.array(vals)

device = torch.device("cuda:3")

experiment = ct_experiments.LimitedAngleCTRecon()

dataset = experiment.get_training_dataset()
dataset = Subset(dataset, [i for i in range(100)])
dataloader = DataLoader(dataset, 4, True)

dataset_val = experiment.get_validation_dataset()
dataset_val = Subset(dataset_val, [i for i in range(100)])
dataloader_val = DataLoader(dataset_val, 4, True)

dataset_test = experiment.get_testing_dataset()
dataset_test = Subset(dataset_test, [i for i in range(100)])
dataloader_test = DataLoader(dataset_test, 1, True)

model = iCTNet(experiment.geo).to(device)

optimizer = Adam(model.parameters())

solver = SupervisedSolver(model, optimizer, nn.MSELoss(), True, experiment.geo, None, device)
solver.set_saving("/store/DAMTP/cs2186/trained_models/test_debugging/", "iCTNet.pt")
solver.set_checkpointing("iCTNet_check_*.pt", 3)
solver.set_loading("/store/DAMTP/cs2186/trained_models/test_debugging", True)
solver.set_training(dataloader)
solver.set_validation(dataloader_val)
solver.set_testing(dataloader_test, ssim)

solver.train(3)

# plot loses
plt.figure()
plt.plot(solver.train_loss, 'r')
plt.plot(solver.validation_loss, 'b')
plt.yscale('log')
plt.legend()
plt.savefig('loss.png')
plt.close()

solver.save_final_results()
solver.clean_checkpoints()

ssims = solver.test()

solver.testing_fn = psnr
psnrs = solver.test()

with open("iCTNetresults.txt", "w") as f:
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

sino, gt = next(iter(solver.test_loader))
noisy_recon = fdk(sino, op)
bad_ssim = my_ssim(noisy_recon, gt)
bad_psnr = my_psnr(noisy_recon, gt)
good_recon = model(sino)
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
    plt.gca().set_title(f"N2I. SSIM: {good_ssim[i]:.2f}. PSNR: {good_psnr[i]:.2f}")
    # reconstruct filepath with suffix i
    plt.savefig(f'n2i2_test{i}walljs.png', dpi=700)
    plt.close()

plt.figure()
plt.semilogy(solver.train_loss[1:])
plt.savefig("loss.png")