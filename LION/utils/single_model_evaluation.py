import matplotlib.pyplot as plt
import pathlib
from skimage.metrics import structural_similarity as ssim

# Torch imports
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

# Lion imports
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
from LION.optimizers.SupervisedSolver import SupervisedSolver
from LION.classical_algorithms.fdk import fdk

# Model imports
from DeepFBP import DeepFBPNetwork

device = torch.device("cuda:2")
torch.cuda.set_device(device)

experiment = ct_experiments.LowDoseCTRecon(dataset="LIDC-IDRI")
lidc_dataset_test = experiment.get_testing_dataset()
lidc_test = DataLoader(lidc_dataset_test, 1, shuffle=False)

model, _, _= DeepFBPNetwork.load('/store/LION/gq217/trained_models/test_debugging/DeepFBP_test.pt')
model.to(device)
model.eval()


def my_ssim(x, y):
    x = x.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    return ssim(x, y, data_range=x.max() - x.min())
    
with torch.no_grad():
    for i, (sino, target) in enumerate(lidc_test):
        sino = sino.to(device)
        target = target.to(device)
        recon = fdk(sino,op=experiment.geometry)
        denoised = model(sino)
        

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(recon[0, 0].cpu(), cmap="gray")
        plt.clim(0, 2)
        plt.axis("off")
        plt.title("Recon: SSIM = {:.2f}".format(my_ssim(target[0], recon[0])), fontsize=6)
        print(ssim.__code__.co_varnames)
        plt.subplot(1, 3, 2)
        plt.imshow(denoised[0, 0].cpu(), cmap="gray")
        plt.clim(0, 2)
        plt.axis("off")
        plt.title("Denoised: SSIM = {:.2f}".format(my_ssim(target[0], denoised[0])), fontsize=6)

        plt.subplot(1, 3, 3)
        plt.imshow(target[0, 0].cpu(), cmap="gray")
        plt.title("Ground Truth", fontsize=6)
        plt.axis("off")
        plt.clim(0, 2)
        plt.savefig(f"test_{i}.png", bbox_inches="tight", dpi=300)
        exit()