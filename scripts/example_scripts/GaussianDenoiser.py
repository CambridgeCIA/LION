# %% This example shows how to train a denoising network with Gaussian noise.
# It will train a Gaussian denoiser on simulated Gaussian noise on the ground truth, rather than usin the noisy data from the data loader.
# This exploits the idea that a simple Gaussian denoiser can be extrapolated to other applications, such as CT reconstruction.
# Ideally you want to use this denoiser as a prox operator or a regularizer in an iterative reconstruction algorithm.

# %% Imports

# Standard imports
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
from LION.optimizers.GaussianDenoiserSolver import GaussianDenoiserSolver
from LION.optimizers.SupervisedSolver import SupervisedSolver

# %%
# % Chose device:
device = torch.device("cuda:0")
torch.cuda.set_device(device)

# Define your data paths
savefolder = pathlib.Path("/store/LION/ab2860/trained_models/LIDC-IDRI/PnP")
# Creates the folders if they does not exist
savefolder.mkdir(parents=True, exist_ok=True)
name = "DRUnet_supervised"
final_result_fname = f"{name}.pt"
checkpoint_fname = f"{name}_check_*.pt"
validation_fname = f"{name}_min_val.pt"
#
# %% Define experiment
experiment = ct_experiments.LowDoseCTRecon(dataset="LIDC-IDRI")
# %% Dataset
lidc_dataset = experiment.get_training_dataset()
lidc_dataset_val = experiment.get_validation_dataset()
lidc_dataset_test = experiment.get_testing_dataset()
# smaller dataset for example. Remove this for full dataset
indices = torch.arange(1000)
lidc_dataset = data_utils.Subset(lidc_dataset, indices)

# %% Define DataLoader
# Use the same amount of training


batch_size = 5
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
lidc_validation = DataLoader(lidc_dataset_val, batch_size, shuffle=False)
lidc_test = DataLoader(lidc_dataset_test, batch_size, shuffle=False)


# %% Model
# Default model is already from the paper.
from LION.models.CNNs.drunet import DRUNet

params = DRUNet.default_parameters()
params.use_noise_level = (
    False  # This is important for the denoiser to work with noise level input
)
model = DRUNet(params)
model.cite()

model.set_normalisation(normalisation_type="dataset", dataset=lidc_dataloader)
# %% Optimizer
train_param = LIONParameter()

# loss fn
loss_fcn = torch.nn.MSELoss()
train_param.optimiser = "adam"

# optimizer
train_param.epochs = 100
train_param.learning_rate = 1e-4
train_param.betas = (0.9, 0.99)
train_param.loss = "MSELoss"
optimiser = torch.optim.Adam(
    model.parameters(), lr=train_param.learning_rate, betas=train_param.betas
)

# %% Train

# Estimate the maximum noise level from one sample of the training data
sino, target = next(iter(lidc_dataloader))
from LION.classical_algorithms.fdk import fdk

recon = fdk(sino, op=experiment.geometry)
diff = recon - target
max_noise_std = diff.std().item()

# create solver
# solver = GaussianDenoiserSolver(
#     model,
#     optimiser,
#     loss_fcn,
#     geometry=experiment.geometry,
#     verbose=True,
#     save_folder=savefolder,
#     noise_level=torch.tensor([0.000, max_noise_std*0.5]),  # Set noise level for denoising
# )
# solver.set_patch_strategy(n_patches=5, patch_size=recon.shape[2]//4)
solver = SupervisedSolver(
    model,
    optimiser,
    loss_fcn,
    geometry=experiment.geometry,
    verbose=True,
    save_folder=savefolder,
)
# set data
solver.set_training(lidc_dataloader)
solver.set_validation(lidc_validation, 10, validation_fname=validation_fname)

# set checkpointing procedure
solver.set_checkpointing(
    checkpoint_fname, 10, load_checkpoint_if_exists=True, save_folder=savefolder
)
# train
solver.train(train_param.epochs)
# delete checkpoints if finished
# solver.clean_checkpoints()
# save final result
solver.save_final_results(final_result_fname, savefolder)

plt.figure()
plt.semilogy(solver.train_loss[1:])
plt.savefig("loss.png")


def my_ssim(x, y):
    x = x.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    return ssim(x, y, data_range=x.max() - x.min())


# Test
from LION.classical_algorithms.fdk import fdk

model = solver.get_model()

model, _, _ = DRUNet.load(savefolder / final_result_fname)
model.to(device)
model.eval()

from LION.reconstructors.PnP import PnP

reconstructor = PnP(experiment.geometry, model, algorithm="FBS")

with torch.no_grad():
    for i, (sino, target) in enumerate(lidc_dataloader):
        for noise_level in [0.001, 0.0001, 0.0000]:
            print(f"Testing with noise level: {noise_level}")
            sino = sino.to(device)
            target = target.to(device)
            recon = fdk(sino, op=experiment.geometry)
            # normalize recon
            recon = model.normalise(recon)
            denoised = model(recon, noise_level=noise_level)
            # unnormalize denoised and recon
            recon = model.unnormalise(recon)
            denoised = model.unnormalise(denoised)

            # PnP recon
            pnp = reconstructor.reconstruct(
                sino.to(device), noise_level=noise_level, max_iter=2
            )

            plt.figure()
            plt.subplot(1, 4, 1)
            plt.imshow(recon[0, 0].cpu(), cmap="gray")
            plt.clim(0, 2)
            plt.axis("off")
            plt.title(
                "Recon: SSIM = {:.2f}".format(my_ssim(target[0], recon[0])), fontsize=6
            )
            plt.subplot(1, 4, 2)
            plt.imshow(denoised[0, 0].cpu(), cmap="gray")
            plt.clim(0, 2)
            plt.axis("off")
            plt.title(
                "Denoised: SSIM = {:.2f}".format(my_ssim(target[0], denoised[0])),
                fontsize=6,
            )
            plt.subplot(1, 4, 3)
            plt.imshow(pnp[0, 0].cpu(), cmap="gray")
            plt.title(
                "PnP: SSIM = {:.2f}".format(my_ssim(target[0], pnp[0])), fontsize=6
            )
            plt.clim(0, 2)
            plt.axis("off")

            plt.subplot(1, 4, 4)
            plt.imshow(target[0, 0].cpu(), cmap="gray")
            plt.title("Ground Truth", fontsize=6)
            plt.axis("off")
            plt.clim(0, 2)
            plt.savefig(f"test_{i}_{noise_level}.png", bbox_inches="tight", dpi=300)
        exit()
