from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pathlib
from LION.CTtools.ct_utils import make_operator
from LION.classical_algorithms.fdk import fdk
from LION.losses.SURE import SURE
from LION.models.iterative_unrolled.ItNet import UNet
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
from skimage.metrics import (
    structural_similarity as ssim,
    peak_signal_noise_ratio as psnr,
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
    x = x.cpu().detach().numpy().squeeze()
    y = y.cpu().detach().numpy().squeeze()
    vals = []
    if x.shape[0] == 1:
        return psnr(x, y, data_range=y.max() - y.min())
    for i in range(x.shape[0]):
        vals.append(psnr(x[i], y[i], data_range=y[i].max() - y[i].min()))
    return np.array(vals).mean()


#%%
# % Chose device:
device = torch.device("cuda:0")
torch.cuda.set_device(device)
savefolder = pathlib.Path("/store/DAMTP/cs2186/trained_models/test_debugging/")

final_result_fname = savefolder.joinpath("SURE_final.pt")
checkpoint_fname = savefolder.joinpath("SURE_check_*.pt")
validation_fname = savefolder.joinpath("SURE_min_val.pt")
#
experiment = ct_experiments.clinicalCTRecon()
op = make_operator(experiment.geo)

lidc_dataset = experiment.get_training_dataset()
lidc_dataset = Subset(lidc_dataset, range(200))

lidc_dataset_val = experiment.get_validation_dataset()
lidc_dataset_val = Subset(lidc_dataset_val, range(50))

lidc_dataset_test = experiment.get_testing_dataset()

batch_size = 2
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
lidc_validation = DataLoader(lidc_dataset_val, 1, shuffle=True)
lidc_testing = DataLoader(lidc_dataset_test, 1, shuffle=True)

model = UNet().to(device)

#%% Optimizer
train_param = LIONParameter()

# loss fn
sample_sino, sample_gt = next(iter(lidc_dataloader))
sample_noisy_recon = fdk(sample_sino, op)
std_estimate = torch.std(sample_gt - sample_noisy_recon)

loss_fcn = SURE(std_estimate.item())
mse = torch.nn.MSELoss()
train_param.optimiser = "adam"

# optimizer
train_param.epochs = 20
train_param.learning_rate = 1e-3
train_param.betas = (0.9, 0.99)
train_param.loss = "MSELoss"
optimiser = torch.optim.Adam(
    model.parameters(), lr=train_param.learning_rate, betas=train_param.betas
)

# learning parameter update
steps = len(lidc_dataloader)
model.train()
min_valid_loss = np.inf
total_loss = np.zeros(train_param.epochs)
start_epoch = 0

# %% Check if there is a checkpoint saved, and if so, start from there.

# If there is a file with the final results, don't run again
# if model.final_file_exists(savefolder.joinpath(final_result_fname)):
#    print("final model exists! You already reached final iter")
#    exit()

# model, optimiser, start_epoch, total_loss, _ = UNet.load_checkpoint_if_exists(
#     checkpoint_fname, model, optimiser, total_loss
# # )

total_loss = np.resize(total_loss, train_param.epochs)
print(f"Starting iteration at epoch {start_epoch}")

#%% train
for epoch in range(start_epoch, train_param.epochs):
    train_loss = 0.0
    for sinogram, _ in tqdm(lidc_dataloader):
        optimiser.zero_grad()
        bad_recon = fdk(sinogram, op)
        loss = loss_fcn(model, bad_recon)

        loss.backward()

        train_loss += loss.item()

        optimiser.step()
    total_loss[epoch] = train_loss
    # Validation
    valid_loss = 0.0
    model.eval()
    for sinogram, target_reconstruction in tqdm(lidc_validation):
        reconstruction = model(fdk(sinogram, op))
        loss = mse(target_reconstruction, reconstruction)
        valid_loss += loss.item()

    print(
        f"Epoch {epoch+1} \t\t Training Loss: {train_loss / (batch_size * len(lidc_dataloader))} \t\t Validation Loss: {valid_loss / (batch_size * len(lidc_validation))}"
    )

    if min_valid_loss > valid_loss:
        print(
            f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model"
        )
        min_valid_loss = valid_loss
        # Saving State Dict
        model.save(
            validation_fname,
            epoch=epoch + 1,
            training=train_param,
            loss=min_valid_loss,
            dataset=experiment.param,
        )

    # Checkpoint every 5 iters anyway
    if epoch % 5 == 0:
        model.save_checkpoint(
            pathlib.Path(str(checkpoint_fname).replace("*", f"{epoch+1:04d}")),
            epoch + 1,
            total_loss,
            optimiser,
            train_param,
            dataset=experiment.param,
        )

ssims = []
psnrs = []
printed = 0
with torch.no_grad():
    for sinogram, gt in tqdm(lidc_testing):

        sinogram = sinogram.to(device)
        gt = gt.to(device)
        bad_recon = fdk(sinogram, op)
        bad_mse = mse(bad_recon, gt)
        bad_ssim = my_ssim(bad_recon, gt)
        bad_psnr = my_psnr(bad_recon, gt)

        output = model(bad_recon)
        cur_mse = mse(output, gt)
        cur_ssim = my_ssim(output, gt)
        cur_psnr = my_psnr(output, gt)

        ssims = np.append(ssims, cur_ssim)
        if np.isfinite(cur_psnr):
            psnrs = np.append(psnrs, cur_psnr)

        if printed < 3:
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
            plt.gca().set_title("SURE")
            plt.text(0, 650, f"{cur_ssim:.2f}\n{cur_psnr:.2f}")
            plt.axis("off")
            # reconstruct filepath with suffix i
            plt.savefig(f"SURE{printed}.png", dpi=700)
            plt.close()

            printed += 1

    print(
        f"Average ssim: {np.mean(ssims)}, Max ssim: {np.max(ssims)}, Min ssim: {np.min(ssims)}"
    )
    print(
        f"Average psnr: {np.mean(psnrs)}, Max psnr: {np.max(psnrs)}, Min psnr: {np.min(psnrs)}"
    )

model.save(
    final_result_fname,
    epoch=train_param.epochs,
    training=train_param,
    dataset=experiment.param,
)
