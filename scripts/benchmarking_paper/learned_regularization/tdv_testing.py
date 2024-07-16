# Standard imports
import matplotlib.pyplot as plt
import pathlib
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Torch imports
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

# Lion imports
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_benchmarking_experiments as ct_benchmarking


# Just a temporary SSIM that takes troch tensors (will be added to LION at some point)
def my_ssim(x: torch.tensor, y: torch.tensor, data_range=None):
    x = x.detach().cpu().numpy().squeeze()
    y = y.detach().cpu().numpy().squeeze()
    if data_range is None:
        data_range = x.max() - x.min()
    elif type(data_range) == torch.Tensor:
        data_range = data_range.cpu().numpy().squeeze()
    return ssim(x, y, data_range=data_range)


def my_psnr(x: torch.tensor, y: torch.tensor, data_range=None):
    x = x.detach().cpu().numpy().squeeze()
    y = y.detach().cpu().numpy().squeeze()
    if data_range is None:
        data_range = x.max() - x.min()
    elif type(data_range) == torch.Tensor:
        data_range = data_range.cpu().numpy().squeeze()
    return psnr(x, y, data_range=data_range)


device = torch.device("cuda:2")
torch.cuda.set_device(device) ####IMPORTANT FOR A

savefolder = pathlib.Path("/store/DAMTP/zs334/LION/eval/")
# use min validation, or final result, whicever you prefer
model_name = "TDV_experiment_name_check_0014.pt"

# from LION.models.post_processing.FBPMSDNet import FBPMS_D
from LION.models.learned_regularizer.TDV import TDV


# model, options, data = ACR.load(savefolder.joinpath(model_name))
model = TDV.load(savefolder.joinpath(model_name))[0]

# model.to(device)
# model.load("/store/DAMTP/zs334/LION/ACR.pt")
print('Model loaded')

# model.eval()
# experiment = ct_benchmarking.FullDataCTRecon()
# Limited angle
# experiment = ct_benchmarking.LimitedAngle150CTRecon()
# experiment = ct_benchmarking.LimitedAngle120CTRecon()
# experiment = ct_benchmarking.LimitedAngle90CTRecon()
# experiment = ct_benchmarking.LimitedAngle60CTRecon()
# # Sparse angle
# experiment = ct_benchmarking.SparseAngle720CTRecon()
# experiment = ct_benchmarking.SparseAngle360CTRecon()
# experiment = ct_benchmarking.SparseAngle180CTRecon()
# experiment = ct_benchmarking.SparseAngle120CTRecon()
# experiment = ct_benchmarking.SparseAngle90CTRecon()
experiment = ct_benchmarking.SparseAngle60CTRecon()

# indices = torch.arange(50)
testing_data = experiment.get_testing_dataset()
# testing_data = data_utils.Subset(testing_data, indices)
testing_dataloader = DataLoader(testing_data, 1, shuffle=False)

# validation_data = experiment.get_validation_dataset()
# indices_val = torch.arange(50)
# validation_data = data_utils.Subset(validation_data, indices_val)
# validation_dataloader = DataLoader(validation_data, 1, shuffle=False)

print(f'Data loaded testing: {len(testing_dataloader)}')


# prepare models/algos
from ts_algorithms import fdk, sirt, tv_min, nag_ls


from LION.CTtools.ct_utils import make_operator

op = make_operator(experiment.geo)

# TEST 1: metrics
test_ssim = np.zeros(len(testing_dataloader))
test_psnr = np.zeros(len(testing_dataloader))

fdk_ssim = np.zeros(len(testing_dataloader))
fdk_psnr = np.zeros(len(testing_dataloader))

print(len(testing_dataloader))

for index, (data, target) in enumerate(testing_dataloader):
    # model
    print(index)
    output = model(data.to(device)).detach()
    # print(index, output, output.shape)
    test_ssim[index] = my_ssim(target, output)
    test_psnr[index] = my_psnr(target, output)
    # standard algo:
    recon = fdk(op, data[0].to(device))
    fdk_ssim[index] = my_ssim(target, recon)
    fdk_psnr[index] = my_psnr(target, recon)

print(f"Testing SSIM: {test_ssim.mean()} +- {test_ssim.std()}")
print(f"FDK SSIM: {fdk_ssim.mean()} +- {fdk_ssim.std()}")
print(f"Testing PSNR: {test_psnr.mean()} +- {test_psnr.std()}")
print(f"FDK PSNR: {fdk_psnr.mean()} +- {fdk_psnr.std()}")


# TEST 2: Display examples

min_idx = np.argmin(test_psnr)
max_idx = np.argmax(test_psnr)


def arg_find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# find the closest to the mean
mean_idx = arg_find_nearest(test_psnr, test_psnr.mean())


import matplotlib.pyplot as plt


# Display options
# This is the scale of the error plot w.r.t. the image DISPLAY. Not data, DISPLAY!
pctg_error = 0.05  # 5%
max_val = 0.010
########################################################
# Just plot code from now one
########################################################

# BEST PSNR
target = testing_data[max_idx][1]
data = testing_data[max_idx][0]
recon = fdk(op, data)
output = model(data.unsqueeze(0).to(device))

plt.figure()
plt.subplot(231)
plt.imshow(target.detach().cpu().numpy().squeeze(), cmap="gray")
plt.title("Ground truth")
plt.axis("off")
plt.clim(0, max_val)
plt.subplot(232)
plt.imshow(recon.detach().cpu().numpy().squeeze(), cmap="gray")
plt.title("FDK, PSNR: {:.2f}".format(fdk_psnr[max_idx]))
plt.axis("off")

plt.clim(0, max_val)
plt.subplot(233)
plt.imshow(output.detach().cpu().numpy().squeeze(), cmap="gray")
plt.title(f"{type(model).__name__}, PSNR: {test_psnr[max_idx]:.2f}")
plt.clim(0, max_val)
plt.axis("off")

plt.subplot(235)
plt.imshow(
    (recon.detach().cpu().numpy().squeeze() - target.detach().cpu().numpy().squeeze()),
    cmap="seismic",
)
plt.title(f"FDK - GT")
plt.axis("off")
cbar = plt.colorbar(fraction=0.046, pad=0.04)
plt.clim(-max_val * pctg_error, max_val * pctg_error)
cbar.set_ticks(
    [-max_val * pctg_error, 0, max_val * pctg_error],
    labels=[f"-{pctg_error*100:.2f}%", "0", f"{pctg_error*100:.2f}%"],
)
cbar.ax.tick_params(labelsize=5)

plt.subplot(236)
plt.imshow(
    (output.detach().cpu().numpy().squeeze() - target.detach().cpu().numpy().squeeze()),
    cmap="seismic",
)
plt.title(f"{type(model).__name__} - GT")
plt.axis("off")
cbar = plt.colorbar(fraction=0.046, pad=0.04)
plt.clim(-max_val * pctg_error, max_val * pctg_error)
cbar.set_ticks(
    [-max_val * pctg_error, 0, max_val * pctg_error],
    labels=[f"-{pctg_error*100:.2f}%", "0", f"{pctg_error*100:.2f}%"],
)
cbar.ax.tick_params(labelsize=5)

plt.suptitle("Best PSNR")
plt.savefig("eval_best_psnr_TDV.png", dpi=300)

del target, data, recon, output

# WORST PSNR

target = testing_data[min_idx][1]
data = testing_data[min_idx][0]
recon = fdk(op, data)
output = model(data.unsqueeze(0).to(device))

plt.figure()
plt.subplot(231)
plt.imshow(target.detach().cpu().numpy().squeeze(), cmap="gray")
plt.title("Ground truth")
plt.axis("off")
plt.clim(0, max_val)
plt.subplot(232)
plt.imshow(recon.detach().cpu().numpy().squeeze(), cmap="gray")
plt.title("FDK, PSNR: {:.2f}".format(fdk_psnr[min_idx]))
plt.axis("off")

plt.clim(0, max_val)
plt.subplot(233)
plt.imshow(output.detach().cpu().numpy().squeeze(), cmap="gray")
plt.title(f"{type(model).__name__}, PSNR: {test_psnr[min_idx]:.2f}")
plt.clim(0, max_val)
plt.axis("off")

plt.subplot(235)
plt.imshow(
    (recon.detach().cpu().numpy().squeeze() - target.detach().cpu().numpy().squeeze()),
    cmap="seismic",
)
plt.title(f"FDK - GT")
plt.axis("off")
cbar = plt.colorbar(fraction=0.046, pad=0.04)
plt.clim(-max_val * pctg_error, max_val * pctg_error)
cbar.set_ticks(
    [-max_val * pctg_error, 0, max_val * pctg_error],
    labels=[f"-{pctg_error*100:.2f}%", "0", f"{pctg_error*100:.2f}%"],
)
cbar.ax.tick_params(labelsize=5)

plt.subplot(236)
plt.imshow(
    (output.detach().cpu().numpy().squeeze() - target.detach().cpu().numpy().squeeze()),
    cmap="seismic",
)
plt.title(f"{type(model).__name__} - GT")
plt.axis("off")
cbar = plt.colorbar(fraction=0.046, pad=0.04)
plt.clim(-max_val * pctg_error, max_val * pctg_error)
cbar.set_ticks(
    [-max_val * pctg_error, 0, max_val * pctg_error],
    labels=[f"-{pctg_error*100:.2f}%", "0", f"{pctg_error*100:.2f}%"],
)
cbar.ax.tick_params(labelsize=5)

plt.suptitle("Worst PSNR")
plt.savefig("eval_worst_psnr_TDV.png", dpi=300)
del target, data, recon, output

# MEAN PSNR
target = testing_data[mean_idx][1]
data = testing_data[mean_idx][0]
recon = fdk(op, data)
output = model(data.unsqueeze(0).to(device))

plt.figure()
plt.subplot(231)
plt.imshow(target.detach().cpu().numpy().squeeze(), cmap="gray")
plt.title("Ground truth")
plt.axis("off")
plt.clim(0, max_val)
plt.subplot(232)
plt.imshow(recon.detach().cpu().numpy().squeeze(), cmap="gray")
plt.title("FDK, PSNR: {:.2f}".format(fdk_psnr[mean_idx]))
plt.axis("off")

plt.clim(0, max_val)
plt.subplot(233)
plt.imshow(output.detach().cpu().numpy().squeeze(), cmap="gray")
plt.title(f"{type(model).__name__}, PSNR: {test_psnr[mean_idx]:.2f}")
plt.clim(0, max_val)
plt.axis("off")

plt.subplot(235)
plt.imshow(
    (recon.detach().cpu().numpy().squeeze() - target.detach().cpu().numpy().squeeze()),
    cmap="seismic",
)
plt.title(f"FDK - GT")
plt.axis("off")
cbar = plt.colorbar(fraction=0.046, pad=0.04)
plt.clim(-max_val * pctg_error, max_val * pctg_error)
cbar.set_ticks(
    [-max_val * pctg_error, 0, max_val * pctg_error],
    labels=[f"-{pctg_error*100:.2f}%", "0", f"{pctg_error*100:.2f}%"],
)
cbar.ax.tick_params(labelsize=5)

plt.subplot(236)
plt.imshow(
    (output.detach().cpu().numpy().squeeze() - target.detach().cpu().numpy().squeeze()),
    cmap="seismic",
)
plt.title(f"{type(model).__name__} - GT")
plt.axis("off")
cbar = plt.colorbar(fraction=0.046, pad=0.04)
plt.clim(-max_val * pctg_error, max_val * pctg_error)
cbar.set_ticks(
    [-max_val * pctg_error, 0, max_val * pctg_error],
    labels=[f"-{pctg_error*100:.2f}%", "0", f"{pctg_error*100:.2f}%"],
)
cbar.ax.tick_params(labelsize=5)

plt.suptitle("Mean PSNR")
plt.savefig("eval_mean_psnr_TDV.png", dpi=300)
