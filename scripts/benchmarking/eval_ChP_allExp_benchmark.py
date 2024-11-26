# Standard imports
import matplotlib.pyplot as plt
import pathlib
import imageio
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
from ts_algorithms import fdk, sirt, tv_min, nag_ls
from ts_algorithms.tv_min import tv_min2d
from LION.CTtools.ct_utils import make_operator

from tqdm import tqdm


# Just a temporary SSIM that takes troch tensors (will be added to LION at some point)
def my_ssim(x: torch.tensor, y: torch.tensor, data_range=None):
    x = x.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    if data_range is None:
        data_range = x.max() - x.min()
    elif type(data_range) == torch.Tensor:
        data_range = data_range.cpu().numpy().squeeze()
    return ssim(x, y, data_range=data_range)


def my_psnr(x: torch.tensor, y: torch.tensor, data_range=None):
    x = x.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    if data_range is None:
        data_range = x.max() - x.min()
    elif type(data_range) == torch.Tensor:
        data_range = data_range.cpu().numpy().squeeze()
    return psnr(x, y, data_range=data_range)

def arg_find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

mean_psnr_indcs = []
mean_ssim_indcs = []

#reg_params_1 = [1e-10,1e-6,1e-4,1e-4,1e-5,1e-5,1e-5,1e-5,1e-7]
#reg_params_2 = [1e-11,1e-7,1e-5,1e-8,1e-6,1e-6,1e-6,1e-6,1e-8]
#reg_params_3 = [1e-12,1e-8,1e-7,1e-9,1e-7,1e-7,1e-7,1e-7,1e-9]
reg_params_f = [1e-10,1e-8,1e-7,1e-8,1e-6,5e-6,5e-6,5e-6,1e-8]

#indices = [72, 161, 312]
#indices = [87, 182, 347]

device = torch.device("cuda:0")

savefolder = pathlib.Path("/export/scratch3/mbk/LION/bm_models/")
savenames = ["/export/scratch3/mbk/LION/ChP_results/ChP_Full Data", "/export/scratch3/mbk/LION/ChP_results/ChP_Limited 120",
"/export/scratch3/mbk/LION/ChP_results/ChP_Limited 90", "/export/scratch3/mbk/LION/ChP_results/ChP_Limited 60",
"/export/scratch3/mbk/LION/ChP_results/ChP_Sparse 360", "/export/scratch3/mbk/LION/ChP_results/ChP_Sparse 120",
"/export/scratch3/mbk/LION/ChP_results/ChP_Sparse 60", "/export/scratch3/mbk/LION/ChP_results/ChP_Low Dose",
"/export/scratch3/mbk/LION/ChP_results/ChP_Beam Hardening"]
# use min validation, or final result, whichever you prefer

# Define all experiments in experiments list
experiments = []

# Standard dataset
experiments.append(ct_benchmarking.FullDataCTRecon())

# Limited angle
experiments.append(ct_benchmarking.LimitedAngle120CTRecon())
experiments.append(ct_benchmarking.LimitedAngle90CTRecon())
experiments.append(ct_benchmarking.LimitedAngle60CTRecon())

# Sparse angle
experiments.append(ct_benchmarking.SparseAngle360CTRecon())
experiments.append(ct_benchmarking.SparseAngle120CTRecon())
experiments.append(ct_benchmarking.SparseAngle60CTRecon())

# Low dose
experiments.append(ct_benchmarking.LowDoseCTRecon())

# Beam Hardening
experiments.append(ct_benchmarking.BeamHardeningCTRecon())


# Define array for storing metrics
models_ssim = np.zeros(len(experiments))
models_ssim_std = np.zeros(len(experiments))
models_psnr = np.zeros(len(experiments))
models_psnr_std = np.zeros(len(experiments))


for exp_idx, experiment in enumerate(tqdm(experiments)):
    
    testing_data = experiments[exp_idx].get_testing_dataset()

    # REMOVE for final tests
    #indices = torch.arange(3)
    #testing_data = data_utils.Subset(testing_data, indices)

    testing_dataloader = DataLoader(testing_data, 1, shuffle=False)

    # prepare models/algos
    op = make_operator(experiments[exp_idx].geo)

    # TEST 1: metrics
    test_ssim = np.zeros(len(testing_dataloader))
    test_psnr = np.zeros(len(testing_dataloader))

    with torch.no_grad():
        for index, (data, target) in enumerate(tqdm(testing_dataloader)):
           # insert classical method
           output = tv_min2d(op, data[0].to(device),lam=reg_params_f[exp_idx])
           test_ssim[index] = my_ssim(target, output)
           test_psnr[index] = my_psnr(target, output)
           #imageio.imwrite(str(savenames[exp_idx]+"_idx"+str(indices[index])+"_regf.tif"), output.detach().cpu().numpy().squeeze())
    
    # Save metrics into metrics arrays
    models_ssim[exp_idx] = test_ssim.mean()
    models_ssim_std[exp_idx] = test_ssim.std()
    models_psnr[exp_idx] = test_psnr.mean()
    models_psnr_std[exp_idx] = test_psnr.std()
    
    mean_psnr_indcs.append(arg_find_nearest(test_psnr, test_psnr.mean()))
    mean_ssim_indcs.append(arg_find_nearest(test_ssim, test_ssim.mean()))



#print(f"SSIM \
#    & {models_ssim[0]:.4f} $\pm$ {models_ssim_std[0]:.4f} \\\\")

#print(f"PSNR \
#    & {models_psnr[0]:.4f} $\pm$ {models_psnr_std[0]:.4f} \\\\")

print(f"SSIM \
     & {models_ssim[0]:.4f} $\pm$ {models_ssim_std[0]:.4f} \
     & {models_ssim[1]:.4f} $\pm$ {models_ssim_std[1]:.4f} \
     & {models_ssim[2]:.4f} $\pm$ {models_ssim_std[2]:.4f} \
     & {models_ssim[3]:.4f} $\pm$ {models_ssim_std[3]:.4f} \
     & {models_ssim[4]:.4f} $\pm$ {models_ssim_std[4]:.4f} \
     & {models_ssim[5]:.4f} $\pm$ {models_ssim_std[5]:.4f} \
     & {models_ssim[6]:.4f} $\pm$ {models_ssim_std[6]:.4f} \
     & {models_ssim[7]:.4f} $\pm$ {models_ssim_std[7]:.4f} \
     & {models_ssim[8]:.4f} $\pm$ {models_ssim_std[8]:.4f} \\\\")

print(f"PSNR \
     & {models_psnr[0]:.4f} $\pm$ {models_psnr_std[0]:.4f} \
     & {models_psnr[1]:.4f} $\pm$ {models_psnr_std[1]:.4f} \
     & {models_psnr[2]:.4f} $\pm$ {models_psnr_std[2]:.4f} \
     & {models_psnr[3]:.4f} $\pm$ {models_psnr_std[3]:.4f} \
     & {models_psnr[4]:.4f} $\pm$ {models_psnr_std[4]:.4f} \
     & {models_psnr[5]:.4f} $\pm$ {models_psnr_std[5]:.4f} \
     & {models_psnr[6]:.4f} $\pm$ {models_psnr_std[6]:.4f} \
     & {models_psnr[7]:.4f} $\pm$ {models_psnr_std[7]:.4f} \
     & {models_psnr[8]:.4f} $\pm$ {models_psnr_std[8]:.4f} \\\\")

print("Mean PSNR indices:",mean_psnr_indcs)
print("Mean SSIM indices:",mean_ssim_indcs)