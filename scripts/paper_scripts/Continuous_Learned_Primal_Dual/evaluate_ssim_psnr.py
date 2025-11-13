#%%
# Basic science imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# %%
def read_ssim_psnr_values(filename):
    tensor_list = torch.load(filename)
    clpd_ssim, lpd_ssim, fdk_ssim, clpd_psnr, lpd_psnr, fdk_psnr = tensor_list
    return clpd_ssim, lpd_ssim, fdk_ssim, clpd_psnr, lpd_psnr, fdk_psnr


#%%
def compute_avg_percentils(t):
    mean = torch.mean(t).item()
    std = torch.std(t).item()
    percentile_25 = torch.quantile(t, 0.25).item()
    percentile_50 = torch.quantile(t, 0.50).item()
    percentile_75 = torch.quantile(t, 0.75).item()
    return mean, std, percentile_25, percentile_50, percentile_75


#%%
def plot_distribution(t, bins=30):
    plt.hist(t.numpy().flatten(), bins=bins)
    plt.title("Distribution of Values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


#%%
def plot_box(tensors, labels, dose, geometry, title, save):
    data = [t.numpy().flatten() for t in tensors]
    plt.boxplot(data, labels=labels)
    plt.title(geometry)
    plt.ylabel(title)
    if save:
        plt.savefig(
            f"/home/cr661/rds/hpc-work/store/LION/trained_models/{DOSE}_dose/results/results_cLPDIN_LPD/{title}_{dose}_{geometry}.png"
        )
    plt.show()


# %%
GEOMETRY = "limited_angle"
DOSE = "extreme_low"
# GEOMETRY = "limited_angle"
# GEOMETRY = "sparse_angle"
FILE_PATH = f"/home/cr661/rds/hpc-work/store/LION/trained_models/{DOSE}_dose/ssim_psnr_values_{DOSE}_{GEOMETRY}_cLPDIN_LPD_corrected.pt"
# FILE_PATH = f"/home/cr661/rds/hpc-work/store/LION/trained_models/extreme_low_dose/results/ssim_psnr_values_extreme_low_{GEOMETRY}.pt"
out = read_ssim_psnr_values(FILE_PATH)
#%%
for i in out:
    mean, std, percentile_25, percentile_50, percentile_75 = compute_avg_percentils(i)
    print(
        f"Mean: {mean:.4f}, Std: {std:.4f}, 25th percentile: {percentile_25}, 50th percentile: {percentile_50}, 75th percentile: {percentile_75}"
    )

# %%
for i in out:
    plot_distribution(i)
# %%
# Plot SSIM
plot_box(
    [out[0], out[1], out[2]],
    ["cLPD", "LPD", "FDK"],
    geometry=GEOMETRY,
    dose=DOSE,
    title="SSIM",
    save=True,
)
# %%
# Plot PSNR
plot_box(
    [out[3], out[4], out[5]],
    ["cLPD", "LPD", "FDK"],
    geometry=GEOMETRY,
    dose=DOSE,
    title="PSNR",
    save=True,
)
# %%
