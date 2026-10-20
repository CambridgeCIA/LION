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

def plot_supplementary_neuroIPS(recon:list, gt:list, indices:list, model_name:str, experiment_name:str,max_val:float=0.008,pctg_error:float=0.05):
    '''
    Recon: len = 3 list of the best, worst and mean reconstructions. They should be a 2D np array.
    gt: len = 3 list of the best, worst and mean ground truth. They should be a 2D np array.
    indices: len = 3 list of slice numbers for best, worst and mean reconstructions. They should be ints.
    model_name
    experiment_name (use the same as in the paper!)
    max_val: max value for the images (do not change)
    pctg_error: percentage of the error plot w.r.t. the image DISPLAY. Not data, DISPLAY! (do not change)
    '''

    titles = ["Best", "Worst", "Mean"]

    fig = plt.figure(figsize=(8, 8))

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=None)
    plt.suptitle(f"{model_name} - {experiment_name}", fontsize=16)
    for i in range(3): #best/worst/mean
        title = titles[i] + f", slice #{indices[i]}"
        plt.subplot(3,3,i+1)
        plt.imshow(gt[i], cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.clim(0, max_val)

        plt.subplot(3,3,(i+1)+3)
        plt.imshow(recon[i], cmap="gray")
        plt.title(f"SSIM: {ssim(gt[i],recon[i],data_range=gt[i].max()-gt[i].min()):.2f}")
        plt.axis("off")
        plt.clim(0, max_val)

        plt.subplot(3,3,(i+1)+6)
        err = plt.imshow((gt[i] -recon[i]),cmap="seismic")
        if i==1:
            plt.title(f"Error")
        plt.axis("off")
        plt.clim(-max_val * pctg_error, max_val * pctg_error)
        cb_ax = fig.add_axes([.2,.08,.6,.012])
        cbar = fig.colorbar(err,orientation='horizontal',cax=cb_ax)
        cbar.set_ticks([-max_val * pctg_error, 0, max_val * pctg_error],labels=[f"-{pctg_error*100:.2f}%", "0", f"{pctg_error*100:.2f}%"])
        cbar.ax.tick_params(labelsize=10)
    plt.savefig(f"eval_{model_name}_{experiment_name}.png", dpi=300,bbox_inches='tight', pad_inches=0)


mean_psnr_indcs = []
mean_ssim_indcs = []

device = torch.device("cuda:1")

savefolder = pathlib.Path("/export/scratch3/mbk/LION/bm_models/")
# use min validation, or final result, whichever you prefer

#model_names = ["LG_Limited60.pt", "LG_Sparse360.pt"]
model_names = ["LG_FullData_min_val.pt",
"LG_Limited120_min_val.pt", "LG_Limited90_min_val.pt", "LG_Limited60_min_val.pt",
"LG_Sparse360_min_val.pt", "LG_Sparse120_min_val.pt", "LG_Sparse60_min_val.pt",
"LG_LowDose_min_val.pt", "LG_BeamHardening_min_val.pt"]

from LION.models.iterative_unrolled.LG import LG


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
models_ssim = np.zeros(len(model_names))
models_ssim_std = np.zeros(len(model_names))
models_psnr = np.zeros(len(model_names))
models_psnr_std = np.zeros(len(model_names))


for model_idx, model_name in enumerate(tqdm(model_names)):
    model, options, data = LG.load(savefolder.joinpath(model_name))
    model.to(device)

    testing_data = experiments[model_idx].get_testing_dataset()
    testing_dataloader = DataLoader(testing_data, 1, shuffle=False)

    model.eval()

    # TEST 1: metrics
    test_ssim = np.zeros(len(testing_dataloader))
    test_psnr = np.zeros(len(testing_dataloader))

    with torch.no_grad():
        for index, (data, target) in enumerate(tqdm(testing_dataloader)):
           # model
           output = model(data.to(device))
           test_ssim[index] = my_ssim(target, output)
           test_psnr[index] = my_psnr(target, output)
    
    # Save metrics into metrics arrays
    models_ssim[model_idx] = test_ssim.mean()
    models_ssim_std[model_idx] = test_ssim.std()
    models_psnr[model_idx] = test_psnr.mean()
    models_psnr_std[model_idx] = test_psnr.std()
    
    mean_psnr_indcs.append(arg_find_nearest(test_psnr, test_psnr.mean()))
    mean_ssim_indcs.append(arg_find_nearest(test_ssim, test_ssim.mean()))

    max_idx = np.argmax(test_ssim)
    min_idx = np.argmin(test_ssim)
    mean_idx = arg_find_nearest(test_ssim, test_ssim.mean())

    gt = [testing_data[max_idx][1].detach().cpu().numpy().squeeze(),testing_data[min_idx][1].detach().cpu().numpy().squeeze(),testing_data[mean_idx][1].detach().cpu().numpy().squeeze()]
    data = [testing_data[max_idx][0].detach().cpu().numpy().squeeze(),testing_data[min_idx][0].detach().cpu().numpy().squeeze(),testing_data[mean_idx][0].detach().cpu().numpy().squeeze()]
    output = [model(testing_data[max_idx][0].unsqueeze(0).to(device)).detach().cpu().numpy().squeeze(),model(testing_data[min_idx][0].unsqueeze(0).to(device)).detach().cpu().numpy().squeeze(),model(testing_data[mean_idx][0].unsqueeze(0).to(device)).detach().cpu().numpy().squeeze()]

    # then call with the appropiate names:
    tmp_str = model_name.partition("_min")[0]
    indices = [max_idx, min_idx, mean_idx]
    plot_supplementary_neuroIPS(output,gt,indices,str(tmp_str.partition("_")[0]+"TV"),tmp_str.partition("_")[2])

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {pytorch_total_params}")



#print(f"SSIM \
#    & {models_ssim[0]:.4f} $\pm$ {models_ssim_std[0]:.4f} \
#    & {models_ssim[1]:.4f} $\pm$ {models_ssim_std[1]:.4f} \\\\")

#print(f"PSNR \
#    & {models_psnr[0]:.4f} $\pm$ {models_psnr_std[0]:.4f} \
#    & {models_psnr[1]:.4f} $\pm$ {models_psnr_std[1]:.4f} \\\\")

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
