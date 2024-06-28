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
import LION.experiments.ct_learned_denoising_experiments as ct_denoising

# Import progress module
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


def plot_recons(ExpNoise_testing_data, ArtNoise_testing_data, idx, model_1_name, model_1, model_2_name, model_2, qual, savename):

    model_1.to(device)

    max_val = 0.010

    Exp_target = ExpNoise_testing_data[idx][1]
    Exp_data = ExpNoise_testing_data[idx][0]
    Exp_fdk_recon = fdk(op, Exp_data)

    Art_target = ArtNoise_testing_data[idx][1]
    Art_data = ArtNoise_testing_data[idx][0]
    Art_fdk_recon = fdk(op, Art_data)

    Exp_clnd_recon_m1 = model_1(Exp_data.unsqueeze(0).to(device))

    Art_clnd_recon_m1 = model_1(Art_data.unsqueeze(0).to(device))    

    model_1.to("cpu")
    model_2.to(device)

    Exp_clnd_recon_m2 = model_2(Exp_data.unsqueeze(0).to(device))
    Art_clnd_recon_m2 = model_2(Art_data.unsqueeze(0).to(device))

    model_2.to("cpu")

    plt.figure()
    plt.subplot(331)
    plt.imshow(Exp_target.detach().cpu().numpy().squeeze(), cmap="gray")
    plt.title("Ground truth")
    plt.axis("off")
    plt.clim(0, max_val)
    
    plt.subplot(332)
    plt.imshow(Exp_fdk_recon.detach().cpu().numpy().squeeze(), cmap="gray")
    plt.title("Exp FDK, PSNR: {:.2f}".format(Exp_fdk_psnr[idx]))
    plt.axis("off")
    plt.clim(0, max_val)

    plt.subplot(333)
    plt.imshow(Art_fdk_recon.detach().cpu().numpy().squeeze(), cmap="gray")
    plt.title("Art FDK, PSNR: {:.2f}".format(Art_fdk_psnr[idx]))
    plt.axis("off")
    plt.clim(0, max_val)
    
    plt.subplot(335)
    plt.imshow(Exp_clnd_recon_m1.detach().cpu().numpy().squeeze(), cmap="gray")
    plt.title(f"ExpDnsd_M1, PSNR: {Exp_FBPUNet_ExpNoise_psnr[idx]:.2f}")
    plt.clim(0, max_val)
    plt.axis("off")

    plt.subplot(336)
    plt.imshow(Art_clnd_recon_m1.detach().cpu().numpy().squeeze(), cmap="gray")
    plt.title(f"ArtDnsd_M1, PSNR: {Art_FBPUNet_ExpNoise_psnr[idx]:.2f}")
    plt.clim(0, max_val)
    plt.axis("off")

    plt.subplot(338)
    plt.imshow(Exp_clnd_recon_m2.detach().cpu().numpy().squeeze(), cmap="gray")
    plt.title(f"ExpDnsd_M2, PSNR: {Exp_FBPUNet_ArtNoise_psnr[idx]:.2f}")
    plt.clim(0, max_val)
    plt.axis("off")

    plt.subplot(339)
    plt.imshow(Art_clnd_recon_m2.detach().cpu().numpy().squeeze(), cmap="gray")
    plt.title(f"ArtDnsd_M2, PSNR: {Art_FBPUNet_ArtNoise_psnr[idx]:.2f}")
    plt.clim(0, max_val)
    plt.axis("off")
    
    plt.suptitle(f"{model_1_name[:-3]} vs. {model_2_name[:-3]} - {qual} PSNR")
    plt.savefig(savename, dpi=300)
    
    return None


# Define your cuda device
device = torch.device("cuda:3")

# Define your data paths
savefolder = pathlib.Path("/export/scratch3/mbk/LION/noise_paper/trained_models/testing_debugging/")
# use min validation, or final result, whicever you prefer

# Make the whole evaluation iterable
#model_names = ["FBPUNet_ExpNoise.pt", "FBPUNet_ArtNoise.pt", "FBPMSDNet_ExpNoise.pt", "FBPMSDNet_ArtNoise.pt"]

model_1_name = "FBPUNet_ExpNoise_min_val.pt"
model_2_name = "FBPUNet_ArtNoise_min_val.pt"

# Set a Flag for whether you want to evaluate on sino or recon domain
eval_domain = "recon"

#%% 2 - Define experiment
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# These are all the experiments we need to run for the noise paper

# Experimental noisy dataset
ExpNoise_experiment = ct_denoising.ExperimentalNoiseDenoisingRecon()

# Simulated noisy dataset
ArtNoise_experiment = ct_denoising.ArtificialNoiseDenoisingRecon()


#Get testing data for ExpNoise
ExpNoise_testing_data = ExpNoise_experiment.get_testing_dataset()
# Delete the following two lines for actual test
#indices = torch.arange(20)
#ExpNoise_testing_data = data_utils.Subset(ExpNoise_testing_data, indices)
ExpNoise_testing_dataloader = DataLoader(ExpNoise_testing_data, 1, shuffle=False)


#Get testing data for ArtNoise
ArtNoise_testing_data = ArtNoise_experiment.get_testing_dataset()
# Delete the following two lines for actual test
#indices = torch.arange(20)
#ArtNoise_testing_data = data_utils.Subset(ArtNoise_testing_data, indices)
ArtNoise_testing_dataloader = DataLoader(ArtNoise_testing_data, 1, shuffle=False)

# prepare models/algos
from ts_algorithms import fdk, sirt, tv_min, nag_ls
from LION.CTtools.ct_utils import make_operator

op = make_operator(ExpNoise_experiment.geo)

# TEST 1: metrics
Exp_FBPUNet_ExpNoise_ssim = np.zeros(len(ExpNoise_testing_dataloader))
Exp_FBPUNet_ExpNoise_psnr = np.zeros(len(ExpNoise_testing_dataloader))

Exp_FBPUNet_ArtNoise_ssim = np.zeros(len(ExpNoise_testing_dataloader))
Exp_FBPUNet_ArtNoise_psnr = np.zeros(len(ExpNoise_testing_dataloader))

Art_FBPUNet_ExpNoise_ssim = np.zeros(len(ArtNoise_testing_dataloader))
Art_FBPUNet_ExpNoise_psnr = np.zeros(len(ArtNoise_testing_dataloader))

Art_FBPUNet_ArtNoise_ssim = np.zeros(len(ArtNoise_testing_dataloader))
Art_FBPUNet_ArtNoise_psnr = np.zeros(len(ArtNoise_testing_dataloader))

Exp_FBPMSDNet_ExpNoise_ssim = np.zeros(len(ExpNoise_testing_dataloader))
Exp_FBPMSDNet_ExpNoise_psnr = np.zeros(len(ExpNoise_testing_dataloader))

Exp_FBPMSDNet_ArtNoise_ssim = np.zeros(len(ExpNoise_testing_dataloader))
Exp_FBPMSDNet_ArtNoise_psnr = np.zeros(len(ExpNoise_testing_dataloader))

Art_FBPMSDNet_ExpNoise_ssim = np.zeros(len(ArtNoise_testing_dataloader))
Art_FBPMSDNet_ExpNoise_psnr = np.zeros(len(ArtNoise_testing_dataloader))

Art_FBPMSDNet_ArtNoise_ssim = np.zeros(len(ArtNoise_testing_dataloader))
Art_FBPMSDNet_ArtNoise_psnr = np.zeros(len(ArtNoise_testing_dataloader))

Exp_fdk_ssim = np.zeros(len(ExpNoise_testing_dataloader))
Exp_fdk_psnr = np.zeros(len(ExpNoise_testing_dataloader))

Art_fdk_ssim = np.zeros(len(ArtNoise_testing_dataloader))
Art_fdk_psnr = np.zeros(len(ArtNoise_testing_dataloader))



from LION.models.post_processing.FBPUNet import FBPUNet

model_1, options, data = FBPUNet.load(savefolder.joinpath(model_1_name))
model_1.to("cpu")
model_1.eval()

model_2, options, data = FBPUNet.load(savefolder.joinpath(model_2_name))
model_2.to("cpu")
model_2.eval()

# Recon domain comparisons
if eval_domain == "recon":
    # Change loaded target data for the experiment to recon domain
    # The other parameters stay the same. We get the config of the detault by
    #ExpNoise_testing_default_parameters = ExpNoise_experiment.default_parameters("2DeteCT")
    #ExpNoise_testing_default_parameters.data_loader_params.task = "sino2recon"
    #ExpNoise_testing_default_parameters.data_loader_params.input_mode = "mode1"
    #ExpNoise_testing_default_parameters.data_loader_params.target_mode = "mode2"
    
    #ExpNoise_experiment = ct_denoising.ExperimentalNoiseDenoising(ExpNoise_testing_default_parameters)

    #Get NEW testing data for ExpNoise
    #ExpNoise_testing_data = ExpNoise_experiment.get_testing_dataset()
    # Delete the following two lines for actual test
    #indices = torch.arange(20)
    #ExpNoise_testing_data = data_utils.Subset(ExpNoise_testing_data, indices)
    ExpNoise_testing_dataloader = DataLoader(ExpNoise_testing_data, 1, shuffle=False)
    

    #ArtNoise_testing_default_parameters = ArtNoise_experiment.default_parameters("2DeteCT")
    #ArtNoise_testing_default_parameters.data_loader_params.task = "sino2recon"
    #ArtNoise_testing_default_parameters.data_loader_params.input_mode = "mode2"
    #ArtNoise_testing_default_parameters.data_loader_params.target_mode = "mode2"
    
    #ArtNoise_experiment = ct_denoising.ArtificialNoiseDenoising(ArtNoise_testing_default_parameters)

    #Get NEW testing data for ArtNoise
    #ArtNoise_testing_data = ArtNoise_experiment.get_testing_dataset()
    # Delete the following two lines for actual test
    #indices = torch.arange(20)
    #ArtNoise_testing_data = data_utils.Subset(ArtNoise_testing_data, indices)
    ArtNoise_testing_dataloader = DataLoader(ArtNoise_testing_data, 1, shuffle=False)

    model_1.to(device)

    with torch.no_grad():
        for index, (sino, target) in enumerate(tqdm(ExpNoise_testing_dataloader)):
            output_m1 = model_1(sino.to(device))
            #sino_recon_m1 = fdk(op, output_m1[0].to(device))
            Exp_FBPUNet_ExpNoise_ssim[index] = my_ssim(target, output_m1)
            Exp_FBPUNet_ExpNoise_psnr[index] = my_psnr(target, output_m1)

            recon = fdk(op, sino[0].to(device))
            Exp_fdk_ssim[index] = my_ssim(target, recon)
            Exp_fdk_psnr[index] = my_psnr(target, recon)

    with torch.no_grad():
        for index, (sino, target) in enumerate(tqdm(ArtNoise_testing_dataloader)):
            output_m1 = model_1(sino.to(device))
            #sino_recon_m1 = fdk(op, output_m1[0].to(device))
            Art_FBPUNet_ExpNoise_ssim[index] = my_ssim(target, output_m1)
            Art_FBPUNet_ExpNoise_psnr[index] = my_psnr(target, output_m1)
    
            recon = fdk(op, sino[0].to(device))
            Art_fdk_ssim[index] = my_ssim(target, recon)
            Art_fdk_psnr[index] = my_psnr(target, recon)

    model_1.to("cpu")
    model_2.to(device)

    with torch.no_grad():
        for index, (sino, target) in enumerate(tqdm(ExpNoise_testing_dataloader)):
            output_m2 = model_2(sino.to(device))
            #sino_recon_m2 = fdk(op, output_m2[0].to(device))
            Exp_FBPUNet_ArtNoise_ssim[index] = my_ssim(target, output_m2)
            Exp_FBPUNet_ArtNoise_psnr[index] = my_psnr(target, output_m2)

    with torch.no_grad():
        for index, (sino, target) in enumerate(tqdm(ArtNoise_testing_dataloader)):
            output_m2 = model_2(sino.to(device))
            #sino_recon_m2 = fdk(op, output_m2[0].to(device))
            Art_FBPUNet_ArtNoise_ssim[index] = my_ssim(target, output_m2)
            Art_FBPUNet_ArtNoise_psnr[index] = my_psnr(target, output_m2)

    model_2.to("cpu")

    print(f"Recon domain results")
    print(f"FBPUNet_ExpNoise Experimental SSIM: {Exp_FBPUNet_ExpNoise_ssim.mean()} +- {Exp_FBPUNet_ExpNoise_ssim.std()}")
    print(f"FBPUNet_ExpNoise Experimental PSNR: {Exp_FBPUNet_ExpNoise_psnr.mean()} +- {Exp_FBPUNet_ExpNoise_psnr.std()}")
    print(f"FBPUNet_ExpNoise Artificial SSIM: {Art_FBPUNet_ExpNoise_ssim.mean()} +- {Art_FBPUNet_ExpNoise_ssim.std()}")
    print(f"FBPUNet_ExpNoise Artificial PSNR: {Art_FBPUNet_ExpNoise_psnr.mean()} +- {Art_FBPUNet_ExpNoise_psnr.std()}")

    print(f"FBPUNet_ArtNoise Experimental SSIM: {Exp_FBPUNet_ArtNoise_ssim.mean()} +- {Exp_FBPUNet_ArtNoise_ssim.std()}")
    print(f"FBPUNet_ArtNoise Experimental PSNR: {Exp_FBPUNet_ArtNoise_psnr.mean()} +- {Exp_FBPUNet_ArtNoise_psnr.std()}")
    print(f"FBPUNet_ArtNoise Artificial SSIM: {Art_FBPUNet_ArtNoise_ssim.mean()} +- {Art_FBPUNet_ArtNoise_ssim.std()}")
    print(f"FBPUNet_ArtNoise Artificial PSNR: {Art_FBPUNet_ArtNoise_psnr.mean()} +- {Art_FBPUNet_ArtNoise_psnr.std()}")


    # TEST 2: Display examples
    min_idx = np.argmin(Exp_FBPUNet_ExpNoise_psnr)
    max_idx = np.argmax(Exp_FBPUNet_ExpNoise_psnr)

    # find the closest to the mean
    mean_idx = arg_find_nearest(Exp_FBPUNet_ExpNoise_psnr, Exp_FBPUNet_ExpNoise_psnr.mean())

    print(f"Recon slice indices: Min_idx: {min_idx}, Max_idx: {max_idx}, Mean_idx: {mean_idx}")

    # Display options
    # This is the scale of the error plot w.r.t. the image DISPLAY. Not data, DISPLAY!
    pctg_error = 0.05  # 5%

    # BEST PSNR
    #plot_recons(ExpNoise_testing_data, ArtNoise_testing_data, max_idx, model_1_name, model_1, model_2_name, model_2, "Best", "FBPUNet_recons_eval_best_psnr.png")

    # WORST PSNR
    #plot_recons(ExpNoise_testing_data, ArtNoise_testing_data, min_idx, model_1_name, model_1, model_2_name, model_2, "Worst", "FBPUNet_recons_eval_worst_psnr.png")

    # MEAN PSNR
    #plot_recons(ExpNoise_testing_data, ArtNoise_testing_data, mean_idx, model_1_name, model_1, model_2_name, model_2, "Mean", "FBPUNet_recons_eval_mean_psnr.png")
