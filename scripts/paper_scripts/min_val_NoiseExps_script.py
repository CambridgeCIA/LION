# Standard imports
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm
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

# Model imports
from LION.models.CNNs.UNets.UNet_3 import UNet
#from LION.models.CNNs.MS_D import MS_D
#from LION.models.post_processing.FBPUNet import FBPUNet
#from LION.models.post_processing.FBPMSDNet import FBPMS_D

device = torch.device("cuda:0")
torch.cuda.set_device(device)
savefolder = pathlib.Path("/export/scratch3/mbk/LION/noise_paper/trained_models/testing_debugging/")

model_names = ["UNet_ExpNoise.pt"]
#model_names = ["MSD_ExpNoise.pt", "MSD_ArtNoise.pt",
#"FBPUNet_ExpNoise.pt", "FBPUNet_ArtNoise.pt", "FBPMSDNet_ExpNoise.pt", "FBPMSDNet_ArtNoise.pt"]

# Define all experiments in experiments list
experiments = []

experiments.append(ct_denoising.ExperimentalNoiseDenoising())
#experiments.append(ct_denoising.ArtificialNoiseDenoising())
#experiments.append(ct_denoising.ExperimentalNoiseDenoising())
#experiments.append(ct_denoising.ArtificialNoiseDenoising())

#experiments.append(ct_denoising.ExperimentalNoiseDenoisingRecon())
#experiments.append(ct_denoising.ArtificialNoiseDenoisingRecon())
#experiments.append(ct_denoising.ExperimentalNoiseDenoisingRecon())
#experiments.append(ct_denoising.ArtificialNoiseDenoisingRecon())

for model_idx, model_name in enumerate(tqdm(model_names)):

    validation_fname = str(model_name[:-3]+"_min_val.pt")
    checkpoint_name = str(model_name[:-3]+"_check_*.pt")

    # Get validation data
    validation_data = experiments[model_idx].get_validation_dataset()
    validation_dataloader = DataLoader(validation_data, 1, shuffle=False)

    # Set loss (same as training)
    loss_fcn = torch.nn.MSELoss()

    checkpoints = sorted(list(savefolder.joinpath(checkpoint_name).parent.glob(savefolder.joinpath(checkpoint_name).name)))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found for {checkpoint_name}")

    validation_loss_list=[]

    for checkpoint in tqdm(checkpoints):
        # CAREFUL: Change model name here (FBPMS_D)
        if model_name.startswith("UNet"):
            model, options, data = UNet.load_checkpoint(savefolder.joinpath(checkpoint_name).parent.joinpath(checkpoint))
        elif model_name.startswith("MSD"):
            model, options, data = MS_D.load_checkpoint(savefolder.joinpath(checkpoint_name).parent.joinpath(checkpoint))
        elif model_name.startswith("FBPUNet"):
            model, options, data = FBPUNet.load_checkpoint(savefolder.joinpath(checkpoint_name).parent.joinpath(checkpoint))
        elif model_name.startswith("FBPMSDNet"):
            model, options, data = FBPMS_D.load_checkpoint(savefolder.joinpath(checkpoint_name).parent.joinpath(checkpoint))
        
        model.eval()

        validation_loss = 0

        for index, (data, target) in enumerate(tqdm(validation_dataloader)):
            with torch.no_grad():
                output = model(data.to(device))
                validation_loss += loss_fcn(output, target.to(device))
            validation_loss /= len(validation_dataloader)
        validation_loss_list.append(validation_loss.detach().cpu().numpy().squeeze())
        print(f"Validation loss for {checkpoint}: {validation_loss.detach().cpu().numpy().squeeze()}")

    min_val_loss = np.argmin(np.array(validation_loss_list))
    print(f"Minimum validation loss at {checkpoints[min_val_loss]} with loss {validation_loss_list[min_val_loss]}")
    model.save(savefolder.joinpath(validation_fname))

