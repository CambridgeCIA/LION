#%% This example shows how to train FBPConvNet for full angle, noisy measurements.


#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pathlib
import AItomotools.CTtools.ct_geometry as ctgeo
import AItomotools.CTtools.ct_utils as ct
from AItomotools.data_loaders.LIDC_IDRI import LIDC_IDRI
from AItomotools.models.ItNet import ItNet, UNet
from AItomotools.utils.parameter import Parameter
from ts_algorithms import fdk


import AItomotools.experiments.ct_experiments as ct_experiments


#%%
# % Chose device:
device = torch.device("cuda:0")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/ab2860/trained_models/low_dose/")
datafolder = pathlib.Path(
    "/store/DAMTP/ab2860/AItomotools/data/AItomotools/processed/LIDC-IDRI/"
)
final_result_fname = savefolder.joinpath("ItNet_final_iter.pt")
checkpoint_fname = savefolder.joinpath("ItNet_check_*.pt")
validation_fname = savefolder.joinpath("ItNet_min_val.pt")
unet_final_result_fname = savefolder.joinpath("ItNet_final_iter.pt")
unet_checkpoint_fname = savefolder.joinpath("ItNet_check_*.pt")
unet_validation_fname = savefolder.joinpath("ItNet_min_val.pt")
#
#%% Define experiment
experiment = ct_experiments.LowDoseCTRecon(datafolder=datafolder)

#%% Dataset
lidc_dataset = experiment.get_training_dataset()
lidc_dataset_val = experiment.get_validation_dataset()

#%% Define DataLoader
# Use the same amount of training
batch_size = 8
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
lidc_validation = DataLoader(lidc_dataset_val, batch_size, shuffle=True)

#%% Model
# Default model is already from the paper.
model = UNet()
# First, pretrain the Unet
if not model.final_file_exists(savefolder.joinpath(unet_final_result_fname)):

    model.to(device)
    train_param = Parameter()
    # loss fn
    loss_fcn = torch.nn.MSELoss()
    train_param.optimiser = "adam"
    # optimizer
    train_param.epochs = 400
    train_param.learning_rate = 1e-3
    train_param.loss = "MSELoss"
    optimiser = torch.optim.Adam(model.parameters(), lr=0.0002)

    # learning parameter update
    model.train()
    min_valid_loss = np.inf
    total_loss = np.zeros(train_param.epochs)
    start_epoch = 0

    model, optimiser, start_epoch, total_loss, _ = UNet.load_checkpoint_if_exists(
        unet_checkpoint_fname, model, optimiser, total_loss
    )

    print(f"Starting Unet iteration at epoch {start_epoch}")
    #%% train
    for epoch in range(start_epoch, train_param.epochs):
        train_loss = 0.0
        for index, (sinogram, target_reconstruction) in tqdm(
            enumerate(lidc_dataloader)
        ):

            optimiser.zero_grad()
            bad_recon = torch.zeros(target_reconstruction.shape, device=device)
            for sino in range(sinogram.shape[0]):
                bad_recon[sino] = fdk(lidc_dataset.operator, sinogram[sino])
            reconstruction = model(bad_recon)
            loss = loss_fcn(reconstruction, target_reconstruction)

            loss.backward()

            train_loss += loss.item()

            optimiser.step()
        total_loss[epoch] = train_loss

        # Validation
        valid_loss = 0.0
        model.eval()
        for index, (sinogram, target_reconstruction) in tqdm(
            enumerate(lidc_validation)
        ):

            bad_recon = torch.zeros(target_reconstruction.shape, device=device)
            for sino in range(sinogram.shape[0]):
                bad_recon[sino] = fdk(lidc_dataset.operator, sinogram[sino])
            reconstruction = model(bad_recon)
            loss = loss_fcn(target_reconstruction, reconstruction)
            valid_loss += loss.item()

        print(
            f"Epoch {epoch+1} \t\t Training Loss: {train_loss / len(lidc_dataloader)} \t\t Validation Loss: {valid_loss / len(lidc_validation)}"
        )

        if min_valid_loss > valid_loss:
            print(
                f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model"
            )
            min_valid_loss = valid_loss
            # Saving State Dict
            model.save(
                savefolder.joinpath(unet_validation_fname),
                epoch=epoch + 1,
                training=train_param,
                loss=min_valid_loss,
                dataset=experiment.param,
            )

        # Checkpoint every 40 iters anyway
        if epoch % 40 == 0:
            model.save_checkpoint(
                pathlib.Path(str(unet_checkpoint_fname).replace("*", f"{epoch+1:04d}")),
                epoch + 1,
                total_loss,
                optimiser,
                train_param,
                dataset=experiment.param,
            )

    model.save(
        unet_final_result_fname,
        epoch=train_param.epochs,
        training=train_param,
        dataset=experiment.param,
    )

##########################################################################################################
#%% Now, UNet has been trained, Lets train ItNet
##########################################################################################################

#%% Model
model = ItNet(experiment.geo).to(device)

# Load all pretrained UNets
for i in range(model.model_parameters.n_iters):
    unet = getattr(model, f"Unet_{i}")
    unet.load(unet_final_result_fname)
    unet.train()


train_param = Parameter()
train_param.epochs = 500
train_param.learning_rate = 1e-3
train_param.loss = "MSELoss"
steps = len(lidc_dataloader)
min_valid_loss = np.inf
total_loss = np.zeros(train_param.epochs)
start_epoch = 0

# %% Check if there is a checkpoint saved, and if so, start from there.
if model.final_file_exists(savefolder.joinpath(final_result_fname)):
    print("final model exists! You already reahced final iter")
    exit()

model, optimiser, start_epoch, total_loss, _ = ItNet.load_checkpoint_if_exists(
    checkpoint_fname, model, optimiser, total_loss
)
print(f"Starting iteration at epoch {start_epoch}")

# loss fn
loss_fcn = torch.nn.MSELoss()
train_param.optimiser = "adam"

# optimizer

optimiser = torch.optim.Adam(model.parameters(), lr=train_param.learning_rate)

model.train()

print(f"Starting iteration at epoch {start_epoch}")
#%% train
for epoch in range(start_epoch, train_param.epochs):
    train_loss = 0.0
    for index, (sinogram, target_reconstruction) in tqdm(enumerate(lidc_dataloader)):
        optimiser.zero_grad()
        reconstruction = model(sinogram)
        loss = loss_fcn(reconstruction, target_reconstruction)
        loss.backward()
        train_loss += loss.item()
        optimiser.step()

    total_loss[epoch] = train_loss
    # Validation
    valid_loss = 0.0
    model.eval()
    for index, (sinogram, target_reconstruction) in tqdm(enumerate(lidc_validation)):
        reconstruction = model(sinogram)
        loss = loss_fcn(target_reconstruction, reconstruction)
        valid_loss += loss.item()

    print(
        f"Epoch {epoch+1} \t\t Training Loss: {train_loss / len(lidc_dataloader)} \t\t Validation Loss: {valid_loss / len(lidc_validation)}"
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
        )

    # Checkpoint every 20 iters anyway
    if epoch % 20 == 0:
        model.save_checkpoint(
            pathlib.Path(str(checkpoint_fname).replace("*", f"{epoch+1:04d}")),
            epoch + 1,
            total_loss,
            optimiser,
            train_param,
        )


model.save(
    final_result_fname,
    epoch=train_param.epochs,
    training=train_param,
)
