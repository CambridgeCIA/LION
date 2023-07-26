#%% Learned Primal Dual


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
import AItomotools.experiments.ct_experiments as ct_experiments
from ts_algorithms import fdk

#%% Chose device:
device = torch.device("cuda:0")
torch.cuda.set_device(device)
savefolder = pathlib.Path("/home/ab2860/rds/hpc-work/models/low_dose_full_angle")
datafolder = pathlib.Path("/home/ab2860/rds/hpc-work/AItomotools/processed/LIDC-IDRI/")

# savefolder = pathlib.Path('/store/DAMTP/ab2860/wip_models/')
# datafolder = pathlib.Path("/local/scratch/public/AItomotools/processed/LIDC-IDRI/")
##################################################
#%% Define experiment
experiment = ct_experiments.LowDoseCTRecon()
experiment.param.data_loader_params.folder = datafolder

#%% Dataset
experiment.param.data_loader_params.max_num_slices_per_patient = 1

lidc_dataset = experiment.get_training_dataset()
lidc_dataset_val = experiment.get_validation_dataset()

# Use the same amount of training
batch_size = 8
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
lidc_validation = DataLoader(lidc_dataset_val, batch_size, shuffle=True)


#%% First we need to train a Unet
my_file = savefolder.joinpath("UNet_final_iter.pt")
if not my_file.is_file():
    print("Training UNet")

    #%% Model
    model = UNet().to(device)
    #%% Optimizer
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

    # %% Check if there is a checkpoint saved, and if so, start from there.
    my_file = savefolder.joinpath("UNet_min_val.pt")
    if my_file.is_file():
        data = torch.load(savefolder.joinpath(my_file))
        min_valid_loss = data["loss"]

    checkpoints = sorted(list(savefolder.glob("UNet_check_*.pt")))
    if checkpoints:
        model, options, data = UNet.load_checkpoint(
            savefolder.joinpath(checkpoints[-1])
        )
        model.to(device)
        optimiser.load_state_dict(data["optimizer_state_dict"])
        start_epoch = data["epoch"]
        total_loss = data["loss"]

    print(f"Starting iteration at epoch {start_epoch}")
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
                savefolder.joinpath(f"UNet_min_val.pt"),
                epoch=epoch + 1,
                training=train_param,
                loss=min_valid_loss,
                dataset=experiment.param,
            )

        # Checkpoint every 10 iters anyway
        if epoch % 40 == 0:
            model.save_checkpoint(
                savefolder.joinpath(f"UNet_check_{epoch+1:04d}.pt"),
                epoch + 1,
                total_loss,
                optimiser,
                train_param,
                dataset=experiment.param,
            )

    model.save(
        savefolder.joinpath("UNet_final_iter.pt"),
        epoch=train_param.epochs,
        training=train_param,
        dataset=experiment.param,
    )

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
#%% Now, UNet has been trained, Lets train ItNet
##########################################################################################################
##########################################################################################################
##########################################################################################################
#%% Optimizer
#%% Model
model = ItNet(experiment.geo).to(device)

for i in range(model.model_parameters.n_iters):
    unet = getattr(model, f"Unet_{i}")
    unet.load(savefolder.joinpath("UNet_final_iter.pt"))
    unet.train()

train_param = Parameter()

train_param.epochs = 500
train_param.learning_rate = 1e-3
train_param.loss = "MSELoss"
# learning parameter update
steps = len(lidc_dataloader)
min_valid_loss = np.inf
total_loss = np.zeros(train_param.epochs)
start_epoch = 0

# %% Check if there is a checkpoint saved, and if so, start from there.

my_file = savefolder.joinpath("ItNet_final_iter.pt")
if my_file.is_file():
    print("Final version found, no need to loop further, exiting")
    exit()
my_file = savefolder.joinpath("ItNet_min_val.pt")
if my_file.is_file():
    data = torch.load(savefolder.joinpath(my_file))
    min_valid_loss = data["loss"]

checkpoints = sorted(list(savefolder.glob("ItNet_check_*.pt")))
if checkpoints:
    model, options, data = ItNet.load_checkpoint(savefolder.joinpath(checkpoints[-1]))
    model.to(device)
    optimiser.load_state_dict(data["optimizer_state_dict"])
    start_epoch = data["epoch"]
    total_loss = data["loss"]

torch.autograd.set_detect_anomaly(True)
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
            savefolder.joinpath(f"ItNet_min_val.pt"),
            epoch=epoch + 1,
            training=train_param,
            loss=min_valid_loss,
        )

    # Checkpoint every 10 iters anyway
    if epoch % 20 == 0:
        model.save_checkpoint(
            savefolder.joinpath(f"ItNet_check_{epoch+1:04d}.pt"),
            epoch + 1,
            total_loss,
            optimiser,
            train_param,
        )


model.save(
    savefolder.joinpath("ItNet_final_iter.pt"),
    epoch=train_param.epochs,
    training=train_param,
)
