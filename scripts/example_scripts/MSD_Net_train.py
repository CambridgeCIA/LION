#%% This example shows how to train FBPConvNet for full angle, noisy measurements.


#%% Imports
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pathlib
from LION.models.CNNs.MSDNets.FBPMS_D import FBPMSD_Net
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
from ts_algorithms import fdk
from LION.CTtools.ct_utils import make_operator


#%%
# % Chose device:
device = torch.device("cuda:0")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/cs2186/trained_models/clinical_dose/")

final_result_fname = savefolder.joinpath("FBPMSDNet_final_iter.pt")
checkpoint_fname = savefolder.joinpath("FBPMSDNet_check_*.pt")  
validation_fname = savefolder.joinpath("FBPMSDNet_min_val.pt")
#
#%% Define experiment
# experiment = ct_experiments.LowDoseCTRecon(datafolder=datafolder)
experiment = ct_experiments.clinicalCTRecon()
#%% Dataset
lidc_dataset = experiment.get_training_dataset()
lidc_dataset_val = experiment.get_validation_dataset()

#%% Define DataLoader
# Use the same amount of training
batch_size = 4
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
lidc_validation = DataLoader(lidc_dataset_val, batch_size, shuffle=True)

#%% Model
# Default model is already from the paper.
model = FBPMSD_Net(geometry_parameters=experiment.geo).to(device)


#%% Optimizer
train_param = LIONParameter()

# loss fn
loss_fcn = torch.nn.MSELoss()
train_param.optimiser = "adam"

# optimizer
train_param.epochs = 2
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
#if model.final_file_exists(savefolder.joinpath(final_result_fname)):
#    print("final model exists! You already reached final iter")
#    exit()

model, optimiser, start_epoch, total_loss, _ = FBPMSD_Net.load_checkpoint_if_exists(
    checkpoint_fname, model, optimiser, total_loss
)
print(f"Starting iteration at epoch {start_epoch}")

#%% train
for epoch in range(start_epoch, train_param.epochs):
    train_loss = 0.0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)
    print("Training...")
    model.train()
    for sinogram, target_reconstruction in tqdm(lidc_dataloader):   
        optimiser.zero_grad()
        
        reconstruction = model(sinogram)
        loss = loss_fcn(reconstruction, target_reconstruction)

        loss.backward()

        train_loss += loss.item()

        optimiser.step()
        scheduler.step()
    total_loss[epoch] = train_loss
    # Validation
    valid_loss = 0.0
    model.eval()
    print("Validating...")
    for sinogram, target_reconstruction in tqdm(lidc_validation):
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
            dataset=experiment.param,
        )

    # Checkpoint every 10 iters anyway
    if epoch % 10 == 0:
        model.save_checkpoint(
            pathlib.Path(str(checkpoint_fname).replace("*", f"{epoch+1:04d}")),
            epoch + 1,
            total_loss,
            optimiser,
            train_param,
            dataset=experiment.param,
        )


model.save(
    final_result_fname,
    epoch=train_param.epochs,
    training=train_param,
    dataset=experiment.param,
    geometry=experiment.geo
)



