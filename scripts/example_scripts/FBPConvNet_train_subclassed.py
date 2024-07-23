#%% This example shows how to train FBPConvNet for full angle, noisy measurements.


#%% Imports
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pathlib
from LION.models.post_processing.FBPConvNet_subclassed import FBPConvNet
import LION.models.LIONmodelSubclasses as LIONmodelSubclasses
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
import os

#%%
# % Chose device:
device = torch.device("cuda:0")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/home/hyt35/trained_models/clinical_dose_subclassed/")
if not os.path.exists(savefolder):
    os.makedirs(savefolder)

final_result_fname = savefolder.joinpath("FBPConvNet_final_iter.pt")
checkpoint_fname = savefolder.joinpath("FBPConvNet_check_*.pt")  
validation_fname = savefolder.joinpath("FBPConvNet_min_val.pt")
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
# lidc_dataloader = DataLoader(Subset(lidc_dataset, range(20), batch_size, shuffle=True)
# lidc_validation = DataLoader(Subset(lidc_dataset_val, range(20)), batch_size, shuffle=True)
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
lidc_validation = DataLoader(lidc_dataset_val, batch_size, shuffle=True)

#%% Model
# Default model is already from the paper.
# model = 
model = FBPConvNet(geometry_parameters=experiment.geo)
# print(model.default_parameters())
# print(model.__class__.__name__)
# print(isinstance(model, LIONmodelSubclasses.LIONmodelPhantom), isinstance(model, LIONmodelSubclasses.LIONmodelSino))
model = LIONmodelSubclasses.Constructor(model).to(device)
# print(model.default_parameters())
# print(isinstance(model, LIONmodelSubclasses.LIONmodelPhantom), isinstance(model, LIONmodelSubclasses.LIONmodelSino))
# print(isinstance(model, LIONmodelSubclasses.LIONmodelPhantom), isinstance(model, LIONmodelSubclasses.LIONmodelSino))

# for sinogram, target_reconstruction in tqdm(lidc_dataloader):
#     bar = sinogram
#     break
# print(model(sinogram))

# fbp = LIONmodelSubclasses.forward_decorator(model, lambda x:x)
# print(fbp(sinogram))
# print(model.phantom2phantom(fbp(sinogram)))
# raise Exception()
#%% Optimizer
train_param = LIONParameter()

# loss fn
loss_fcn = torch.nn.MSELoss()
train_param.optimiser = "adam"

# optimizer
train_param.epochs = 5
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
if model.final_file_exists(savefolder.joinpath(final_result_fname)):
    print("final model exists! You already reached final iter")
    exit()

model, optimiser, start_epoch, total_loss, _ = FBPConvNet.load_checkpoint_if_exists(
    checkpoint_fname, model, optimiser, total_loss
)
print(f"Starting iteration at epoch {start_epoch}")

#%% train
for epoch in range(start_epoch, train_param.epochs):
    train_loss = 0.0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)
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
    with torch.no_grad():
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
)



