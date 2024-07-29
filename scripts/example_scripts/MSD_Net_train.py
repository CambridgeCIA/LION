#%% This example shows how to train MSDNet for full angle, noisy measurements.


#%% Imports
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pathlib
from LION.experiments import ct_benchmarking_experiments
from LION.models.CNNs.MSDNets.MS_D2 import MSD_Net, MSDParams
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
from torchviz import make_dot


#%%
# % Chose device:
device = torch.device("cuda:1")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/cs2186/trained_models/clinical_dose/")

final_result_fname = savefolder.joinpath("FBPMSDNetw1d30lball_final_iter.pt")
checkpoint_fname = savefolder.joinpath("FBPMSDNetw1d30lball_check_*.pt")
validation_fname = savefolder.joinpath("FBPMSDNetw1d30lball_min_val.pt")
#
#%% Define experiment
# experiment = ct_experiments.LowDoseCTRecon(datafolder=datafolder)
experiment = ct_benchmarking_experiments.FullDataCTRecon()
#%% Dataset
lidc_dataset = experiment.get_training_dataset()
lidc_dataset_val = experiment.get_validation_dataset()

#%% Define DataLoader
# Use the same amount of training
batch_size = 3
lidc_dataset = Subset(lidc_dataset, [i for i in range(50)])
lidc_dataset_val = Subset(lidc_dataset_val, [i for i in range(50)])
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
lidc_validation = DataLoader(lidc_dataset_val, batch_size, shuffle=True)

#%% Model
width, depth = 1, 3
dilations = []
for i in range(depth):
    for j in range(width):
        dilations.append((((i * width) + j) % 10) + 1)
model_params = MSDParams(
    in_channels=1,
    width=width,
    depth=depth,
    dilations=dilations,
    look_back_depth=-1,
    final_look_back_depth=-1,
    activation=nn.ReLU(),
)
model = MSD_Net(geometry_parameters=experiment.geo, model_parameters=model_params).to(
    device
)

#%% Optimizer
train_param = LIONParameter()

# loss fn
loss_fcn = torch.nn.MSELoss()
train_param.optimiser = "adam"

# optimizer
train_param.epochs = 1
train_param.learning_rate = 1e-3
train_param.betas = (0.9, 0.99)
train_param.loss = "MSELoss"
train_param.accumulation_steps = 1
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
# if model.final_file_exists(savefolder.joinpath(final_result_fname)):
#    print("final model exists! You already reached final iter")
#    exit()

# model, optimiser, start_epoch, total_loss, _ = FBPMSD_Net.load_checkpoint_if_exists(
#     checkpoint_fname, model, optimiser, total_loss
# )
total_loss = np.resize(total_loss, train_param.epochs)
print(f"Starting iteration at epoch {start_epoch}")
optimiser.zero_grad()

scaler = torch.cuda.amp.grad_scaler.GradScaler()

start_time = time.time()

#%% train
for epoch in range(start_epoch, train_param.epochs):
    train_loss = 0.0
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)
    print("Training...")
    model.train()
    for idx, (sinogram, target_reconstruction) in enumerate(tqdm(lidc_dataloader)):
        sinogram = sinogram.to(device)
        target_reconstruction = target_reconstruction.to(device)
        # with torch.autocast("cuda"):
        reconstruction = model(sinogram)
        make_dot(reconstruction, params=dict(list(model.named_parameters()))).render(
            "msdnet1_3_alt_torchviz", format="png"
        )
        quit()
        loss = loss_fcn(reconstruction, target_reconstruction)
        loss = loss / train_param.accumulation_steps

        scaler.scale(loss).backward()
        # loss.backward()

        train_loss += loss.item()

        if (idx + 1) % train_param.accumulation_steps == 0:
            scaler.step(optimiser)
            scaler.update()
            # scheduler.step()
            optimiser.zero_grad()

    total_loss[epoch] = train_loss
    print(f"Model took {time.time() - start_time}s to train")
    print(f"Model achieved training loss of {train_loss}")
    # Validation
    valid_loss = 0.0
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        print("Validating...")
        for sinogram, target_reconstruction in tqdm(lidc_validation):
            reconstruction = model(sinogram)
            loss = loss_fcn(target_reconstruction, reconstruction)
            valid_loss += loss.item()

    print(
        f"Epoch {epoch+1} \t\t Training Loss: {train_loss / len(lidc_dataloader)} \t\t Validation Loss: {valid_loss / len(lidc_validation)}"
    )

    print(f"Model took {time.time() - start_time}s to validate")
    print(f"Model achieved validation loss of {valid_loss}")

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
    if (epoch + 1) % 1 == 0:
        print("Checkpointing")
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
    geometry=experiment.geo,
)
