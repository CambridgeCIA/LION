#%% Noise2Inverse train

#%% Imports

# Basic science imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# basic python imports
from tqdm import tqdm
import pathlib

# LION imports
import LION.CTtools.ct_utils as ct
from LION.models.MS_D import MS_D
from LION.utils.parameter import Parameter
import LION.experiments.ct_experiments as ct_experiments
from ts_algorithms import fdk


#%%
# % Chose device:
device = torch.device("cuda:0")
torch.cuda.set_device(device)
# Define your data paths
savefolder = pathlib.Path("/store/DAMTP/ab2860/low_dose/")
datafolder = pathlib.Path("/local/scratch/public/AItomotools/processed/LIDC-IDRI/")
final_result_fname = savefolder.joinpath("Noise2Inverse_final_iter.pt")
checkpoint_fname = savefolder.joinpath("Noise2Inverse_check_*.pt")
#
#%% Define experiment
experiment = ct_experiments.LowDoseCTRecon(datafolder=datafolder)

#%% Dataset
lidc_dataset = experiment.get_training_dataset()

# Use the same amount of training
batch_size = 16
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)

#%% Model
# Default model is already from the paper.
model = MS_D().to(device)

#%% Operators for self-supervised training

# number of data splits
k = 4
op = []
geo = experiment.geo
angles = geo.angles.copy()
for i in range(k):
    geo.angles = angles[i:-1:k]
    op.append(ct.make_operator(geo))  # list of operators for each subsampling of angles
#%% Optimizer
train_param = Parameter()

# loss fn
loss_fcn = torch.nn.MSELoss()
train_param.optimiser = "adam"

# optimizer
train_param.epochs = 100
train_param.learning_rate = 1e-3
train_param.loss = "MSELoss"
optimiser = torch.optim.Adam(model.parameters(), lr=train_param.learning_rate)

# learning parameter update
steps = len(lidc_dataloader)
model.train()
total_loss = np.zeros(train_param.epochs)
start_epoch = 0

# %% Check if there is a checkpoint saved, and if so, start from there.

# If there is a file with the final results, don't run again
# if model.final_file_exists(savefolder.joinpath(final_result_fname)):
#     print("final model exists! You already reahced final iter")
#     exit()

model, optimiser, start_epoch, total_loss, _ = MS_D.load_checkpoint_if_exists(
    checkpoint_fname, model, optimiser, total_loss
)
print(f"Starting iteration at epoch {start_epoch}")

#%% train
for epoch in range(start_epoch, train_param.epochs):
    train_loss = 0.0
    for index, (sinogram, target_reconstruction) in tqdm(enumerate(lidc_dataloader)):
        # 1: Reset the optimizer
        optimiser.zero_grad()

        # 2: Perform FDK reconstruction for each subset of the training data.
        size_noise2inv = list(target_reconstruction.shape)
        size_noise2inv.insert(0, k)
        bad_recon = torch.zeros(size_noise2inv, device=device)
        for sino in range(sinogram.shape[0]):
            for split in range(k):
                bad_recon[split, sino] = fdk(op[split], sinogram[sino, 0, split:-1:k])

        # 3: we train K->1 so, pick one of these to be the target, at random.
        label_array = torch.zeros(target_reconstruction.shape, device=device)
        target = torch.zeros(target_reconstruction.shape, device=device)
        for sino in range(sinogram.shape[0]):
            indices = np.arange(k)
            label = np.random.randint(k)
            target[sino] = bad_recon[label, sino].detach().clone()
            label_array[sino] = torch.mean(
                bad_recon[np.delete(indices, label), sino].detach().clone(), axis=0
            )

        # 4: Do the optimizer step
        reconstruction = model(label_array)
        loss = loss_fcn(reconstruction, target)
        loss.backward()
        train_loss += loss.item()
        optimiser.step()

    # end bach loop
    total_loss[epoch] = train_loss

    # Checkpoint every 10 iters anyway
    if epoch % 10 == 0:
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
